"""检索层级化 —— 父子分块（small-to-big）。

问题：块切得小，检索准，但交给大模型的上下文不完整；块切得大，向量被无关内容稀释，检索变差。
做法：
  1. 每篇文档先切成"父块"（~1200 字，接近一个小节），每个父块再切成"子块"（~300 字）；
  2. 只给子块建 BM25 + 向量索引（检索粒度细）；
  3. 检索到子块后映射回父块：同一父块的多个子块取最高分（max）或求和（sum），去重；
  4. 把父块交给重排和大模型（上下文完整）。
另一种层级：先用文档摘要/标题检索出相关文档，再在这些文档内部检索块（doc -> chunk 两阶段），这里没有实现。

评测（scripts/eval_hierarchical.py）：
  - qrels 标注在文档级别，所以把父块再按 max 聚合回文档，和"整篇文档建索引"（现在的做法）比 doc 级 nDCG@10 / Recall；
  - 同时记录送给大模型的前 k 条资料平均多少字：父子分块的好处是"检索按小块、给模型的是完整父块"，
    代价是上下文变长（或者被 max_chars_per_doc 截断）。
  - CMedQA 大部分文档是几百字的医生回答，一篇只有一个父块、一两个子块，差别主要来自几万篇长网页；
    在自己的长文档语料（手册、规章、论文）上效果差别会更明显。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..fusion import rrf_fuse
from ..text import chunk_text
from ..types import Hits


@dataclass
class Chunk:
    id: str
    parent_id: str
    text: str


def build_parent_child(
    docs: list[dict],
    parent_size: int = 1200,
    child_size: int = 300,
    child_overlap: int = 50,
) -> tuple[list[dict], list[Chunk]]:
    """docs 是 [{"id", "text", ...}]。返回 (父块列表, 子块列表)。

    约定：父块 {"id": f"{doc_id}#p{i}", "text": ..., "doc_id": doc_id, "title": 文档标题}；
    子块 Chunk(id=f"{parent_id}#c{j}", parent_id, text)。每个子块完整地落在一个父块内（是父块文本的子串）。
    父块之间不重叠（一段话只属于一个父块，聚合时不会重复计分）；子块之间有 child_overlap 字重叠，避免答案被切断。
    """
    if child_size > parent_size:
        raise ValueError("child_size 不能大于 parent_size")
    parents: list[dict] = []
    children: list[Chunk] = []
    for doc in docs:
        doc_id = str(doc["id"])
        for i, ptext in enumerate(chunk_text(doc.get("text") or "", parent_size, 0)):
            pid = f"{doc_id}#p{i}"
            parents.append({"id": pid, "text": ptext, "doc_id": doc_id, "title": doc.get("title") or ""})
            overlap = min(child_overlap, child_size - 1)
            for j, ctext in enumerate(chunk_text(ptext, child_size, overlap)):
                children.append(Chunk(id=f"{pid}#c{j}", parent_id=pid, text=ctext))
    return parents, children


def children_to_parents(child_hits: Hits, child_to_parent: dict[str, str], agg: str = "max", top_k: int = 20) -> Hits:
    """把子块的检索结果聚合成父块排序。agg="max" 取最高分；agg="sum" 求和（偏向命中多个子块的父块）。

    注意：BM25 分数和余弦分数量纲不同，聚合前先融合（RRF）成一个列表再调用本函数。
    同一个函数也能把父块聚合成文档（传 父块 -> 文档 的映射）。不在映射里的 id 原样保留。
    """
    if agg not in ("max", "sum"):
        raise ValueError(f"agg 只能是 max 或 sum，收到 {agg!r}")
    scores: dict[str, float] = {}
    for cid, s in child_hits:
        pid = child_to_parent.get(cid, cid)
        if pid not in scores:
            scores[pid] = float(s)
        elif agg == "max":
            scores[pid] = max(scores[pid], float(s))
        else:
            scores[pid] += float(s)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k] if top_k else ranked


class ParentChildRetriever:
    """子块上做 BM25 + Dense 混合检索 -> 聚合到父块 ->（可选）重排父块。

    child_bm25 / child_dense 是建在子块上的 rag.bm25.BM25Retriever / rag.dense.DenseRetriever，其中一个可以是 None（单路）。
    两路都有时先用 RRF 融合子块列表（weights 同 retrieval.weights，[bm25, dense]），再聚合 —— 聚合前分数必须同一量纲。
    reranker 不为 None 时，重排聚合后的前 rerank_top_n 个父块（用父块全文，和"给大模型的就是父块"一致），
    其余的按原顺序接在后面（同 rag.pipeline.rerank_head：头部是重排分、尾部是融合分，只有顺序有意义）。
    """

    def __init__(self, child_bm25, child_dense, child_to_parent: dict[str, str], parents: dict[str, dict],
                 rrf_k: int = 60, weights: list[float] | None = None, agg: str = "max",
                 reranker=None, rerank_top_n: int = 50):
        self.child_bm25 = child_bm25
        self.child_dense = child_dense
        self.child_to_parent = child_to_parent
        self.parents = parents
        self.rrf_k = rrf_k
        self.weights = weights
        self.agg = agg
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n

    def search_children(self, queries: list[str], child_k: int = 100) -> list[Hits]:
        """子块级的（融合后）排序。"""
        lists: list[list[Hits]] = []
        if self.child_bm25 is not None:
            lists.append(self.child_bm25.search(queries, child_k))
        if self.child_dense is not None:
            lists.append(self.child_dense.search(queries, child_k))
        if not lists:
            raise ValueError("child_bm25 和 child_dense 至少要有一个")
        if len(lists) == 1:
            return lists[0]
        return [rrf_fuse(list(per_q), k=self.rrf_k, weights=self.weights, top_k=child_k) for per_q in zip(*lists)]

    def search(self, queries: list[str], top_k: int = 20, child_k: int = 100) -> list[Hits]:
        out: list[Hits] = []
        keep = max(top_k, self.rerank_top_n) if self.reranker is not None else top_k
        for q, hits in zip(queries, self.search_children(queries, child_k)):
            parents = children_to_parents(hits, self.child_to_parent, self.agg, keep)
            if self.reranker is not None and parents:
                head = parents[: self.rerank_top_n]
                cands = [(pid, parent_text(self.parents[pid])) for pid, _ in head]
                parents = self.reranker.rerank([q], [cands])[0] + parents[self.rerank_top_n:]
            out.append(parents[:top_k])
        return out


def parent_text(parent: dict) -> str:
    """重排和给大模型时用的父块文本：有标题就拼在前面（同 rag.data.doc_text）。"""
    title = (parent.get("title") or "").strip()
    return f"{title}\n{parent['text']}" if title else parent["text"]


# ---------------------------------------------------------------- 子块索引的存放


def pc_index_dir(cfg: dict) -> Path:
    """子块索引目录：indexes/<数据集>_pc<父块>_<子块>_<重叠>/，里面是 bm25_<模式>/、dense/<模型>/ 和 parents.jsonl / children.jsonl。"""
    h = cfg.get("hierarchical", {}) or {}
    p, c, o = int(h.get("parent_size", 1200)), int(h.get("child_size", 300)), int(h.get("child_overlap", 50))
    return Path(f"{cfg['index_dir']}_pc{p}_{c}_{o}")


def child_corpus(parents: list[dict], children: list[Chunk]) -> list[dict]:
    """给子块建索引用的"语料"：格式同 corpus.jsonl；标题沿用文档标题（doc_text 会把它拼在前面）。"""
    titles = {p["id"]: p.get("title", "") for p in parents}
    return [{"id": c.id, "text": c.text, "title": titles.get(c.parent_id, "")} for c in children]


def save_parent_child(out_dir: str | Path, parents: list[dict], children: list[Chunk], params: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "parents.jsonl", "w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(out / "children.jsonl", "w", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps({"id": c.id, "parent_id": c.parent_id, "text": c.text}, ensure_ascii=False) + "\n")
    with open(out / "pc_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)


def load_parent_child(out_dir: str | Path) -> tuple[dict[str, dict], dict[str, str], dict]:
    """返回 (父块 id -> 父块, 子块 id -> 父块 id, 切块参数)。"""
    out = Path(out_dir)
    if not (out / "pc_params.json").exists():
        raise FileNotFoundError(f"找不到父子分块 {out}，先运行 python scripts/eval_hierarchical.py build")
    parents: dict[str, dict] = {}
    with open(out / "parents.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            parents[p["id"]] = p
    c2p: dict[str, str] = {}
    with open(out / "children.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            c2p[c["id"]] = c["parent_id"]
    with open(out / "pc_params.json", encoding="utf-8") as f:
        params = json.load(f)
    return parents, c2p, params


def chunk_stats(docs: list[dict], parents: list[dict], children: list[Chunk]) -> dict:
    """切块统计：文档长度分布、每篇文档几个父块、每个父块几个子块。"""
    import numpy as np

    lens = np.array([len(d.get("text") or "") for d in docs]) if docs else np.zeros(1)
    per_doc: dict[str, int] = defaultdict(int)
    for p in parents:
        per_doc[p["doc_id"]] += 1
    per_parent: dict[str, int] = defaultdict(int)
    for c in children:
        per_parent[c.parent_id] += 1
    return {
        "docs": len(docs), "parents": len(parents), "children": len(children),
        "doc_chars_p50": float(np.percentile(lens, 50)), "doc_chars_p90": float(np.percentile(lens, 90)),
        "doc_chars_p99": float(np.percentile(lens, 99)), "doc_chars_mean": float(lens.mean()),
        "docs_multi_parent_rate": sum(1 for v in per_doc.values() if v > 1) / max(1, len(per_doc)),
        "parents_multi_child_rate": sum(1 for v in per_parent.values() if v > 1) / max(1, len(per_parent)),
        "children_per_doc": len(children) / max(1, len(docs)),
    }
