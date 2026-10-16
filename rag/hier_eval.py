"""父子分块（rag/advanced/hierarchical.py）的建索引和评测：和"整篇文档建索引"（现在的做法）在同一批查询上比 doc 级指标。

模式（都输出文档级排序，用原始 qrels 算指标）：
  doc_dense / doc_hybrid / doc_hybrid_rerank   整篇文档建索引（就是 evaluate.py 里的 dense / hybrid / hybrid+rerank）
  pc_dense  / pc_hybrid  / pc_hybrid_rerank    子块建索引 -> 聚合到父块（max / sum）->（重排父块）-> 父块按 max 聚合回文档

另外统计"送给大模型的前 k 条资料"：平均字数（截断前 / 按 max_chars_per_doc 截断后）、
前 k 条里有没有相关文档（文档模式看文档，父子模式看父块所属文档）。
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path

from .advanced.hierarchical import (
    ParentChildRetriever, build_parent_child, child_corpus, chunk_stats, load_parent_child,
    parent_text, pc_index_dir, save_parent_child,
)
from .data import doc_text, load_corpus, select_queries
from .fusion import fuse
from .indexing import build_bm25, build_dense, load_bm25, load_dense
from .metrics import evaluate_run, hit_at_k, per_query
from .pipeline import rerank_head
from .stats import format_test, paired_test

MODES = ("doc_dense", "doc_hybrid", "doc_hybrid_rerank", "pc_dense", "pc_hybrid", "pc_hybrid_rerank")
LABELS = {
    "doc_dense": "整篇 · Dense", "doc_hybrid": "整篇 · Hybrid", "doc_hybrid_rerank": "整篇 · Hybrid + Rerank",
    "pc_dense": "父子 · Dense", "pc_hybrid": "父子 · Hybrid", "pc_hybrid_rerank": "父子 · Hybrid + Rerank",
}


def pc_params(cfg: dict) -> dict:
    h = cfg.get("hierarchical", {}) or {}
    return {"parent_size": int(h.get("parent_size", 1200)), "child_size": int(h.get("child_size", 300)),
            "child_overlap": int(h.get("child_overlap", 50))}


def child_cfg(cfg: dict) -> dict:
    """子块索引用的配置：只把 index_dir 换成子块目录，其余（分词、模型、Faiss、融合）和整篇文档完全一样。"""
    sub = copy.deepcopy(cfg)
    sub["index_dir"] = str(pc_index_dir(cfg))
    return sub


def build_pc(cfg: dict, embedder=None, only: str | None = None, corpus: list[dict] | None = None,
             reencode: bool = False, show_progress: bool = True) -> dict:
    """切父子块、存 parents.jsonl / children.jsonl，给子块建 BM25 和向量索引。返回切块统计。"""
    corpus = corpus if corpus is not None else load_corpus(cfg["data_dir"])
    params = pc_params(cfg)
    t0 = time.perf_counter()
    parents, children = build_parent_child(corpus, **params)
    stats = chunk_stats(corpus, parents, children)
    stats["chunk_seconds"] = time.perf_counter() - t0
    sub = child_cfg(cfg)
    save_parent_child(sub["index_dir"], parents, children, {**params, "data_dir": str(cfg["data_dir"]), **stats})
    ccorpus = child_corpus(parents, children)
    if only in (None, "bm25"):
        build_bm25(sub, ccorpus, show_progress=show_progress)
    if only in (None, "dense"):
        if embedder is None:
            from .embedder import Embedder

            embedder = Embedder.from_config(cfg["embedder"])
        build_dense(sub, embedder, ccorpus, reencode=reencode, show_progress=show_progress)
    return stats


def _doc_level(parent_hits, parents: dict[str, dict], k: int):
    """父块排序 -> 文档排序：每篇文档取它排得最靠前的父块的位置（= 按顺序去重）。
    不按分数取 max：重排后的列表头部是重排分、尾部是融合分，量纲不同，只有顺序可信。"""
    out, seen = [], set()
    for pid, s in parent_hits:
        d = parents[pid]["doc_id"]
        if d not in seen:
            seen.add(d)
            out.append((d, s))
    return out[:k]


def run_pc_eval(cfg: dict, embedder=None, reranker=None, modes=MODES, split: str = "dev",
                max_queries: int | None = None, agg: str | None = None, child_k: int | None = None) -> dict:
    """跑各模式，返回 {"qids", "qrels", "lists": 模式 -> 每条查询的文档级排序, "contexts": 模式 -> 资料统计, "seconds"}。"""
    from .embedder import Embedder

    h = cfg.get("hierarchical", {}) or {}
    agg = agg or h.get("agg", "max")
    child_k = int(child_k or h.get("child_k", 200))
    r = cfg["retrieval"]
    g = cfg.get("generation", {}) or {}
    ctx_k, max_chars = int(g.get("context_top_k", 5)), int(g.get("max_chars_per_doc", 600))
    rcfg = cfg.get("reranker", {}) or {}
    top_n = int(rcfg.get("top_n", 50))
    depth = int(r.get("fused_top_k", 100))
    modes = [m for m in MODES if m in modes]
    if any(m.endswith("_rerank") for m in modes) and reranker is None:
        if rcfg.get("enabled", True):
            from .reranker import Reranker

            reranker = Reranker.from_config(rcfg)
        else:
            modes = [m for m in modes if not m.endswith("_rerank")]

    qids, queries, qrels = select_queries(cfg["data_dir"], split, max_queries, int((cfg.get("eval") or {}).get("seed", 42)))
    if not qids:
        raise ValueError(f"划分 {split!r} 里没有带标注的查询")
    texts = [queries[q] for q in qids]
    docs = {d["id"]: d for d in load_corpus(cfg["data_dir"])}
    embedder = embedder or Embedder.from_config(cfg["embedder"])
    q_emb = embedder.encode_queries(texts)

    lists: dict[str, list] = {}
    contexts: dict[str, dict] = {}
    secs: dict[str, float] = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        secs[name] = time.perf_counter() - t0
        return out

    # ---- 整篇文档
    if any(m.startswith("doc_") for m in modes):
        bm25, dense = load_bm25(cfg), load_dense(cfg, embedder)
        d_hits = timed("doc_dense", lambda: dense.search(texts, int(r.get("dense_top_k", 100)), query_emb=q_emb))
        lists["doc_dense"] = d_hits
        if "doc_hybrid" in modes or "doc_hybrid_rerank" in modes:
            b_hits = timed("doc_bm25", lambda: bm25.search(texts, int(r.get("bm25_top_k", 100))))
            lists["doc_hybrid"] = timed("doc_fuse", lambda: [fuse(r, b, d) for b, d in zip(b_hits, d_hits)])
            secs["doc_hybrid"] = secs["doc_dense"] + secs["doc_bm25"] + secs["doc_fuse"]
        if "doc_hybrid_rerank" in modes:
            lists["doc_hybrid_rerank"] = timed(
                "doc_rerank", lambda: rerank_head(reranker, texts, lists["doc_hybrid"], docs, top_n))
            secs["doc_hybrid_rerank"] = secs["doc_hybrid"] + secs["doc_rerank"]
        for m in ("doc_dense", "doc_hybrid", "doc_hybrid_rerank"):
            if m in lists:
                ctx = [[doc_text(docs[d]) for d, _ in hits[:ctx_k]] for hits in lists[m]]
                hit = [hit_at_k([d for d, _ in hits], qrels[q], ctx_k) for q, hits in zip(qids, lists[m])]
                contexts[m] = _ctx_stats(ctx, hit, max_chars)

    # ---- 父子分块
    if any(m.startswith("pc_") for m in modes):
        sub = child_cfg(cfg)
        parents, c2p, params = load_parent_child(sub["index_dir"])
        if params.get("data_dir") and Path(params["data_dir"]) != Path(cfg["data_dir"]):
            raise ValueError(f"子块索引是从 {params['data_dir']} 切出来的，当前数据是 {cfg['data_dir']}")
        c_bm25, c_dense = load_bm25(sub), load_dense(sub, embedder)
        weights = None
        if r.get("rrf_use_weights", False):
            w = r.get("weights", {}) or {}
            weights = [float(w.get("bm25", 0.5)), float(w.get("dense", 0.5))]
        rrf_k = int(r.get("rrf_k", 60))

        class _QE:  # 子块 dense 检索复用已经编码好的查询向量
            def search(self, qs, k):
                return c_dense.search(qs, k, query_emb=q_emb)

        variants = {
            "pc_dense": ParentChildRetriever(None, _QE(), c2p, parents, rrf_k, None, agg),
            "pc_hybrid": ParentChildRetriever(c_bm25, _QE(), c2p, parents, rrf_k, weights, agg),
            "pc_hybrid_rerank": ParentChildRetriever(c_bm25, _QE(), c2p, parents, rrf_k, weights, agg,
                                                     reranker=reranker, rerank_top_n=top_n),
        }
        for m, ret in variants.items():
            if m not in modes:
                continue
            p_hits = timed(m, lambda ret=ret: ret.search(texts, top_k=depth, child_k=child_k))
            lists[m] = [_doc_level(ph, parents, depth) for ph in p_hits]
            ctx = [[parent_text(parents[pid]) for pid, _ in ph[:ctx_k]] for ph in p_hits]
            hit = [hit_at_k([parents[pid]["doc_id"] for pid, _ in ph], qrels[q], ctx_k) for q, ph in zip(qids, p_hits)]
            contexts[m] = _ctx_stats(ctx, hit, max_chars)
    return {"qids": qids, "qrels": qrels, "lists": {m: lists[m] for m in modes if m in lists},
            "contexts": contexts, "seconds": secs, "agg": agg, "child_k": child_k, "split": split,
            "ctx_k": ctx_k, "max_chars": max_chars}


def _ctx_stats(ctx: list[list[str]], hit: list[float], max_chars: int) -> dict:
    n = max(1, len(ctx))
    return {
        "chars": sum(sum(len(t) for t in c) for c in ctx) / n,
        "chars_truncated": sum(sum(min(len(t), max_chars) for t in c) for c in ctx) / n,
        "truncated_rate": sum(sum(int(len(t) > max_chars) for t in c) for c in ctx) / max(1, sum(len(c) for c in ctx)),
        "gold_in_context": sum(hit) / n,
    }


def summarize(res: dict, recall_ks=(5, 10, 20, 100), baseline: str = "doc_dense", metric: str = "nDCG@10") -> dict:
    qids, qrels = res["qids"], res["qrels"]
    runs = {m: {q: [d for d, _ in hits] for q, hits in zip(qids, lists)} for m, lists in res["lists"].items()}
    table = {}
    for m, run in runs.items():
        row = evaluate_run(run, qrels, recall_ks=recall_ks)
        row["ms/q"] = 1000 * res["seconds"].get(m, 0.0) / len(qids)
        table[m] = row
    tests = {}
    pairs = [(baseline, m) for m in runs if m != baseline]
    # 同一档（dense / hybrid / rerank）的整篇 vs 父子，是最直接的对比
    for a, b in (("doc_hybrid", "pc_hybrid"), ("doc_hybrid_rerank", "pc_hybrid_rerank")):
        if a in runs and b in runs and (a, b) not in pairs:
            pairs.append((a, b))
    for a, b in pairs:
        if a in runs:
            tests[f"{b} vs {a}"] = paired_test(per_query(runs[a], qrels, metric), per_query(runs[b], qrels, metric),
                                               n_resamples=5000)
    return {"table": table, "tests": tests, "metric": metric, "contexts": res["contexts"]}


def render(summary: dict, res: dict, cfg: dict, stats: dict | None, tag: str) -> str:
    p = pc_params(cfg)
    md = [f"# 父子分块评测 {tag}", "",
          f"- 数据：`{cfg['data_dir']}`，划分 `{res['split']}`，{len(res['qids'])} 条查询；嵌入 `{cfg['embedder']['model']}`",
          f"- 父块 {p['parent_size']} 字 / 子块 {p['child_size']} 字（重叠 {p['child_overlap']}）；子块召回 {res['child_k']} 条，"
          f"子块 -> 父块聚合 `{res['agg']}`，父块 -> 文档取排得最靠前的父块；融合和重排参数同整篇文档"]
    if stats:
        md.append(f"- 切块：{stats['docs']} 篇文档 -> {stats['parents']} 个父块 / {stats['children']} 个子块"
                  f"（每篇平均 {stats['children_per_doc']:.2f} 个子块；文档字数 P50 {stats['doc_chars_p50']:.0f} / "
                  f"P90 {stats['doc_chars_p90']:.0f} / P99 {stats['doc_chars_p99']:.0f}；"
                  f"{100 * stats['docs_multi_parent_rate']:.1f}% 的文档切出多个父块，"
                  f"{100 * stats['parents_multi_child_rate']:.1f}% 的父块有多个子块）")
    cols = list(next(iter(summary["table"].values())).keys())
    md += ["", "| 方法 | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
    for m, row in summary["table"].items():
        md.append(f"| {LABELS.get(m, m)} | " + " | ".join(f"{v:.1f}" if c == "ms/q" else f"{100 * v:.1f}"
                                                          for c, v in row.items()) + " |")
    md += ["", "指标 ×100，文档级（原始 qrels）；ms/q 为批处理下平均每条查询的检索耗时（不含查询编码）。", "",
           f"## 显著性（{summary['metric']}，配对随机化检验 + bootstrap 95% CI）", ""]
    md += [f"- {LABELS.get(k.split(' vs ')[0])} vs {LABELS.get(k.split(' vs ')[1])}：{format_test(t)}"
           for k, t in summary["tests"].items()]
    md += ["", f"## 送给大模型的前 {res['ctx_k']} 条资料", "",
           f"| 方法 | 平均总字数 | 截断到 {res['max_chars']} 字/条后 | 被截断的条目 | 前 {res['ctx_k']} 条含相关文档 |",
           "| --- | --- | --- | --- | --- |"]
    for m, c in summary["contexts"].items():
        md.append(f"| {LABELS.get(m, m)} | {c['chars']:.0f} | {c['chars_truncated']:.0f} | "
                  f"{100 * c['truncated_rate']:.1f}% | {100 * c['gold_in_context']:.1f}% |")
    md += ["", "怎么读：父子分块想要的是\"检索指标不降、给模型的资料更完整（少被截断）\"。",
           "整篇模式下长文档会被截断到前若干字，答案在后半段时模型看不到；父子模式给的是命中的那一节。",
           "注意：父子模式的前 k 条可能来自同一篇文档的不同父块，\"含相关文档\"按父块所属文档算。", ""]
    return "\n".join(md)


def save(summary: dict, res: dict, cfg: dict, stats: dict | None, out_dir: str | Path, tag: str | None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"hier_{tag + '_' if tag else ''}{datetime.now():%Y%m%d-%H%M%S}"
    md = render(summary, res, cfg, stats, name)
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"split": res["split"], "n_queries": len(res["qids"]), "params": pc_params(cfg), "agg": res["agg"],
                   "child_k": res["child_k"], "chunk_stats": stats, **summary}, f, ensure_ascii=False, indent=2)
    return out / f"{name}.md"
