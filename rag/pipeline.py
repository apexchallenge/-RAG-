"""在线问答流程：问题 -> BM25 + 向量双路召回 -> 融合 -> 重排 -> 拼提示词 -> 大模型回答。"""

from __future__ import annotations

from .data import doc_text, load_corpus
from .embedder import Embedder
from .fusion import fuse
from .generator import Generator
from .indexing import load_bm25, load_dense
from .reranker import Reranker
from .types import Hits
from .utils import timer


def rerank_head(reranker: Reranker, queries: list[str], ranked_lists: list[Hits], docs: dict, top_n: int) -> list[Hits]:
    """只重排每个列表的前 top_n 个；其余的按原顺序接在后面（分数仍是融合分，只表示顺序）。

    这样 Recall@100 这类"深层"指标和重排前保持可比，重排只影响前 top_n 的顺序。
    """
    candidates = [[(d, doc_text(docs[d])) for d, _ in hits[:top_n]] for hits in ranked_lists]
    heads = reranker.rerank(queries, candidates)
    return [head + list(hits[top_n:]) for head, hits in zip(heads, ranked_lists)]


class RAGPipeline:
    def __init__(
        self,
        cfg: dict,
        embedder=None,
        reranker=None,
        generator=None,
        use_reranker: bool = True,
        bm25=None,
        dense=None,
        docs: dict | None = None,
    ):
        self.cfg = cfg
        self.docs = docs if docs is not None else {d["id"]: d for d in load_corpus(cfg["data_dir"])}
        self.bm25 = bm25 or load_bm25(cfg)
        self.embedder = embedder or (dense.embedder if dense is not None else Embedder.from_config(cfg["embedder"]))
        self.dense = dense or load_dense(cfg, self.embedder)
        rcfg = cfg.get("reranker", {}) or {}
        if reranker is not None:
            self.reranker = reranker
        elif use_reranker and rcfg.get("enabled", True):
            self.reranker = Reranker.from_config(rcfg)
        else:
            self.reranker = None
        self.generator = generator or Generator(cfg.get("generation"))

    def retrieve(self, question: str) -> dict:
        r = self.cfg["retrieval"]
        t: dict[str, float] = {}
        with timer(t, "bm25"):
            bm25_hits = self.bm25.search([question], int(r.get("bm25_top_k", 100)))[0]
        with timer(t, "dense"):
            dense_hits = self.dense.search([question], int(r.get("dense_top_k", 100)))[0]
        with timer(t, "fusion"):
            fused = fuse(r, bm25_hits, dense_hits)
        final = fused
        if self.reranker is not None:
            top_n = int(self.cfg["reranker"].get("top_n", 50))
            with timer(t, "rerank"):
                final = rerank_head(self.reranker, [question], [fused], self.docs, top_n)[0]
        return {"bm25": bm25_hits, "dense": dense_hits, "fused": fused, "final": final, "timings": t}

    def ask(self, question: str) -> dict:
        res = self.retrieve(question)
        gcfg = self.cfg.get("generation", {}) or {}
        k = int(gcfg.get("context_top_k", 5))
        res["contexts"] = [{"id": d, "score": s, "text": doc_text(self.docs[d])} for d, s in res["final"][:k]]
        res["answer"] = None
        if self.generator.available:
            with timer(res["timings"], "llm"):
                res["answer"] = self.generator.answer(question, [c["text"] for c in res["contexts"]])
        return res

    def warmup(self) -> None:
        """第一次调用 GPU 有初始化开销（几十到几百毫秒），交互问答前先跑一次，耗时统计才准。"""
        for comp in (getattr(self.embedder, "warmup", None), getattr(self.reranker, "warmup", None)):
            if comp is not None:
                comp()

    def agent_llm(self):
        """agent 判断步骤（充分性检查、改写查询）用的大模型；思考模式和输出上限读 agent 配置。"""
        a = self.cfg.get("agent", {}) or {}
        return self.generator.as_llm(thinking=a.get("thinking", False), max_tokens=int(a.get("max_tokens", 1024)))

    def ask_with_check(self, question: str, exclude: set[str] | None = None) -> dict:
        """检索 -> 判断资料（充分/部分/部分（查不到）/无关）-> 值得的话补检索 -> 回答；只有无关才拒答（见 rag/advanced/query_rewrite.py）。"""
        from .advanced.query_rewrite import answer_with_check

        a = self.cfg.get("agent", {}) or {}
        return answer_with_check(self, self.agent_llm(), question, int(a.get("max_rounds", 2)), exclude=exclude,
                                 refine=a.get("refine", "hyde"),
                                 max_irrelevant_retries=int(a.get("max_irrelevant_retries", 1)),
                                 merge=a.get("merge", "slots"), new_slots=int(a.get("new_slots", 2)))
