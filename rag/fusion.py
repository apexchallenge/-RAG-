"""多路召回融合：把 BM25 和向量检索的结果合成一个列表。

RRF（Reciprocal Rank Fusion, Cormack et al. 2009）：
    score(d) = Σ_i  w_i / (k + rank_i(d))      rank 从 1 开始，k 通常取 60
只用排名、不用原始分数，所以不用管 BM25 分数（0~几十）和余弦相似度（-1~1）量纲不同。

加权融合：每路先做 min-max 归一化到 [0,1]，再按权重相加。权重要在 dev 集上调，不能在 test 上调。
"""

from __future__ import annotations

from collections import defaultdict

from .types import Hits


def _sorted(scores: dict[str, float], top_k: int | None) -> Hits:
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # 分数相同时按 id 排，结果可复现
    return ranked[:top_k] if top_k else ranked


def rrf_fuse(ranked_lists: list[Hits], k: int = 60, weights: list[float] | None = None, top_k: int | None = None) -> Hits:
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights 的个数要和召回路数一致")
    scores: dict[str, float] = defaultdict(float)
    for li, hits in enumerate(ranked_lists):
        w = 1.0 if weights is None else float(weights[li])
        for rank, (doc_id, _) in enumerate(hits, start=1):
            scores[doc_id] += w / (k + rank)
    return _sorted(scores, top_k)


def weighted_fuse(ranked_lists: list[Hits], weights: list[float], top_k: int | None = None) -> Hits:
    if len(weights) != len(ranked_lists):
        raise ValueError("weights 的个数要和召回路数一致")
    scores: dict[str, float] = defaultdict(float)
    for hits, w in zip(ranked_lists, weights):
        if not hits:
            continue
        values = [s for _, s in hits]
        lo, hi = min(values), max(values)
        for doc_id, s in hits:
            norm = (s - lo) / (hi - lo) if hi > lo else 1.0
            scores[doc_id] += float(w) * norm
    return _sorted(scores, top_k)


def fuse(retrieval_cfg: dict, bm25_hits: Hits, dense_hits: Hits, method: str | None = None) -> Hits:
    """按配置融合两路结果。method 不传时用 retrieval.fusion。"""
    method = method or retrieval_cfg.get("fusion", "rrf")
    w = retrieval_cfg.get("weights", {}) or {}
    weights = [float(w.get("bm25", 0.5)), float(w.get("dense", 0.5))]
    top_k = int(retrieval_cfg.get("fused_top_k", 100))
    if method == "rrf":
        rrf_weights = weights if retrieval_cfg.get("rrf_use_weights", False) else None
        return rrf_fuse([bm25_hits, dense_hits], k=int(retrieval_cfg.get("rrf_k", 60)), weights=rrf_weights, top_k=top_k)
    if method == "weighted":
        return weighted_fuse([bm25_hits, dense_hits], weights, top_k=top_k)
    raise ValueError(f"未知融合方式 {method!r}，可选 rrf / weighted")
