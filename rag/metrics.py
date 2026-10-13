"""检索评测指标（和 BEIR / MTEB 用的 pytrec_eval 口径一致：nDCG 用线性增益）。

    Recall@k   前 k 个结果里找回了多少比例的相关文档        —— 召回阶段看它
    MRR@k      第一个相关文档排在第几（取倒数），没进前 k 记 0  —— 用户只看第一条时看它
    nDCG@k     考虑相关文档的位置和相关等级，越靠前越好        —— 精排阶段看它
    Hit@k      前 k 个里是否至少有一个相关文档
    P@k        前 k 个里相关文档的比例（补标之后才有意义：原始 qrels 每题只有 1~2 个正例）
"""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    rel = {d for d, s in relevant.items() if s > 0}
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / len(rel)


def hit_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    return 1.0 if any(relevant.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if relevant.get(d, 0) > 0) / k


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if relevant.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if relevant.get(d, 0) > 0)
    ideal = sorted((s for s in relevant.values() if s > 0), reverse=True)[:k]
    idcg = sum(s / math.log2(i + 1) for i, s in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


METRICS = {"recall": recall_at_k, "hit": hit_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k, "precision": precision_at_k}
_NAMES = {"r": "recall", "recall": "recall", "hit": "hit", "mrr": "mrr", "ndcg": "ndcg", "p": "precision", "precision": "precision"}


def parse_metric(name: str) -> tuple[str, int]:
    """"nDCG@10" -> ("ndcg", 10)；"R@100" -> ("recall", 100)。"""
    if "@" not in name:
        raise ValueError(f"指标名应形如 nDCG@10 / R@10 / MRR@10 / Hit@10，收到 {name!r}")
    head, k = name.split("@", 1)
    key = _NAMES.get(head.strip().lower())
    if key is None or not k.strip().isdigit():
        raise ValueError(f"无法解析指标名 {name!r}")
    return key, int(k)


def per_query(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], metric: str = "nDCG@10") -> dict[str, float]:
    """每条查询各自的指标值（做显著性检验要用逐条结果，不能只看平均数）。"""
    key, k = parse_metric(metric)
    fn = METRICS[key]
    return {q: fn(run[q], qrels[q], k) for q in run if any(s > 0 for s in qrels.get(q, {}).values())}


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    recall_ks=(5, 10, 20, 100),
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float]:
    """run: {query_id: [doc_id 按排名]}。返回各指标在所有查询上的平均值。"""
    qids = [q for q in run if any(s > 0 for s in qrels.get(q, {}).values())]
    if not qids:
        raise ValueError("没有可评测的查询（qrels 里找不到这些查询的相关文档）")
    out: dict[str, float] = {}
    for k in recall_ks:
        out[f"R@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in qids) / len(qids)
    out[f"MRR@{mrr_k}"] = sum(mrr_at_k(run[q], qrels[q], mrr_k) for q in qids) / len(qids)
    out[f"nDCG@{ndcg_k}"] = sum(ndcg_at_k(run[q], qrels[q], ndcg_k) for q in qids) / len(qids)
    return out
