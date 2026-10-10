"""离线评测：一次跑完 BM25 / Dense / Hybrid / +Rerank 的消融对比。

产出三样东西（简历和面试都要用）：
1. 消融表：每加一个组件，指标和耗时怎么变；
2. 失败案例：最终前 k 名里仍没有相关文档的查询，分成"召回失败 / 截断失败 / 排序失败"；
3. 在线延迟：逐条查询跑完整检索流程，统计 P50 / P95。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import dump_config
from .data import doc_text, load_corpus, select_queries
from .embedder import Embedder
from .fusion import fuse
from .indexing import load_bm25, load_dense
from .metrics import evaluate_run, hit_at_k
from .pipeline import rerank_head
from .reranker import Reranker
from .types import Hits
from .utils import timer

MODES = ("bm25", "dense", "hybrid_rrf", "hybrid_weighted", "dense_rerank", "hybrid_rerank")
LABELS = {
    "bm25": "BM25 (jieba)",
    "dense": "Dense (BGE + Faiss)",
    "hybrid_rrf": "Hybrid · RRF",
    "hybrid_weighted": "Hybrid · 加权",
    "dense_rerank": "Dense + Rerank",
    "hybrid_rerank": "Hybrid + Rerank",
}


@dataclass
class EvalResult:
    split: str
    qids: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    docs: dict[str, dict]
    fusion: str
    lists: dict[str, list[Hits]] = field(default_factory=dict)  # 模式 -> 每个查询的排序结果
    table: dict[str, dict[str, float]] = field(default_factory=dict)  # 模式 -> 指标
    bm25: object = None
    dense: object = None
    reranker: object = None


def run_eval(
    cfg: dict,
    embedder=None,
    reranker=None,
    modes=MODES,
    split: str | None = None,
    max_queries: int | None = None,
) -> EvalResult:
    ecfg = cfg.get("eval", {}) or {}
    split = split or ecfg.get("split", "test")
    if max_queries is None:
        max_queries = ecfg.get("max_queries")
    qids, queries, qrels = select_queries(cfg["data_dir"], split, max_queries, int(ecfg.get("seed", 42)))
    if not qids:
        raise ValueError(f"划分 {split!r} 里没有带标注的查询")
    texts = [queries[q] for q in qids]
    docs = {d["id"]: d for d in load_corpus(cfg["data_dir"])}
    r = cfg["retrieval"]
    rcfg = cfg.get("reranker", {}) or {}
    method = r.get("fusion", "rrf")

    modes = [m for m in MODES if m in modes]
    if any(m.endswith("_rerank") for m in modes) and reranker is None and not rcfg.get("enabled", True):
        print("[eval] reranker.enabled=false，跳过带重排的模式")
        modes = [m for m in modes if not m.endswith("_rerank")]
    res = EvalResult(split=split, qids=qids, queries=queries, qrels=qrels, docs=docs, fusion=method)
    t: dict[str, float] = {}
    lists = res.lists

    if any(m in modes for m in ("bm25", "hybrid_rrf", "hybrid_weighted", "hybrid_rerank")):
        res.bm25 = load_bm25(cfg)
        with timer(t, "bm25"):
            lists["bm25"] = res.bm25.search(texts, int(r.get("bm25_top_k", 100)))
    if any(m != "bm25" for m in modes):
        res.dense = load_dense(cfg, embedder or Embedder.from_config(cfg["embedder"]))
        res.dense.embedder.warmup()  # 预热放在计时之外，否则第一批查询的耗时会偏大
        with timer(t, "dense"):
            lists["dense"] = res.dense.search(texts, int(r.get("dense_top_k", 100)))
    for m in ("rrf", "weighted"):
        if f"hybrid_{m}" in modes or ("hybrid_rerank" in modes and method == m):
            with timer(t, f"fuse_{m}"):
                lists[f"hybrid_{m}"] = [fuse(r, b, d, method=m) for b, d in zip(lists["bm25"], lists["dense"])]
    if any(m.endswith("_rerank") for m in modes):
        res.reranker = reranker or Reranker.from_config(rcfg)
        res.reranker.warmup()
        top_n = int(rcfg.get("top_n", 50))
        if "dense_rerank" in modes:
            with timer(t, "rerank_dense"):
                lists["dense_rerank"] = rerank_head(res.reranker, texts, lists["dense"], docs, top_n)
        if "hybrid_rerank" in modes:
            with timer(t, "rerank_hybrid"):
                lists["hybrid_rerank"] = rerank_head(res.reranker, texts, lists[f"hybrid_{method}"], docs, top_n)

    # 每种模式的总耗时 = 它用到的各阶段耗时之和（批处理口径，除以查询数得到 ms/q）
    g = t.get
    cost = {
        "bm25": g("bm25", 0),
        "dense": g("dense", 0),
        "hybrid_rrf": g("bm25", 0) + g("dense", 0) + g("fuse_rrf", 0),
        "hybrid_weighted": g("bm25", 0) + g("dense", 0) + g("fuse_weighted", 0),
        "dense_rerank": g("dense", 0) + g("rerank_dense", 0),
        "hybrid_rerank": g("bm25", 0) + g("dense", 0) + g(f"fuse_{method}", 0) + g("rerank_hybrid", 0),
    }
    recall_ks = tuple(ecfg.get("recall_ks", (5, 10, 20, 100)))
    for m in modes:
        run = {q: [d for d, _ in hits] for q, hits in zip(qids, lists[m])}
        metrics = evaluate_run(run, qrels, recall_ks=recall_ks)
        metrics["ms/q"] = 1000.0 * cost[m] / len(qids)
        res.table[m] = metrics
    return res


def format_table(res: EvalResult) -> str:
    if not res.table:
        return "(没有结果)"
    cols = list(next(iter(res.table.values())).keys())
    lines = ["| 方法 | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
    for m, metrics in res.table.items():
        cells = [f"{v:.1f}" if c == "ms/q" else f"{100 * v:.1f}" for c, v in metrics.items()]
        lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _best_rank(hits: Hits, gold: set[str]) -> int | None:
    for i, (d, _) in enumerate(hits, start=1):
        if d in gold:
            return i
    return None


def default_mode(res: EvalResult) -> str | None:
    """本次评测里"最完整"的那条流水线，失败分析和延迟默认都看它。"""
    prefer = ("hybrid_rerank", "dense_rerank", f"hybrid_{res.fusion}", "hybrid_rrf", "hybrid_weighted", "dense", "bm25")
    return next((m for m in prefer if m in res.table), None)


def analyze_failures(res: EvalResult, cfg: dict, mode: str | None = None, k: int = 10, limit: int = 30):
    """找出最终前 k 名没有任何相关文档的查询，并判断问题出在哪一环。

    召回失败：候选列表里根本没有相关文档 -> 优化召回（分词词典、换/微调嵌入模型、加大召回 K）
    截断失败：相关文档召回到了，但排在 rerank top_n 之外 -> 调大 top_n 或改进融合
    排序失败：相关文档进了重排候选，却被排到 k 名以外 -> 重排模型/输入截断的问题
    """
    mode = mode or default_mode(res)
    if mode is None or mode not in res.lists:
        return {}, []
    source = {"hybrid_rerank": f"hybrid_{res.fusion}", "dense_rerank": "dense"}.get(mode)
    top_n = int((cfg.get("reranker") or {}).get("top_n", 50))
    stages = [s for s in ("bm25", "dense", f"hybrid_{res.fusion}", mode) if s in res.lists]
    stages = list(dict.fromkeys(stages))
    counts: Counter = Counter()
    rows = []
    for i, q in enumerate(res.qids):
        final = [d for d, _ in res.lists[mode][i]]
        if hit_at_k(final, res.qrels[q], k):
            continue
        gold = {d for d, s in res.qrels[q].items() if s > 0}
        ranks = {s: _best_rank(res.lists[s][i], gold) for s in stages}
        if source:  # 带重排的模式：看重排前的候选
            pre = ranks.get(source)
            category = "召回失败" if pre is None else ("截断失败" if pre > top_n else "排序失败")
        else:
            category = "召回失败" if ranks[mode] is None else "排序失败"
        counts[category] += 1
        if len(rows) < limit:
            rows.append({
                "qid": q,
                "query": res.queries[q],
                "category": category,
                "gold_rank": ranks,
                "gold": [{"id": d, "text": doc_text(res.docs[d])[:200]} for d in sorted(gold) if d in res.docs][:3],
                "top3": [
                    {"id": d, "score": round(s, 4), "text": doc_text(res.docs[d])[:200]}
                    for d, s in res.lists[mode][i][:3]
                ],
            })
    summary = {"mode": mode, "k": k, "failed": sum(counts.values()), "total": len(res.qids), **counts}
    return summary, rows


def measure_latency(cfg: dict, res: EvalResult, mode: str | None = None, n: int = 50, warmup: int = 3) -> dict[str, dict[str, float]]:
    """逐条查询跑某个模式（默认是评测里最完整的那条流水线）的检索阶段，统计各阶段和总耗时的 P50 / P95（毫秒）。

    不含大模型生成；只用 run_eval 已经加载好的组件。
    """
    mode = mode or default_mode(res)
    if mode is None:
        return {}
    r = cfg["retrieval"]
    top_n = int((cfg.get("reranker") or {}).get("top_n", 50))
    method = mode.split("_")[1] if mode in ("hybrid_rrf", "hybrid_weighted") else res.fusion
    use_bm25 = mode == "bm25" or mode.startswith("hybrid")
    use_dense = mode != "bm25"
    questions = [res.queries[q] for q in res.qids[: n + warmup]]
    samples: dict[str, list[float]] = defaultdict(list)
    for i, question in enumerate(questions):
        t: dict[str, float] = {}
        if use_bm25:
            with timer(t, "bm25"):
                bm25_hits = res.bm25.search([question], int(r.get("bm25_top_k", 100)))[0]
        if use_dense:
            with timer(t, "dense"):
                dense_hits = res.dense.search([question], int(r.get("dense_top_k", 100)))[0]
        hits = None
        if use_bm25 and use_dense:
            with timer(t, "fusion"):
                hits = fuse(r, bm25_hits, dense_hits, method=method)
        if mode.endswith("_rerank"):
            with timer(t, "rerank"):
                rerank_head(res.reranker, [question], [hits if hits is not None else dense_hits], res.docs, top_n)
        if i < warmup:  # 预热：首次调用 GPU 有初始化开销，不计入
            continue
        for stage, sec in t.items():
            samples[stage].append(1000 * sec)
        samples["total"].append(1000 * sum(t.values()))
    return {
        s: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n": len(v)}
        for s, v in samples.items()
    }


def save_results(
    res: EvalResult,
    cfg: dict,
    out_dir: str | Path = "results",
    tag: str | None = None,
    failures: tuple[dict, list] | None = None,
    latency: dict | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{tag or Path(cfg['data_dir']).name}_{datetime.now():%Y%m%d-%H%M%S}"
    name, i = base, 2
    while (out / f"{name}.md").exists():  # 同一秒内跑两次也不会互相覆盖
        name, i = f"{base}-{i}", i + 1
    summary = {
        "split": res.split,
        "n_queries": len(res.qids),
        "n_docs": len(res.docs),
        "table": res.table,
        "failures": failures[0] if failures else None,
        "latency_ms": latency,
        "config": cfg,
    }
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = [
        f"# 评测结果 {name}",
        "",
        f"- 数据：`{cfg['data_dir']}`，划分 `{res.split}`，{len(res.qids)} 条查询，{len(res.docs)} 篇文档",
        f"- 嵌入：`{cfg['embedder']['model']}`；Faiss：`{cfg['faiss'].get('index_type')}`；融合：`{res.fusion}`；"
        f"重排：`{(cfg.get('reranker') or {}).get('model')}` top_n={(cfg.get('reranker') or {}).get('top_n')}",
        "",
        format_table(res),
        "",
        "指标 ×100；ms/q 为批处理下平均每条查询耗时。",
    ]
    if latency:
        md += ["", f"## 在线延迟（{LABELS.get(default_mode(res), '')}，逐条查询，ms）", "",
               "| 阶段 | P50 | P95 |", "| --- | --- | --- |"]
        md += [f"| {s} | {v['p50']:.1f} | {v['p95']:.1f} |" for s, v in latency.items()]
    if failures and failures[0]:
        s, rows = failures
        md += ["", f"## 失败分析（{s.get('mode')}，前 {s.get('k')} 名无相关文档）", ""]
        md += [f"- {key}: {val}" for key, val in s.items() if key not in ("mode", "k")]
        with open(out / f"{name}_failures.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md += ["", "<details><summary>完整配置</summary>", "", "```yaml", dump_config(cfg), "```", "</details>"]
    (out / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    return out / f"{name}.md"
