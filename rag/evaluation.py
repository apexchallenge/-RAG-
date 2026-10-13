"""离线评测：一次跑完 BM25 / Dense / Hybrid / +Rerank 的消融对比。

产出三样东西（简历和面试都要用）：
1. 消融表：每加一个组件，指标和耗时怎么变；
2. 失败案例：最终前 k 名里仍没有相关文档的查询，分成"召回失败 / 截断失败 / 排序失败"；
3. 在线延迟：逐条查询跑完整检索流程，统计 P50 / P95。
"""

from __future__ import annotations

import copy
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
from .metrics import evaluate_run, hit_at_k, per_query
from .pipeline import rerank_head
from .reranker import Reranker
from .runs import write_runs
from .stats import format_test, paired_test
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


def _within(hits: Hits, gold: set[str], window: int) -> bool:
    r = _best_rank(hits, gold)
    return r is not None and r <= window


def default_mode(res: EvalResult) -> str | None:
    """本次评测里"最完整"的那条流水线，失败分析和延迟默认都看它。"""
    prefer = ("hybrid_rerank", "dense_rerank", f"hybrid_{res.fusion}", "hybrid_rrf", "hybrid_weighted", "dense", "bm25")
    return next((m for m in prefer if m in res.table), None)


# 失败类别（按"问题出在哪一环"排列）
FAILURE_CATEGORIES = ("召回失败", "融合丢失", "截断失败", "重排挤出", "排序失败")


def classify_failure(ranks: dict[str, int | None], mode: str, fusion: str, k: int, top_n: int) -> str:
    """根据正确答案在各阶段的最好名次，判断失败出在哪一环。ranks 里是 None 表示该阶段的列表里没有它。

    召回失败：所有单路召回都没找到
    融合丢失：至少一路召回把它排进了"窗口"（重排模式是 top_n，否则是 k），融合后却掉出了窗口
              —— 说明融合方式/权重有问题，单用那一路反而不会丢
    截断失败：（重排模式）召回到了，但排在重排窗口 top_n 之外，且没有哪一路排进窗口
    重排挤出：（重排模式）重排前已经在前 k 名，重排后被挤出前 k
    排序失败：其余情况 —— 在窗口内但始终没进前 k（重排没把它提上来，或单路/融合本身排得太靠后）
    """
    channels = [c for c in ("bm25", "dense") if c in ranks and (mode.startswith("hybrid") or mode.startswith(c))]
    best_channel = min((ranks[c] for c in channels if ranks[c] is not None), default=None)
    if best_channel is None:
        return "召回失败"
    if mode.endswith("_rerank"):
        source = "dense" if mode == "dense_rerank" else f"hybrid_{fusion}"
        pre = ranks.get(source)
        if pre is None or pre > top_n:
            if mode.startswith("hybrid") and best_channel <= top_n:
                return "融合丢失"
            return "截断失败"
        return "重排挤出" if pre <= k else "排序失败"
    if mode.startswith("hybrid"):
        final = ranks.get(mode)
        if (final is None or final > k) and best_channel <= k:
            return "融合丢失"
    return "排序失败"


def analyze_failures(res: EvalResult, cfg: dict, mode: str | None = None, k: int = 10, limit: int | None = None):
    """找出最终前 k 名没有任何相关文档的查询，判断问题出在哪一环（类别见 classify_failure）。

    limit=None 保存全部失败案例；每条记录存全文（不截断），方便事后人工或大模型复核。
    处理建议：
      召回失败 -> 查询改写/清洗、分词词典、换/微调嵌入模型、加大召回 K
      融合丢失 -> 调融合权重（scripts/tune_fusion.py），或在这个数据集上不用弱的那一路
      截断失败 -> 调大 reranker.top_n
      重排挤出 -> 重排模型偏好有问题（偏长文本、广告），可以把重排分和召回分加权融合；也可能是标注不全
      排序失败 -> 重排/排序模型能力不足，或正确答案本身和问题不太对得上
    """
    mode = mode or default_mode(res)
    if mode is None or mode not in res.lists:
        return {}, []
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
        category = classify_failure(ranks, mode, res.fusion, k, top_n)
        counts[category] += 1
        if limit is None or len(rows) < limit:
            rows.append({
                "qid": q,
                "query": res.queries[q],
                "category": category,
                "gold_rank": ranks,
                "gold": [
                    {"id": d, "rank": {s: _best_rank(res.lists[s][i], {d}) for s in stages}, "text": doc_text(res.docs[d])}
                    for d in sorted(gold) if d in res.docs
                ],
                "top3": [
                    {"id": d, "score": round(s, 4), "text": doc_text(res.docs[d])}
                    for d, s in res.lists[mode][i][:3]
                ],
                "top10_ids": [d for d, _ in res.lists[mode][i][:k]],
            })
    summary = {"mode": mode, "k": k, "failed": sum(counts.values()), "total": len(res.qids)}
    summary.update({c: counts[c] for c in FAILURE_CATEGORIES if counts[c]})
    return summary, rows


def fusion_loss(res: EvalResult, fused_mode: str, window: int) -> dict:
    """融合丢失率：有多少查询，某一路召回把正确答案排进了前 window，融合后的前 window 里却没有它。

    window 一般取重排的 top_n（没进重排窗口，重排再好也救不回来）。也统计反方向：融合"救回"了多少。
    """
    lost = rescued = 0
    for i, q in enumerate(res.qids):
        gold = {d for d, s in res.qrels[q].items() if s > 0}
        in_channel = any(_within(res.lists[c][i], gold, window) for c in ("bm25", "dense") if c in res.lists)
        in_fused = _within(res.lists[fused_mode][i], gold, window)
        lost += int(in_channel and not in_fused)
        rescued += int(in_fused and not in_channel)
    n = len(res.qids)
    return {"window": window, "lost": lost, "lost_rate": lost / n, "rescued": rescued, "n": n}


def significance(res: EvalResult, baseline: str | None = None, metric: str = "nDCG@10",
                 n_resamples: int = 10000, seed: int = 42) -> dict:
    """每个模式相对 baseline（默认 dense，没有就取第一个）的配对显著性检验。"""
    modes = list(res.table)
    if len(modes) < 2:
        return {}
    baseline = baseline if baseline in res.table else ("dense" if "dense" in res.table else modes[0])
    runs = {m: {q: [d for d, _ in hits] for q, hits in zip(res.qids, res.lists[m])} for m in modes}
    base = per_query(runs[baseline], res.qrels, metric)
    out = {"baseline": baseline, "metric": metric, "tests": {}}
    for m in modes:
        if m != baseline:
            out["tests"][m] = paired_test(base, per_query(runs[m], res.qrels, metric), n_resamples, seed)
    return out


FUSION_GRID = {
    "rrf_k": (10, 30, 60),
    "bm25_weight": (0.0, 0.1, 0.2, 0.3, 0.5),  # dense 权重 = 1 - bm25_weight；0.5 且不加权就是原来的等权 RRF
}


def tune_fusion(res: EvalResult, cfg: dict, metric: str = "nDCG@10", grid: dict | None = None,
                window: int | None = None, n_resamples: int = 2000) -> list[dict]:
    """在已经算好的 BM25 / dense 结果上，枚举融合方式和参数，返回按 metric 从高到低排序的结果。

    融合几乎不耗时，所以召回只跑一次，几十组参数几秒钟就比完。只应该在 dev 上用！
    每组结果都带：主要指标、R@100、融合丢失率、相对 dense 的配对检验，以及可以直接复制的 --set 参数。
    """
    if "bm25" not in res.lists or "dense" not in res.lists:
        raise ValueError("tune_fusion 需要先跑 bm25 和 dense 两路")
    grid = grid or FUSION_GRID
    r = cfg["retrieval"]
    window = window or int((cfg.get("reranker") or {}).get("top_n", 50))
    recall_ks = tuple((cfg.get("eval") or {}).get("recall_ks", (5, 10, 20, 100)))
    dense_run = {q: [d for d, _ in h] for q, h in zip(res.qids, res.lists["dense"])}
    dense_pq = per_query(dense_run, res.qrels, metric)

    candidates: list[tuple[str, dict, list[str]]] = [("dense only", {}, [])]
    for k_ in grid["rrf_k"]:
        for w in grid["bm25_weight"]:
            if w == 0.0:
                continue  # 权重 0 就等于只用 dense，上面已经有了
            rc = {**r, "fusion": "rrf", "rrf_k": k_, "rrf_use_weights": True, "weights": {"bm25": w, "dense": 1 - w}}
            sets = ["retrieval.fusion=rrf", f"retrieval.rrf_k={k_}", "retrieval.rrf_use_weights=true",
                    f"retrieval.weights={{bm25: {w:g}, dense: {1 - w:g}}}"]
            candidates.append((f"RRF k={k_} bm25={w:g}", rc, sets))
    for w in grid["bm25_weight"]:
        if w == 0.0:
            continue
        rc = {**r, "fusion": "weighted", "weights": {"bm25": w, "dense": 1 - w}}
        sets = ["retrieval.fusion=weighted", f"retrieval.weights={{bm25: {w:g}, dense: {1 - w:g}}}"]
        candidates.append((f"加权 bm25={w:g}", rc, sets))

    rows = []
    saved = dict(res.lists)
    try:
        for name, rc, sets in candidates:
            if not rc:
                fused = res.lists["dense"]
            else:
                fused = [fuse(rc, b, d) for b, d in zip(res.lists["bm25"], res.lists["dense"])]
            run = {q: [d for d, _ in h] for q, h in zip(res.qids, fused)}
            metrics = evaluate_run(run, res.qrels, recall_ks=recall_ks)
            res.lists["_tmp"] = fused
            loss = fusion_loss(res, "_tmp", window)
            test = paired_test(dense_pq, per_query(run, res.qrels, metric), n_resamples) if rc else None
            rows.append({"name": name, "metrics": metrics, "fusion_loss": loss, "vs_dense": test, "set": sets})
    finally:
        res.lists.clear()
        res.lists.update(saved)
    rows.sort(key=lambda x: -x["metrics"][metric])
    return rows


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


def snapshot_config(cfg: dict, res: EvalResult) -> dict:
    """存进结果文件的配置：把这次实际用的划分和查询数写进去（命令行 --split 覆盖的值原来不会出现在快照里）。"""
    snap = copy.deepcopy(cfg)
    ev = snap.setdefault("eval", {}) or {}
    snap["eval"] = ev
    ev["split"] = res.split
    ev["n_queries"] = len(res.qids)
    return snap


def save_results(
    res: EvalResult,
    cfg: dict,
    out_dir: str | Path = "results",
    tag: str | None = None,
    failures: tuple[dict, list] | None = None,
    latency: dict | None = None,
    sig: dict | None = None,
    runs_k: int = 100,
) -> Path:
    """写出 <name>.md / .json / _failures.jsonl，以及 runs_k > 0 时的 _runs.tsv.gz（每条查询每个模式的前 runs_k 名）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{tag or Path(cfg['data_dir']).name}_{datetime.now():%Y%m%d-%H%M%S}"
    name, i = base, 2
    while (out / f"{name}.md").exists():  # 同一秒内跑两次也不会互相覆盖
        name, i = f"{base}-{i}", i + 1
    snap = snapshot_config(cfg, res)
    top_n = int((cfg.get("reranker") or {}).get("top_n", 50))
    losses = {m: fusion_loss(res, m, top_n) for m in res.table if m in ("hybrid_rrf", "hybrid_weighted")}
    runs_file = None
    if runs_k and res.lists:
        runs_file = write_runs(out / f"{name}_runs.tsv.gz", res.qids, {m: res.lists[m] for m in res.table}, k=runs_k)
    summary = {
        "split": res.split,
        "n_queries": len(res.qids),
        "n_docs": len(res.docs),
        "table": res.table,
        "significance": sig or None,
        "fusion_loss": losses or None,
        "failures": failures[0] if failures else None,
        "latency_ms": latency,
        "runs_file": runs_file.name if runs_file else None,
        "config": snap,
    }
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = [
        f"# 评测结果 {name}",
        "",
        f"- 数据：`{cfg['data_dir']}`，划分 `{res.split}`，{len(res.qids)} 条查询，{len(res.docs)} 篇文档",
        f"- 嵌入：`{cfg['embedder']['model']}`；Faiss：`{cfg['faiss'].get('index_type')}`；融合：`{res.fusion}`；"
        f"重排：`{(cfg.get('reranker') or {}).get('model')}` top_n={top_n}",
        "",
        format_table(res),
        "",
        "指标 ×100；ms/q 为批处理下平均每条查询耗时。",
    ]
    if sig and sig.get("tests"):
        md += ["", f"## 显著性（相对 {LABELS.get(sig['baseline'], sig['baseline'])}，{sig['metric']}，"
               "配对随机化检验 + bootstrap 95% CI）", "", "| 方法 | 差值（×100） |", "| --- | --- |"]
        md += [f"| {LABELS.get(m, m)} | {format_test(t)} |" for m, t in sig["tests"].items()]
    if losses:
        md += ["", f"## 融合丢失（某一路把正确答案排进前 {top_n}，融合后却掉出前 {top_n}）", ""]
        md += [f"- {LABELS.get(m, m)}：丢失 {v['lost']} 条（{100 * v['lost_rate']:.1f}%），救回 {v['rescued']} 条"
               for m, v in losses.items()]
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
    if runs_file:
        md += ["", f"逐条排序结果：`{runs_file.name}`（每个模式前 {runs_k} 名，可用 scripts/compare_runs.py 做跨评测对比）"]
    md += ["", "<details><summary>完整配置</summary>", "", "```yaml", dump_config(snap), "```", "</details>"]
    (out / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    return out / f"{name}.md"
