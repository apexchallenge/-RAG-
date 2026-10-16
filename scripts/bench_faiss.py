"""Faiss 索引对比：同一批文档向量，比较 flat / hnsw / ivf / ivfpq 在不同查询参数下的
召回（相对精确搜索）、nDCG@10、单条查询延迟（P50/P95）、索引大小、构建耗时。

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/bench_faiss.py                 # 默认 dev 1000 条
    python scripts/bench_faiss.py --types hnsw --set faiss.hnsw_m=16 --tag hnsw_m16       # 换建索引参数单独比

先跑过 build_index.py（需要存好的向量）。结果写到 results/faiss_<tag>_<时间>.md / .json：
表格 + 机器信息 + 每种索引"召回 ≥ 95%（相对 Flat）时最快的参数"，就是面试里"为什么选 HNSW / IVF-PQ"的依据。
注意延迟和机器强相关：写进简历时带上 md 里记录的 CPU / 线程数。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from rag.config import load_config  # noqa: E402
from rag.data import select_queries  # noqa: E402
from rag.dense import build_faiss_index, build_params, load_vectors, search_index, set_search_params  # noqa: E402
from rag.indexing import dense_dir  # noqa: E402
from rag.metrics import ndcg_at_k  # noqa: E402
from rag.paths import resolve  # noqa: E402

SWEEP = {
    "flat": [{}],
    "hnsw": [{"hnsw_ef_search": v} for v in (16, 32, 64, 128, 256)],
    "ivf": [{"ivf_nprobe": v} for v in (1, 4, 16, 64)],
    "ivfpq": [{"ivf_nprobe": v} for v in (4, 16, 64)],
}
PARAM_NAMES = {"hnsw_ef_search": "efSearch", "ivf_nprobe": "nprobe"}


def bench(cfg: dict, q_emb: np.ndarray, qids: list[str], qrels: dict, doc_ids: list[str], emb: np.ndarray,
          types: list[str], k: int = 10, latency_queries: int = 200, threads: int = 1) -> list[dict]:
    exact = search_index(build_faiss_index(emb, {"index_type": "flat"}), doc_ids, q_emb, k)
    exact_ids = [{d for d, _ in hits} for hits in exact]
    rows = []
    for kind in types:
        base = {**cfg["faiss"], "index_type": kind}
        start = time.perf_counter()
        try:
            index = build_faiss_index(emb, base)
        except ValueError as e:  # 例如语料太小，IVF-PQ 训练不了
            print(f"[bench] 跳过 {kind}：{e}")
            continue
        build_s = time.perf_counter() - start
        size_mb = faiss.serialize_index(index).nbytes / 2**20
        for params in SWEEP[kind]:
            set_search_params(index, {**base, **params})
            hits = search_index(index, doc_ids, q_emb, k)
            ann_recall = np.mean([len(exact_ids[i] & {d for d, _ in h}) / max(1, len(exact_ids[i]))
                                  for i, h in enumerate(hits)])
            ndcg = np.mean([ndcg_at_k([d for d, _ in h], qrels[q], k) for q, h in zip(qids, hits)])
            # 单条查询延迟：线上通常一次只来一个查询，单线程测更贴近真实情况
            old = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(threads)
            lat = []
            try:
                for i in range(min(latency_queries, len(q_emb))):
                    t0 = time.perf_counter()
                    index.search(q_emb[i : i + 1], k)
                    lat.append(1000 * (time.perf_counter() - t0))
            finally:
                faiss.omp_set_num_threads(old)
            rows.append({
                "index": kind,
                "params": ",".join(f"{PARAM_NAMES[a]}={b}" for a, b in params.items()) or "-",
                "build_params": build_params(base),
                "build_s": build_s,
                "size_mb": size_mb,
                "p50_ms": float(np.percentile(lat, 50)),
                "p95_ms": float(np.percentile(lat, 95)),
                "recall_vs_flat": float(ann_recall),
                "ndcg@10": float(ndcg),
            })
    return rows


def pick_settings(rows: list[dict], min_recall: float = 0.95) -> dict[str, dict | None]:
    """每种索引里，召回（相对 Flat）≥ min_recall 的设置中单条 P50 最低的一个；一个都达不到时为 None。"""
    out: dict[str, dict | None] = {}
    for r in rows:
        out.setdefault(r["index"], None)
        if r["recall_vs_flat"] >= min_recall:
            best = out[r["index"]]
            if best is None or r["p50_ms"] < best["p50_ms"]:
                out[r["index"]] = r
    return out


def machine_info(threads: int) -> dict:
    cpu = platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {"platform": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(),
            "faiss": getattr(faiss, "__version__", "?"), "latency_threads": threads}


def format_rows(rows: list[dict]) -> list[str]:
    lines = ["| 索引 | 查询参数 | 构建 s | 大小 MB | 单条 P50 ms | P95 ms | Recall@10 vs Flat | nDCG@10 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['index']} | {r['params']} | {r['build_s']:.1f} | {r['size_mb']:.1f} | {r['p50_ms']:.2f} | "
                     f"{r['p95_ms']:.2f} | {100 * r['recall_vs_flat']:.1f} | {100 * r['ndcg@10']:.1f} |")
    return lines


def save_bench(rows: list[dict], meta: dict, out_dir: str | Path, tag: str | None = None,
               min_recall: float = 0.95) -> Path:
    """写出 faiss_<tag>_<时间>.json / .md，返回 md 路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = f"faiss_{tag + '_' if tag else ''}{datetime.now():%Y%m%d-%H%M%S}"
    name, i = base, 2
    while (out / f"{name}.md").exists():
        name, i = f"{base}-{i}", i + 1
    picks = pick_settings(rows, min_recall)
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "min_recall": min_recall, "rows": rows,
                   "picks": {k: (v["params"] if v else None) for k, v in picks.items()}}, f, ensure_ascii=False, indent=2)
    m = meta.get("machine", {})
    flat = next((r for r in rows if r["index"] == "flat"), None)
    md = [
        f"# Faiss 索引对比 {name}",
        "",
        f"- 向量：`{meta.get('model')}`，{meta.get('n_docs')} 篇 × {meta.get('dim')} 维；"
        f"查询：划分 `{meta.get('split')}` {meta.get('n_queries')} 条",
        f"- 机器：{m.get('cpu')}（{m.get('cpu_count')} 核），{m.get('platform')}；faiss {m.get('faiss')}；"
        f"延迟按单条查询、{m.get('latency_threads')} 线程测",
        f"- 建索引参数：{json.dumps(meta.get('build_cfg', {}), ensure_ascii=False)}",
        "",
        *format_rows(rows),
        "",
        "Recall@10 vs Flat：近似索引前 10 名和精确搜索前 10 名的重合比例（衡量\"近似\"丢了多少）；nDCG@10 用原始标注。",
        "",
        f"## 召回 ≥ {100 * min_recall:.0f}%（相对 Flat）时最快的设置",
        "",
    ]
    for kind, r in picks.items():
        if r is None:
            md.append(f"- {kind}：扫描的参数里没有一个达到 {100 * min_recall:.0f}%")
            continue
        speed = f"，比 Flat 快 {flat['p50_ms'] / r['p50_ms']:.1f} 倍" if flat and r is not flat and r["p50_ms"] > 0 else ""
        mem = f"，大小为 Flat 的 {100 * r['size_mb'] / flat['size_mb']:.0f}%" if flat and flat["size_mb"] > 0 else ""
        md.append(f"- {kind}（{r['params']}）：P50 {r['p50_ms']:.2f} ms，召回 {100 * r['recall_vs_flat']:.1f}%，"
                  f"nDCG@10 {100 * r['ndcg@10']:.1f}{speed}{mem}")
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out / f"{name}.md"


def main() -> None:
    p = argparse.ArgumentParser(description="Faiss 索引速度/精度对比")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="dev", help="默认 dev：挑索引参数也是调参，不要在 test 上挑")
    p.add_argument("--max-queries", type=int, default=1000)
    p.add_argument("--types", default="flat,hnsw,ivf,ivfpq")
    p.add_argument("--threads", type=int, default=1, help="测单条延迟时 Faiss 用的线程数")
    p.add_argument("--latency-queries", type=int, default=200)
    p.add_argument("--min-recall", type=float, default=0.95, help="挑\"最快设置\"时要求的召回下限（相对 Flat）")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--tag", default=None, help="结果文件名里的标签")
    p.add_argument("--out", default="results")
    p.add_argument("--no-save", action="store_true", help="只打印，不写结果文件")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    qids, queries, qrels = select_queries(cfg["data_dir"], args.split, args.max_queries)
    if not qids:
        raise SystemExit(f"{args.split} 划分里没有带标注的查询，可改用 --split test 或 --split all")
    doc_ids, emb, meta = load_vectors(dense_dir(cfg))

    from rag.embedder import Embedder

    embedder = Embedder.from_config(cfg["embedder"])
    if meta["model"] != embedder.name:
        raise SystemExit(f"向量是 {meta['model']} 编码的，配置是 {embedder.name}")
    q_emb = embedder.encode_queries([queries[q] for q in qids])

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    rows = bench(cfg, q_emb, qids, qrels, doc_ids, emb, types, latency_queries=args.latency_queries,
                 threads=args.threads)
    print(f"\n{len(doc_ids)} 篇文档 × {emb.shape[1]} 维，{len(qids)} 条查询\n")
    print("\n".join(format_rows(rows)))
    if not args.no_save:
        info = {"model": meta["model"], "n_docs": len(doc_ids), "dim": int(emb.shape[1]), "split": args.split,
                "n_queries": len(qids), "build_cfg": {t: build_params({**cfg["faiss"], "index_type": t}) for t in types},
                "machine": machine_info(args.threads)}
        path = save_bench(rows, info, resolve(args.out), args.tag, args.min_recall)
        print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
