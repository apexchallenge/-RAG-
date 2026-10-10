"""Faiss 索引对比：同一批文档向量，比较 flat / hnsw / ivf / ivfpq 在不同查询参数下的
召回（相对精确搜索）、nDCG@10、单条查询延迟、索引大小。

    python scripts/bench_faiss.py --config configs/default.yaml --max-queries 1000

先跑过 build_index.py（需要存好的向量）。输出的表格就是面试里"为什么选 HNSW / IVF-PQ"的依据。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from rag.config import load_config  # noqa: E402
from rag.data import select_queries  # noqa: E402
from rag.dense import build_faiss_index, load_vectors, search_index, set_search_params  # noqa: E402
from rag.indexing import dense_dir  # noqa: E402
from rag.metrics import ndcg_at_k  # noqa: E402

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
            for i in range(min(latency_queries, len(q_emb))):
                t0 = time.perf_counter()
                index.search(q_emb[i : i + 1], k)
                lat.append(1000 * (time.perf_counter() - t0))
            faiss.omp_set_num_threads(old)
            rows.append({
                "index": kind,
                "params": ",".join(f"{PARAM_NAMES[a]}={b}" for a, b in params.items()) or "-",
                "build_s": build_s,
                "size_mb": size_mb,
                "p50_ms": float(np.percentile(lat, 50)),
                "recall_vs_flat": float(ann_recall),
                "ndcg@10": float(ndcg),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Faiss 索引速度/精度对比")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="dev", help="默认 dev：挑索引参数也是调参，不要在 test 上挑")
    p.add_argument("--max-queries", type=int, default=1000)
    p.add_argument("--types", default="flat,hnsw,ivf,ivfpq")
    p.add_argument("--threads", type=int, default=1, help="测单条延迟时 Faiss 用的线程数")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
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
    rows = bench(cfg, q_emb, qids, qrels, doc_ids, emb, types, threads=args.threads)
    print(f"\n{len(doc_ids)} 篇文档 × {emb.shape[1]} 维，{len(qids)} 条查询\n")
    print("| 索引 | 参数 | 构建 s | 大小 MB | 单条 P50 ms | Recall@10 vs Flat | nDCG@10 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['index']} | {r['params']} | {r['build_s']:.1f} | {r['size_mb']:.1f} | {r['p50_ms']:.2f} | "
              f"{100 * r['recall_vs_flat']:.1f} | {100 * r['ndcg@10']:.1f} |")


if __name__ == "__main__":
    main()
