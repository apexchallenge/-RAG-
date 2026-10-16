"""bench_faiss.py：小规模随机向量上跑通对比，并检查结果文件和"召回达标时最快的设置"。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_faiss  # noqa: E402


class TestBenchFaiss(unittest.TestCase):
    def test_bench_and_save(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(400, 16)).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        ids = [f"d{i}" for i in range(400)]
        qids = [f"q{i}" for i in range(8)]
        qrels = {q: {f"d{i}": 1} for i, q in enumerate(qids)}
        cfg = {"faiss": {"ivf_nlist": 4, "pq_m": 4, "pq_nbits": 4, "hnsw_m": 8}}
        rows = bench_faiss.bench(cfg, emb[:8] + 0.01, qids, qrels, ids, emb, ["flat", "hnsw", "ivf"], latency_queries=4)
        self.assertEqual(rows[0]["index"], "flat")
        self.assertAlmostEqual(rows[0]["recall_vs_flat"], 1.0)
        for r in rows:
            self.assertGreaterEqual(r["p95_ms"], r["p50_ms"])
        picks = bench_faiss.pick_settings(rows, min_recall=0.0)
        self.assertEqual(set(picks), {"flat", "hnsw", "ivf"})
        self.assertIsNone(bench_faiss.pick_settings([dict(rows[1], recall_vs_flat=0.5)], 0.95)["hnsw"])
        with tempfile.TemporaryDirectory() as d:
            meta = {"model": "m", "n_docs": 400, "dim": 16, "split": "dev", "n_queries": 8,
                    "machine": bench_faiss.machine_info(1)}
            path = bench_faiss.save_bench(rows, meta, d, tag="t")
            self.assertTrue(path.name.startswith("faiss_t_"))
            self.assertTrue(path.with_suffix(".json").exists())
            self.assertIn("最快的设置", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
