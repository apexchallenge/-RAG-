"""评测工具单测：失败分类、融合丢失、显著性检验、run 文件读写、融合调参。全部用手造的小数据，不需要模型和索引。"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.evaluation import EvalResult, classify_failure, fusion_loss, significance, snapshot_config, tune_fusion  # noqa: E402
from rag.metrics import evaluate_run, parse_metric, per_query  # noqa: E402
from rag.runs import read_runs, write_runs  # noqa: E402
from rag.stats import paired_test  # noqa: E402


def hits(*ids):
    return [(d, float(len(ids) - i)) for i, d in enumerate(ids)]


class TestClassifyFailure(unittest.TestCase):
    def test_categories(self):
        k, n = 10, 50
        c = lambda r, m="hybrid_rerank": classify_failure(r, m, "rrf", k, n)  # noqa: E731
        self.assertEqual(c({"bm25": None, "dense": None, "hybrid_rrf": None, "hybrid_rerank": None}), "召回失败")
        # dense 排第 30（在重排窗口内），融合后掉出前 50 -> 融合丢失
        self.assertEqual(c({"bm25": None, "dense": 30, "hybrid_rrf": 70, "hybrid_rerank": 70}), "融合丢失")
        self.assertEqual(c({"bm25": None, "dense": 85, "hybrid_rrf": None, "hybrid_rerank": None}), "截断失败")
        self.assertEqual(c({"bm25": None, "dense": 3, "hybrid_rrf": 6, "hybrid_rerank": 12}), "重排挤出")
        self.assertEqual(c({"bm25": None, "dense": 20, "hybrid_rrf": 39, "hybrid_rerank": 32}), "排序失败")
        self.assertEqual(c({"dense": 3, "dense_rerank": 21}, "dense_rerank"), "重排挤出")
        self.assertEqual(c({"dense": 60, "dense_rerank": 60}, "dense_rerank"), "截断失败")
        self.assertEqual(c({"dense": None, "dense_rerank": None}, "dense_rerank"), "召回失败")
        # 不带重排的融合：某一路在前 k，融合后不在 -> 融合丢失
        self.assertEqual(c({"bm25": None, "dense": 4, "hybrid_rrf": 15}, "hybrid_rrf"), "融合丢失")
        self.assertEqual(c({"bm25": 30, "dense": 20, "hybrid_rrf": 15}, "hybrid_rrf"), "排序失败")
        self.assertEqual(c({"dense": 15}, "dense"), "排序失败")
        self.assertEqual(c({"bm25": None}, "bm25"), "召回失败")


def toy_result():
    qids = ["q1", "q2", "q3"]
    qrels = {"q1": {"g1": 1}, "q2": {"g2": 1}, "q3": {"g3": 1}}
    bm25 = [hits("x1", "x2", "x3", "g1"), hits("y1", "y2", "y3"), hits("g3", "z1", "z2")]
    dense = [hits("g1", "x1", "x2"), hits("g2", "y1", "y2"), hits("z1", "z2", "z3", "g3")]
    res = EvalResult(split="dev", qids=qids, queries={q: q for q in qids}, qrels=qrels, docs={}, fusion="rrf")
    res.lists = {"bm25": bm25, "dense": dense}
    return res


class TestFusionTools(unittest.TestCase):
    def test_fusion_loss(self):
        res = toy_result()
        # 一个只看 bm25 的"融合"：q1、q2 的正确答案在 dense 第 1，融合后的前 2 里没有 -> 丢失；q3 的 g3 在 bm25 第 1，没丢
        res.lists["hybrid_rrf"] = res.lists["bm25"]
        loss = fusion_loss(res, "hybrid_rrf", window=2)
        self.assertEqual((loss["lost"], loss["rescued"], loss["n"]), (2, 0, 3))  # q1、q2 丢失

    def test_tune_fusion_rows(self):
        res = toy_result()
        cfg = {"retrieval": {"fusion": "rrf", "rrf_k": 60, "weights": {"bm25": 0.5, "dense": 0.5}, "fused_top_k": 10},
               "reranker": {"top_n": 2}, "eval": {"recall_ks": [1, 2, 10]}}
        rows = tune_fusion(res, cfg, metric="nDCG@10", grid={"rrf_k": (60,), "bm25_weight": (0.0, 0.5)}, n_resamples=200)
        names = [r["name"] for r in rows]
        self.assertIn("dense only", names)
        self.assertIn("RRF k=60 bm25=0.5", names)
        self.assertIn("加权 bm25=0.5", names)
        self.assertEqual(set(res.lists), {"bm25", "dense"})  # 调参不能改坏原来的结果
        scores = [r["metrics"]["nDCG@10"] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for r in rows:
            self.assertEqual(r["vs_dense"] is None, r["name"] == "dense only")

    def test_significance_and_snapshot(self):
        res = toy_result()
        res.table = {"bm25": {}, "dense": {}}
        sig = significance(res, n_resamples=200)
        self.assertEqual(sig["baseline"], "dense")
        self.assertIn("bm25", sig["tests"])
        snap = snapshot_config({"eval": {"split": "test"}, "x": 1}, res)
        self.assertEqual(snap["eval"]["split"], "dev")
        self.assertEqual(snap["eval"]["n_queries"], 3)


class TestStatsAndRuns(unittest.TestCase):
    def test_paired_test(self):
        base = {f"q{i}": 0.5 for i in range(200)}
        same = paired_test(base, dict(base), n_resamples=500)
        self.assertEqual(same["delta"], 0.0)
        self.assertGreater(same["p"], 0.9)
        better = {q: v + (0.1 if i % 4 else -0.05) for i, (q, v) in enumerate(base.items())}
        t = paired_test(base, better, n_resamples=500)
        self.assertGreater(t["delta"], 0)
        self.assertLess(t["p"], 0.01)
        self.assertLess(t["ci95"][0], t["delta"])
        self.assertGreater(t["ci95"][1], t["delta"])
        with self.assertRaises(ValueError):
            paired_test({"a": 1.0}, {"b": 1.0})

    def test_metric_parsing_and_per_query(self):
        self.assertEqual(parse_metric("nDCG@10"), ("ndcg", 10))
        self.assertEqual(parse_metric("R@100"), ("recall", 100))
        with self.assertRaises(ValueError):
            parse_metric("ndcg")
        run, qrels = {"a": ["d1", "d2"], "b": ["d3"]}, {"a": {"d2": 1}, "b": {"d9": 1}}
        pq = per_query(run, qrels, "MRR@10")
        self.assertEqual(pq, {"a": 0.5, "b": 0.0})
        self.assertAlmostEqual(evaluate_run(run, qrels)["MRR@10"], sum(pq.values()) / 2)

    def test_runs_roundtrip(self):
        res = toy_result()
        with tempfile.TemporaryDirectory() as d:
            path = write_runs(Path(d) / "r.tsv.gz", res.qids, {"bm25": res.lists["bm25"], "dense": res.lists["dense"]}, k=2)
            runs = read_runs(path)
        self.assertEqual(set(runs), {"bm25", "dense"})
        self.assertEqual([d for d, _ in runs["dense"]["q1"]], ["g1", "x1"])  # 只存前 k=2
        self.assertEqual(len(runs["bm25"]["q3"]), 2)
        with self.assertRaises(ValueError):
            write_runs(Path(tempfile.gettempdir()) / "bad.tsv.gz", ["q1"], {"dense": res.lists["dense"]})


if __name__ == "__main__":
    unittest.main()
