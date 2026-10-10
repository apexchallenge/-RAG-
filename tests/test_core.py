"""核心模块单测：指标、融合、分词与切块、数据读写、配置覆盖。运行：python -m unittest discover -s tests -v"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import set_by_path  # noqa: E402
from rag.data import load_qrels, load_split_ids, make_splits, read_jsonl, select_queries, write_jsonl, write_qrels  # noqa: E402
from rag.fusion import fuse, rrf_fuse, weighted_fuse  # noqa: E402
from rag.metrics import evaluate_run, hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from rag.text import ChineseTokenizer, chunk_text, split_sentences  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_recall_hit_mrr(self):
        ranked = ["a", "b", "c"]
        rel = {"b": 1, "d": 1}
        self.assertEqual(recall_at_k(ranked, rel, 1), 0.0)
        self.assertEqual(recall_at_k(ranked, rel, 3), 0.5)
        self.assertEqual(hit_at_k(ranked, rel, 1), 0.0)
        self.assertEqual(hit_at_k(ranked, rel, 2), 1.0)
        self.assertEqual(mrr_at_k(ranked, rel, 10), 0.5)
        self.assertEqual(mrr_at_k(ranked, rel, 1), 0.0)

    def test_ndcg_graded(self):
        ranked = ["a", "b", "c"]
        rel = {"a": 2, "c": 1}
        dcg = 2 / math.log2(2) + 1 / math.log2(4)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(ranked, rel, 3), dcg / idcg)
        self.assertAlmostEqual(ndcg_at_k(["a", "c"], rel, 2), 1.0)
        self.assertEqual(ndcg_at_k(["x"], rel, 1), 0.0)

    def test_evaluate_run_skips_unlabeled(self):
        run = {"q1": ["a", "b"], "q2": ["c"]}
        qrels = {"q1": {"b": 1}, "q2": {"c": 0}}
        out = evaluate_run(run, qrels, recall_ks=(1, 2))
        self.assertEqual(out["R@1"], 0.0)
        self.assertEqual(out["R@2"], 1.0)
        self.assertEqual(out["MRR@10"], 0.5)
        with self.assertRaises(ValueError):
            evaluate_run({"q2": ["c"]}, qrels)


class TestFusion(unittest.TestCase):
    def test_rrf_order_and_scores(self):
        a = [("x", 9.0), ("y", 5.0)]
        b = [("y", 0.9), ("z", 0.8)]
        fused = rrf_fuse([a, b], k=60)
        self.assertEqual([d for d, _ in fused], ["y", "x", "z"])
        self.assertAlmostEqual(dict(fused)["y"], 1 / 62 + 1 / 61)
        self.assertEqual(len(rrf_fuse([a, b], top_k=2)), 2)

    def test_rrf_weights(self):
        a = [("x", 1.0)]
        b = [("z", 1.0)]
        fused = rrf_fuse([a, b], weights=[0.2, 0.8])
        self.assertEqual(fused[0][0], "z")
        with self.assertRaises(ValueError):
            rrf_fuse([a, b], weights=[1.0])

    def test_weighted_minmax(self):
        a = [("x", 10.0), ("y", 0.0)]
        b = [("y", 0.5)]
        fused = dict(weighted_fuse([a, b], [0.5, 0.5]))
        self.assertAlmostEqual(fused["x"], 0.5)  # 10 -> 1.0
        self.assertAlmostEqual(fused["y"], 0.5)  # 0 -> 0.0；单元素列表 -> 1.0

    def test_fuse_dispatch(self):
        cfg = {"fusion": "rrf", "weights": {"bm25": 0.3, "dense": 0.7}, "fused_top_k": 5}
        self.assertEqual(len(fuse(cfg, [("a", 1.0)], [("b", 1.0)])), 2)
        self.assertEqual(fuse(cfg, [("a", 1.0)], [("b", 1.0)], method="weighted")[0][0], "b")
        with self.assertRaises(ValueError):
            fuse(cfg, [], [], method="nope")


class TestText(unittest.TestCase):
    def test_tokenizer_filters_and_modes(self):
        tok = ChineseTokenizer("accurate")
        words = tok("混合检索结合了BM25和BGE，效果很好！")
        self.assertIn("bm25", words)
        self.assertNotIn("，", words)
        self.assertNotIn("！", words)
        search = ChineseTokenizer("search")("中华人民共和国成立")
        accurate = ChineseTokenizer("accurate")("中华人民共和国成立")
        self.assertGreater(len(search), len(accurate))
        with self.assertRaises(ValueError):
            ChineseTokenizer("fast")

    def test_stopwords_and_user_dict(self):
        with tempfile.TemporaryDirectory() as d:
            sw = Path(d) / "sw.txt"
            sw.write_text("的\n了\n", encoding="utf-8")
            ud = Path(d) / "ud.txt"
            ud.write_text("幽门螺杆菌\n", encoding="utf-8")
            tok = ChineseTokenizer("accurate", user_dict=str(ud), stopwords=str(sw))
            words = tok("感染了幽门螺杆菌的症状")
            self.assertIn("幽门螺杆菌", words)
            self.assertNotIn("了", words)

    def test_whitespace_english_and_fullwidth(self):
        self.assertEqual(split_sentences("BM25\nBGE"), ["BM25\n", "BGE"])
        self.assertEqual(chunk_text("BM25\nBGE", chunk_size=100, overlap=10), ["BM25\nBGE"])  # 换行不会被吃掉
        self.assertEqual(split_sentences("Hello world. Next one. 3.5 stays"), ["Hello world. ", "Next one. ", "3.5 stays"])
        tok = ChineseTokenizer("accurate")
        self.assertIn("28", tok("学分上限是２８分"))  # 全角数字先转半角
        words = tok("BM25\nBGE")
        self.assertIn("bm25", words)
        self.assertIn("bge", words)

    def test_chunk_overlap_falls_back_to_characters(self):
        text = ("一" * 29 + "。") * 4  # 每句 30 字，放不进 20 字的整句重叠
        chunks = chunk_text(text, chunk_size=50, overlap=20)
        self.assertTrue(all(len(c) <= 50 for c in chunks))
        self.assertTrue(chunks[1].startswith(chunks[0][-20:]))

    def test_chunking(self):
        text = "第一句话。" * 30 + "这是一个特别长的句子" * 30 + "。结尾。"
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertEqual(chunks[1][:5], "第一句话。")  # 下一块以上一块末尾的整句开头（重叠）
        self.assertIn("结尾", chunks[-1])
        self.assertEqual(split_sentences("你好！再见。\n\n好"), ["你好！", "再见。\n\n", "好"])  # 段落换行保留
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=10, overlap=10)


class TestDataAndConfig(unittest.TestCase):
    def test_splits_deterministic(self):
        ids = [f"q{i}" for i in range(100)]
        s1, s2 = make_splits(ids, 30, 20, seed=1), make_splits(ids, 30, 20, seed=1)
        self.assertEqual(s1, s2)
        self.assertEqual((len(s1["test"]), len(s1["dev"]), len(s1["train"])), (30, 20, 50))
        self.assertFalse(set(s1["test"]) & set(s1["train"]))

    def test_qrels_roundtrip_and_headerless(self):
        with tempfile.TemporaryDirectory() as d:
            write_qrels(Path(d) / "qrels.tsv", {"q1": {"d1": 1, "d2": 0}})
            self.assertEqual(load_qrels(d), {"q1": {"d1": 1, "d2": 0}})
            (Path(d) / "qrels.tsv").write_text("q9\td9\t2\n", encoding="utf-8")
            self.assertEqual(load_qrels(d), {"q9": {"d9": 2}})

    def test_select_queries(self):
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(Path(d) / "queries.jsonl", [{"id": f"q{i}", "text": f"问题{i}"} for i in range(6)])
            write_qrels(Path(d) / "qrels.tsv", {f"q{i}": {"d": 1 if i != 5 else 0} for i in range(6)})
            (Path(d) / "splits.json").write_text(json.dumps({"test": ["q0", "q1", "q5"], "dev": ["q2"], "train": []}))
            qids, queries, _ = select_queries(d, "test")
            self.assertEqual(qids, ["q0", "q1"])  # q5 没有正例，被过滤
            self.assertEqual(len(select_queries(d, "all")[0]), 5)
            self.assertEqual(select_queries(d, "all", max_queries=3, seed=7), select_queries(d, "all", max_queries=3, seed=7))

    def test_missing_splits_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_split_ids(d, "dev")  # 不能悄悄退回"全部查询"
            self.assertIsNone(load_split_ids(d, "all"))

    def test_bom_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "q.jsonl").write_bytes('{"id": "q1", "text": "问题"}\n'.encode("utf-8-sig"))
            self.assertEqual(read_jsonl(Path(d) / "q.jsonl")[0]["id"], "q1")
            (Path(d) / "qrels.tsv").write_bytes("q1\td1\t1\n".encode("utf-8-sig"))
            self.assertEqual(load_qrels(d), {"q1": {"d1": 1}})

    def test_set_by_path(self):
        cfg = {"reranker": {"top_n": 50}}
        set_by_path(cfg, "reranker.top_n=20")
        set_by_path(cfg, "retrieval.weights={bm25: 0.4, dense: 0.6}")
        set_by_path(cfg, "reranker.enabled=false")
        self.assertEqual(cfg["reranker"], {"top_n": 20, "enabled": False})
        self.assertEqual(cfg["retrieval"]["weights"]["dense"], 0.6)
        with self.assertRaises(ValueError):
            set_by_path(cfg, "no_equals_sign")


if __name__ == "__main__":
    unittest.main()
