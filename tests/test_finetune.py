"""finetune_embedder.py 里不依赖 GPU / sentence-transformers 训练组件的部分：构造训练对、泄漏检查、负例过滤。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_embedder import build_pair_lists, drop_known_positive_negatives  # noqa: E402
from rag.data import write_jsonl, write_qrels  # noqa: E402


def make_data(d: Path, splits: dict, queries: dict, qrels: dict, docs: dict) -> None:
    write_jsonl(d / "corpus.jsonl", [{"id": k, "text": v, "title": ""} for k, v in docs.items()])
    write_jsonl(d / "queries.jsonl", [{"id": k, "text": v} for k, v in queries.items()])
    write_qrels(d / "qrels.tsv", qrels)
    (d / "splits.json").write_text(json.dumps(splits), encoding="utf-8")


class TestPairs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairs_and_leak_checks(self):
        make_data(
            self.d,
            {"train": ["t1", "t2", "t3"], "dev": ["v1"], "test": ["s1"]},
            {"t1": "头疼怎么办", "t2": "咳嗽吃什么药", "t3": "胃疼", "v1": "胃疼", "s1": "发烧"},
            {"t1": {"d1": 1, "d2": 1, "d0": 0}, "t2": {"d3": 1, "dx": 1}, "t3": {"d4": 1},
             "v1": {"d4": 1}, "s1": {"d2": 1}},
            {"d0": "无关", "d1": "多休息", "d2": "量体温", "d3": "止咳糖浆", "d4": "胃药"},
        )
        lists, stats = build_pair_lists(str(self.d))
        # t3 的文本和 dev 查询重复 -> 整条丢掉；t1-d2 的 d2 是 test 正例 -> 默认丢掉；dx 不在语料；d0 分数为 0
        self.assertEqual(list(zip(lists["qid"], lists["doc_id"])), [("t1", "d1"), ("t2", "d3")])
        self.assertEqual(stats["dup_query_text"], 1)
        self.assertEqual(stats["eval_doc_pairs_dropped"], 1)
        self.assertEqual(stats["missing_docs"], 1)
        self.assertEqual(stats["queries"], 2)
        self.assertEqual(lists["anchor"][0], "头疼怎么办")
        self.assertEqual(lists["positive"][1], "止咳糖浆")
        kept, _ = build_pair_lists(str(self.d), exclude_eval_docs=False)
        self.assertIn("d2", kept["doc_id"])

    def test_split_overlap_raises(self):
        make_data(self.d, {"train": ["q"], "dev": ["q"], "test": []}, {"q": "x"}, {"q": {"d": 1}}, {"d": "y"})
        with self.assertRaises(ValueError):
            build_pair_lists(str(self.d))

    def test_drop_known_positive_negatives(self):
        rows = [{"anchor": "a", "positive": "p1", "negative_1": "p2", "negative_2": "n"},
                {"anchor": "a", "positive": "p2", "negative_1": "n", "negative_2": "m"},
                {"anchor": "b", "positive": "p3", "negative_1": "p1", "negative_2": "n"}]
        kept, dropped = drop_known_positive_negatives(rows, {"a": {"p1", "p2"}, "b": {"p3"}})
        self.assertEqual(dropped, 1)
        self.assertEqual([r["positive"] for r in kept], ["p2", "p3"])  # b 的负例 p1 是 a 的正例，不算假负例


if __name__ == "__main__":
    unittest.main()
