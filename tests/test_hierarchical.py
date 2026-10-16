"""父子分块：切块、聚合、ParentChildRetriever，以及在 demo 数据上用假模型跑通 建子块索引 -> 评测。"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.advanced.hierarchical import (  # noqa: E402
    ParentChildRetriever, build_parent_child, children_to_parents, chunk_stats, load_parent_child, pc_index_dir,
)
from rag.hier_eval import MODES, _doc_level, build_pc, run_pc_eval, save, summarize  # noqa: E402
from tests.fakes import demo_config, fake_embedder, fake_reranker  # noqa: E402


class FakeRet:
    def __init__(self, hits):
        self.hits = hits

    def search(self, queries, k):
        return [self.hits[:k] for _ in queries]


class TestParentChild(unittest.TestCase):
    def test_long_doc_gets_multiple_parents_and_overlap(self):
        text = "".join(f"第{i}句话讲一件事。" for i in range(200))
        parents, children = build_parent_child([{"id": "d", "text": text, "title": "标题"}], 300, 80, 20)
        self.assertGreater(len(parents), 1)
        self.assertEqual("".join(p["text"] for p in parents), text)  # 父块不重叠、不丢字
        self.assertTrue(all(p["title"] == "标题" for p in parents))
        by_parent = {}
        for c in children:
            by_parent.setdefault(c.parent_id, []).append(c.text)
        first = by_parent[parents[0]["id"]]
        self.assertGreater(len(first), 1)
        self.assertTrue(first[1].startswith(first[0][-9:]) or first[0][-9:] in first[1])  # 相邻子块有重叠

    def test_empty_doc_and_bad_sizes(self):
        parents, children = build_parent_child([{"id": "e", "text": "  "}])
        self.assertEqual((parents, children), ([], []))
        with self.assertRaises(ValueError):
            build_parent_child([{"id": "x", "text": "a"}], parent_size=100, child_size=200)

    def test_aggregation(self):
        with self.assertRaises(ValueError):
            children_to_parents([("c", 1.0)], {}, agg="mean")
        # 不在映射里的 id 原样保留；分数相同按 id 排
        self.assertEqual(children_to_parents([("x", 0.5), ("c1", 0.5)], {"c1": "p1"}), [("p1", 0.5), ("x", 0.5)])

    def test_retriever_fuses_then_aggregates(self):
        c2p = {"a#c0": "A", "a#c1": "A", "b#c0": "B", "c#c0": "C"}
        parents = {pid: {"id": pid, "text": f"{pid} 的内容", "doc_id": pid.lower()} for pid in "ABC"}
        bm25 = FakeRet([("b#c0", 9.0), ("a#c1", 5.0)])
        dense = FakeRet([("a#c0", 0.9), ("c#c0", 0.8), ("b#c0", 0.7)])
        r = ParentChildRetriever(bm25, dense, c2p, parents, rrf_k=60, agg="sum")
        top = r.search(["q"], top_k=3)[0]
        self.assertEqual(top[0][0], "A")  # A 的两个子块各被一路召回，求和后第一
        self.assertEqual({p for p, _ in top}, {"A", "B", "C"})
        only_dense = ParentChildRetriever(None, dense, c2p, parents, agg="max").search(["q"], top_k=2)[0]
        self.assertEqual(only_dense, [("A", 0.9), ("C", 0.8)])
        with self.assertRaises(ValueError):
            ParentChildRetriever(None, None, c2p, parents).search(["q"])

    def test_retriever_rerank_head_only(self):
        class Rer:
            def rerank(self, queries, cands):
                return [[(pid, float(len(cands[0]) - i)) for i, (pid, _) in enumerate(reversed(cands[0]))]]

        c2p = {f"{x}#c0": x for x in "ABCD"}
        parents = {x: {"id": x, "text": x, "doc_id": x} for x in "ABCD"}
        dense = FakeRet([("A#c0", 0.9), ("B#c0", 0.8), ("C#c0", 0.7), ("D#c0", 0.6)])
        top = ParentChildRetriever(None, dense, c2p, parents, reranker=Rer(), rerank_top_n=2).search(["q"], top_k=4)[0]
        self.assertEqual([p for p, _ in top], ["B", "A", "C", "D"])

    def test_doc_level_keeps_order(self):
        parents = {"d1#p0": {"doc_id": "d1"}, "d1#p1": {"doc_id": "d1"}, "d2#p0": {"doc_id": "d2"}}
        # 头部是重排分（负数），尾部是融合分（正数）：按顺序去重，不能按分数取 max
        self.assertEqual(_doc_level([("d2#p0", -1.0), ("d1#p1", -2.0), ("d1#p0", 0.03)], parents, 10),
                         [("d2", -1.0), ("d1", -2.0)])

    def test_chunk_stats(self):
        docs = [{"id": "a", "text": "短。"}, {"id": "b", "text": "长句子。" * 100}]
        parents, children = build_parent_child(docs, 200, 50, 10)
        s = chunk_stats(docs, parents, children)
        self.assertEqual(s["docs"], 2)
        self.assertAlmostEqual(s["docs_multi_parent_rate"], 0.5)
        self.assertEqual(s["children"], len(children))


class TestHierEvalEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rag.indexing import build_bm25, build_dense

        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = demo_config(Path(cls.tmp.name) / "indexes")
        cls.cfg["hierarchical"] = {"parent_size": 40, "child_size": 20, "child_overlap": 5, "agg": "max", "child_k": 50}
        cls.embedder, cls.reranker = fake_embedder(), fake_reranker()
        build_bm25(cls.cfg, show_progress=False)
        build_dense(cls.cfg, cls.embedder, show_progress=False)
        cls.stats = build_pc(cls.cfg, embedder=cls.embedder, show_progress=False)
        cls.res = run_pc_eval(cls.cfg, embedder=cls.embedder, reranker=cls.reranker, split="all")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_index_layout(self):
        d = pc_index_dir(self.cfg)
        self.assertTrue(str(d).endswith("indexes_pc40_20_5"))
        parents, c2p, params = load_parent_child(d)
        self.assertEqual(params["children"], len(c2p))
        self.assertGreater(self.stats["children"], self.stats["docs"])  # demo 文档 50~80 字，会切成多个子块
        self.assertTrue(set(c2p.values()) <= set(parents))

    def test_all_modes_and_report(self):
        self.assertEqual(list(self.res["lists"]), list(MODES))
        for m, lists in self.res["lists"].items():
            for hits in lists:
                ids = [d for d, _ in hits]
                self.assertEqual(len(ids), len(set(ids)), m)  # 文档级去重
                self.assertFalse(any("#" in d for d in ids), m)  # 已经聚合回文档
        s = summarize(self.res, recall_ks=(1, 3, 5))
        for row in s["table"].values():
            self.assertTrue(0.0 <= row["nDCG@10"] <= 1.0)
        self.assertIn("pc_hybrid vs doc_hybrid", s["tests"])
        ctx = s["contexts"]
        self.assertLess(ctx["pc_dense"]["chars"], ctx["doc_dense"]["chars"])  # 父块（40 字）比整篇短
        path = save(s, self.res, self.cfg, self.stats, Path(self.tmp.name) / "results", "t")
        text = path.read_text(encoding="utf-8")
        self.assertIn("父子 · Hybrid + Rerank", text)
        self.assertTrue(path.with_suffix(".json").exists())

    def test_wrong_data_dir_is_rejected(self):
        import copy

        cfg = copy.deepcopy(self.cfg)
        cfg["data_dir"] = str(Path(self.tmp.name) / "other")
        with self.assertRaises(Exception):
            run_pc_eval(cfg, embedder=self.embedder, reranker=self.reranker, modes=("pc_dense",), split="all")


if __name__ == "__main__":
    unittest.main()
