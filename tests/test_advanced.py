"""进阶部分的单测：函数还是 TODO（抛 NotImplementedError）时自动跳过；实现之后应该全部通过。

先让这些测试通过，再接到评测里看指标 —— 这就是"测试驱动"地做进阶功能。
"""

import functools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.advanced import hierarchical, query_rewrite, router  # noqa: E402


def todo_ok(fn):
    """被测函数还没实现时跳过，而不是报错。"""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            self.skipTest(f"TODO 未实现：{e}")

    return wrapper


class TestHierarchical(unittest.TestCase):
    @todo_ok
    def test_parent_child_containment(self):
        docs = [{"id": "doc1", "text": "第一节内容。" * 300}, {"id": "doc2", "text": "很短的文档。"}]
        parents, children = hierarchical.build_parent_child(docs, parent_size=600, child_size=150, child_overlap=30)
        by_id = {p["id"]: p for p in parents}
        self.assertTrue(all(p["id"].startswith(p["doc_id"] + "#p") for p in parents))
        self.assertTrue(all(len(p["text"]) <= 600 for p in parents))
        for c in children:
            self.assertIn(c.parent_id, by_id)
            self.assertTrue(c.id.startswith(c.parent_id + "#c"))
            self.assertIn(c.text, by_id[c.parent_id]["text"])
            self.assertLessEqual(len(c.text), 150)
        self.assertTrue(any(p["doc_id"] == "doc2" for p in parents))

    @todo_ok
    def test_children_to_parents(self):
        hits = [("c1", 0.9), ("c2", 0.8), ("c3", 0.5)]
        mapping = {"c1": "p1", "c2": "p2", "c3": "p2"}
        self.assertEqual(hierarchical.children_to_parents(hits, mapping, agg="max"), [("p1", 0.9), ("p2", 0.8)])
        top = hierarchical.children_to_parents(hits, mapping, agg="sum")
        self.assertEqual(top[0][0], "p2")
        self.assertAlmostEqual(top[0][1], 1.3)
        self.assertEqual(len(hierarchical.children_to_parents(hits, mapping, top_k=1)), 1)


class TestQueryRewrite(unittest.TestCase):
    @todo_ok
    def test_parse_lines(self):
        text = "1. 第一种说法\n2、第二种说法\n\n- 第三种说法\n• 第一种说法\n4) 第四种"
        self.assertEqual(query_rewrite.parse_lines(text, max_n=3), ["第一种说法", "第二种说法", "第三种说法"])

    @todo_ok
    def test_multi_query_keeps_original_first(self):
        fake_llm = lambda prompt: "改写一\n改写二\n改写三\n改写四"  # noqa: E731
        out = query_rewrite.multi_query(fake_llm, "原问题", n=3)
        self.assertEqual(out[0], "原问题")
        self.assertEqual(len(out), 4)

    @todo_ok
    def test_parse_check(self):
        self.assertEqual(query_rewrite.parse_check("充分"), (True, ""))
        ok, missing = query_rewrite.parse_check("不足：缺少报销比例")
        self.assertFalse(ok)
        self.assertIn("报销比例", missing)
        self.assertFalse(query_rewrite.parse_check("我也不知道")[0])
        self.assertFalse(query_rewrite.parse_check("不充分：缺少报销比例")[0])  # 别用 "充分" in text 判断


class TestRouter(unittest.TestCase):
    @todo_ok
    def test_parse_route(self):
        self.assertEqual(router.parse_route("simple"), "simple")
        self.assertEqual(router.parse_route("类别：Complex。"), "complex")
        self.assertEqual(router.parse_route("out_of_scope"), "out_of_scope")
        self.assertEqual(router.parse_route("嗯……不确定"), "complex")


if __name__ == "__main__":
    unittest.main()
