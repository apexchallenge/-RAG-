"""补标工具单测：分数解析、候选池、合并标注、一致率、缓存续跑、并发打分。用假的大模型，不联网。"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.judge import (  # noqa: E402
    JudgeAbort, JudgmentCache, build_judge_prompt, build_pool, cohen_kappa, judge_pairs, judged_at_k, merge_qrels, parse_grade,
)


class TestJudge(unittest.TestCase):
    def test_parse_grade(self):
        for text, want in [("2", 2), (" 1\n", 1), ("评分：0", 0), ("0。", 0), ("**2**", 2),
                           ("<think>怀孕39周，耻骨疼……</think>2", 2), ("这个回答比较相关，评分为1", 1),
                           ("", None), (None, None), ("3", None), ("我无法判断", None)]:
            self.assertEqual(parse_grade(text), want, text)

    def test_prompt_truncates_doc(self):
        p = build_judge_prompt("问题", "很长" * 1000, max_chars=10)
        self.assertIn("很长很长很长很长很长\n", p + "\n")
        self.assertNotIn("很长" * 6, p)

    def test_pool_merge_judged(self):
        runs = {"dense": {"q": ["a", "b", "c"]}, "rerank": {"q": ["c", "d", "a"]}}
        pool = build_pool(runs, ["q"], ["dense", "rerank"], depth=2)
        self.assertEqual(pool["q"], ["a", "b", "c", "d"])
        qrels = {"q": {"a": 1}}
        grades = {("q", "b"): 2, ("q", "c"): 1, ("q", "d"): None, ("q", "a"): 0, ("x", "z"): 2}
        strict = merge_qrels(qrels, grades, threshold=2)
        self.assertEqual(strict, {"q": {"a": 1, "b": 1, "c": 0}})  # 原正例不被覆盖；None 不算已判；未知查询忽略
        self.assertEqual(merge_qrels(qrels, grades, threshold=1)["q"]["c"], 1)
        self.assertEqual(qrels, {"q": {"a": 1}})  # 不改原字典
        self.assertAlmostEqual(judged_at_k({"q": ["a", "b", "d", "e"]}, {"q": set(strict["q"])}, k=4), 0.5)

    def test_kappa(self):
        self.assertAlmostEqual(cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]), 1.0)
        self.assertAlmostEqual(cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0]), -1.0)
        self.assertAlmostEqual(cohen_kappa([1, 1, 1], [1, 1, 1]), 1.0)
        with self.assertRaises(ValueError):
            cohen_kappa([1], [1, 2])

    def test_judge_pairs_and_cache_resume(self):
        queries = {"q1": "宝宝咳嗽", "q2": "耻骨疼"}
        docs = {"a": "宝宝咳嗽多喝水", "b": "XX", "c": "耻骨联合疼痛"}
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            if prompt.endswith("【回答】\nXX"):
                raise RuntimeError("网络错误")
            return "2" if "宝宝咳嗽多喝水" in prompt else "1"

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "j.jsonl"
            cache = JudgmentCache(path, model="m")
            pairs = [("q1", "a"), ("q1", "b"), ("q2", "c"), ("q1", "a")]
            stats = judge_pairs(pairs, queries, docs, fake_llm, cache, workers=2, show_progress=False, probe_n=1)
            self.assertEqual((stats["requested"], stats["grade_2"], stats["grade_1"], stats["failed"]), (3, 1, 1, 1))
            self.assertEqual(cache.grades[("q1", "a")], 2)
            self.assertIsNone(cache.grades[("q1", "b")])
            # 重新加载：成功的跳过，失败的重试；换模型则全部重来
            cache2 = JudgmentCache(path, model="m")
            self.assertIn(("q1", "a"), cache2)
            self.assertNotIn(("q1", "b"), cache2)
            n = len(calls)
            stats2 = judge_pairs(pairs, queries, docs, fake_llm, cache2, workers=1, show_progress=False, probe_n=0)
            self.assertEqual(stats2["requested"], 1)
            self.assertEqual(len(calls), n + 1)
            self.assertEqual(JudgmentCache(path, model="other").grades, {})
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"qid": "q1", "doc_')  # 模拟中断时写了半行
            self.assertIn(("q2", "c"), JudgmentCache(path, model="m"))

    def test_abort_when_probe_all_fail(self):
        calls = []

        def empty_llm(prompt):
            calls.append(prompt)
            return ""  # 推理模型输出上限太小时就是这样

        with tempfile.TemporaryDirectory() as d:
            cache = JudgmentCache(Path(d) / "j.jsonl", model="m")
            pairs = [(f"q{i}", "a") for i in range(50)]
            with self.assertRaises(JudgeAbort):
                judge_pairs(pairs, {f"q{i}": "问题" for i in range(50)}, {"a": "回答"}, empty_llm, cache,
                            show_progress=False, probe_n=5)
        self.assertEqual(len(calls), 5)  # 只试了 5 条就停

    def test_max_calls_and_resume(self):
        queries = {f"q{i}": "问题" for i in range(40)}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "j.jsonl"
            cache = JudgmentCache(path, model="m")
            pairs = [(f"q{i}", "a") for i in range(40)]
            stats = judge_pairs(pairs, queries, {"a": "回答"}, lambda p: "1", cache, show_progress=False, max_calls=10)
            self.assertEqual((stats["requested"], stats["remaining"], stats["grade_1"]), (10, 30, 10))
            self.assertEqual(len(JudgmentCache(path, model="m").grades), 10)  # 每条立刻落盘
            stats = judge_pairs(pairs, queries, {"a": "回答"}, lambda p: "2", JudgmentCache(path, model="m"),
                                show_progress=False)
            self.assertEqual((stats["cached"], stats["requested"], stats["remaining"]), (10, 30, 0))


if __name__ == "__main__":
    unittest.main()
