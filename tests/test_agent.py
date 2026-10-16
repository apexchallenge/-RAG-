"""answer_with_check（agent）和 agent 评测工具的单测：用假的检索流程和假的大模型，不需要 GPU、不调用 API。"""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.advanced.query_rewrite import (  # noqa: E402
    IRRELEVANT, PARTIAL, SUFFICIENT, answer_with_check, parse_check_detail, parse_verdict, refine_query, retrieve_multi,
    similar,
)
from rag.agent_eval import (  # noqa: E402
    RecordStore, build_sets, compare_tags, estimate, parse_answer_judge, run_task, summarize,
)
from rag.generator import REFUSAL  # noqa: E402


class FakeReranker:
    """按文档里含"答案"的程度排序。"""

    def rerank(self, queries, candidates):
        return [sorted([(d, float(t.count("答案"))) for d, t in c], key=lambda x: -x[1]) for c in candidates]


class FakeGenerator:
    available = True

    def __init__(self):
        self.calls = []

    def answer(self, question, contexts, usage=None):
        self.calls.append((question, contexts))
        if usage is not None:
            usage["calls"] += 1
        return f"回答[1]：{contexts[0][:6]}"

    def as_llm(self, thinking=False, max_tokens=None, usage=None):
        def llm(prompt):
            if usage is not None:
                usage["calls"] += 1
            return "充分"
        return llm


class FakePipeline:
    def __init__(self, routes, reranker=None):
        self.cfg = {"generation": {"context_top_k": 2, "max_chars_per_doc": 50}, "agent": {"max_rounds": 2},
                    "retrieval": {"fusion": "rrf", "rrf_k": 60, "fused_top_k": 10, "bm25_top_k": 10, "dense_top_k": 10}}
        self.docs = {d: {"id": d, "text": t} for d, t in {
            "a": "相关话题但没说到点子上", "b": "另一条泛泛而谈", "g": "这里有答案答案", "x": "广告", "gold": "正确答案"}.items()}
        self.routes = routes
        self.reranker = reranker
        self.generator = FakeGenerator()
        self.retrieved = []

    def retrieve(self, q):
        self.retrieved.append(q)
        return {"final": [(d, 1.0 - i / 10) for i, d in enumerate(self.routes.get(q, ["a", "b"]))]}


def scripted(*outputs):
    """按顺序返回预设输出的假大模型，并记录收到的提示词。"""
    seq, prompts = list(outputs), []

    def llm(prompt):
        prompts.append(prompt)
        return seq.pop(0) if seq else "充分"
    llm.prompts = prompts
    return llm


class TestAnswerWithCheck(unittest.TestCase):
    def test_sufficient_first_round(self):
        pipe = FakePipeline({"问题": ["g", "a"]})
        res = answer_with_check(pipe, scripted("充分"), "问题")
        self.assertFalse(res["refused"])
        self.assertEqual((res["rounds"], res["llm_calls"], res["queries"]), (1, 2, ["问题"]))
        self.assertEqual([c["id"] for c in res["contexts"]], ["g", "a"])
        self.assertEqual(len(pipe.generator.calls), 1)

    def test_partial_then_refine_then_answer(self):
        pipe = FakePipeline({"问题": ["a", "b"], "新查询": ["x", "g"]}, reranker=FakeReranker())
        llm = scripted("部分：缺少用药剂量", "新查询", "充分")
        res = answer_with_check(pipe, llm, "问题", max_rounds=2, refine="query")
        self.assertFalse(res["refused"])
        self.assertEqual(res["queries"], ["问题", "新查询"])
        self.assertEqual((res["rounds"], res["llm_calls"]), (2, 4))  # 检查、改写、检查、回答
        # k=2：原资料留前 1 条，新文档按"原问题+缺失"排序后占 1 个名额（含答案的 g 排在 x 前面）
        self.assertEqual([c["id"] for c in res["contexts"]], ["a", "g"])
        self.assertEqual(res["trace"][0]["added"], ["g"])
        self.assertEqual((res["context_changed"], res["stop_reason"]), (True, "sufficient"))
        self.assertEqual(res["trace"][0]["missing"], "缺少用药剂量")
        self.assertEqual((res["statuses"], res["status"], res["improved"]), ([PARTIAL, SUFFICIENT], SUFFICIENT, True))
        self.assertIn("缺少用药剂量", llm.prompts[1])  # 改写时看到了缺什么
        self.assertIn("【问题】问题", llm.prompts[2])  # 第二次检查仍然针对原问题

    def test_slots_vs_rerank_merge(self):
        routes = {"问题": ["g", "a"], "新查询": ["x", "b"]}
        # 旧做法：新旧合在一起按原问题重排，原来的前 k 本来就是按原问题排出来的，新文档进不来 -> 资料不变，提前停
        pipe = FakePipeline(routes, reranker=FakeReranker())
        res = answer_with_check(pipe, scripted("部分：缺饮食", "新查询"), "问题", refine="query", merge="rerank")
        self.assertEqual(([c["id"] for c in res["contexts"]], res["stop_reason"], res["llm_calls"]),
                         (["g", "a"], "context_unchanged", 3))
        # 新做法：保留名额，新文档替换掉原资料里排最后的
        pipe = FakePipeline(routes, reranker=FakeReranker())
        res = answer_with_check(pipe, scripted("部分：缺饮食", "新查询", "充分"), "问题", refine="query")
        self.assertEqual([c["id"] for c in res["contexts"]], ["g", "x"])
        self.assertTrue(res["context_changed"])

    def test_hyde_refine_uses_answer_style_text(self):
        pipe = FakePipeline({}, reranker=FakeReranker())
        llm = scripted("部分：没说饮食", "这种情况一般 建议\n清淡饮食，多喝水。", "充分")
        res = answer_with_check(pipe, llm, "问题", max_rounds=2)  # 默认 refine="hyde"
        self.assertEqual(res["queries"][1], "这种情况一般 建议 清淡饮食，多喝水。")  # 多行合成一段
        self.assertIn("口语化回答", llm.prompts[1])
        self.assertNotIn("之前已经用这些文字检索过", llm.prompts[1])  # 第一轮没有"之前的文字"

    def test_not_retrievable_partial_answers_without_retrieval(self):
        pipe = FakePipeline({})
        res = answer_with_check(pipe, scripted("部分（查不到）：提问者本人的检查结果"), "问题")
        self.assertFalse(res["refused"])
        self.assertEqual((res["stop_reason"], res["rounds"], res["llm_calls"], len(pipe.retrieved)),
                         ("not_retrievable", 1, 2, 1))
        self.assertFalse(res["trace"][0]["retrievable"])

    def test_context_unchanged_stops_early(self):
        pipe = FakePipeline({"问题": ["a", "b"], "换个说法": ["b", "a"]})
        res = answer_with_check(pipe, scripted("部分：缺剂量", "换个说法"), "问题", refine="query")
        self.assertEqual((res["stop_reason"], res["rounds"], res["llm_calls"]), ("context_unchanged", 1, 3))
        self.assertFalse(res["context_changed"])

    def test_duplicate_refine_stops(self):
        pipe = FakePipeline({"问题": ["a", "b"], "清淡饮食多喝水注意休息": ["g", "x"]}, reranker=FakeReranker())
        llm = scripted("部分：缺饮食", "清淡饮食多喝水注意休息", "部分：还缺", "清淡饮食多喝水注意休息吧")
        res = answer_with_check(pipe, llm, "问题", max_rounds=3)
        self.assertEqual((res["stop_reason"], res["rounds"], res["llm_calls"]), ("duplicate_refine", 2, 5))
        self.assertIn("之前已经用这些文字检索过", llm.prompts[3])  # 第二次写检索文字时看到了第一次写的
        self.assertIn("清淡饮食多喝水注意休息", llm.prompts[3])

    def test_worse_after_refine_reverts_to_best_round(self):
        # 第一轮"部分"，补检索把文档换掉后反而判"无关"：不拒答，用第一轮的资料回答
        pipe = FakePipeline({"问题": ["a", "b"], "换个说法": ["x", "g"]}, reranker=FakeReranker())
        res = answer_with_check(pipe, scripted("部分：缺剂量", "换个说法", "无关：资料讲的是别的"), "问题",
                                max_rounds=1, refine="query")
        self.assertFalse(res["refused"])
        self.assertEqual(len(pipe.generator.calls), 1)
        self.assertEqual(([c["id"] for c in res["contexts"]], res["status"], res["reverted_to_round"]), (["a", "b"], PARTIAL, 0))
        self.assertFalse(res["context_changed"])
        self.assertEqual(res["statuses"], [PARTIAL, IRRELEVANT])

    def test_no_progress_stops(self):
        pipe = FakePipeline({"问题": ["a", "b"], "第一次": ["g", "x"]}, reranker=FakeReranker())
        llm = scripted("部分：缺少饮食建议", "第一次", "部分：仍缺少饮食建议")
        res = answer_with_check(pipe, llm, "问题", max_rounds=3, refine="query")
        self.assertEqual((res["stop_reason"], res["rounds"], res["llm_calls"]), ("no_progress", 2, 4))  # 不再写第三次
        self.assertFalse(res["refused"])

    def test_still_partial_answers_instead_of_refusing(self):
        pipe = FakePipeline({"问题": ["a", "b"], "换个说法": ["g", "x"]})
        res = answer_with_check(pipe, scripted("部分：缺少报销比例", "换个说法", "部分：仍缺报销比例"), "问题",
                                max_rounds=1, refine="query")
        self.assertFalse(res["refused"])
        self.assertEqual(len(pipe.generator.calls), 1)  # 仍用合并后的资料回答
        self.assertEqual((res["status"], res["improved"], res["llm_calls"], res["rounds"], res["stop_reason"]),
                         (PARTIAL, False, 4, 2, "max_rounds"))

    def test_irrelevant_retries_then_refuses_without_answer_call(self):
        pipe = FakePipeline({"问题": ["a", "b"], "换个说法": ["g", "x"]})
        llm = scripted("无关：资料讲的是别的病", "换个说法", "无关：问的是某医院排班")
        res = answer_with_check(pipe, llm, "问题", max_rounds=3, refine="query", max_irrelevant_retries=1)
        self.assertTrue(res["refused"])
        self.assertTrue(res["answer"].startswith(REFUSAL))
        self.assertIn("某医院排班", res["answer"])
        self.assertEqual(pipe.generator.calls, [])
        self.assertEqual((res["rounds"], res["llm_calls"], res["statuses"], res["stop_reason"]),
                         (2, 3, [IRRELEVANT, IRRELEVANT], "irrelevant"))

    def test_irrelevant_no_retry(self):
        pipe = FakePipeline({})
        res = answer_with_check(pipe, scripted("无关：不相干"), "问题", max_rounds=2, max_irrelevant_retries=0)
        self.assertTrue(res["refused"])
        self.assertEqual((res["rounds"], len(pipe.retrieved)), (1, 1))

    def test_irrelevant_then_partial_answers(self):
        pipe = FakePipeline({"问题": ["a", "b"], "换个说法": ["g", "x"]})
        res = answer_with_check(pipe, scripted("无关：没找对", "换个说法", "部分：缺剂量"), "问题", max_rounds=1, refine="query")
        self.assertFalse(res["refused"])
        self.assertTrue(res["improved"])

    def test_zero_extra_rounds_and_unparseable_check(self):
        pipe = FakePipeline({})
        res = answer_with_check(pipe, scripted("我也说不清"), "问题", max_rounds=0)
        self.assertFalse(res["refused"])  # 解析不了按"部分"处理：照常回答，不因格式问题拒答
        self.assertEqual(res["status"], PARTIAL)
        self.assertEqual(len(pipe.retrieved), 1)

    def test_parse_verdict(self):
        self.assertEqual(parse_verdict("充分"), (SUFFICIENT, ""))
        self.assertEqual(parse_verdict("**部分**：缺少剂量"), (PARTIAL, "缺少剂量"))
        self.assertEqual(parse_verdict("不充分：缺少剂量"), (PARTIAL, "缺少剂量"))  # 不会被当成"充分"
        self.assertEqual(parse_verdict("不足：旧写法"), (PARTIAL, "旧写法"))
        self.assertEqual(parse_verdict("<think>想想</think>无关：问的是排班\n多余的话"), (IRRELEVANT, "问的是排班"))
        self.assertEqual(parse_verdict(""), (PARTIAL, ""))
        self.assertEqual(parse_check_detail("部分（查不到）：本人的检查结果"), (PARTIAL, "本人的检查结果", False))
        self.assertEqual(parse_check_detail("**部分(查不到)**：医院安排"), (PARTIAL, "医院安排", False))
        self.assertEqual(parse_check_detail("部分：剂量"), (PARTIAL, "剂量", True))
        self.assertEqual(parse_check_detail("无关（查不到）：x")[0], IRRELEVANT)

    def test_similar(self):
        self.assertGreater(similar("清淡饮食多喝水注意休息", "清淡饮食多喝水注意休息吧"), 0.8)
        self.assertLess(similar("清淡饮食多喝水", "建议做胃镜检查"), 0.2)
        self.assertEqual(similar("", "x"), 0.0)

    def test_exclude_masks_documents(self):
        pipe = FakePipeline({"问题": ["gold", "a", "b"]})
        res = answer_with_check(pipe, scripted("充分"), "问题", exclude={"gold"})
        self.assertEqual([c["id"] for c in res["contexts"]], ["a", "b"])

    def test_model_refusal_is_detected(self):
        pipe = FakePipeline({})
        res = answer_with_check(pipe, scripted("充分"), "问题", answer_fn=lambda q, c: REFUSAL + "。")
        self.assertTrue(res["refused"])
        partial = REFUSAL + "具体剂量，但资料提示：这种情况一般先清淡饮食、多喝水，症状加重再就医[1]。"
        res = answer_with_check(pipe, scripted("充分"), "问题", answer_fn=lambda q, c: partial)
        self.assertFalse(res["refused"])  # "无法回答 X，但……"是部分回答，不算拒答

    def test_refine_fallback_when_repeated(self):
        self.assertEqual(refine_query(lambda p: "问题", "问题", "剂量", ["问题"], mode="query"), "问题 剂量")
        self.assertEqual(refine_query(lambda p: "", "问题", "剂量", ["问题"]), "问题 剂量")  # hyde 写不出来也退回
        self.assertEqual(refine_query(lambda p: "1. 新的查询\n2. 另一个", "问题", "剂量", ["问题"], mode="query"), "新的查询")

    def test_retrieve_multi_merges_queries(self):
        class Ret:
            def __init__(self, table):
                self.table = table

            def search(self, qs, k):
                return [self.table.get(q, []) for q in qs]

        pipe = FakePipeline({})
        pipe.bm25 = Ret({"q1": [("a", 3.0)], "q2": [("b", 2.0)]})
        pipe.dense = Ret({"q1": [("a", 0.9), ("g", 0.5)], "q2": [("g", 0.8)]})
        merged = retrieve_multi(pipe, ["q1", "q2"], top_k=3)
        self.assertEqual({d for d, _ in merged}, {"a", "b", "g"})
        self.assertLessEqual(len(merged), 3)


class TestAgentEval(unittest.TestCase):
    def test_parse_answer_judge(self):
        self.assertEqual(parse_answer_judge('{"refused": 0, "helpful": 2, "grounded": 1}'),
                         {"refused": 0, "helpful": 2, "grounded": 1})
        self.assertEqual(parse_answer_judge('结果：```json\n{"refused":1,"helpful":0,"grounded":1}\n```'),
                         {"refused": 1, "helpful": 0, "grounded": 1})
        self.assertEqual(parse_answer_judge("refused: 0, helpful: 1, grounded: 0"),
                         {"refused": 0, "helpful": 1, "grounded": 0})
        self.assertIsNone(parse_answer_judge('{"refused": 0, "helpful": 5, "grounded": 1}'))
        self.assertIsNone(parse_answer_judge("不知道"))

    def test_build_sets(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            data = d / "toy"
            data.mkdir()
            with open(data / "queries.jsonl", "w", encoding="utf-8") as f:
                for i in range(10):
                    f.write(json.dumps({"id": f"q{i}", "text": f"问题{i}"}, ensure_ascii=False) + "\n")
            (data / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\td{i}\t1\n" for i in range(10)), encoding="utf-8")
            (data / "splits.json").write_text(json.dumps({"dev": [f"q{i}" for i in range(10)], "test": [], "train": []}), encoding="utf-8")
            rel = d / "relabel"
            rel.mkdir()
            with open(rel / "toy_judgments.jsonl", "w", encoding="utf-8") as f:
                for i in range(10):
                    f.write(json.dumps({"qid": f"q{i}", "doc_id": f"x{i}", "grade": 2}) + "\n")
                    f.write(json.dumps({"qid": f"q{i}", "doc_id": f"y{i}", "grade": 0}) + "\n")
            ood = d / "ood.jsonl"
            ood.write_text(json.dumps({"id": "ood-001", "text": "知识库外"}, ensure_ascii=False) + "\n", encoding="utf-8")
            hard = d / "hard.jsonl"
            hard.write_text("".join(json.dumps({"id": f"hood-{i}", "text": f"新药{i}"}, ensure_ascii=False) + "\n"
                                    for i in range(3)), encoding="utf-8")
            items = build_sets(data, "dev", 4, 3, ood, 5, rel, seed=1, hard_ood_path=hard, n_hard_ood=2)
            self.assertNotIn("hard_ood", {it["set"] for it in build_sets(data, "dev", 1, 0, ood, 1, rel, hard_ood_path=hard)})
        sets = Counter(it["set"] for it in items)
        self.assertEqual(sets, Counter({"answerable": 4, "masked": 3, "ood": 1, "hard_ood": 2}))
        ans = {it["qid"] for it in items if it["set"] == "answerable"}
        masked = [it for it in items if it["set"] == "masked"]
        self.assertFalse(ans & {it["qid"] for it in masked})  # 两组不重叠
        for it in masked:
            i = it["qid"][1:]
            self.assertEqual(set(it["exclude"]), {f"d{i}", f"x{i}"})  # 原正例 + 判为相关的，不含判 0 分的

    def test_run_task_summarize_and_resume(self):
        pipe = FakePipeline({"问题": ["g", "a"]}, reranker=FakeReranker())
        judge = scripted(*['{"refused": 0, "helpful": 2, "grounded": 1}'] * 20)
        item = {"set": "answerable", "qid": "q1", "question": "问题", "exclude": []}
        recs = [run_task(pipe, judge, item, s) for s in ("baseline", "agent")]
        for r in recs:
            self.assertEqual(r["judge"], {"refused": 0, "helpful": 2, "grounded": 1})
            self.assertNotIn("contexts", r)
        self.assertEqual(recs[1]["usage"]["calls"], 2)  # agent：检查 + 回答（评审单独统计）
        self.assertEqual((recs[1]["status"], recs[1]["statuses"], recs[1]["improved"]), (SUFFICIENT, [SUFFICIENT], False))
        self.assertEqual((recs[1]["stop_reason"], recs[1]["context_changed"]), ("sufficient", False))
        bad = run_task(pipe, lambda p: "乱码", item, "baseline")
        self.assertIsNone(bad["judge"])
        with tempfile.TemporaryDirectory() as d:
            store = RecordStore(Path(d) / "records.jsonl")
            for r in recs + [bad]:
                store.add(r)
            again = RecordStore(Path(d) / "records.jsonl")
            self.assertTrue(again.done(("answerable", "q1", "agent")))
            self.assertEqual(len(again.records), 2)

        records = []
        for i in range(10):
            for s, ref, helpful in (("baseline", 0, 1), ("agent", int(i < 3), 2 if i >= 3 else 0)):
                r = {"set": "answerable", "qid": f"q{i}", "system": s, "rounds": 1,
                     "seconds": 1.0, "usage": {"calls": 2}, "judge": {"refused": ref, "helpful": helpful, "grounded": 1}}
                if s == "agent":  # 前 4 题触发了补检索，其中 2 题从"部分"变"充分"
                    st = [PARTIAL, SUFFICIENT] if i in (2, 3) else ([PARTIAL, PARTIAL] if i < 2 else [SUFFICIENT])
                    r.update({"rounds": len(st), "statuses": st, "status": st[-1], "improved": st != [PARTIAL, PARTIAL] and len(st) > 1})
                records.append(r)
        summ = summarize(records)
        ag = summ["by_set"]["answerable"]["agent"]
        self.assertAlmostEqual(ag["refusal_rate"], 0.3)
        self.assertAlmostEqual(summ["by_set"]["answerable"]["baseline"]["helpful_mean"], 1.0)
        self.assertIn("helpful", summ["tests"]["answerable"])
        self.assertAlmostEqual(ag["triggered_rate"], 0.4)
        self.assertAlmostEqual(ag["improved_rate"], 0.5)
        self.assertAlmostEqual(ag["triggered_helpful_delta"], (-1 - 1 - 1 + 1) / 4)  # q0~q2 helpful 0，q3 helpful 2；baseline 都是 1
        self.assertEqual(ag["first_status"], {PARTIAL: 4, SUFFICIENT: 6})
        self.assertNotIn("triggered_rate", summ["by_set"]["answerable"]["baseline"])
        # v3 记录：按 trace 里有没有 next_query 判断是否触发，统计资料是否真的变了、停止原因、"查不到"单独计
        v3 = [{"set": "masked", "qid": f"m{i}", "system": sy, "rounds": 1, "seconds": 1.0, "usage": {},
               "judge": {"refused": 0, "helpful": 1, "grounded": 1}} for i in range(3) for sy in ("baseline",)]
        v3 += [{"set": "masked", "qid": "m0", "system": "agent", "rounds": 1, "seconds": 1.0, "usage": {},
                "judge": {"refused": 0, "helpful": 1, "grounded": 1}, "statuses": [PARTIAL], "improved": False,
                "context_changed": False, "stop_reason": "context_unchanged",
                "trace": [{"status": PARTIAL, "retrievable": True, "next_query": "x"}]},
               {"set": "masked", "qid": "m1", "system": "agent", "rounds": 2, "seconds": 1.0, "usage": {},
                "judge": {"refused": 0, "helpful": 2, "grounded": 1}, "statuses": [PARTIAL, SUFFICIENT], "improved": True,
                "context_changed": True, "stop_reason": "sufficient",
                "trace": [{"status": PARTIAL, "retrievable": True, "next_query": "y"}, {"status": SUFFICIENT}]},
               {"set": "masked", "qid": "m2", "system": "agent", "rounds": 1, "seconds": 1.0, "usage": {},
                "judge": {"refused": 0, "helpful": 1, "grounded": 1}, "statuses": [PARTIAL], "improved": False,
                "context_changed": False, "stop_reason": "not_retrievable",
                "trace": [{"status": PARTIAL, "retrievable": False}]}]
        m = summarize(v3)["by_set"]["masked"]["agent"]
        self.assertAlmostEqual(m["triggered_rate"], 2 / 3)
        self.assertAlmostEqual(m["context_changed_rate"], 0.5)
        self.assertEqual(m["first_status"], {PARTIAL: 2, "partial_nr": 1})
        self.assertEqual(m["stop_reason"], {"context_unchanged": 1, "sufficient": 1, "not_retrievable": 1})
        est = estimate([item] * 3, pipe.cfg)
        self.assertLess(est["calls_min"], est["calls_max"])

    def test_compare_tags(self):
        def rec(q, sys_, helpful, out_tokens, s="answerable"):
            return {"set": s, "qid": q, "system": sys_, "seconds": 1.0,
                    "usage": {"calls": 1, "prompt_tokens": 100, "completion_tokens": out_tokens},
                    "judge": {"refused": 0, "helpful": helpful, "grounded": 1}}
        a = [rec(f"q{i}", "baseline", 1, 1000) for i in range(6)] + [rec("x", "agent", 2, 10)]
        b = [rec(f"q{i}", "baseline", 2 if i < 3 else 1, 300) for i in range(6)] + [rec("q9", "baseline", 0, 1)]
        cmp = compare_tags(a, b)
        row = cmp["answerable"]["baseline"]
        self.assertEqual(row["n"], 6)  # 只比两边都有的；q9 只在 B 里
        self.assertAlmostEqual(row["b"]["helpful_mean"] - row["a"]["helpful_mean"], 0.5)
        self.assertAlmostEqual(row["tests"]["helpful"]["delta"], 0.5)
        self.assertEqual((row["a"]["completion_tokens_mean"], row["b"]["completion_tokens_mean"]), (1000, 300))
        self.assertNotIn("agent", cmp["answerable"])  # agent 只在 A 里有


if __name__ == "__main__":
    unittest.main()
