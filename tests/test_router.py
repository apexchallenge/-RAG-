"""大小模型路由（rag/advanced/router.py）和路由评测（rag/router_eval.py）的单测：假检索、假大模型，不调用 API。"""

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.advanced.router import (  # noqa: E402
    CLASSIFY_PROMPT, ROUTES, Router, classify, make_small_generator, parse_route, prompt_answer_fn,
)
from rag.generator import REFUSAL  # noqa: E402
from rag.router_eval import (  # noqa: E402
    build_router_sets, cost_of, run_classify, run_router_task, summarize_classify, summarize_full,
)

ROOT = Path(__file__).resolve().parents[1]


class FakePipeline:
    def __init__(self):
        self.cfg = {"generation": {"context_top_k": 2, "max_chars_per_doc": 50}}
        self.docs = {d: {"id": d, "text": f"文档{d}的内容"} for d in ("a", "b", "c")}
        self.retrieved = []

    def retrieve(self, q):
        self.retrieved.append(q)
        return {"final": [("a", 0.9), ("b", 0.8), ("c", 0.7)]}


def fake_small(route: str):
    """分类提示词返回 route，其余（闲聊、写回答）返回固定文本；记录收到的提示词。"""
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        if prompt.startswith("你是问答系统的路由器"):
            return route
        if prompt.startswith("[SYSTEM]"):
            return "小模型回答[1]"
        return "<think>想一想</think>你好！有问题可以问我。"
    llm.prompts = prompts
    return llm


def fake_big():
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "大模型回答[1][2]"
    llm.prompts = prompts
    return llm


class TestParseRoute(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(parse_route("simple"), "simple")
        self.assertEqual(parse_route("类别：Complex。"), "complex")
        self.assertEqual(parse_route("out_of_scope"), "out_of_scope")
        self.assertEqual(parse_route("Out of scope"), "out_of_scope")
        self.assertEqual(parse_route("**chitchat**"), "chitchat")
        self.assertEqual(parse_route("<think>可能是 simple，也可能是 complex</think>\nchitchat"), "chitchat")
        self.assertEqual(parse_route("这是闲聊"), "chitchat")
        self.assertEqual(parse_route("complex\n（因为不是 simple）"), "complex")
        self.assertEqual(parse_route("嗯……不确定"), "complex")
        self.assertEqual(parse_route(""), "complex")
        self.assertEqual(parse_route(None), "complex")
        # 思考被截断、没有结论：不能从思考里捞一个类别出来
        self.assertEqual(parse_route("<think>用户在打招呼，应该是 chitchat 吧"), "complex")

    def test_classify_prompt(self):
        llm = fake_small("simple")
        self.assertEqual(classify(llm, "感冒能喝酒吗", "医疗问答"), "simple")
        self.assertIn("感冒能喝酒吗", llm.prompts[0])
        self.assertIn("医疗问答", llm.prompts[0])
        self.assertTrue(set(ROUTES) <= {w for w in ROUTES if w in CLASSIFY_PROMPT})


class TestRouter(unittest.TestCase):
    def make(self, route):
        pipe = FakePipeline()
        small, big = fake_small(route), fake_big()
        return Router(small, big, pipe, "医疗问答"), pipe, small, big

    def test_chitchat_no_retrieval(self):
        r, pipe, small, big = self.make("chitchat")
        res = r.answer("你好")
        self.assertEqual(res["route"], "chitchat")
        self.assertEqual(res["model"], "small")
        self.assertEqual(res["contexts"], [])
        self.assertEqual(pipe.retrieved, [])
        self.assertEqual(big.prompts, [])
        self.assertNotIn("<think>", res["answer"])
        self.assertIn("total", res["timings"])

    def test_out_of_scope_no_model_no_retrieval(self):
        r, pipe, small, big = self.make("out_of_scope")
        res = r.answer("协和医院周三谁出诊")
        self.assertEqual(res["route"], "out_of_scope")
        self.assertIsNone(res["model"])
        self.assertTrue(res["answer"].startswith(REFUSAL))
        self.assertEqual(len(small.prompts), 1)  # 只有分类
        self.assertEqual(pipe.retrieved, [])
        self.assertEqual(big.prompts, [])

    def test_simple_uses_small(self):
        r, pipe, small, big = self.make("simple")
        res = r.answer("感冒能喝酒吗", exclude={"a"})
        self.assertEqual(res["model"], "small")
        self.assertEqual([c["id"] for c in res["contexts"]], ["b", "c"])  # 屏蔽 a，取前 k=2
        self.assertEqual(big.prompts, [])
        self.assertTrue(small.prompts[-1].startswith("[SYSTEM]"))
        self.assertIn("文档b的内容", small.prompts[-1])
        self.assertEqual(res["answer"], "小模型回答[1]")
        self.assertIn("retrieve", res["timings"])

    def test_complex_and_unparseable_use_big(self):
        for out in ("complex", "我不知道"):
            r, pipe, small, big = self.make(out)
            res = r.answer("头疼发烧还咳嗽，吃了药没用，要不要去医院")
            self.assertEqual(res["route"], "complex")
            self.assertEqual(res["model"], "big")
            self.assertEqual(len(big.prompts), 1)
            self.assertEqual(pipe.retrieved, ["头疼发烧还咳嗽，吃了药没用，要不要去医院"])

    def test_custom_answer_fns(self):
        pipe = FakePipeline()
        seen = []
        r = Router(fake_small("complex"), fake_big(), pipe, "医疗问答",
                   big_answer=lambda q, ctx: seen.append((q, ctx)) or "带 system 的回答")
        self.assertEqual(r.answer("问题")["answer"], "带 system 的回答")
        self.assertEqual(seen[0][1], ["文档a的内容", "文档b的内容"])

    def test_prompt_answer_fn_truncates(self):
        got = []
        fn = prompt_answer_fn(lambda p: got.append(p) or "ok", max_chars_per_doc=3)
        fn("问", ["一二三四五"])
        self.assertIn("[1] 一二三\n", got[0] + "\n")
        self.assertNotIn("四", got[0])

    def test_make_small_generator_overrides(self):
        cfg = {"generation": {"base_url": "https://api.x.com", "model": "big", "context_top_k": 5},
               "router": {"small": {"base_url": "http://localhost:11434/v1", "model": "qwen3:4b"}}}
        g = make_small_generator(cfg)
        self.assertEqual(g.model, "qwen3:4b")
        self.assertEqual(g.cfg["context_top_k"], 5)
        self.assertTrue(g.available)  # 本地服务不需要 key


class TestPreflight(unittest.TestCase):
    def test_preflight_explains_failure_and_passes(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import eval_router

        class Small:
            def __init__(self, exc=None):
                self.exc = exc

            def as_llm(self, **kw):
                def llm(p):
                    if self.exc:
                        raise self.exc
                    return "<think>嗯</think>simple"
                return llm

        cfg = {"router": {"small": {"model": "qwen3:4b", "base_url": "http://localhost:11434/v1"}}}
        item = {"question": "感冒能喝酒吗"}
        with self.assertRaises(SystemExit) as cm:
            eval_router.preflight(Small(ConnectionError("Connection refused")), cfg, "kb", item)
        self.assertIn("Ollama", str(cm.exception.code))
        with self.assertRaises(SystemExit) as cm:
            eval_router.preflight(Small(RuntimeError("Error code: 404 - model 'qwen3:4b' not found")), cfg, "kb", item)
        self.assertIn("ollama list", str(cm.exception.code))
        eval_router.preflight(Small(), cfg, "kb", item)  # 正常时不抛异常


class TestRouterEval(unittest.TestCase):
    def test_build_sets_has_chitchat(self):
        items = build_router_sets(ROOT / "data" / "demo", "all", 3, ROOT / "eval_sets" / "ood_questions.jsonl", 2,
                                  ROOT / "eval_sets" / "ood_hard_questions.jsonl", 1,
                                  ROOT / "eval_sets" / "chitchat_questions.jsonl", 4)
        self.assertEqual(Counter(it["set"] for it in items), {"answerable": 3, "ood": 2, "hard_ood": 1, "chitchat": 4})

    def test_run_classify_and_summarize(self):
        items = [{"set": "answerable", "qid": "q1", "question": "a"}, {"set": "answerable", "qid": "q2", "question": "b"},
                 {"set": "chitchat", "qid": "c1", "question": "你好"}, {"set": "ood", "qid": "o1", "question": "排班"}]
        outs = {"a": "simple", "b": "out_of_scope", "你好": "chitchat", "排班": "complex"}

        def llm(prompt):
            q = prompt.rsplit("用户问题：", 1)[1]
            return outs[q]

        def boom(prompt):
            raise TimeoutError("超时")

        recs = [run_classify(llm, it, "kb") for it in items]
        recs.append(run_classify(boom, {"set": "ood", "qid": "o2", "question": "x"}, "kb"))
        self.assertIsNone(recs[-1]["route"])
        s = summarize_classify(recs)
        self.assertAlmostEqual(s["overall"]["misblock_rate"], 0.5)
        self.assertAlmostEqual(s["overall"]["simple_rate"], 0.5)
        self.assertEqual(s["overall"]["n_errors"], 1)
        self.assertAlmostEqual(s["by_set"]["answerable"]["expected_rate"], 0.5)
        self.assertAlmostEqual(s["by_set"]["chitchat"]["expected_rate"], 1.0)
        self.assertAlmostEqual(s["by_set"]["ood"]["expected_rate"], 0.0)

    def test_run_router_task_and_summarize_full(self):
        class Gen:
            available = True
            thinking = None
            cfg = {"temperature": 0.2}

            def __init__(self, route=None):
                self.route = route

            def answer(self, q, ctx, usage=None):
                if usage is not None:
                    usage.update({"calls": 1, "prompt_tokens": 100, "completion_tokens": 50})
                return "回答[1]"

            def as_llm(self, thinking=False, max_tokens=None, temperature=0.0, usage=None):
                def llm(p):
                    if usage is not None:
                        usage.update({"calls": 1, "prompt_tokens": 10, "completion_tokens": 1})
                    return self.route or "x"
                return llm

        pipe = FakePipeline()
        pipe.generator = Gen()
        judge = lambda p: '{"refused": 0, "helpful": 2, "grounded": 1}'  # noqa: E731
        recs = []
        for i, route in enumerate(["simple", "complex", "simple", "chitchat", "complex", "simple"]):
            it = {"set": "answerable", "qid": f"q{i}", "question": f"问题{i}", "exclude": []}
            for sys_ in ("big_only", "router"):
                rec = run_router_task(pipe, Gen(route), judge, it, sys_, pipe.cfg)
                self.assertIsNotNone(rec["judge"], rec.get("error"))
                recs.append(rec)
        router_recs = [r for r in recs if r["system"] == "router"]
        self.assertEqual(router_recs[0]["route"], "simple")
        self.assertEqual(router_recs[0]["usage"]["big"], {})
        self.assertEqual(router_recs[0]["usage"]["small"]["calls"], 2)  # 分类 + 写回答
        self.assertEqual(router_recs[1]["usage"]["big"]["calls"], 1)
        s = summarize_full(recs, prices={"big": (2.0, 8.0), "small": (0.0, 0.0)})
        o = s["overall"]
        self.assertEqual(o["n_pairs"], 6)
        self.assertEqual(o["big_tokens_big_only"], 6 * 150)
        self.assertEqual(o["big_tokens_router"], 2 * 150)
        self.assertAlmostEqual(o["big_token_saving"], 1 - 2 / 6)
        self.assertAlmostEqual(o["cost_per_q_big_only"], (100 * 2 + 50 * 8) / 1e6)
        self.assertEqual(s["by_route"]["simple"]["n"], 3)
        self.assertEqual(s["by_set"]["answerable"]["router"]["routes"], {"simple": 3, "complex": 2, "chitchat": 1})
        self.assertIn("helpful", s["tests"]["answerable"])

    def test_cost_of(self):
        rec = {"usage": {"small": {"prompt_tokens": 1000, "completion_tokens": 0},
                         "big": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}}}
        self.assertAlmostEqual(cost_of(rec, {"big": (1.0, 2.0), "small": (0.5, 0.5)}), 1.0 + 1.0 + 0.0005)

    def test_error_is_recorded(self):
        class Bad:
            cfg = {}
            docs = {}
            generator = None

            def retrieve(self, q):
                raise RuntimeError("索引坏了")

        rec = run_router_task(Bad(), None, None, {"set": "answerable", "qid": "q", "question": "x"}, "big_only", {})
        self.assertIsNone(rec["judge"])
        self.assertIn("Error", rec["error"])


if __name__ == "__main__":
    unittest.main()
