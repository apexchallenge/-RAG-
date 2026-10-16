"""大小模型分工 ——"分小模型"：小模型先做意图分类，再决定走哪条路。

思路：大部分问题用不着最贵的模型。让一个小模型（本地 Ollama 上的 Qwen3-1.7B / 4B，或 API 上便宜的模型）先分类：
  chitchat      打招呼、闲聊           -> 不检索，小模型一两句话回应
  out_of_scope  知识库不可能有答案     -> 不检索，直接说明超出范围（省掉检索 + 重排 + 大模型）
  simple        一个事实 / 一条建议    -> 检索 + 小模型写回答
  complex       要结合情况分析、综合   -> 检索 + 大模型写回答（和不路由时完全一样）
解析不了的分类结果一律当 complex：宁可多花钱，不要答错。

评测（scripts/eval_router.py）：
  - classify 模式：只跑小模型分类（不检索、不用大模型），看各组问题被分到哪条路 ——
    可回答问题被误判成 chitchat / out_of_scope 的比例（误拦率，直接丢答案）是最要紧的数字；
  - full 模式：同一批问题对比"全部走大模型"（big_only）和"路由"（router）的回答质量（盲评 helpful / 拒答 / 有依据）、
    延迟、大小模型各自的 token。写简历的是"质量不降的前提下，大模型 token 省了多少、延迟降了多少"。
"""

from __future__ import annotations

import re
import time
from typing import Callable, Literal

from ..data import doc_text
from ..generator import REFUSAL, build_messages
from ..utils import timer

Route = Literal["chitchat", "simple", "complex", "out_of_scope"]
ROUTES: tuple[str, ...] = ("chitchat", "simple", "complex", "out_of_scope")
LLM = Callable[[str], str]
AnswerFn = Callable[[str, list[str]], str]

DEFAULT_KB = "网上医疗问答社区里医生对患者提问的回答（常见病的症状、病因、用药、检查、日常护理和饮食建议）"

CLASSIFY_PROMPT = """你是问答系统的路由器。知识库内容：{kb_description}
把用户问题分到以下一类，只输出类别英文名：
chitchat     —— 打招呼、闲聊、问你是谁，与知识库无关
simple       —— 一个事实或一条通用建议就能回答（是什么、正常值是多少、能不能做某事、一般怎么处理）
complex      —— 需要结合提问者的具体情况分析、比较多种可能、多方面综合，或者问题里有好几问
out_of_scope —— 与知识库主题无关的问题；或者问题要的是知识库里不可能有的具体信息：
                某家医院 / 机构的排班和安排、某个人的账户或记录、今年 / 最新 / 实时的价格、政策和新闻
用户问题：{question}"""

CHITCHAT_PROMPT = """你是一个{kb_description}方面的问答助手。用户在和你闲聊，请用一两句话友好地回应，
并提醒可以向你咨询这方面的问题。不要编造任何医学建议。
用户：{question}"""

OUT_OF_SCOPE_REPLY = REFUSAL + "：这个问题需要的信息不在知识库范围内（知识库是{kb_description}），建议咨询相关机构或查看官方渠道。"

_THINK = re.compile(r"<think>.*?</think>", re.S)
# 每一类可能的写法；按出现位置取最早的一个（"类别：Complex。" -> complex）
_ALIASES: dict[str, tuple[str, ...]] = {
    "out_of_scope": ("out_of_scope", "out-of-scope", "out of scope", "outofscope", "超出范围", "知识库外"),
    "chitchat": ("chitchat", "chit-chat", "chit chat", "闲聊"),
    "simple": ("simple", "简单"),
    "complex": ("complex", "复杂"),
}


def parse_route(text: str) -> Route:
    """从小模型输出里解析类别（可能带思考过程、标点、大小写、解释文字）。解析不了时返回 "complex"：宁可多花钱，不要答错。"""
    t = _THINK.sub("", text or "")
    if "<think>" in t:  # 思考没写完就被截断：没有结论
        t = t.split("<think>")[0]
    low = t.lower()
    first = low.strip().splitlines()[0].strip(" \t*`'\"。.：:") if low.strip() else ""
    if first in ROUTES:
        return first  # type: ignore[return-value]
    best: tuple[int, str] | None = None
    for route, names in _ALIASES.items():
        for name in names:
            i = low.find(name)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, route)
    return best[1] if best else "complex"  # type: ignore[return-value]


def classify(small_llm: LLM, question: str, kb_description: str) -> Route:
    return parse_route(small_llm(CLASSIFY_PROMPT.format(kb_description=kb_description, question=question)))


def prompt_answer_fn(llm: LLM, max_chars_per_doc: int = 600) -> AnswerFn:
    """只有 llm(prompt) -> str 时，把带引用的回答提示词（系统 + 用户）拼成一段文本交给它。
    格式和 offline_agent_eval.py 的写回答提示词一致（[SYSTEM] / [USER]），离线回放时可以共用缓存。"""

    def answer(question: str, contexts: list[str]) -> str:
        msgs = build_messages(question, contexts, max_chars_per_doc)
        return llm(f"[SYSTEM]\n{msgs[0]['content']}\n\n[USER]\n{msgs[1]['content']}")

    return answer


class Router:
    """small_llm 做分类和闲聊回复；写回答默认也用 small_llm / big_llm 拼提示词，
    传入 small_answer / big_answer（例如 Generator.answer，带 system 消息）时用它们。"""

    def __init__(self, small_llm: LLM, big_llm: LLM, pipeline, kb_description: str = DEFAULT_KB,
                 small_answer: AnswerFn | None = None, big_answer: AnswerFn | None = None,
                 context_top_k: int | None = None, max_chars_per_doc: int | None = None):
        self.small_llm = small_llm
        self.big_llm = big_llm
        self.pipeline = pipeline
        self.kb_description = kb_description
        gcfg = (getattr(pipeline, "cfg", None) or {}).get("generation", {}) or {}
        self.k = int(context_top_k or gcfg.get("context_top_k", 5))
        chars = int(max_chars_per_doc or gcfg.get("max_chars_per_doc", 600))
        self.small_answer = small_answer or prompt_answer_fn(small_llm, chars)
        self.big_answer = big_answer or prompt_answer_fn(big_llm, chars)

    @classmethod
    def from_config(cls, pipeline, cfg: dict | None = None, small=None, small_usage=None, big_usage=None) -> "Router":
        """按配置建路由：大模型 = pipeline.generator（generation 段），小模型 = generation 段被 router.small 覆盖后的结果。
        small 传入已建好的小模型 Generator 时复用它（评测时每条问题建一个 Router、各自记 token，不必重复建客户端）。"""
        cfg = cfg or pipeline.cfg
        rcfg = cfg.get("router", {}) or {}
        big = pipeline.generator
        small = small or make_small_generator(cfg)
        if not small.available:
            raise RuntimeError("小模型不可用：检查 router.small 的 base_url / model / api_key_env")
        return cls(
            small_classify_llm(small, cfg, usage=small_usage),
            big.as_llm(thinking=big.thinking, max_tokens=None, temperature=float(big.cfg.get("temperature", 0.2)),
                       usage=big_usage),
            pipeline,
            rcfg.get("kb_description") or DEFAULT_KB,
            small_answer=lambda q, ctx: small.answer(q, ctx, usage=small_usage),
            big_answer=lambda q, ctx: big.answer(q, ctx, usage=big_usage),
        )

    def answer(self, question: str, exclude: set[str] | None = None) -> dict:
        """返回 {"route", "answer", "contexts", "timings", "model"}，方便统计每条路由的成本和延迟。

        model：写回答用的模型 "small" / "big"；out_of_scope 不调用任何模型写回答，记 None。
        timings：classify / retrieve / llm / total，单位秒。
        """
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        with timer(t, "classify"):
            route = classify(self.small_llm, question, self.kb_description)
        contexts: list[dict] = []
        model: str | None = None
        if route == "chitchat":
            model = "small"
            with timer(t, "llm"):
                answer = _THINK.sub("", self.small_llm(
                    CHITCHAT_PROMPT.format(kb_description=self.kb_description, question=question))).strip()
        elif route == "out_of_scope":
            answer = OUT_OF_SCOPE_REPLY.format(kb_description=self.kb_description)
        else:
            exclude = exclude or set()
            with timer(t, "retrieve"):
                final = self.pipeline.retrieve(question)["final"]
            hits = [(d, s) for d, s in final if d not in exclude][: self.k]
            contexts = [{"id": d, "score": s, "text": doc_text(self.pipeline.docs[d])} for d, s in hits]
            model = "small" if route == "simple" else "big"
            fn = self.small_answer if route == "simple" else self.big_answer
            with timer(t, "llm"):
                answer = _THINK.sub("", fn(question, [c["text"] for c in contexts])).strip()
        t["total"] = time.perf_counter() - t0
        return {"route": route, "answer": answer, "contexts": contexts, "timings": t, "model": model}


def make_small_generator(cfg: dict):
    """小模型的 Generator：generation 段的字段被 router.small 覆盖（没写的沿用 generation，例如 context_top_k）。"""
    from ..generator import Generator

    small_cfg = {**(cfg.get("generation") or {}), **((cfg.get("router") or {}).get("small") or {})}
    return Generator(small_cfg)


def small_classify_llm(small, cfg: dict, usage=None) -> LLM:
    """分类 / 闲聊用的小模型调用：温度 0；思考模式读 router.small.thinking；输出上限读 router.classify_max_tokens
    （Qwen3 这类默认开思考的模型，要么关思考，要么把上限留够，否则思考被截断就解析不出类别）。"""
    rcfg = cfg.get("router", {}) or {}
    return small.as_llm(thinking=(rcfg.get("small") or {}).get("thinking"),
                        max_tokens=rcfg.get("classify_max_tokens"), temperature=0.0, usage=usage)
