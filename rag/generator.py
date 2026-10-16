"""生成：把重排后的前几条资料拼进提示词，调用任意 OpenAI 兼容接口。

DeepSeek、通义千问(DashScope 兼容模式)、硅基流动、本地 Ollama / vLLM 都提供 OpenAI 兼容接口，
只要改 configs 里的 base_url 和 model 即可切换。

思考模式（generation.thinking / agent.thinking）：
    null  不传参数，用服务商默认（deepseek-flash 默认开思考）
    false 关闭思考：更快更便宜，适合"格式化判断"类的步骤（充分性检查、改写查询）
    true  开启思考
参数按 DeepSeek 的写法放在 extra_body={"thinking": {"type": "enabled|disabled"}} 里；
别的服务商如果不认这个参数，保持 null 即可（null 时完全不传）。
别的服务商关思考的写法不一样：用配置里的 extra_body（原样并进请求体）或 prompt_suffix（拼在用户消息末尾，
例如 Qwen3 的 " /no_think"）。本地 Ollama 哪种管用，用 python scripts/eval_router.py probe 实测。
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Callable

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请只依据【参考资料】回答用户的问题：\n"
    "1. 每个结论后用 [编号] 标注出处，例如 [1][3]；\n"
    "2. 资料只能回答一部分时，回答能回答的部分，并说明哪些方面资料没有涉及；"
    "资料和问题完全无关时，直接回答“根据现有资料无法回答”；不要编造资料里没有的内容；\n"
    "3. 先给结论，再给依据，语言简洁。"
)
REFUSAL = "根据现有资料无法回答"


def build_messages(question: str, contexts: list[str], max_chars_per_doc: int = 600) -> list[dict]:
    blocks = [f"[{i}] {text[:max_chars_per_doc]}" for i, text in enumerate(contexts, start=1)]
    user = "【参考资料】\n" + "\n\n".join(blocks) + f"\n\n【问题】{question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def thinking_kwargs(thinking: bool | None) -> dict:
    if thinking is None:
        return {}
    return {"extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}


class Generator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.base_url = self.cfg.get("base_url", "")
        self.model = self.cfg.get("model", "")
        key = os.environ.get(self.cfg.get("api_key_env", "LLM_API_KEY"), "")
        self.is_local = any(h in self.base_url for h in ("localhost", "127.0.0.1"))
        self.api_key = key or ("EMPTY" if self.is_local else "")  # 本地服务一般不校验 key
        self.thinking = self.cfg.get("thinking")
        self.usage: Counter = Counter()  # 累计调用次数和 token，用来统计成本
        self._lock = threading.Lock()
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def _get_client(self):
        if not self.available:
            raise RuntimeError(
                f"没有可用的大模型：请设置环境变量 {self.cfg.get('api_key_env', 'LLM_API_KEY')}（或写进 .env），"
                "或在配置里把 base_url 指向本地 Ollama/vLLM"
            )
        if self._client is None:
            from openai import OpenAI

            kwargs = {}
            if self.is_local:
                # 本地服务（Ollama 等）不走代理：开着 Clash 之类的系统代理时，httpx 会把 localhost 的请求也发给代理，
                # 结果每次调用都失败（502 / 连接被拒）。trust_env=False 让这个客户端忽略代理环境变量和系统代理设置。
                import httpx

                kwargs["http_client"] = httpx.Client(trust_env=False, timeout=float(self.cfg.get("timeout", 300)))
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=float(self.cfg.get("timeout", 300 if self.is_local else 120)),
                                  max_retries=int(self.cfg.get("max_retries", 1 if self.is_local else 3)), **kwargs)
        return self._client

    def complete(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None,
                 thinking: bool | None | str = "default", usage: Counter | None = None) -> str:
        """通用调用；thinking="default" 表示用配置里的 generation.thinking；usage 传入时额外记一份（按任务统计成本）。"""
        thinking = self.thinking if thinking == "default" else thinking
        kwargs = {"model": self.model, "messages": messages,
                  "temperature": float(self.cfg.get("temperature", 0.2) if temperature is None else temperature)}
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        suffix = self.cfg.get("prompt_suffix")
        if suffix and messages and messages[-1].get("role") == "user":
            # 例如 Qwen3 的 " /no_think"：写在最后一条用户消息末尾，混合思考模型就不输出思考过程
            kwargs["messages"] = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + suffix}]
        extra = thinking_kwargs(thinking)
        if self.cfg.get("extra_body"):
            # 服务商自己的参数，原样放进请求体，例如 Ollama 的 {"think": false} / {"reasoning_effort": "none"}
            extra = {"extra_body": {**extra.get("extra_body", {}), **self.cfg["extra_body"]}}
        resp = self._get_client().chat.completions.create(**kwargs, **extra)
        u = getattr(resp, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0) if u is not None else 0
        ct = int(getattr(u, "completion_tokens", 0) or 0) if u is not None else 0
        with self._lock:
            for c in (self.usage, usage):
                if c is not None:
                    c["calls"] += 1
                    c["prompt_tokens"] += pt
                    c["completion_tokens"] += ct
        return resp.choices[0].message.content or ""

    def answer(self, question: str, contexts: list[str], usage: Counter | None = None) -> str:
        messages = build_messages(question, contexts, int(self.cfg.get("max_chars_per_doc", 600)))
        return self.complete(messages, usage=usage)

    def as_llm(self, thinking: bool | None = False, max_tokens: int | None = 1024, temperature: float = 0.0,
               usage: Counter | None = None) -> Callable[[str], str]:
        """包装成 llm(prompt) -> str，给 agent 里的判断步骤（充分性检查、改写）和评审用。"""

        def llm(prompt: str) -> str:
            return self.complete([{"role": "user", "content": prompt}], temperature=temperature,
                                 max_tokens=max_tokens, thinking=thinking, usage=usage)

        return llm
