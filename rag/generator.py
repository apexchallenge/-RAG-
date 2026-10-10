"""生成：把重排后的前几条资料拼进提示词，调用任意 OpenAI 兼容接口。

DeepSeek、通义千问(DashScope 兼容模式)、硅基流动、本地 Ollama / vLLM 都提供 OpenAI 兼容接口，
只要改 configs 里的 base_url 和 model 即可切换。
"""

from __future__ import annotations

import os

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请只依据【参考资料】回答用户的问题：\n"
    "1. 每个结论后用 [编号] 标注出处，例如 [1][3]；\n"
    "2. 资料不足以回答时，直接回答“根据现有资料无法回答”，不要编造；\n"
    "3. 先给结论，再给依据，语言简洁。"
)


def build_messages(question: str, contexts: list[str], max_chars_per_doc: int = 600) -> list[dict]:
    blocks = [f"[{i}] {text[:max_chars_per_doc]}" for i, text in enumerate(contexts, start=1)]
    user = "【参考资料】\n" + "\n\n".join(blocks) + f"\n\n【问题】{question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class Generator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.base_url = self.cfg.get("base_url", "")
        self.model = self.cfg.get("model", "")
        key = os.environ.get(self.cfg.get("api_key_env", "LLM_API_KEY"), "")
        is_local = any(h in self.base_url for h in ("localhost", "127.0.0.1"))
        self.api_key = key or ("EMPTY" if is_local else "")  # 本地服务一般不校验 key

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def answer(self, question: str, contexts: list[str]) -> str:
        if not self.available:
            raise RuntimeError(
                f"没有可用的大模型：请设置环境变量 {self.cfg.get('api_key_env', 'LLM_API_KEY')}，"
                "或在配置里把 base_url 指向本地 Ollama/vLLM"
            )
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        messages = build_messages(question, contexts, int(self.cfg.get("max_chars_per_doc", 600)))
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=float(self.cfg.get("temperature", 0.2)),
        )
        return resp.choices[0].message.content or ""
