"""TODO（你来实现）：大小模型分工 ——"分小模型"。

思路：大部分问题用不着最贵的模型。让一个小模型（本地 Qwen3-1.7B 之类，或 API 上便宜的模型）先做：
  - 意图分类：闲聊 / 简单事实 / 复杂推理 / 超出知识库范围
  - 查询改写（可以复用 query_rewrite.py）
然后路由：
  - chitchat、out_of_scope -> 不检索，直接简短回复或拒答
  - simple                 -> 检索 + 小模型生成
  - complex                -> 检索 + 大模型（或推理模型）生成
评测：同一批 100~200 个问题，对比"全部走大模型"和"路由"两种方案的
     答案正确率（LLM 打分 + 人工抽查）、平均延迟、token 成本。写简历的是这三列数字。
"""

from __future__ import annotations

from typing import Callable, Literal

Route = Literal["chitchat", "simple", "complex", "out_of_scope"]
ROUTES: tuple[str, ...] = ("chitchat", "simple", "complex", "out_of_scope")
LLM = Callable[[str], str]

CLASSIFY_PROMPT = """你是问答系统的路由器。知识库内容：{kb_description}
把用户问题分到以下一类，只输出类别英文名：
chitchat     —— 打招呼、闲聊，与知识库无关
simple       —— 一个事实就能回答（时间、数字、规定是什么）
complex      —— 需要比较、推理、多条规定综合，或问"为什么/怎么办"
out_of_scope —— 与知识库主题无关的专业问题
用户问题：{question}"""


def parse_route(text: str) -> Route:
    """从小模型输出里解析类别（可能带标点、大小写、解释文字）。解析不了时返回 "complex"：宁可多花钱，不要答错。"""
    raise NotImplementedError("TODO: 解析路由类别")


def classify(small_llm: LLM, question: str, kb_description: str) -> Route:
    raise NotImplementedError("TODO: 小模型意图分类")


class Router:
    def __init__(self, small_llm: LLM, big_llm: LLM, pipeline, kb_description: str):
        self.small_llm = small_llm
        self.big_llm = big_llm
        self.pipeline = pipeline
        self.kb_description = kb_description

    def answer(self, question: str) -> dict:
        """返回 {"route", "answer", "contexts", "timings", "model"}，方便统计每条路由的成本和延迟。"""
        raise NotImplementedError("TODO: 按类别路由")
