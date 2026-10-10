"""TODO（你来实现）：查询改写 + 推理（"加点 reasoning"）。

四个由浅入深的功能，每个都要用评测说话（在 dev 上调，在 test 上报告；同时记录多花的延迟和 token）：
  1. multi_query：大模型把问题改写成 3 个不同说法，分别检索后用 RRF 合并 —— 解决口语化、表述不全
  2. hyde：大模型先"假想"一段答案，用它做向量检索（HyDE, Gao et al. 2022）—— 短问题找长文档
  3. decompose：把需要多条证据的问题拆成子问题，逐个检索再汇总（IRCoT 的思路）
  4. answer_with_check：回答前让大模型判断"资料够不够"；不够就改写再检索，最多 max_rounds 轮，
     仍不够就拒答（CRAG / Self-RAG 的简化版）—— 这一步最能体现"reasoning"，也最能降低幻觉

llm 统一约定为一个函数 llm(prompt: str) -> str，单测里用假模型替换；真实调用可以包一下 rag.generator 里的 OpenAI 客户端。
"""

from __future__ import annotations

from typing import Callable

from ..types import Hits

LLM = Callable[[str], str]

REWRITE_PROMPT = """请把下面的用户问题改写成 {n} 个意思相同、但措辞不同的检索查询，便于在知识库中搜索。
要求：补全省略的主语和关键词，把口语换成书面语；每行一个，不要编号，不要解释。
用户问题：{question}"""

HYDE_PROMPT = """请针对下面的问题，写一段约 150 字、像知识库原文一样的回答段落（可以不完全准确，只用于检索）。
问题：{question}"""

DECOMPOSE_PROMPT = """下面的问题可能需要多条资料才能回答。请把它拆成最多 {n} 个可以独立检索的子问题，每行一个，不要编号。
如果问题本身已经足够简单，只输出原问题。
问题：{question}"""

CHECK_PROMPT = """根据【参考资料】判断能否完整回答【问题】。只输出一行：
"充分" —— 资料足以回答；或 "不足：<还缺什么信息>"。
【参考资料】
{contexts}
【问题】{question}"""


def parse_lines(text: str, max_n: int = 3) -> list[str]:
    """把大模型输出的多行文本解析成干净的列表：去掉 "1." "1、" "-" "•" 等前缀、空行和重复项，最多 max_n 条。"""
    raise NotImplementedError("TODO: 解析多行输出")


def multi_query(llm: LLM, question: str, n: int = 3) -> list[str]:
    """返回 [原问题] + 最多 n 个改写（原问题放第一个，保证不会比不改写更差太多）。"""
    raise NotImplementedError("TODO: 多查询改写")


def hyde(llm: LLM, question: str) -> str:
    raise NotImplementedError("TODO: HyDE")


def decompose(llm: LLM, question: str, n: int = 3) -> list[str]:
    raise NotImplementedError("TODO: 问题分解")


def retrieve_multi(pipeline, queries: list[str], top_k: int = 100) -> Hits:
    """每个查询分别走 pipeline 的 BM25 + 向量 + 融合，再用 rag.fusion.rrf_fuse 合并成一个列表。"""
    raise NotImplementedError("TODO: 多路查询检索 + RRF")


def parse_check(text: str) -> tuple[bool, str]:
    """解析 CHECK_PROMPT 的输出：返回 (是否充分, 缺失信息)。解析不了时按"不足"处理，宁可多查一轮。"""
    raise NotImplementedError("TODO: 解析充分性判断")


def answer_with_check(pipeline, llm: LLM, question: str, max_rounds: int = 2) -> dict:
    """检索 -> 判断充分性 -> 不足则根据缺失信息改写查询再检索 -> 回答或拒答。返回答案、轮数、每轮的查询。"""
    raise NotImplementedError("TODO: 带充分性检查的回答")
