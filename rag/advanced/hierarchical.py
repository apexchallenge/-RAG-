"""TODO（你来实现）：检索层级化 —— 父子分块（small-to-big）。

问题：块切得小，检索准，但交给大模型的上下文不完整；块切得大，向量被无关内容稀释，检索变差。
做法：
  1. 每篇文档先切成"父块"（~1200 字，接近一个小节），每个父块再切成"子块"（~300 字）；
  2. 只给子块建 BM25 + 向量索引（检索粒度细）；
  3. 检索到子块后映射回父块：同一父块的多个子块取最高分（max）或求和（sum），去重；
  4. 把父块交给重排和大模型（上下文完整）。
另一种层级：先用文档摘要/标题检索出相关文档，再在这些文档内部检索块（doc -> chunk 两阶段）。

评测（在你自己的长文档语料上做，公开数据集的短段落用不上）：
  - qrels 标注到文档或父块级别，对比"普通 400 字切块"和"父子切块"的 doc 级 Recall@5；
  - 大模型答案的完整性：同一批问题，人工或 LLM 打分（见 playbook）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Hits


@dataclass
class Chunk:
    id: str
    parent_id: str
    text: str


def build_parent_child(
    docs: list[dict],
    parent_size: int = 1200,
    child_size: int = 300,
    child_overlap: int = 50,
) -> tuple[list[dict], list[Chunk]]:
    """docs 是 [{"id", "text", ...}]。返回 (父块列表, 子块列表)。

    约定：父块 {"id": f"{doc_id}#p{i}", "text": ..., "doc_id": doc_id}；子块 Chunk(id=f"{parent_id}#c{j}", parent_id, text)。
    每个子块必须完整地落在一个父块内。提示：直接复用 rag.text.chunk_text。
    """
    raise NotImplementedError("TODO: 父子分块")


def children_to_parents(child_hits: Hits, child_to_parent: dict[str, str], agg: str = "max", top_k: int = 20) -> Hits:
    """把子块的检索结果聚合成父块排序。agg="max" 取最高分；agg="sum" 求和（偏向命中多个子块的父块）。

    注意：BM25 分数和余弦分数量纲不同，聚合前先融合（RRF）成一个列表再调用本函数。
    """
    raise NotImplementedError("TODO: 子块 -> 父块聚合")


class ParentChildRetriever:
    """子块上做 BM25 + Dense 混合检索 -> 聚合到父块 ->（可选）重排父块。

    建议直接复用 rag.bm25.BM25Retriever、rag.dense.DenseRetriever 和 rag.fusion.rrf_fuse，
    给子块单独建一套索引（例如 indexes/<数据集>_children/）。
    """

    def __init__(self, child_bm25, child_dense, child_to_parent: dict[str, str], parents: dict[str, dict]):
        self.child_bm25 = child_bm25
        self.child_dense = child_dense
        self.child_to_parent = child_to_parent
        self.parents = parents

    def search(self, queries: list[str], top_k: int = 20, child_k: int = 100) -> list[Hits]:
        raise NotImplementedError("TODO: 子块混合检索 + 聚合到父块")
