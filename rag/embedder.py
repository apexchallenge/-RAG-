"""稠密向量编码：用 sentence-transformers 加载 BGE（或任何同类模型）。

双塔（bi-encoder）：查询和文档分别编码成向量，文档向量可以离线算好存进 Faiss，
所以在线只需编码一个查询，速度快；代价是查询和文档之间没有逐词交互，精度不如重排模型。
"""

from __future__ import annotations

import numpy as np


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def to_unit_float32(x) -> np.ndarray:
    """转成 float32 并做 L2 归一化：归一化后内积 = 余弦相似度，Faiss 用内积索引即可。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.clip(norms, 1e-12, None))


class Embedder:
    """对 SentenceTransformer 薄薄包一层，统一处理：查询指令、归一化、float32。"""

    def __init__(
        self,
        model,
        name: str = "",
        query_instruction: str = "",
        query_prompt_name: str | None = None,
        batch_size: int = 64,
    ):
        self.model = model
        self.name = name
        self.query_instruction = query_instruction or ""
        self.query_prompt_name = query_prompt_name
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, cfg: dict) -> "Embedder":
        from sentence_transformers import SentenceTransformer  # 延迟导入：只跑 BM25 时不用装 torch

        from .paths import resolve_model

        device = resolve_device(cfg.get("device"))
        model = SentenceTransformer(resolve_model(cfg["model"]), device=device)
        if cfg.get("max_seq_length"):
            model.max_seq_length = int(cfg["max_seq_length"])
        if cfg.get("fp16", True) and device.startswith("cuda"):
            model.half()  # 半精度：显存减半、速度约翻倍，检索精度几乎不变
        print(f"[embedder] {cfg['model']} on {device}, max_seq_length={model.max_seq_length}")
        return cls(
            model,
            name=cfg["model"],
            query_instruction=cfg.get("query_instruction") or "",
            query_prompt_name=cfg.get("query_prompt_name"),
            batch_size=int(cfg.get("batch_size", 64)),
        )

    @property
    def max_seq_length(self) -> int | None:
        return getattr(self.model, "max_seq_length", None)

    def warmup(self) -> None:
        """第一次调用 GPU 有初始化开销，计时前先跑一次。"""
        self.encode_queries(["预热"])

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        if self.query_prompt_name:  # 例如 Qwen3-Embedding：模型自带 "query" 提示词
            emb = self.model.encode(
                texts,
                prompt_name=self.query_prompt_name,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:  # BGE-zh-v1.5：查询前拼一句指令（短查询找长文档时通常更好，可做消融）
            emb = self.model.encode(
                [self.query_instruction + t for t in texts],
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return to_unit_float32(emb)

    def encode_docs(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return to_unit_float32(emb)
