"""重排：交叉编码器（cross-encoder）把"查询 + 文档"拼在一起过一遍模型，直接输出相关性分数。

和双塔相比：能看到查询和文档之间逐词的交互，所以更准；但每个 (查询, 文档) 对都要跑一次模型，
没法离线预算，只能用在召回之后的前 top_n 个候选上。这就是"召回 -> 精排"两阶段的原因。
"""

from __future__ import annotations

import numpy as np

from .embedder import resolve_device
from .types import Hits


class Reranker:
    def __init__(self, model, batch_size: int = 32, name: str = ""):
        self.model = model
        self.batch_size = batch_size
        self.name = name

    @classmethod
    def from_config(cls, cfg: dict) -> "Reranker":
        from sentence_transformers import CrossEncoder

        from .paths import resolve_model

        device = resolve_device(cfg.get("device"))
        model = CrossEncoder(resolve_model(cfg["model"]), max_length=int(cfg.get("max_length", 512)), device=device)
        if cfg.get("fp16", True) and device.startswith("cuda"):
            model.half()
        print(f"[reranker] {cfg['model']} on {device}")
        return cls(model, batch_size=int(cfg.get("batch_size", 32)), name=cfg["model"])

    def warmup(self) -> None:
        """第一次调用 GPU 有初始化开销，计时前先跑一次。"""
        self.score_pairs([("预热", "预热")])

    def score_pairs(self, pairs: list[tuple[str, str]], show_progress: bool = False) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=show_progress, convert_to_numpy=True
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(
        self,
        queries: list[str],
        candidates: list[list[tuple[str, str]]],
        show_progress: bool = False,
    ) -> list[Hits]:
        """candidates[i] 是第 i 个查询的 [(doc_id, 文本), ...]。

        所有查询的 (查询, 文档) 对一次性送进模型，批次填得满，GPU 利用率更高。
        """
        pairs: list[tuple[str, str]] = []
        owners: list[tuple[int, str]] = []
        for qi, (q, cands) in enumerate(zip(queries, candidates)):
            for doc_id, text in cands:
                pairs.append((q, text))
                owners.append((qi, doc_id))
        scores = self.score_pairs(pairs, show_progress=show_progress)
        results: list[Hits] = [[] for _ in queries]
        for (qi, doc_id), s in zip(owners, scores):
            results[qi].append((doc_id, float(s)))
        for r in results:
            r.sort(key=lambda x: (-x[1], x[0]))
        return results
