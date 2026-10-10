"""测试用的假模型：接口和 SentenceTransformer / CrossEncoder 一致，但不需要 torch 和下载权重，几毫秒就能跑完。"""

from __future__ import annotations

import copy
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"


def bigrams(text: str) -> list[str]:
    return [text[i : i + 2] for i in range(len(text) - 1)] or [text]


class FakeSentenceTransformer:
    """字符 bigram 哈希成 dim 维向量，字面越像向量越近。"""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.max_seq_length = 512
        self.calls: list[list[str]] = []

    def encode(self, texts, prompt_name=None, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=False, **kwargs):
        texts = list(texts)
        self.calls.append(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for g in bigrams(t):
                out[i, zlib.crc32(g.encode("utf-8")) % self.dim] += 1.0
        if normalize_embeddings:
            out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return out


class FakeCrossEncoder:
    """分数 = 查询 bigram 在文档中出现的比例。"""

    def predict(self, pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True, **kwargs):
        scores = []
        for q, d in pairs:
            qg = set(bigrams(q))
            scores.append(len(qg & set(bigrams(d))) / max(1, len(qg)))
        return np.asarray(scores, dtype=np.float32)


def fake_embedder(name: str = "fake-embedder"):
    from rag.embedder import Embedder

    return Embedder(FakeSentenceTransformer(), name=name, query_instruction="")


def fake_reranker():
    from rag.reranker import Reranker

    return Reranker(FakeCrossEncoder(), batch_size=16, name="fake-reranker")


def demo_config(index_dir: str | Path, data_dir: str | Path = DEMO_DIR) -> dict:
    from rag.config import load_config

    cfg = copy.deepcopy(load_config(ROOT / "configs" / "demo.yaml"))
    cfg["data_dir"] = str(data_dir)
    cfg["index_dir"] = str(index_dir)
    cfg["embedder"]["model"] = "fake-embedder"
    cfg["reranker"]["model"] = "fake-reranker"
    return cfg
