"""BM25 稀疏检索：jieba 分词 + bm25s（scipy 稀疏矩阵实现，比 rank_bm25 快一到两个数量级）。

BM25 打分（面试常问）：
    score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
    k1 控制词频饱和（同一个词出现第 10 次不比第 2 次重要多少），b 控制长文档惩罚。
bm25s 在建索引时就把每个 (词, 文档) 的分数算好了，所以改 k1 / b / 分词方式都要重建索引。
"""

from __future__ import annotations

import json
from pathlib import Path

import bm25s

from .text import ChineseTokenizer
from .types import Hits
from .utils import progress

PARAMS_FILE = "build_params.json"


class BM25Retriever:
    def __init__(self, tokenizer: ChineseTokenizer, k1: float = 1.2, b: float = 0.75):
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.model: bm25s.BM25 | None = None
        self.doc_ids: list[str] = []

    @property
    def build_params(self) -> dict:
        return {**self.tokenizer.params, "k1": self.k1, "b": self.b}

    def build(self, doc_ids: list[str], texts: list[str], show_progress: bool = True) -> None:
        corpus_tokens = [self.tokenizer(t) for t in progress(texts, "jieba 分词", not show_progress)]
        self.model = bm25s.BM25(k1=self.k1, b=self.b)
        self.model.index(corpus_tokens, show_progress=show_progress)
        self.doc_ids = list(doc_ids)

    def search(self, queries: list[str], top_k: int) -> list[Hits]:
        if self.model is None:
            raise RuntimeError("BM25 索引还没建，先运行 scripts/build_index.py")
        k = min(top_k, len(self.doc_ids))
        query_tokens = [self.tokenizer(q) for q in queries]
        docs, scores = self.model.retrieve(query_tokens, k=k, show_progress=False)
        results = []
        for row_docs, row_scores in zip(docs, scores):
            # 分数为 0 的文档和查询没有任何共同词，丢掉；否则融合时它们会白拿排名分
            results.append([(self.doc_ids[int(i)], float(s)) for i, s in zip(row_docs, row_scores) if s > 0])
        return results

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out))
        with open(out / "doc_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False)
        with open(out / PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.build_params, f)

    @classmethod
    def load(cls, out_dir: str | Path, tokenizer: ChineseTokenizer, k1: float | None = None, b: float | None = None) -> "BM25Retriever":
        """加载索引，并核对建索引时的参数（分词模式、词典、停用词、k1、b）和当前配置一致。"""
        out = Path(out_dir)
        if not (out / PARAMS_FILE).exists():
            raise FileNotFoundError(f"找不到 BM25 索引 {out}，先运行 scripts/build_index.py")
        with open(out / PARAMS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        wanted = {**tokenizer.params, "k1": saved["k1"] if k1 is None else float(k1), "b": saved["b"] if b is None else float(b)}
        diff = {k: (saved.get(k), v) for k, v in wanted.items() if saved.get(k) != v}
        if diff:
            detail = "，".join(f"{k}: 索引={old} / 配置={new}" for k, (old, new) in diff.items())
            raise ValueError(f"BM25 索引的参数和当前配置不一致（{detail}）。"
                             "请带上同样的 --set 重新运行 build_index.py --only bm25")
        obj = cls(tokenizer, k1=saved["k1"], b=saved["b"])
        obj.model = bm25s.BM25.load(str(out))
        with open(out / "doc_ids.json", encoding="utf-8") as f:
            obj.doc_ids = json.load(f)
        return obj
