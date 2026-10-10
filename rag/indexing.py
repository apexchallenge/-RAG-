"""建索引 / 读索引，以及索引存放位置的约定：

    indexes/<数据集>/bm25_<jieba模式>/     BM25 稀疏矩阵 + doc_ids + 建索引参数
    indexes/<数据集>/dense/<模型名>/       向量(float16) + 各类 Faiss 索引 + 指纹

不同嵌入模型放不同目录，换模型做对比时互不覆盖。加载时会核对参数，配置变了而索引没重建会直接报错，
不会悄悄用旧索引得出"新配置"的结果。
"""

from __future__ import annotations

import time
from pathlib import Path

from .bm25 import BM25Retriever
from .data import doc_text, load_corpus
from .dense import DenseRetriever, index_kind, load_vectors
from .text import ChineseTokenizer
from .utils import slugify


def bm25_dir(cfg: dict) -> Path:
    return Path(cfg["index_dir"]) / f"bm25_{cfg['bm25'].get('jieba_mode', 'search')}"


def dense_dir(cfg: dict, model_name: str | None = None) -> Path:
    return Path(cfg["index_dir"]) / "dense" / slugify(model_name or cfg["embedder"]["model"])


def build_bm25(cfg: dict, corpus: list[dict] | None = None, show_progress: bool = True) -> BM25Retriever:
    corpus = corpus if corpus is not None else load_corpus(cfg["data_dir"])
    bcfg = cfg["bm25"]
    retriever = BM25Retriever(ChineseTokenizer.from_config(bcfg), k1=float(bcfg.get("k1", 1.2)), b=float(bcfg.get("b", 0.75)))
    start = time.perf_counter()
    retriever.build([d["id"] for d in corpus], [doc_text(d) for d in corpus], show_progress=show_progress)
    out = bm25_dir(cfg)
    retriever.save(out)
    print(f"[bm25] {len(corpus)} 篇文档，词表 {len(retriever.model.vocab_dict)} 个词，"
          f"耗时 {time.perf_counter() - start:.1f}s -> {out}")
    return retriever


def load_bm25(cfg: dict) -> BM25Retriever:
    bcfg = cfg["bm25"]
    return BM25Retriever.load(bm25_dir(cfg), ChineseTokenizer.from_config(bcfg),
                              k1=bcfg.get("k1", 1.2), b=bcfg.get("b", 0.75))


def build_dense(
    cfg: dict,
    embedder,
    corpus: list[dict] | None = None,
    reencode: bool = False,
    show_progress: bool = True,
) -> DenseRetriever:
    """编码全部文档并建 Faiss 索引。

    向量已存在、且模型 / max_seq_length / 语料都没变时，只重建 Faiss 索引（换 index_type 很快）；
    否则重新编码。--reencode 强制重新编码。
    """
    corpus = corpus if corpus is not None else load_corpus(cfg["data_dir"])
    out = dense_dir(cfg, embedder.name)
    retriever = DenseRetriever(embedder, cfg["faiss"])
    ids = [d["id"] for d in corpus]
    start = time.perf_counter()
    reused = False
    if not reencode and (out / "meta.json").exists():
        doc_ids, embeddings, meta = load_vectors(out)
        same = (meta["model"] == embedder.name and doc_ids == ids
                and meta.get("max_seq_length") == getattr(embedder, "max_seq_length", None))
        if same:
            retriever.doc_ids, retriever.embeddings = doc_ids, embeddings
            retriever.vectors_id = meta.get("vectors_id", "")
            retriever.rebuild_index()
            reused = True
    if reused:
        retriever.save_index(out)  # 向量没变，只写索引
    else:
        retriever.build(ids, [doc_text(d) for d in corpus], show_progress=show_progress)
        retriever.save(out)
    what = "复用已有向量，重建" if reused else "编码文档并建"
    print(f"[dense] {what} {index_kind(cfg['faiss'])} 索引：{retriever.embeddings.shape}，"
          f"耗时 {time.perf_counter() - start:.1f}s -> {out}")
    return retriever


def load_dense(cfg: dict, embedder) -> DenseRetriever:
    return DenseRetriever.load(dense_dir(cfg, embedder.name), embedder, cfg["faiss"])
