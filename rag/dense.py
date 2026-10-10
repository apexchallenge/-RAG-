"""稠密检索：文档向量 + Faiss 索引。

四种索引，方便做"速度 / 内存 / 召回"的取舍实验（scripts/bench_faiss.py）：
    flat   精确暴力搜索，召回 100%，是其他索引的对照组
    hnsw   图索引：查询快、召回高，但内存比 flat 还大（要存图的边）
    ivf    倒排：先 k-means 聚成 nlist 个簇，查询只搜最近的 nprobe 个簇
    ivfpq  倒排 + 乘积量化：向量压缩成 pq_m 个字节，内存最省，召回有损失

磁盘上的布局（indexes/<数据集>/dense/<模型名>/）：
    embeddings.npy + doc_ids.json + meta.json      文档向量，只在换模型/语料/max_seq_length 时重算
    faiss_<类型>.index + faiss_<类型>.json        由向量建出的索引 + "指纹"（向量版本 + 建索引参数）
指纹对不上（向量重算过、参数改过）就自动用向量重建索引，避免读到过期的缓存。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import faiss
import numpy as np

from .types import Hits

INDEX_TYPES = ("flat", "hnsw", "ivf", "ivfpq")
DEFAULTS = {"hnsw_m": 32, "hnsw_ef_construction": 200, "ivf_nlist": 1024, "pq_m": 64, "pq_nbits": 8,
            "hnsw_ef_search": 128, "ivf_nprobe": 32}
BUILD_KEYS = {"flat": (), "hnsw": ("hnsw_m", "hnsw_ef_construction"), "ivf": ("ivf_nlist",),
              "ivfpq": ("ivf_nlist", "pq_m", "pq_nbits")}


def _get(cfg: dict, key: str) -> int:
    return int(cfg.get(key, DEFAULTS[key]))


def index_kind(cfg: dict) -> str:
    kind = str(cfg.get("index_type", "flat")).lower()
    if kind not in INDEX_TYPES:
        raise ValueError(f"未知 index_type={kind!r}，可选 {INDEX_TYPES}")
    return kind


def build_params(cfg: dict) -> dict:
    """影响索引内容的参数（efSearch / nprobe 是查询期参数，不在其中）。"""
    kind = index_kind(cfg)
    return {"index_type": kind, **{k: _get(cfg, k) for k in BUILD_KEYS[kind]}}


def _safe_nlist(n: int, nlist: int) -> int:
    # Faiss 建议每个聚类中心至少 39 个训练点，语料太小时自动缩小 nlist
    return max(1, min(nlist, n // 39))


def build_faiss_index(emb: np.ndarray, cfg: dict) -> faiss.Index:
    """emb 必须是 L2 归一化后的 float32 矩阵；用内积度量（= 余弦相似度）。"""
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    n, d = emb.shape
    kind = index_kind(cfg)
    metric = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, _get(cfg, "hnsw_m"), metric)
        index.hnsw.efConstruction = _get(cfg, "hnsw_ef_construction")
    else:
        nlist = _safe_nlist(n, _get(cfg, "ivf_nlist"))
        quantizer = faiss.IndexFlatIP(d)
        if kind == "ivf":
            index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
        else:
            m, nbits = _get(cfg, "pq_m"), _get(cfg, "pq_nbits")
            if d % m:
                raise ValueError(f"pq_m={m} 必须能整除向量维度 {d}（768 维可用 32/48/64/96）")
            if n < 2**nbits:
                raise ValueError(f"IVF-PQ 训练至少需要 {2**nbits} 条向量（pq_nbits={nbits}），当前只有 {n} 条；"
                                 "小语料请用 flat / hnsw，或调小 pq_nbits")
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric)
        index.train(emb)
    index.add(emb)
    set_search_params(index, cfg)
    return index


def set_search_params(index: faiss.Index, cfg: dict) -> None:
    """查询期参数：HNSW 的 efSearch、IVF 的 nprobe。越大越准越慢。"""
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = _get(cfg, "hnsw_ef_search")
    ivf = faiss.try_extract_index_ivf(index)
    if ivf is not None:
        ivf.nprobe = min(_get(cfg, "ivf_nprobe"), ivf.nlist)


def search_index(index: faiss.Index, doc_ids: list[str], query_emb: np.ndarray, top_k: int) -> list[Hits]:
    k = min(top_k, len(doc_ids))
    scores, idx = index.search(np.ascontiguousarray(query_emb, dtype=np.float32), k)
    # IVF 的 nprobe 太小时可能凑不满 k 个结果，Faiss 用 -1 占位
    return [
        [(doc_ids[i], float(s)) for s, i in zip(srow, irow) if i >= 0]
        for srow, irow in zip(scores, idx)
    ]


def write_index(index: faiss.Index, path: Path) -> None:
    # 先序列化成字节再用 Python 写文件：faiss.write_index 在中文 Windows 路径下会打不开文件
    path.write_bytes(faiss.serialize_index(index).tobytes())


def read_index(path: Path) -> faiss.Index:
    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))


def load_vectors(out_dir: str | Path) -> tuple[list[str], np.ndarray, dict]:
    """读取 build_index 存下的 doc_ids、向量（float16 -> float32）和元信息。"""
    out = Path(out_dir)
    if not (out / "meta.json").exists():
        raise FileNotFoundError(f"找不到向量索引 {out}，先运行 scripts/build_index.py（换了嵌入模型也要重新建）")
    with open(out / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(out / "doc_ids.json", encoding="utf-8") as f:
        doc_ids = json.load(f)
    embeddings = np.ascontiguousarray(np.load(out / "embeddings.npy").astype(np.float32))
    if embeddings.shape[0] != len(doc_ids):
        raise ValueError(f"{out} 里向量数 {embeddings.shape[0]} 和文档数 {len(doc_ids)} 不一致，请 --reencode 重建")
    return doc_ids, embeddings, meta


class DenseRetriever:
    def __init__(self, embedder, faiss_cfg: dict):
        self.embedder = embedder
        self.cfg = faiss_cfg
        self.index: faiss.Index | None = None
        self.doc_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.vectors_id = ""  # 每次重新编码生成一个新 id，索引指纹靠它判断是否过期

    def build(self, doc_ids: list[str], texts: list[str], show_progress: bool = True) -> None:
        self.doc_ids = list(doc_ids)
        self.embeddings = self.embedder.encode_docs(texts, show_progress=show_progress)
        self.vectors_id = uuid.uuid4().hex
        self.index = build_faiss_index(self.embeddings, self.cfg)

    def rebuild_index(self) -> None:
        """换索引类型/参数时复用已经算好的向量，不用重新跑模型。"""
        self.index = build_faiss_index(self.embeddings, self.cfg)

    def search(self, queries: list[str], top_k: int, query_emb: np.ndarray | None = None) -> list[Hits]:
        if self.index is None:
            raise RuntimeError("Faiss 索引还没建，先运行 scripts/build_index.py")
        if query_emb is None:
            query_emb = self.embedder.encode_queries(queries)
        return search_index(self.index, self.doc_ids, query_emb, top_k)

    def index_path(self, out_dir: str | Path) -> Path:
        return Path(out_dir) / f"faiss_{index_kind(self.cfg)}.index"

    def fingerprint(self) -> dict:
        return {"vectors_id": self.vectors_id, **build_params(self.cfg)}

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", self.embeddings.astype(np.float16))  # 半精度存盘，省一半空间
        with open(out / "doc_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False)
        meta = {"model": self.embedder.name, "max_seq_length": getattr(self.embedder, "max_seq_length", None),
                "dim": int(self.embeddings.shape[1]), "n": len(self.doc_ids), "vectors_id": self.vectors_id}
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        self.save_index(out)

    def save_index(self, out_dir: str | Path) -> None:
        path = self.index_path(out_dir)
        write_index(self.index, path)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.fingerprint(), f)

    @classmethod
    def load(cls, out_dir: str | Path, embedder, faiss_cfg: dict) -> "DenseRetriever":
        out = Path(out_dir)
        doc_ids, embeddings, meta = load_vectors(out)
        if embedder is not None:
            if meta["model"] != embedder.name:
                raise ValueError(f"向量是用 {meta['model']} 编码的，当前模型是 {embedder.name}，请重新 build_index")
            msl = getattr(embedder, "max_seq_length", None)
            if meta.get("max_seq_length") and msl and meta["max_seq_length"] != msl:
                raise ValueError(f"向量是用 max_seq_length={meta['max_seq_length']} 编码的，当前是 {msl}，"
                                 "请运行 build_index.py --only dense --reencode")
        obj = cls(embedder, faiss_cfg)
        obj.doc_ids, obj.embeddings, obj.vectors_id = doc_ids, embeddings, meta.get("vectors_id", "")
        path = obj.index_path(out)
        fp_path = path.with_suffix(".json")
        cached = None
        if path.exists() and fp_path.exists():
            with open(fp_path, encoding="utf-8") as f:
                cached = json.load(f)
        if cached == obj.fingerprint():
            obj.index = read_index(path)
            set_search_params(obj.index, faiss_cfg)
        else:  # 没有缓存，或缓存过期（向量重算过 / 建索引参数变了）：用现有向量重建
            print(f"[dense] {path.name} 不存在或已过期，用已有向量重建索引")
            obj.rebuild_index()
            obj.save_index(out)
        return obj
