"""建索引（BM25 + 向量/Faiss）。

    python scripts/build_index.py --config configs/default.yaml
    python scripts/build_index.py --config configs/default.yaml --only dense --set faiss.index_type=hnsw
向量已经算过时，换 index_type 只会重建 Faiss 索引，不会重新跑模型；加 --reencode 强制重新编码。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.data import load_corpus  # noqa: E402
from rag.indexing import build_bm25, build_dense  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="建 BM25 和向量索引")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--only", choices=["bm25", "dense"], default=None)
    p.add_argument("--reencode", action="store_true", help="强制重新编码全部文档")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置，可重复")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    corpus = load_corpus(cfg["data_dir"])
    print(f"[build] {cfg['data_dir']}: {len(corpus)} 篇文档")
    if args.only in (None, "bm25"):
        build_bm25(cfg, corpus)
    if args.only in (None, "dense"):
        from rag.embedder import Embedder

        build_dense(cfg, Embedder.from_config(cfg["embedder"]), corpus, reencode=args.reencode)


if __name__ == "__main__":
    main()
