"""准备数据，统一成 corpus.jsonl / queries.jsonl / qrels.tsv / splits.json。

公开数据集（MTEB 格式，国内先设置 HF_ENDPOINT=https://hf-mirror.com）：
    python scripts/prepare_data.py --hf mteb/CmedqaRetrieval --out data/cmedqa
调试用小语料（指标会虚高，只用来跑通流程）：
    python scripts/prepare_data.py --hf mteb/CmedqaRetrieval --out data/cmedqa_small --max-corpus 20000
你自己的文档（一个放 .txt/.md 的文件夹）：
    python scripts/prepare_data.py --docs-dir my_docs --out data/mydomain --chunk-size 400 --overlap 80
写好 queries.jsonl 和 qrels.tsv 之后，生成 train/dev/test 划分：
    python scripts/prepare_data.py --make-splits data/mydomain --n-test 200 --n-dev 50
相对路径一律相对项目根目录。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.paths import resolve  # noqa: E402
from rag.prep import make_splits_for, prepare_docs, prepare_hf  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="准备检索评测数据")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="HuggingFace 数据集名，例如 mteb/CmedqaRetrieval")
    src.add_argument("--docs-dir", help="自有文档目录（.txt / .md）")
    src.add_argument("--make-splits", metavar="DATA_DIR", help="给已有 queries/qrels 的目录生成 splits.json")
    p.add_argument("--out", help="输出目录，例如 data/cmedqa（--hf / --docs-dir 时必填）")
    p.add_argument("--n-test", type=int, default=1000, help="测试集查询数（只用于最终报告）")
    p.add_argument("--n-dev", type=int, default=500, help="开发集查询数（用于调参）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-corpus", type=int, default=None, help="只保留这么多文档（调试用）")
    p.add_argument("--hf-split", default=None, help="数据集有多个划分时指定，例如 dev")
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    args = p.parse_args()

    if args.make_splits:
        make_splits_for(resolve(args.make_splits), args.n_test, args.n_dev, args.seed)
        return
    if not args.out:
        p.error("--hf / --docs-dir 需要 --out")
    if args.hf:
        prepare_hf(args.hf, resolve(args.out), args.n_test, args.n_dev, args.seed, args.max_corpus, args.hf_split)
    else:
        prepare_docs(resolve(args.docs_dir), resolve(args.out), args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
