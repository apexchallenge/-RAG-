"""父子分块（small-to-big）：建子块索引、和"整篇文档建索引"比 doc 级指标。原理见 rag/advanced/hierarchical.py。

    # ① 先看切块统计（不加载模型，几十秒）：文档长度分布、多少文档会被切成多个父块
    python scripts/eval_hierarchical.py stats
    # ② 切块 + 给子块建 BM25 和向量索引（向量要编码全部子块，和第一次 build_index 差不多久）
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/eval_hierarchical.py build
    # ③ 在 dev 上比较（整篇 vs 父子 × dense / hybrid / hybrid+rerank），只在 dev 上调切块大小和聚合方式
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/eval_hierarchical.py eval --split dev
    python scripts/eval_hierarchical.py eval --split dev --agg sum --tag sum           # 换聚合方式
    python scripts/eval_hierarchical.py build --set hierarchical.child_size=150         # 换子块大小（另建一套索引）

子块索引放在 indexes/<数据集>_pc<父块>_<子块>_<重叠>/，不影响原来的索引。结果写到 results/hier_<tag>_<时间>.md / .json。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.advanced.hierarchical import build_parent_child, chunk_stats, pc_index_dir  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.data import load_corpus  # noqa: E402
from rag.hier_eval import MODES, build_pc, pc_params, run_pc_eval, save, summarize  # noqa: E402
from rag.paths import resolve  # noqa: E402


def print_stats(s: dict) -> None:
    print(f"{s['docs']} 篇文档 -> {s['parents']} 个父块 / {s['children']} 个子块（每篇平均 {s['children_per_doc']:.2f} 个子块）")
    print(f"文档字数：均值 {s['doc_chars_mean']:.0f}，P50 {s['doc_chars_p50']:.0f}，P90 {s['doc_chars_p90']:.0f}，"
          f"P99 {s['doc_chars_p99']:.0f}")
    print(f"切出多个父块的文档 {100 * s['docs_multi_parent_rate']:.1f}%；有多个子块的父块 {100 * s['parents_multi_child_rate']:.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="父子分块：建索引 / 评测")
    p.add_argument("cmd", choices=["stats", "build", "eval"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--only", choices=["bm25", "dense"], default=None, help="build：只建一种索引")
    p.add_argument("--reencode", action="store_true", help="build：强制重新编码子块")
    p.add_argument("--split", default="dev")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--agg", choices=["max", "sum"], default=None, help="子块 -> 父块的聚合，默认读 hierarchical.agg")
    p.add_argument("--child-k", type=int, default=None, help="子块召回条数，默认读 hierarchical.child_k")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    if args.cmd == "stats":
        corpus = load_corpus(cfg["data_dir"])
        parents, children = build_parent_child(corpus, **pc_params(cfg))
        print(f"参数 {pc_params(cfg)}")
        print_stats(chunk_stats(corpus, parents, children))
        return
    if args.cmd == "build":
        stats = build_pc(cfg, only=args.only, reencode=args.reencode)
        print_stats(stats)
        print(f"已保存到 {pc_index_dir(cfg)}")
        return
    if args.split == "test":
        print("[提示] 切块大小、聚合方式请先在 dev 上定好，test 只跑一次。")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    res = run_pc_eval(cfg, modes=modes, split=args.split, max_queries=args.max_queries, agg=args.agg,
                      child_k=args.child_k)
    summary = summarize(res, tuple((cfg.get("eval") or {}).get("recall_ks", (5, 10, 20, 100))))
    stats_path = pc_index_dir(cfg) / "pc_params.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else None
    path = save(summary, res, cfg, stats, resolve(args.out), args.tag)
    print(path.read_text(encoding="utf-8"))
    print(f"已保存：{path}")


if __name__ == "__main__":
    main()
