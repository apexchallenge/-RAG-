"""消融评测：BM25 / Dense / Hybrid(RRF、加权) / Dense+Rerank / Hybrid+Rerank 一次跑完。

    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --split dev --set reranker.top_n=20 --tag topn20      # 调参只在 dev 上
    python scripts/evaluate.py --modes dense --set embedder.model=BAAI/bge-large-zh-v1.5
    python scripts/evaluate.py --latency 100                                          # 顺便测 P50/P95 延迟
结果写到 results/ 下：<名字>.md（表格，可直接贴进 README）、.json、_failures.jsonl（失败案例）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.paths import resolve  # noqa: E402
from rag.evaluation import MODES, analyze_failures, format_table, measure_latency, run_eval, save_results  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="检索消融评测")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default=None, help="test / dev / train / all，默认取配置 eval.split")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--modes", default=",".join(MODES), help=f"逗号分隔，可选 {','.join(MODES)}")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置，可重复")
    p.add_argument("--tag", default=None, help="结果文件名前缀")
    p.add_argument("--failures", type=int, default=30, help="保存多少条失败案例，0 表示不分析")
    p.add_argument("--latency", type=int, default=0, help="逐条测延迟的查询数，0 表示不测")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        p.error(f"未知模式 {unknown}")

    res = run_eval(cfg, modes=modes, split=args.split, max_queries=args.max_queries)
    print(f"\n划分 {res.split}：{len(res.qids)} 条查询，{len(res.docs)} 篇文档（指标 ×100）\n")
    print(format_table(res))

    failures = analyze_failures(res, cfg, limit=args.failures) if args.failures else None
    if failures and failures[0]:
        print(f"\n失败分析：{failures[0]}")
    latency = None
    if args.latency:
        try:
            latency = measure_latency(cfg, res, n=args.latency)
            print("\n在线延迟 (ms)：" + "，".join(f"{s} P50={v['p50']:.1f}/P95={v['p95']:.1f}" for s, v in latency.items()))
        except Exception as e:  # 测延迟出错也不要丢掉前面算好的评测结果
            print(f"\n[警告] 延迟测量失败，已跳过：{e!r}")
    path = save_results(res, cfg, resolve(args.out), args.tag, failures, latency)
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
