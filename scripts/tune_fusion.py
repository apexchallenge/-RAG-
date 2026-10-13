"""融合调参：BM25 和向量各跑一次，然后枚举融合方式 / RRF 的 k / 两路权重，几秒钟比完几十组参数。

    python scripts/tune_fusion.py --split dev                       # 只在 dev 上调！
    python scripts/tune_fusion.py --split dev --set embedder.model=BAAI/bge-large-zh-v1.5

输出（终端 + results/fusion_<划分>_<时间>.md/.json）：每组参数的 nDCG@10、R@10、R@100、
融合丢失率（某一路把正确答案排进重排窗口 top_n、融合后却掉出去的查询比例）、相对"只用 dense"的配对显著性检验，
以及最优参数对应的 --set 写法。选定之后，在 test 上用 evaluate.py 带上这些 --set 跑一次作为最终报告。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import dump_config, load_config  # noqa: E402
from rag.evaluation import run_eval, snapshot_config, tune_fusion  # noqa: E402
from rag.paths import resolve  # noqa: E402
from rag.stats import format_test  # noqa: E402


def render(rows: list[dict], metric: str, window: int) -> list[str]:
    lines = [f"| 融合方式 | {metric} | R@10 | R@100 | 融合丢失率(前{window}) | 相对 dense（{metric}，×100） |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        m = r["metrics"]
        vs = format_test(r["vs_dense"]) if r["vs_dense"] else "（基线）"
        lines.append(f"| {r['name']} | {100 * m[metric]:.1f} | {100 * m.get('R@10', 0):.1f} | "
                     f"{100 * m.get('R@100', 0):.1f} | {100 * r['fusion_loss']['lost_rate']:.1f}% | {vs} |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="融合方式与权重调参（只在 dev 上用）")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="dev")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--metric", default="nDCG@10")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    if args.split == "test":
        print("[警告] 在 test 上调参会让最终结果偏乐观；调参请用 --split dev，test 只用来报告一次。")

    cfg = load_config(args.config, args.set)
    res = run_eval(cfg, modes=["bm25", "dense"], split=args.split, max_queries=args.max_queries)
    window = int((cfg.get("reranker") or {}).get("top_n", 50))
    rows = tune_fusion(res, cfg, metric=args.metric, window=window)

    table = render(rows, args.metric, window)
    print(f"\n划分 {res.split}：{len(res.qids)} 条查询（指标 ×100，按 {args.metric} 从高到低）\n")
    print("\n".join(table))
    best = rows[0]
    flags = " ".join(f'--set "{s}"' for s in best["set"])
    if best["set"]:
        print(f"\n最优：{best['name']}。在 test 上报告时加上：\n  {flags}")
    else:
        print("\n最优是只用 dense：在这个数据集上加 BM25 没有带来提升，最终报告可以只跑 dense / dense_rerank。")

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.tag or 'fusion'}_{res.split}_{datetime.now():%Y%m%d-%H%M%S}"
    snap = snapshot_config(cfg, res)
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"split": res.split, "n_queries": len(res.qids), "metric": args.metric, "window": window,
                   "rows": rows, "config": snap}, f, ensure_ascii=False, indent=2)
    md = [f"# 融合调参 {name}", "", f"- 数据：`{cfg['data_dir']}`，划分 `{res.split}`，{len(res.qids)} 条查询",
          f"- 嵌入：`{cfg['embedder']['model']}`；BM25：jieba {cfg['bm25'].get('jieba_mode')}", "", *table, "",
          f"最优：{best['name']}" + (f"，参数 `{flags}`" if best["set"] else "（只用 dense）"), "",
          "<details><summary>完整配置</summary>", "", "```yaml", dump_config(snap), "```", "</details>"]
    (out / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n已保存：{out / (name + '.md')}")


if __name__ == "__main__":
    main()
