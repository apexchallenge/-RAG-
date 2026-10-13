"""跨两次评测做配对显著性检验，例如 bge-base vs bge-large（两次 evaluate.py 各自保存了 _runs.tsv.gz）：

    python scripts/compare_runs.py \
        --a results/cmedqa_<base 那次>_runs.tsv.gz:dense \
        --b results/cmedqa_<large 那次>_runs.tsv.gz:dense

冒号后面是模式名（bm25 / dense / hybrid_rrf / dense_rerank ...），省略时两边取同名模式。
只比较两边都有的查询；qrels 从 --config 的 data_dir 读。输出 B - A 的差值、95% CI、p 值。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.data import load_qrels  # noqa: E402
from rag.metrics import per_query  # noqa: E402
from rag.paths import resolve  # noqa: E402
from rag.runs import read_runs  # noqa: E402
from rag.stats import format_test, paired_test  # noqa: E402


def load_side(spec: str, default_mode: str | None) -> tuple[str, dict[str, list[str]]]:
    path, _, mode = spec.rpartition(":") if ":" in Path(spec).name else (spec, "", "")
    mode = mode or default_mode
    runs = read_runs(resolve(path))
    if mode is None:
        if len(runs) != 1:
            raise SystemExit(f"{path} 里有多个模式 {sorted(runs)}，请写成 路径:模式")
        mode = next(iter(runs))
    if mode not in runs:
        raise SystemExit(f"{path} 里没有模式 {mode!r}，可选 {sorted(runs)}")
    return f"{Path(path).name}:{mode}", {q: [d for d, _ in hits] for q, hits in runs[mode].items()}


def main() -> None:
    p = argparse.ArgumentParser(description="两次评测结果的配对显著性检验（B - A）")
    p.add_argument("--a", required=True, help="基线：路径[:模式]")
    p.add_argument("--b", required=True, help="对比：路径[:模式]")
    p.add_argument("--mode", default=None, help="两边都用这个模式（--a/--b 没写模式时）")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--metrics", default="nDCG@10,R@10,MRR@10,R@100")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    qrels = load_qrels(cfg["data_dir"])
    name_a, run_a = load_side(args.a, args.mode)
    name_b, run_b = load_side(args.b, args.mode)
    common = sorted(set(run_a) & set(run_b) & set(qrels))
    if not common:
        raise SystemExit("两次结果没有共同的、带标注的查询（是不是用了不同的划分或数据集？）")
    if len(common) < max(len(run_a), len(run_b)):
        print(f"[提示] 只比较共同的 {len(common)} 条查询（A 有 {len(run_a)} 条，B 有 {len(run_b)} 条）")
    run_a = {q: run_a[q] for q in common}
    run_b = {q: run_b[q] for q in common}
    print(f"A = {name_a}\nB = {name_b}\n{len(common)} 条查询，B - A（×100）：\n")
    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        pa, pb = per_query(run_a, qrels, metric), per_query(run_b, qrels, metric)
        mean_a, mean_b = 100 * sum(pa.values()) / len(pa), 100 * sum(pb.values()) / len(pb)
        print(f"  {metric:8s} A={mean_a:.1f}  B={mean_b:.1f}  {format_test(paired_test(pa, pb))}")


if __name__ == "__main__":
    main()
