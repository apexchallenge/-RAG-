"""比较两次 agent 评测（两个 --tag）：同一批问题、同一个系统，B − A 配对检验。用来做"只改一个配置"的消融。

    # 写回答关思考（B） vs 默认开思考（A）
    python scripts/compare_agent_runs.py v2_nothink v2_gen_nothink
    # 两次配置完全相同的运行：差值就是重跑噪声，用来判断别的差值算不算数
    python scripts/compare_agent_runs.py v2_nothink v2_think --systems baseline

结果打印出来，同时保存到 results/agent/compare_<A>_vs_<B>.md。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.agent_eval import SET_LABELS, SYSTEMS, compare_tags, load_records  # noqa: E402
from rag.paths import resolve  # noqa: E402
from rag.stats import format_test  # noqa: E402


def render(cmp: dict, tag_a: str, tag_b: str) -> str:
    md = [f"# {tag_b} − {tag_a}", "", f"A = `{tag_a}`，B = `{tag_b}`；只比较两次都有评审结果的同一批问题。", ""]
    for s, by_sys in cmp.items():
        md += [f"## {SET_LABELS[s]}", "",
               "| 系统 | 条数 | helpful A→B | 拒答率 A→B | 无依据回答率 A→B | 调用 A→B | 输入 token A→B | 输出 token A→B | 耗时 P50(s) A→B |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sys_, row in by_sys.items():
            a, b = row["a"], row["b"]
            md.append(f"| {sys_} | {row['n']} | {a['helpful_mean']:.2f} → {b['helpful_mean']:.2f} | "
                      f"{100 * a['refusal_rate']:.1f}% → {100 * b['refusal_rate']:.1f}% | "
                      f"{100 * a['ungrounded_answer_rate']:.1f}% → {100 * b['ungrounded_answer_rate']:.1f}% | "
                      f"{a['calls_mean']:.1f} → {b['calls_mean']:.1f} | {a['prompt_tokens_mean']:.0f} → {b['prompt_tokens_mean']:.0f} | "
                      f"{a['completion_tokens_mean']:.0f} → {b['completion_tokens_mean']:.0f} | "
                      f"{a['seconds_p50']:.1f} → {b['seconds_p50']:.1f} |")
        for sys_, row in by_sys.items():
            t = row.get("tests")
            if t:
                md += ["", f"{sys_}（B − A 配对检验）：", "",
                       f"- helpful：{format_test(t['helpful'], scale=1)}",
                       f"- 拒答率：{format_test(t['refused'])}（×100）",
                       f"- 无依据回答率：{format_test(t['ungrounded'])}（×100）"]
        md.append("")
    md.append("token 和调用次数不含评审；输出 token 包括思考过程（思考越多，输出 token 越多）。")
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser(description="比较两次 agent 评测（B − A）")
    p.add_argument("tag_a")
    p.add_argument("tag_b")
    p.add_argument("--systems", default="baseline,agent")
    p.add_argument("--out", default="results/agent")
    args = p.parse_args()
    root = resolve(args.out)
    recs = []
    for tag in (args.tag_a, args.tag_b):
        path = root / tag / "records.jsonl"
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        recs.append(load_records(path))
    systems = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEMS]
    md = render(compare_tags(recs[0], recs[1], systems), args.tag_a, args.tag_b)
    out = root / f"compare_{args.tag_a}_vs_{args.tag_b}.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n已保存：{out}")


if __name__ == "__main__":
    main()
