"""评测 answer_with_check（agent）vs 检索一次直接回答（baseline）。原理和四组问题的含义见 rag/agent_eval.py 开头。

    python scripts/eval_agent.py --dry-run                         # 只看要调用多少次大模型（不花钱，不加载模型）
    python scripts/eval_agent.py --n 20 --n-ood 10 --n-masked 10   # 小规模试跑
    python scripts/eval_agent.py -y                                 # 默认：dev 200 条 + 知识库外 50 条 + 看似可答的知识库外 30 条 + 屏蔽 100 条

随时 Ctrl+C 暂停，已完成的逐条存盘；用同一个 --tag 重跑会跳过已完成的。
结果在 results/agent/<tag>/：records.jsonl（每条的回答、检索过程、评审结果）、report.md / report.json。
改了 agent 配置（max_rounds、thinking、提示词……）想重新比，换一个 --tag。
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.agent_eval import (  # noqa: E402
    SET_LABELS, SETS, SYSTEMS, LockedPipeline, RecordStore, build_sets, estimate, run_task, summarize,
)
from rag.config import dump_config, load_config  # noqa: E402
from rag.paths import resolve  # noqa: E402
from rag.stats import format_test  # noqa: E402
from rag.utils import progress  # noqa: E402


def render(summary: dict, cfg: dict, tag: str, split: str) -> str:
    a = cfg.get("agent", {}) or {}
    g = cfg.get("generation", {}) or {}
    names = {"sufficient": "充分", "partial": "部分", "partial_nr": "部分（查不到）", "irrelevant": "无关",
             "not_retrievable": "缺的查不到", "context_unchanged": "资料没变化", "duplicate_refine": "补检索文字重复", "no_progress": "补检索后没进展",
             "max_rounds": "到最大轮数"}

    def dist(d: dict) -> str:
        return "、".join(f"{names.get(k, k)} {v}" for k, v in sorted(d.items(), key=lambda x: -x[1])) or "—"

    def pct(x) -> str:
        return "—" if x is None else f"{100 * x:.1f}%"

    md = [f"# agent 评测 {tag}", "",
          f"- 数据：`{cfg['data_dir']}`，划分 `{split}`；大模型 `{g.get('model')}`；"
          f"写回答 thinking={g.get('thinking')}；agent 判断步骤 thinking={a.get('thinking', False)}，max_rounds={a.get('max_rounds', 2)}，"
          f"refine={a.get('refine', 'hyde')}，merge={a.get('merge', 'slots')}，new_slots={a.get('new_slots', 2)}，"
          f"max_irrelevant_retries={a.get('max_irrelevant_retries', 1)}",
          "- baseline：检索一次取前 k 条直接回答（提示词允许部分回答和拒答）；agent：answer_with_check"
          "（资料判成 充分/部分/部分（查不到）/无关；只有'部分'才补检索，新文档保留名额；无关才拒答）",
          "- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)，见 rag/agent_eval.py", ""]
    for s in SETS:
        rows = summary["by_set"].get(s)
        if not rows:
            continue
        md += [f"## {SET_LABELS[s]}", "",
               "| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sys_ in SYSTEMS:
            m = rows.get(sys_)
            if m:
                gr = m["grounded_rate_answered"]
                gr = "—" if gr is None else f"{100 * gr:.1f}%"
                md.append(f"| {sys_} | {m['n']} | {100 * m['refusal_rate']:.1f}% | {m['helpful_mean']:.2f} | "
                          f"{100 * m['helpful2_rate']:.1f}% | {gr} | "
                          f"{100 * m['ungrounded_answer_rate']:.1f}% | {m['rounds_mean']:.2f} | {m['llm_calls_mean']:.1f} | "
                          f"{m['tokens_mean']:.0f} | {m['seconds_p50']:.1f} |")
        ag = rows.get("agent")
        if ag and "triggered_rate" in ag:
            md += ["", f"再检索有没有起作用：触发补检索 {pct(ag['triggered_rate'])}（{ag['n_triggered']} 条）；"
                   f"其中资料真的变了 {pct(ag.get('context_changed_rate'))}、判断变好 {pct(ag['improved_rate'])}；"
                   "这些题上 agent − baseline 的 helpful "
                   + ("—" if ag["triggered_helpful_delta"] is None else f"{ag['triggered_helpful_delta']:+.2f}") + "。",
                   f"第一轮判断：{dist(ag['first_status'])}；最后判断：{dist(ag['final_status'])}"]
            if ag.get("stop_reason"):
                md.append(f"循环停止原因：{dist(ag['stop_reason'])}")
        tests = summary["tests"].get(s)
        if tests:
            md += ["", "agent − baseline（配对检验）：", "",
                   f"- helpful：{format_test(tests['helpful'], scale=1)}",
                   f"- 拒答率：{format_test(tests['refused'])}（×100）",
                   f"- 无依据回答率：{format_test(tests['ungrounded'])}（×100）"]
        md.append("")
    md += ["怎么读：可回答组希望 helpful 不降、拒答率不明显升高；知识库外 / 屏蔽组希望拒答率高、无依据回答率低；"
           "看似可答组最看重无依据回答率（拿沾边资料答一部分、并说明缺什么，也算合格）。",
           "再检索那一行：判断变好的比例低、helpful 差值接近 0，说明多花的调用没换来东西。",
           "耗时含排队（检索和重排在 GPU 上串行），只适合同一次运行内两系统之间比较。", "",
           "<details><summary>完整配置</summary>", "", "```yaml", dump_config(cfg), "```", "</details>"]
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser(description="agent（answer_with_check）vs baseline 评测")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=200, help="可回答组条数")
    p.add_argument("--n-ood", type=int, default=50, help="知识库外问题条数（来自 --ood-file）")
    p.add_argument("--n-masked", type=int, default=100, help="屏蔽相关文档组条数（需要先做过补标）")
    p.add_argument("--n-hard-ood", type=int, default=30, help="看似可答的知识库外问题条数（来自 --hard-ood-file）")
    p.add_argument("--ood-file", default="eval_sets/ood_questions.jsonl")
    p.add_argument("--hard-ood-file", default="eval_sets/ood_hard_questions.jsonl")
    p.add_argument("--relabel-dir", default="results/relabel")
    p.add_argument("--systems", default="baseline,agent")
    p.add_argument("--judge-thinking", default="false", choices=["true", "false", "null"], help="评审是否开思考")
    p.add_argument("--workers", type=int, default=4, help="并发条数（大模型调用并发，检索串行）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None, help="结果目录名；同名重跑会续跑")
    p.add_argument("--reuse-baseline", default=None, metavar="TAG",
                   help="从另一个 tag 复制 baseline 的结果（只改了 agent 配置时用，省掉重跑 baseline；检索和写回答的配置必须相同）")
    p.add_argument("--out", default="results/agent")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    if args.split == "test":
        print("[提示] agent 的参数（max_rounds、提示词）请先在 dev 上定好，test 只跑一次。")
    items = build_sets(cfg["data_dir"], args.split, args.n, args.n_masked, resolve(args.ood_file), args.n_ood,
                       resolve(args.relabel_dir), args.seed, resolve(args.hard_ood_file), args.n_hard_ood)
    systems = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEMS]
    tag = args.tag or f"{Path(cfg['data_dir']).name}_{args.split}"
    out_dir = resolve(args.out) / tag
    store = RecordStore(out_dir / "records.jsonl")
    if args.reuse_baseline:
        src = RecordStore(resolve(args.out) / args.reuse_baseline / "records.jsonl")
        wanted = {(it["set"], it["qid"], "baseline") for it in items}
        copied = [r for k, r in src.records.items() if k in wanted and not store.done(k)]
        if args.dry_run:  # 只预览，不写盘
            store.records.update({(r["set"], r["qid"], r["system"]): r for r in copied})
        else:
            for r in copied:
                store.add(dict(r, reused_from=args.reuse_baseline))
        print(f"从 {args.reuse_baseline} {'可复制' if args.dry_run else '复制了'} {len(copied)} 条 baseline 结果"
              "（请确认检索和写回答的配置没变）")
    tasks = [(it, s) for it in items for s in systems if not store.done((it["set"], it["qid"], s))]

    counts = {s: sum(1 for it in items if it["set"] == s) for s in SETS}
    print(f"问题：{', '.join(f'{SET_LABELS[s]} {n} 条' for s, n in counts.items())}；系统：{', '.join(systems)}")
    if args.n_masked and counts["masked"] < args.n_masked:
        print(f"[提示] 屏蔽组只凑到 {counts['masked']} 条（只用补标过的查询）")
    est = estimate(items, cfg, done=len(items) * len(systems) - len(tasks))
    print(f"共 {est['tasks']} 个任务，已完成 {est['done']}，本次 {len(tasks)} 个；"
          f"大模型调用约 {est['calls_min']}~{est['calls_max']} 次（含评审，全部未完成时），"
          f"输入上限约 {est['input_chars_min'] // 10000}~{est['input_chars_max'] // 10000} 万字。"
          "agent 判断步骤关思考时输出很短；写回答和评审的输出按服务商价格另算")
    if args.dry_run or not tasks:
        if not tasks:
            print("没有未完成的任务。")
        else:
            return
    else:
        if not args.yes and input("开始？[y/N] ").strip().lower() != "y":
            return
        from rag.pipeline import RAGPipeline

        pipe = RAGPipeline(cfg)
        if not pipe.generator.available:
            raise SystemExit("没有可用的大模型：请设置 LLM_API_KEY（或写进 .env）")
        pipe.warmup()
        locked = LockedPipeline(pipe)
        jt = {"true": True, "false": False, "null": None}[args.judge_thinking]
        judge_llm = pipe.generator.as_llm(thinking=jt, max_tokens=1024)
        max_chars = int((cfg.get("generation") or {}).get("max_chars_per_doc", 600))

        ex = ThreadPoolExecutor(max_workers=max(1, args.workers))
        failed = 0
        try:
            futs = [ex.submit(run_task, locked, judge_llm, it, s, max_chars) for it, s in tasks]
            for fut in progress(as_completed(futs), "评测", total=len(futs)):
                rec = fut.result()
                store.add(rec)
                failed += int(rec.get("judge") is None)
        except KeyboardInterrupt:
            ex.shutdown(wait=True, cancel_futures=True)
            raise SystemExit(f"\n[已暂停] 已完成的都已保存到 {store.path}，用同一个 --tag 重跑继续。")
        ex.shutdown(wait=True)
        print(f"本次完成 {len(tasks) - failed} 个，失败 {failed} 个（重跑同一命令会重试）；"
              f"大模型累计 {dict(pipe.generator.usage)}")

    keys = {(it["set"], it["qid"], s) for it in items for s in systems}
    records = [r for k, r in store.records.items() if k in keys]
    summary = summarize(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render(summary, cfg, tag, args.split)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "split": args.split, "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                   "counts": counts, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(md.split("<details>")[0])
    print(f"已保存：{out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
