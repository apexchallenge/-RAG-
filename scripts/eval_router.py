"""评测大小模型路由（rag/advanced/router.py）。原理和问题组见 rag/router_eval.py 开头。

小模型配置在 configs/default.yaml 的 router.small（默认本地 Ollama 的 qwen3:4b，先 `ollama pull qwen3:4b`）。

    # ⓪ 小模型开着思考会很慢：先试哪种写法能关掉思考，把推荐的那行写进 router.small
    python scripts/eval_router.py probe
    # ① 只跑分类（只调小模型，不检索、不用大模型、不评审）：看路由分布和"可回答问题被误拦"的比例
    python scripts/eval_router.py classify --tag r1
    # ② 完整对比：全部走大模型 vs 路由，盲评质量 + 延迟 + 大小模型 token（需要大模型 API）
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/eval_router.py full --tag r1 --price-big 1,2 -y
    python scripts/eval_router.py full --tag r1 --dry-run        # 只看规模，不调用

结果在 results/router/<tag>/：classify.jsonl + classify.md；records.jsonl + report.md / report.json。
同一个 --tag 重跑会跳过已完成的。改了路由提示词或小模型，换一个 --tag。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.agent_eval import LockedPipeline, RecordStore, _load_jsonl  # noqa: E402
from rag.config import dump_config, load_config  # noqa: E402
from rag.paths import resolve  # noqa: E402
from rag.router_eval import (  # noqa: E402
    BLOCKING, SET_LABELS, SETS, SYSTEMS, build_router_sets, run_classify, run_router_task, save_jsonl,
    summarize_classify, summarize_full,
)
from rag.stats import format_test  # noqa: E402
from rag.utils import progress  # noqa: E402

ROUTE_NAMES = {"chitchat": "闲聊", "simple": "简单", "complex": "复杂", "out_of_scope": "知识库外"}


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def small_label(cfg: dict) -> str:
    s = {**(cfg.get("generation") or {}), **((cfg.get("router") or {}).get("small") or {})}
    return f"{s.get('model')} @ {s.get('base_url')}"


def render_classify(summary: dict, cfg: dict, tag: str, split: str) -> str:
    md = [f"# 路由分类评测 {tag}", "",
          f"- 小模型：`{small_label(cfg)}`；划分 `{split}`；只跑分类，不检索、不写回答", "",
          "| 问题组 | 条数 | 闲聊 | 简单 | 复杂 | 知识库外 | 分对比例 | 被拦下（不检索） | 分类耗时 P50(s) | token/题 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in SETS:
        r = summary["by_set"].get(s)
        if not r:
            continue
        n = r["n"]
        cells = [f"{r['routes'].get(k, 0)}（{100 * r['routes'].get(k, 0) / n:.0f}%）" for k in ROUTE_NAMES]
        md.append(f"| {SET_LABELS[s]} | {n} | " + " | ".join(cells) +
                  f" | {pct(r.get('expected_rate'))} | {pct(r['blocked_rate'])} | {r['seconds_p50']:.2f} | {r['tokens_mean']:.0f} |")
    o = summary["overall"]
    md += ["",
           f"- **误拦率**（可回答问题被分成闲聊 / 知识库外、直接不检索）：{pct(o.get('misblock_rate'))} —— 这部分等于直接丢答案，越低越好",
           f"- 可回答问题分到\"简单\"（交给小模型写回答）的比例：{pct(o.get('simple_rate'))} —— 这是路由能省大模型调用的上限",
           f"- 分类失败（调用出错）：{o.get('n_errors', 0)} 条", "",
           "分对比例：闲聊组 = 分成闲聊；知识库外组 = 分成知识库外；可回答组 = 分成简单或复杂。看似可答组不设标准答案。", ""]
    return "\n".join(md)


def render_full(summary: dict, cfg: dict, tag: str, split: str, prices: dict | None) -> str:
    g = cfg.get("generation", {}) or {}
    md = [f"# 路由评测 {tag}", "",
          f"- 大模型：`{g.get('model')}`（写回答 thinking={g.get('thinking')}）；小模型：`{small_label(cfg)}`；划分 `{split}`",
          "- big_only：检索一次 + 大模型写回答（= agent 评测的 baseline）；router：小模型分类 → 闲聊 / 知识库外不检索，"
          "简单题小模型写回答，复杂题和 big_only 一样",
          "- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)（同 agent 评测），评审的 token 不计入系统成本", ""]
    for s in SETS:
        rows = summary["by_set"].get(s)
        if not rows:
            continue
        md += [f"## {SET_LABELS[s]}", "",
               "| 系统 | 条数 | 拒答率 | helpful 均值 | 无依据回答率 | 耗时 P50(s) | 耗时均值(s) | 大模型 token/题 | 小模型 token/题 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sys_ in SYSTEMS:
            m = rows.get(sys_)
            if m:
                md.append(f"| {sys_} | {m['n']} | {pct(m['refusal_rate'])} | {m['helpful_mean']:.2f} | "
                          f"{pct(m['ungrounded_answer_rate'])} | {m['seconds_p50']:.1f} | {m['seconds_mean']:.1f} | "
                          f"{m['big_tokens_mean']:.0f} | {m['small_tokens_mean']:.0f} |")
        if rows.get("router", {}).get("routes"):
            md += ["", "路由分布：" + "、".join(f"{ROUTE_NAMES.get(k, k)} {v}" for k, v in
                                           sorted(rows["router"]["routes"].items(), key=lambda x: -x[1]))]
        t = summary["tests"].get(s)
        if t:
            md += ["", "router − big_only（配对检验）：", "",
                   f"- helpful：{format_test(t['helpful'], scale=1)}",
                   f"- 拒答率：{format_test(t['refused'])}（×100）",
                   f"- 无依据回答率：{format_test(t['ungrounded'])}（×100）"]
        md.append("")
    if summary["by_route"]:
        md += ["## 可回答组按路由拆开", "", "| 路由 | 条数 | helpful：big_only → router | router 拒答率 |", "| --- | --- | --- | --- |"]
        for route, r in summary["by_route"].items():
            md.append(f"| {ROUTE_NAMES.get(route, route)} | {r['n']} | {r['helpful_big_only']:.2f} → {r['helpful_router']:.2f} | "
                      f"{pct(r['refusal_router'])} |")
        md += ["", "\"简单\"那一行就是\"小模型写回答\"的代价；\"闲聊 / 知识库外\"那两行是误拦（不检索就回答 / 拒答）。", ""]
    o = summary["overall"]
    if o:
        md += ["## 整体", "",
               f"- 两系统都完成的 {o['n_pairs']} 题：大模型 token {o['big_tokens_big_only']} → {o['big_tokens_router']}"
               f"（节省 {pct(o['big_token_saving'])}），路由多花小模型 token {o['small_tokens_router']}",
               f"- 平均耗时：{o['seconds_mean_big_only']:.2f}s → {o['seconds_mean_router']:.2f}s"]
        if prices:
            md.append(f"- 每题成本（元，按 {json.dumps(prices)} 元/百万 token）："
                      f"{o['cost_per_q_big_only']:.5f} → {o['cost_per_q_router']:.5f}")
        md.append("")
    md += ["怎么读：可回答组希望 helpful 不降、拒答率不升；闲聊和知识库外组看耗时和大模型 token 降了多少；"
           "看似可答组看无依据回答率（小模型更容易拿沾边资料硬答）。",
           "注意：分到\"复杂\"的题 router 和 big_only 的提示词完全一样，差别只来自生成的随机性，可当作重跑噪声的参照。",
           "耗时含排队（检索和重排串行），只适合同一次运行内两系统之间比较。", "",
           "<details><summary>完整配置</summary>", "", "```yaml", dump_config(cfg), "```", "</details>"]
    return "\n".join(md)


def parse_price(s: str | None):
    if not s:
        return None
    a, b = (float(x) for x in s.split(","))
    return a, b


def preflight(small, cfg: dict, kb: str, item: dict) -> None:
    """正式跑之前先调一次小模型：连不上、模型名不对、走了代理、太慢，都在这里一次说清楚，而不是每条都报错。"""
    import time

    from rag.advanced.router import CLASSIFY_PROMPT, parse_route, small_classify_llm

    print(f"[检查] 先试调一次小模型 {small_label(cfg)} ……")
    t0 = time.perf_counter()
    try:
        raw = small_classify_llm(small, cfg)(CLASSIFY_PROMPT.format(kb_description=kb, question=item["question"]))
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        hints = []
        low = msg.lower()
        if "connect" in low or "refused" in low:
            hints.append("连不上服务：确认 Ollama 在运行（托盘有图标，或终端 `ollama serve`），"
                         "浏览器打开 http://localhost:11434 应显示 Ollama is running")
        if "404" in msg or "not found" in low:
            hints.append("模型名不对或没下载：`ollama list` 看实际名字，和 router.small.model 保持一致")
        if "502" in msg or "proxy" in low or "503" in msg:
            hints.append("可能走了代理：先关掉系统代理再试，或设置 NO_PROXY=localhost,127.0.0.1")
        if "timeout" in low or "timed out" in low:
            hints.append("超时：模型可能在 CPU 上跑或思考太长，换 qwen3:1.7b，或加 --set router.small.timeout=600")
        raise SystemExit(f"[检查失败] {msg}\n" + "\n".join(f"  - {h}" for h in hints or ["把上面这行错误发出来排查"]))
    sec = time.perf_counter() - t0
    tail = raw.strip().replace("\n", " ")[-120:]
    print(f"[检查] 通过，用时 {sec:.1f}s，解析为 {parse_route(raw)}；原始输出结尾：{tail!r}")
    if sec > 30:
        print("[提示] 单条超过 30 秒，多半在 CPU 上跑或思考很长；可以换更小的模型，或调低 --workers 避免排队")


PROBE_VARIANTS = [
    ("默认（不加参数）", {}),
    ("extra_body: {think: false}", {"extra_body": {"think": False}}),
    ("extra_body: {reasoning_effort: none}", {"extra_body": {"reasoning_effort": "none"}}),
    ("prompt_suffix: ' /no_think'", {"prompt_suffix": " /no_think"}),
]


def cmd_probe(args, cfg, items, out_dir: Path) -> None:
    """试几种"关思考"的写法：小模型开着思考时，分类一条要几十秒、上千 token，路由反而比直接用大模型还慢。
    哪种写法让输出 token 降到几个、类别又不变，就把它写进 router.small。"""
    import copy
    import time

    from rag.advanced.router import CLASSIFY_PROMPT, DEFAULT_KB, make_small_generator, parse_route, small_classify_llm

    kb = (cfg.get("router") or {}).get("kb_description") or DEFAULT_KB
    qs = [it for it in items if it["set"] == "answerable"][: max(1, args.n_probe)]
    print(f"小模型 {small_label(cfg)}；用 {len(qs)} 道可回答题，每种写法各调一次（串行）")
    rows = []
    for name, extra in PROBE_VARIANTS:
        c = copy.deepcopy(cfg)
        c.setdefault("router", {}).setdefault("small", {}).update(extra)
        small = make_small_generator(c)
        usage: Counter = Counter()
        llm = small_classify_llm(small, c, usage=usage)
        routes, secs, err = [], [], None
        for it in qs:
            t0 = time.perf_counter()
            try:
                routes.append(parse_route(llm(CLASSIFY_PROMPT.format(kb_description=kb, question=it["question"]))))
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:150]
                break
            secs.append(time.perf_counter() - t0)
        n = max(1, usage.get("calls", 0))
        rows.append({"name": name, "extra": extra, "routes": routes, "err": err,
                     "sec": sum(secs) / len(secs) if secs else None,
                     "out_tokens": usage.get("completion_tokens", 0) / n})
        r = rows[-1]
        print(f"  {name:38s} " + (f"出错：{err}" if err else
              f"平均 {r['sec']:.1f}s/题，输出 {r['out_tokens']:.0f} token/题，类别 {routes}"))
    base = rows[0]["routes"]
    ok = [r for r in rows[1:] if not r["err"] and r["out_tokens"] < 50]
    if not ok:
        print("\n没有一种写法能关掉思考（输出仍然上百 token）。建议换不带思考的小模型，"
              "例如 Qwen3-4B-Instruct-2507 或 qwen2.5:3b，再跑一次 probe。")
        return
    best = min(ok, key=lambda r: r["sec"])
    same = sum(a == b for a, b in zip(best["routes"], base))
    print(f"\n推荐：{best['name']}（{best['sec']:.1f}s/题，和开思考时的类别一致 {same}/{len(base)}）。写进 configs/default.yaml 的 router.small：")
    for k, v in best["extra"].items():
        print(f"    {k}: {json.dumps(v, ensure_ascii=False)}")
    print("类别不一致的多，说明关掉思考后分类变差了；以 classify 跑全量时的误拦率为准。")


def cmd_classify(args, cfg, items, out_dir: Path) -> None:
    from rag.advanced.router import DEFAULT_KB, make_small_generator, small_classify_llm

    path = out_dir / "classify.jsonl"
    done = {(r["set"], r["qid"]): r for r in _load_jsonl(path) if r.get("route")}
    todo = [it for it in items if (it["set"], it["qid"]) not in done]
    print(f"问题 {len(items)} 条（{dict(Counter(it['set'] for it in items))}），已完成 {len(items) - len(todo)}，本次 {len(todo)}")
    if todo and not args.dry_run:
        small = make_small_generator(cfg)
        if not small.available:
            raise SystemExit(f"小模型不可用：{small_label(cfg)}（检查 router.small；本地 Ollama 要先启动）")
        kb = (cfg.get("router") or {}).get("kb_description") or DEFAULT_KB
        preflight(small, cfg, kb, todo[0])

        def one(it):
            usage = Counter()
            return run_classify(small_classify_llm(small, cfg, usage=usage), it, kb, usage)

        failed = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(one, it) for it in todo]
            for fut in progress(as_completed(futs), "分类", total=len(futs)):
                r = fut.result()
                if r.get("route"):
                    done[(r["set"], r["qid"])] = r
                else:
                    failed.append(r)
        save_jsonl(path, list(done.values()))
        save_jsonl(out_dir / "classify_errors.jsonl", failed)  # 失败的也存下来，方便排查
        if failed:
            kinds = Counter(str(r.get("error", "")).split(":")[0] for r in failed)
            print(f"[注意] {len(failed)} 条失败（{dict(kinds)}），详情见 {out_dir / 'classify_errors.jsonl'}；"
                  f"例如：{failed[0].get('error')}")
    if args.dry_run:
        return
    records = [done[k] for k in ((it["set"], it["qid"]) for it in items) if k in done]
    errors = _load_jsonl(out_dir / "classify_errors.jsonl")
    summary = summarize_classify(records + [e for e in errors if (e["set"], e["qid"]) not in done])
    md = render_classify(summary, cfg, args.tag, args.split)
    (out_dir / "classify.md").write_text(md, encoding="utf-8")
    with open(out_dir / "classify_report.json", "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "split": args.split, "small": small_label(cfg), "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    print(md)
    blocked = [r for r in records if r["set"] == "answerable" and r["route"] in BLOCKING]
    for r in blocked[:10]:
        print(f"  误拦：[{r['route']}] {r['question'][:60]}")
    print(f"已保存：{out_dir / 'classify.md'}")


def cmd_full(args, cfg, items, out_dir: Path) -> None:
    store = RecordStore(out_dir / "records.jsonl")
    systems = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEMS]
    tasks = [(it, s) for it in items for s in systems if not store.done((it["set"], it["qid"], s))]
    prices = None
    if args.price_big or args.price_small:
        prices = {"big": parse_price(args.price_big) or (0.0, 0.0), "small": parse_price(args.price_small) or (0.0, 0.0)}
    print(f"问题 {len(items)} 条（{dict(Counter(it['set'] for it in items))}）；系统 {systems}；"
          f"未完成 {len(tasks)} 个任务（每个任务 = 回答 + 评审，路由多一次小模型分类）")
    if tasks and not args.dry_run:
        if not args.yes and input("开始？[y/N] ").strip().lower() != "y":
            return
        from rag.advanced.router import make_small_generator
        from rag.pipeline import RAGPipeline

        pipe = RAGPipeline(cfg)
        if not pipe.generator.available:
            raise SystemExit("没有可用的大模型：请设置 LLM_API_KEY（或写进 .env）")
        small = make_small_generator(cfg)
        if not small.available:
            raise SystemExit(f"小模型不可用：{small_label(cfg)}")
        pipe.warmup()
        locked = LockedPipeline(pipe)
        judge_llm = pipe.generator.as_llm(thinking=False, max_tokens=1024)
        max_chars = int((cfg.get("generation") or {}).get("max_chars_per_doc", 600))
        failed = 0
        ex = ThreadPoolExecutor(max_workers=max(1, args.workers))
        try:
            futs = [ex.submit(run_router_task, locked, small, judge_llm, it, s, cfg, max_chars) for it, s in tasks]
            for fut in progress(as_completed(futs), "评测", total=len(futs)):
                rec = fut.result()
                store.add(rec)
                failed += int(rec.get("judge") is None)
        except KeyboardInterrupt:
            ex.shutdown(wait=True, cancel_futures=True)
            raise SystemExit(f"\n[已暂停] 已完成的都已保存到 {store.path}，用同一个 --tag 重跑继续。")
        ex.shutdown(wait=True)
        print(f"本次完成 {len(tasks) - failed} 个，失败 {failed} 个（重跑同一命令会重试）")
    if args.dry_run:
        return
    keys = {(it["set"], it["qid"], s) for it in items for s in systems}
    records = [r for k, r in store.records.items() if k in keys]
    summary = summarize_full(records, prices)
    md = render_full(summary, cfg, args.tag, args.split, prices)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "split": args.split, "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                   "small": small_label(cfg), "prices": prices, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(md.split("<details>")[0])
    print(f"已保存：{out_dir / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description="大小模型路由评测")
    p.add_argument("mode", choices=["probe", "classify", "full"])
    p.add_argument("--n-probe", type=int, default=3, help="probe：用几道题试")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=200, help="可回答组条数（和 agent 评测同样的抽样）")
    p.add_argument("--n-chitchat", type=int, default=20)
    p.add_argument("--n-ood", type=int, default=50)
    p.add_argument("--n-hard-ood", type=int, default=30)
    p.add_argument("--chitchat-file", default="eval_sets/chitchat_questions.jsonl")
    p.add_argument("--ood-file", default="eval_sets/ood_questions.jsonl")
    p.add_argument("--hard-ood-file", default="eval_sets/ood_hard_questions.jsonl")
    p.add_argument("--systems", default="big_only,router")
    p.add_argument("--price-big", default=None, help="大模型价格：输入,输出（元/百万 token），例如 1,2")
    p.add_argument("--price-small", default=None, help="小模型价格，本地模型不填（按 0 算）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default="results/router")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    args.tag = args.tag or f"{Path(cfg['data_dir']).name}_{args.split}"
    items = build_router_sets(cfg["data_dir"], args.split, args.n, resolve(args.ood_file), args.n_ood,
                              resolve(args.hard_ood_file), args.n_hard_ood, resolve(args.chitchat_file),
                              args.n_chitchat, args.seed)
    out_dir = resolve(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    {"probe": cmd_probe, "classify": cmd_classify, "full": cmd_full}[args.mode](args, cfg, items, out_dir)


if __name__ == "__main__":
    main()
