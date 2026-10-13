"""大模型补标 + 重算指标 + 人工抽查一致率。原理和注意事项见 rag/judge.py 开头。

第 1 步：补标并重算指标（用 evaluate.py 保存的 run 文件，不用重跑检索和重排）
    python scripts/relabel.py judge --runs results/dev_v3_<时间>_runs.tsv.gz --dry-run      # 先看要打多少分
    python scripts/relabel.py judge --runs results/dev_v3_<时间>_runs.tsv.gz --max-queries 50   # 小规模试跑
    python scripts/relabel.py judge --runs results/dev_v3_<时间>_runs.tsv.gz                 # 全量（已打过的自动跳过）
    随时按 Ctrl+C 暂停（已返回的结果都已存盘），重跑同一命令从断点继续；--max-calls 2000 可以分批跑

第 2 步：人工盲评抽样（judge 会生成 results/relabel/<名字>_human.csv，用 Excel 打开，只填"人工评分"一列 0/1/2）
    python scripts/relabel.py agree --csv results/relabel/<名字>_human.csv

输出都在 results/relabel/：<数据集>_judgments.jsonl（打分缓存）、<名字>.md/.json（原始 vs 补标指标）、
<名字>_human.csv + _key.csv（盲评表和答案表，分开放，评的时候别打开 key）。
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.data import doc_text, load_corpus, load_qrels, load_queries  # noqa: E402
from rag.judge import (  # noqa: E402
    PROMPT_VERSION, JudgeAbort, JudgmentCache, build_judge_prompt, build_pool, cohen_kappa, judge_pairs, judged_at_k, make_llm,
    merge_qrels,
)
from rag.metrics import evaluate_run, per_query  # noqa: E402
from rag.paths import resolve  # noqa: E402
from rag.runs import read_runs  # noqa: E402
from rag.stats import format_test, paired_test  # noqa: E402

METRICS = ("nDCG@10", "MRR@10", "P@10", "Hit@10")
LABELS = {"bm25": "BM25", "dense": "Dense", "hybrid_rrf": "Hybrid·RRF", "hybrid_weighted": "Hybrid·加权",
          "dense_rerank": "Dense+Rerank", "hybrid_rerank": "Hybrid+Rerank"}


def metrics_for(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    out = {}
    for m in METRICS:
        pq = per_query(run, qrels, m)
        out[m] = sum(pq.values()) / len(pq) if pq else 0.0
    return out


def write_human_sample(out: Path, name: str, rows: list[dict], n: int, seed: int) -> tuple[Path, Path] | None:
    """按大模型分数分层抽样，生成盲评表（不含大模型分数）和答案表。"""
    if n <= 0 or not rows:
        return None
    rng = random.Random(seed)
    by_grade: dict[int, list[dict]] = {}
    for r in rows:
        by_grade.setdefault(r["llm_grade"], []).append(r)
    picked: list[dict] = []
    per = max(1, n // max(1, len(by_grade)))
    for g in sorted(by_grade):
        picked += rng.sample(by_grade[g], min(per, len(by_grade[g])))
    rest = [r for r in rows if r not in picked]
    picked += rng.sample(rest, min(n - len(picked), len(rest))) if len(picked) < n else []
    rng.shuffle(picked)
    human, key = out / f"{name}_human.csv", out / f"{name}_key.csv"
    with open(human, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig：Excel 直接打开不乱码
        w = csv.writer(f)
        w.writerow(["编号", "问题", "回答", "人工评分（0/1/2）"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r["query"], r["doc"], ""])
    with open(key, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["编号", "qid", "doc_id", "来源", "大模型评分"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r["qid"], r["doc_id"], r["source"], r["llm_grade"]])
    return human, key


def load_fallback(path: Path) -> tuple[dict, set]:
    """备用评审：{(qid, doc_id): 分数}，以及评审名。文件不存在就是空的。"""
    grades, models = {}, set()
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("grade") in (0, 1, 2):
                    grades[(r["qid"], r["doc_id"])] = int(r["grade"])
                    models.add(str(r.get("model", "未知")))
    return grades, models


def cmd_judge(args) -> None:
    cfg = load_config(args.config, args.set)
    data_dir = cfg["data_dir"]
    runs_path = resolve(args.runs)
    runs = {m: {q: [d for d, _ in hits] for q, hits in per_q.items()} for m, per_q in read_runs(runs_path).items()}
    modes = [m.strip() for m in args.pool_modes.split(",")] if args.pool_modes else list(runs)
    missing = [m for m in modes if m not in runs]
    if missing:
        raise SystemExit(f"run 文件里没有模式 {missing}，可选 {sorted(runs)}")
    qrels_all = load_qrels(data_dir)
    queries = load_queries(data_dir)
    qids = sorted(q for q in set().union(*(runs[m].keys() for m in modes)) if q in qrels_all and q in queries)
    if args.max_queries and len(qids) > args.max_queries:
        qids = sorted(random.Random(args.seed).sample(qids, args.max_queries))
    qrels = {q: qrels_all[q] for q in qids}

    pool = build_pool(runs, qids, modes, args.depth)
    pool_pairs = [(q, d) for q in qids for d in pool[q] if qrels[q].get(d, 0) <= 0]
    gold_pairs = [(q, d) for q in qids for d, s in qrels[q].items() if s > 0]
    gold_sample = random.Random(args.seed).sample(gold_pairs, min(args.judge_gold, len(gold_pairs)))

    gen_cfg = dict(cfg.get("generation") or {})
    if args.model:
        gen_cfg["model"] = args.model
    model_name = gen_cfg.get("model", "")
    out = resolve(args.out)
    cache = JudgmentCache(out / f"{Path(data_dir).name}_judgments.jsonl", model_name)
    pairs = pool_pairs + gold_sample
    fb_path = resolve(args.fallback) if args.fallback else out / f"{Path(data_dir).name}_judgments_fallback.jsonl"
    fallback, fb_models = load_fallback(fb_path)
    failed_before = {p for p in pairs if p in cache.grades and cache.grades[p] is None}
    todo = [p for p in dict.fromkeys(pairs)
            if p not in cache and p not in fallback and not (args.skip_failed and p in failed_before)]
    if fallback:
        print(f"备用评审：{fb_path.name} 里有 {len(fallback)} 对（{', '.join(sorted(fb_models))}），"
              "只用于主评审没给出分数的对，这些对不再重新调用主评审")

    print(f"{len(qids)} 条查询，候选池：{', '.join(modes)} 各取前 {args.depth} 名")
    print(f"需要打分：池中未标注 {len(pool_pairs)} 对 + 原正例抽查 {len(gold_sample)} 对；其中已缓存 {len(pairs) - len(todo)}，"
          f"本次要调用 {len(todo)} 次（模型 {model_name}，提示词 {PROMPT_VERSION}）")
    if args.max_queries:
        print("[提示] 只抽了部分查询时，补标指标只代表这些查询；正式报告请用整个划分。")
    corpus = None
    if todo:
        corpus = {d["id"]: doc_text(d) for d in load_corpus(data_dir)}
        chars = sum(len(build_judge_prompt(queries[q], corpus[d], args.max_chars)) for q, d in todo)
        print(f"输入约 {chars} 字（中文大致 1 字 ≈ 0.6~1 token），费用请按服务商当前价格估算")
        if args.dry_run:
            return
        if not args.yes and input("开始调用大模型？[y/N] ").strip().lower() != "y":
            return
        try:
            stats = judge_pairs(todo, queries, corpus, make_llm(gen_cfg, max_tokens=args.max_tokens), cache,
                                workers=args.workers, max_chars=args.max_chars, max_calls=args.max_calls)
        except JudgeAbort as e:
            raise SystemExit(f"[停止] {e}")
        except KeyboardInterrupt:
            done = sum(1 for p in todo if p in cache)
            raise SystemExit(f"\n[已暂停] 本次已保存 {done} 条，还剩 {len(todo) - done} 条。重跑同一命令会从断点继续。")
        print(f"本次结果：{stats}")
        if stats.get("remaining"):
            print(f"[分批] 受 --max-calls 限制，还有 {stats['remaining']} 对没打；重跑同一命令继续。补齐之前不输出报告。")
            return
        if stats.get("failed"):
            print(f"[提示] {stats['failed']} 条没取到分数（网络错误或输出无法解析），重跑同一命令会自动重试；"
                  "如果几乎全部失败，看 judgments.jsonl 里的 raw 字段：推理型模型需要加大 --max-tokens")
    elif args.dry_run:
        return

    primary = {p: cache.grades.get(p) for p in pairs}
    fb_used = {p for p, g in primary.items() if g is None and fallback.get(p) is not None}
    grades = {p: (fallback[p] if p in fb_used else g) for p, g in primary.items()}
    pool_set, gold_set = set(pool_pairs), set(gold_sample)
    pool_grades = {p: g for p, g in grades.items() if p in pool_set}
    n_unjudged = sum(1 for g in pool_grades.values() if g is None)
    n_fb_pool = sum(1 for p in fb_used if p in pool_set)
    if pool_pairs and n_unjudged > 0.05 * len(pool_pairs):
        raise SystemExit(f"[停止] 候选池 {len(pool_pairs)} 对里有 {n_unjudged} 对没取到分数（超过 5%），补标指标不可信，没有输出报告。"
                         f"看 {cache.path.name} 里 grade 为 null 的 raw 字段找原因，修好后重跑同一命令（成功的会跳过）。")
    variants = {
        "原始标注": qrels,
        "补标·严格(≥2)": merge_qrels(qrels, pool_grades, threshold=2),
        "补标·宽松(≥1)": merge_qrels(qrels, pool_grades, threshold=1),
    }
    judged = {q: {d for d in variants["补标·严格(≥2)"][q]} for q in qids}
    mode_runs = {m: {q: runs[m].get(q, []) for q in qids} for m in modes}

    table = {m: {v: metrics_for(mode_runs[m], qr) for v, qr in variants.items()} for m in modes}
    judged10 = {m: judged_at_k(mode_runs[m], judged, 10) for m in modes}
    baseline = args.baseline if args.baseline in modes else modes[0]
    sig = {}
    for v, qr in variants.items():
        base = per_query(mode_runs[baseline], qr, "nDCG@10")
        sig[v] = {m: paired_test(base, per_query(mode_runs[m], qr, "nDCG@10"), 5000) for m in modes if m != baseline}
    dist_pool = Counter(g for g in pool_grades.values() if g is not None)
    # 敏感性：备用评审补的那些对如果按"没判"（不相关）处理，结论变不变
    sens = {}
    if n_fb_pool:
        no_fb = {p: g for p, g in pool_grades.items() if p not in fb_used}
        for v, th in (("补标·严格(≥2)", 2), ("补标·宽松(≥1)", 1)):
            qr = merge_qrels(qrels, no_fb, threshold=th)
            base = per_query(mode_runs[baseline], qr, "nDCG@10")
            sens[v] = {m: {"ndcg": metrics_for(mode_runs[m], qr)["nDCG@10"],
                           "vs_baseline": paired_test(base, per_query(mode_runs[m], qr, "nDCG@10"), 5000) if m != baseline else None}
                       for m in modes}
    dist_gold = Counter(grades[p] for p in gold_sample if grades.get(p) is not None)

    # ---------- 输出 ----------
    name = f"{runs_path.name.replace('_runs.tsv.gz', '')}_relabel_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    pct = lambda c, k: f"{100 * c.get(k, 0) / max(1, sum(c.values())):.0f}%"  # noqa: E731
    md = [f"# 大模型补标 {name}", "",
          f"- run 文件：`{runs_path.name}`；{len(qids)} 条查询；候选池：{', '.join(modes)} 各前 {args.depth} 名",
          f"- 评审模型：`{model_name}`，提示词 {PROMPT_VERSION}；池中未标注 {len(pool_pairs)} 对"
          + (f"（{n_unjudged} 对没取到分数，按不相关处理）" if n_unjudged else "")
          + (f"；其中 {n_fb_pool} 对主评审没给出分数，由备用评审 {'、'.join(sorted(fb_models))} 按同一评分标准补判"
             f"（见 `{fb_path.name}`）" if n_fb_pool else ""),
          f"- 池中未标注文档的大模型评分{'（含备用评审）' if n_fb_pool else ''}：2 分 {pct(dist_pool, 2)}，1 分 {pct(dist_pool, 1)}，0 分 {pct(dist_pool, 0)}",
          f"- 原标注正例抽查 {sum(dist_gold.values())} 对：2 分 {pct(dist_gold, 2)}，1 分 {pct(dist_gold, 1)}，0 分 {pct(dist_gold, 0)}"
          + ("（原正例里也有大模型判 0 分的，说明原标注本身也有噪声）" if dist_gold.get(0) else ""),
          "- 前 10 名被判过的比例（Judged@10，越接近 100% 说明候选池越完整）："
          + "，".join(f"{LABELS.get(m, m)} {100 * judged10[m]:.0f}%" for m in modes), ""]
    for v in variants:
        md += [f"## {v}", "", "| 方法 | " + " | ".join(METRICS) + " |", "| --- |" + " --- |" * len(METRICS)]
        for m in modes:
            cells = [f"{100 * table[m][v][k]:.1f}" for k in METRICS]
            md.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
        md += ["", f"相对 {LABELS.get(baseline, baseline)} 的 nDCG@10 差值（×100，配对检验）：", ""]
        md += [f"- {LABELS.get(m, m)}：{format_test(t)}" for m, t in sig[v].items()]
        md.append("")
    if sens:
        md += [f"## 敏感性：去掉备用评审补判的 {n_fb_pool} 对（按不相关处理）", "",
               "| 方法 | 严格 nDCG@10（含 → 不含） | 宽松 nDCG@10（含 → 不含） |", "| --- | --- | --- |"]
        for m in modes:
            a = [f"{100 * table[m][v]['nDCG@10']:.1f} → {100 * sens[v][m]['ndcg']:.1f}" for v in ("补标·严格(≥2)", "补标·宽松(≥1)")]
            md.append(f"| {LABELS.get(m, m)} | {a[0]} | {a[1]} |")
        md += ["", f"不含备用评审时，相对 {LABELS.get(baseline, baseline)} 的严格 nDCG@10 差值：", ""]
        md += [f"- {LABELS.get(m, m)}：{format_test(sens['补标·严格(≥2)'][m]['vs_baseline'])}" for m in modes if m != baseline]
        md.append("")
    md += ["指标 ×100。严格：大模型 2 分才算相关；宽松：1 分及以上算相关。原标注正例在所有版本里都算相关。",
           "补标只用于评估，不要用补标结果调参；报告时原始和补标指标都给出，并附人工抽查的一致率。"]

    sample_rows = []
    if args.human_sample:
        corpus = corpus or {d["id"]: doc_text(d) for d in load_corpus(data_dir)}
        for (q, d) in pairs:
            g = primary.get((q, d))  # 人工盲评只抽主评审打的分，用来检验主评审
            if g is not None:
                sample_rows.append({"qid": q, "doc_id": d, "query": queries[q], "doc": corpus[d], "llm_grade": g,
                                    "source": "原正例" if (q, d) in gold_set else "候选池"})
    files = write_human_sample(out, name, sample_rows, args.human_sample, args.seed)
    if files:
        md += ["", f"人工盲评：用 Excel 打开 `{files[0].name}` 填\"人工评分\"，然后运行 "
               f"`python scripts/relabel.py agree --csv {Path(args.out) / files[0].name}`"]
    (out / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"runs": runs_path.name, "n_queries": len(qids), "modes": modes, "depth": args.depth,
                   "model": model_name, "prompt_version": PROMPT_VERSION, "pool_pairs": len(pool_pairs),
                   "unjudged": n_unjudged, "fallback_used": n_fb_pool, "fallback_models": sorted(fb_models),
                   "sensitivity_without_fallback": sens or None, "grade_dist_pool": dist_pool, "grade_dist_gold": dist_gold,
                   "table": table, "judged@10": judged10, "baseline": baseline, "significance": sig},
                  f, ensure_ascii=False, indent=2)
    print("\n".join(md))
    print(f"\n已保存：{out / (name + '.md')}")


def read_csv_rows(path: Path) -> list[dict]:
    """Excel 在中文 Windows 上另存 CSV 默认是 GBK 编码，两种都试一下。"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"读不了 {path}：请另存为 \"CSV UTF-8（逗号分隔）\"")


def cmd_agree(args) -> None:
    human = resolve(args.csv)
    key = resolve(args.key) if args.key else human.with_name(human.name.replace("_human.csv", "_key.csv"))
    llm = {row["编号"].strip(): row for row in read_csv_rows(key)}
    pairs = []
    for row in read_csv_rows(human):
        h = (row.get("人工评分（0/1/2）") or "").strip()
        i = (row.get("编号") or "").strip()
        if h in ("0", "1", "2") and i in llm:
            pairs.append((int(h), int(llm[i]["大模型评分"]), llm[i]["来源"]))
    if not pairs:
        raise SystemExit("没有读到人工评分：请在\"人工评分（0/1/2）\"一列填 0、1 或 2 后保存（保持 CSV 格式）")
    hs, ls = [p[0] for p in pairs], [p[1] for p in pairs]
    lines = [f"人工评了 {len(pairs)} 对", "",
             f"三档完全一致：{100 * sum(a == b for a, b in zip(hs, ls)) / len(pairs):.0f}%，Cohen's kappa = {cohen_kappa(hs, ls):.2f}"]
    for th, label in ((2, "严格(≥2)"), (1, "宽松(≥1)")):
        hb, lb = [int(x >= th) for x in hs], [int(x >= th) for x in ls]
        lines.append(f"{label} 是否相关：一致 {100 * sum(a == b for a, b in zip(hb, lb)) / len(pairs):.0f}%，kappa = {cohen_kappa(hb, lb):.2f}")
    lines += ["", "混淆矩阵（行 = 人工，列 = 大模型）：", "", "| 人工 \\ 大模型 | 0 | 1 | 2 |", "| --- | --- | --- | --- |"]
    conf = Counter(zip(hs, ls))
    lines += [f"| {h} | " + " | ".join(str(conf[(h, m)]) for m in (0, 1, 2)) + " |" for h in (0, 1, 2)]
    gold = [p for p in pairs if p[2] == "原正例"]
    if gold:
        lines += ["", f"其中原标注正例 {len(gold)} 对，人工认为相关（≥1）的占 {100 * sum(p[0] >= 1 for p in gold) / len(gold):.0f}%"]
    lines += ["", "kappa 参考：0.4~0.6 中等，0.6~0.8 较好，>0.8 很好。严格口径 kappa 偏低时，报告里以宽松口径为主或修改提示词重评。"]
    text = "\n".join(lines)
    print(text)
    (human.with_name(human.name.replace("_human.csv", "_agreement.md"))).write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="大模型补标 / 人工一致率")
    sub = p.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge", help="补标并重算指标")
    j.add_argument("--runs", required=True, help="evaluate.py 生成的 _runs.tsv.gz")
    j.add_argument("--config", default="configs/default.yaml")
    j.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    j.add_argument("--pool-modes", default=None, help="逗号分隔；默认 run 文件里的全部模式（要比较的方法都应该在池里）")
    j.add_argument("--depth", type=int, default=10, help="每个模式取前多少名进候选池")
    j.add_argument("--max-queries", type=int, default=None, help="只抽这么多条查询（试跑用）")
    j.add_argument("--judge-gold", type=int, default=100, help="顺便抽查多少对原标注正例（看原标注本身的噪声）")
    j.add_argument("--human-sample", type=int, default=100, help="生成多少对人工盲评样本，0 表示不生成")
    j.add_argument("--baseline", default="dense")
    j.add_argument("--model", default=None, help="评审模型，默认用 configs 里 generation.model")
    j.add_argument("--workers", type=int, default=8, help="并发请求数")
    j.add_argument("--max-chars", type=int, default=800, help="回答截断长度")
    j.add_argument("--max-tokens", type=int, default=1024, help="大模型最多输出多少 token。普通模型只输出一个数字，不受影响；推理型模型会先花 token 思考，太小会得到空输出")
    j.add_argument("--seed", type=int, default=42)
    j.add_argument("--out", default="results/relabel")
    j.add_argument("--fallback", default=None, help="备用评审结果（jsonl，格式同缓存），只填补主评审没给出分数的对；"
                   "默认 results/relabel/<数据集>_judgments_fallback.jsonl（存在时）")
    j.add_argument("--skip-failed", action="store_true", help="之前失败过的对不再重试（直接出报告，或留给备用评审）")
    j.add_argument("--max-calls", type=int, default=None, help="本次最多调用多少次（分批控制花费），下次重跑接着打")
    j.add_argument("--dry-run", action="store_true", help="只统计要打多少分，不调用大模型")
    j.add_argument("-y", "--yes", action="store_true", help="不询问，直接开始调用")
    j.set_defaults(func=cmd_judge)
    a = sub.add_parser("agree", help="人工盲评和大模型评分的一致率")
    a.add_argument("--csv", required=True, help="填好人工评分的 _human.csv")
    a.add_argument("--key", default=None, help="默认同目录下的 _key.csv")
    a.set_defaults(func=cmd_agree)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
