"""answer_with_check（agent）对比"检索一次直接回答"（baseline）的评测。

四组问题：
  answerable  dev 里的普通问题：看回答质量会不会变好、会不会误拒答、多花多少调用和时间
  ood         eval_sets/ood_questions.jsonl：知识库里不可能有答案的问题（具体医院排班、当年政策价格、个人记录、实时信息）
              —— 正确做法是拒答，看谁更会说"不知道"。这组比较容易：问题一看就不是医学知识
  hard_ood    eval_sets/ood_hard_questions.jsonl：看起来是普通医疗问题，但涉及语料之后才出现的东西（新冠后遗症、
              司美格鲁肽、猴痘、九价 HPV 扩龄……），知识库里只有沾边的资料 —— 看会不会拿沾边资料硬答、编造
  masked      dev 里的普通问题，但把已知的相关文档（原标注正例 + 补标判为相关的）从检索结果里屏蔽掉，模拟"知识库缺了这块"
              —— 注意：只屏蔽了判过分的文档，更深处可能还有没判过的相关文档，所以这一组看趋势，不当绝对结论

每条回答再让大模型评审打分（看不到是哪个系统的回答）：
  refused   是否拒答
  helpful   0/1/2，是否回答了用户问题的核心（和补标用同一套口径）
  grounded  回答里的结论能否在系统给出的参考资料里找到依据（编造记 0）
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path

from .data import doc_text, load_qrels, load_queries, load_split_ids
from .stats import paired_test

SETS = ("answerable", "ood", "hard_ood", "masked")
SYSTEMS = ("baseline", "agent")
SET_LABELS = {"answerable": "可回答（dev 普通问题）", "ood": "知识库外问题（明显）", "hard_ood": "看似可答的知识库外问题",
              "masked": "屏蔽相关文档"}

ANSWER_JUDGE_PROMPT = """你是医疗问答系统的评审员。下面是用户问题、系统回答时能看到的全部参考资料，以及系统的回答。

请输出一行 JSON：{{"refused": 0或1, "helpful": 0或1或2, "grounded": 0或1}}
- refused：回答是否拒答（整体表示资料不足、无法回答）。回答了一部分、同时说明某些方面资料没有涉及的，不算拒答。拒答时 helpful 记 0、grounded 记 1。
- helpful：是否回答了用户问题的核心。2 = 针对用户的具体情况直接回答了核心问题；1 = 相关但笼统、只答了一部分；0 = 没有回答到问题。
- grounded：回答里的结论是否都能在参考资料里找到依据。编造了资料里没有的具体说法（数字、药名、结论等）记 0。
只看是否切题、是否有依据，不评判医学观点本身是否正确。只输出 JSON。

【用户问题】
{question}

【参考资料】
{contexts}

【系统回答】
{answer}"""


def parse_answer_judge(text: str) -> dict | None:
    """解析评审输出；取不到三项就返回 None（下次重跑会重试）。"""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*?\}", t, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            out = {"refused": int(d["refused"]), "helpful": int(d["helpful"]), "grounded": int(d["grounded"])}
            if out["refused"] in (0, 1) and out["helpful"] in (0, 1, 2) and out["grounded"] in (0, 1):
                return out
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass
    vals = {}
    for key, pat in (("refused", r"refused\D{0,5}([01])"), ("helpful", r"helpful\D{0,5}([012])"), ("grounded", r"grounded\D{0,5}([01])")):
        mm = re.search(pat, t)
        if mm:
            vals[key] = int(mm.group(1))
    return vals if len(vals) == 3 else None


# ---------------------------------------------------------------- 问题集


def load_relevant_judged(relabel_dir: Path, dataset: str, min_grade: int = 1) -> dict[str, set[str]]:
    """补标结果里判为相关（>= min_grade）的文档，包括主评审和备用评审。"""
    rel: dict[str, set[str]] = {}
    for name in (f"{dataset}_judgments.jsonl", f"{dataset}_judgments_fallback.jsonl"):
        path = relabel_dir / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("grade") is not None and int(r["grade"]) >= min_grade:
                    rel.setdefault(r["qid"], set()).add(r["doc_id"])
    return rel


def _load_jsonl(path: str | Path | None) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_sets(data_dir: str | Path, split: str, n_answerable: int, n_masked: int, ood_path: str | Path | None,
               n_ood: int, relabel_dir: str | Path, seed: int = 42, hard_ood_path: str | Path | None = None,
               n_hard_ood: int = 0) -> list[dict]:
    """返回评测条目 [{set, qid, question, exclude}]。answerable 和 masked 从同一划分里不重叠地抽。"""
    queries, qrels = load_queries(data_dir), load_qrels(data_dir)
    ids = load_split_ids(data_dir, split)
    qids = sorted(q for q in (ids if ids is not None else queries) if q in queries and q in qrels)
    rng = random.Random(seed)
    rng.shuffle(qids)
    items: list[dict] = []
    for q in qids[:n_answerable]:
        items.append({"set": "answerable", "qid": q, "question": queries[q], "exclude": []})
    if n_masked:
        judged = load_relevant_judged(Path(relabel_dir), Path(data_dir).name)
        pool = [q for q in qids[n_answerable:] if q in judged]  # 只用补标过的查询，屏蔽得更彻底
        for q in pool[:n_masked]:
            ex = {d for d, s in qrels[q].items() if s > 0} | judged.get(q, set())
            items.append({"set": "masked", "qid": q, "question": queries[q], "exclude": sorted(ex)})
    for name, path, n in (("ood", ood_path, n_ood), ("hard_ood", hard_ood_path, n_hard_ood)):
        for r in _load_jsonl(path)[:n] if n else []:
            items.append({"set": name, "qid": r["id"], "question": r["text"], "exclude": []})
    return items


# ---------------------------------------------------------------- 运行


class LockedPipeline:
    """多线程并发调用大模型时，检索和重排（GPU）串行执行，其余照常。"""

    def __init__(self, pipe):
        self._p = pipe
        self._lock = threading.Lock()
        self.cfg, self.docs, self.generator = pipe.cfg, pipe.docs, pipe.generator
        self.bm25, self.dense = getattr(pipe, "bm25", None), getattr(pipe, "dense", None)
        self.reranker = _LockedReranker(pipe.reranker, self._lock) if pipe.reranker is not None else None

    def retrieve(self, question: str) -> dict:
        with self._lock:
            return self._p.retrieve(question)


class _LockedReranker:
    def __init__(self, reranker, lock):
        self._r, self._lock = reranker, lock

    def rerank(self, *args, **kwargs):
        with self._lock:
            return self._r.rerank(*args, **kwargs)


def _looks_refused(answer: str) -> bool:
    from .advanced.query_rewrite import _looks_refused as f

    return f(answer)


def run_baseline(pipe, item: dict, usage: Counter) -> dict:
    """检索一次，取前 k 条直接回答（回答提示词本身也允许拒答）。"""
    t0 = time.perf_counter()
    k = int((pipe.cfg.get("generation") or {}).get("context_top_k", 5))
    exclude = set(item.get("exclude") or [])
    hits = [(d, s) for d, s in pipe.retrieve(item["question"])["final"] if d not in exclude][:k]
    texts = [doc_text(pipe.docs[d]) for d, _ in hits]
    answer = pipe.generator.answer(item["question"], texts, usage=usage)
    return {"answer": answer, "self_refused": _looks_refused(answer), "doc_ids": [d for d, _ in hits],
            "contexts": texts, "rounds": 1, "queries": [item["question"]], "trace": [],
            "seconds": time.perf_counter() - t0}


def run_agent(pipe, item: dict, usage: Counter) -> dict:
    from .advanced.query_rewrite import answer_with_check

    t0 = time.perf_counter()
    a = pipe.cfg.get("agent", {}) or {}
    llm = pipe.generator.as_llm(thinking=a.get("thinking", False), max_tokens=int(a.get("max_tokens", 1024)), usage=usage)
    res = answer_with_check(pipe, llm, item["question"], int(a.get("max_rounds", 2)),
                            answer_fn=lambda q, ctx: pipe.generator.answer(q, ctx, usage=usage),
                            exclude=set(item.get("exclude") or []), refine=a.get("refine", "hyde"),
                            max_irrelevant_retries=int(a.get("max_irrelevant_retries", 1)),
                            merge=a.get("merge", "slots"), new_slots=int(a.get("new_slots", 2)))
    return {"answer": res["answer"], "self_refused": res["refused"], "doc_ids": [c["id"] for c in res["contexts"]],
            "contexts": [c["text"] for c in res["contexts"]], "rounds": res["rounds"], "queries": res["queries"],
            "status": res["status"], "statuses": res["statuses"], "improved": res["improved"],
            "context_changed": res["context_changed"], "stop_reason": res["stop_reason"],
            "trace": res["trace"], "seconds": time.perf_counter() - t0}


def judge_answer(judge_llm, question: str, contexts: list[str], answer: str, max_chars: int = 600) -> tuple[dict | None, str]:
    ctx = "\n\n".join(f"[{i}] {t[:max_chars]}" for i, t in enumerate(contexts, start=1)) or "（无）"
    raw = judge_llm(ANSWER_JUDGE_PROMPT.format(question=question, contexts=ctx, answer=answer))
    return parse_answer_judge(raw), raw


class RecordStore:
    """结果逐条追加写盘；重跑同一个 --tag 时跳过已完成的（set, qid, system）。"""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[tuple[str, str, str], dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("judge") is not None:  # 评审没成功的下次重跑
                        self.records[(r["set"], r["qid"], r["system"])] = r

    def done(self, key: tuple[str, str, str]) -> bool:
        return key in self.records

    def add(self, rec: dict) -> None:
        with self._lock:
            if rec.get("judge") is not None:
                self.records[(rec["set"], rec["qid"], rec["system"])] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_task(pipe, judge_llm, item: dict, system: str, max_chars: int = 600) -> dict:
    usage: Counter = Counter()
    rec = {"set": item["set"], "qid": item["qid"], "question": item["question"], "system": system}
    try:
        out = (run_agent if system == "agent" else run_baseline)(pipe, item, usage)
        verdict, raw = judge_answer(judge_llm, item["question"], out.pop("contexts"), out["answer"], max_chars)
        rec.update(out)
        rec.update({"usage": dict(usage), "judge": verdict, "judge_raw": raw[:200]})
    except Exception as e:  # 单条失败不影响整体，记下来，下次重跑重试
        rec.update({"error": f"{type(e).__name__}: {e}"[:300], "judge": None})
    return rec


# ---------------------------------------------------------------- 汇总


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(records: list[dict]) -> dict:
    """按 (问题集, 系统) 汇总，并在每个问题集上做 agent vs baseline 的配对检验。"""
    out: dict = {"by_set": {}, "tests": {}}
    for s in SETS:
        rows = {sys_: {r["qid"]: r for r in records if r["set"] == s and r["system"] == sys_ and r.get("judge")} for sys_ in SYSTEMS}
        if not any(rows.values()):
            continue
        out["by_set"][s] = {}
        for sys_, rs in rows.items():
            vals = list(rs.values())
            if not vals:
                continue
            answered = [r for r in vals if not r["judge"]["refused"]]
            secs = sorted(r["seconds"] for r in vals)
            out["by_set"][s][sys_] = {
                "n": len(vals),
                "refusal_rate": _mean([r["judge"]["refused"] for r in vals]),
                "helpful_mean": _mean([r["judge"]["helpful"] for r in vals]),
                "helpful2_rate": _mean([int(r["judge"]["helpful"] == 2) for r in vals]),
                "grounded_rate_answered": _mean([r["judge"]["grounded"] for r in answered]) if answered else None,
                "ungrounded_answer_rate": _mean([int(not r["judge"]["refused"] and not r["judge"]["grounded"]) for r in vals]),
                "rounds_mean": _mean([r["rounds"] for r in vals]),
                "llm_calls_mean": _mean([r.get("usage", {}).get("calls", 0) for r in vals]),
                "tokens_mean": _mean([r.get("usage", {}).get("prompt_tokens", 0) + r.get("usage", {}).get("completion_tokens", 0) for r in vals]),
                "seconds_p50": secs[len(secs) // 2] if secs else 0.0,
                "seconds_mean": _mean(secs),
            }
            if sys_ == "agent":
                out["by_set"][s][sys_].update(_retrieval_effect(vals, rows["baseline"]))
        common = sorted(set(rows["baseline"]) & set(rows["agent"]))
        if len(common) >= 5:
            tests = {}
            for metric, fn in (("helpful", lambda r: r["judge"]["helpful"]),
                               ("refused", lambda r: r["judge"]["refused"]),
                               ("ungrounded", lambda r: int(not r["judge"]["refused"] and not r["judge"]["grounded"]))):
                base = {q: float(fn(rows["baseline"][q])) for q in common}
                ag = {q: float(fn(rows["agent"][q])) for q in common}
                tests[metric] = paired_test(base, ag, n_resamples=5000)
            out["tests"][s] = tests
    return out


def _retrieval_effect(agent_rows: list[dict], baseline: dict[str, dict]) -> dict:
    """再检索有没有起作用：
    triggered_rate        触发了补检索（写了补检索文字）的比例
    context_changed_rate  触发的里面，最终给大模型的资料和第一轮不一样的比例 —— 补检索有没有真的改变资料
    improved_rate         触发的里面，判断结果变好了的比例（无关→部分/充分，部分→充分）
    triggered_helpful_delta  触发的那些问题上，agent 的 helpful 减 baseline 的 helpful（同一题配对）
    first_status / final_status  第一轮、最后一轮的判断分布；"查不到"的部分单独记为 partial_nr
    stop_reason           循环为什么停（v3 起才有）
    旧版本的记录缺少的字段会被跳过。
    """
    with_status = [r for r in agent_rows if r.get("statuses")]
    trig = [r for r in agent_rows if _triggered(r)]
    paired = [(r, baseline[r["qid"]]) for r in trig if r["qid"] in baseline]
    changed = [r for r in trig if "context_changed" in r]
    return {
        "triggered_rate": _mean([int(_triggered(r)) for r in agent_rows]),
        "n_triggered": len(trig),
        "context_changed_rate": _mean([int(r["context_changed"]) for r in changed]) if changed else None,
        "improved_rate": _mean([int(bool(r.get("improved"))) for r in trig]) if trig else None,
        "triggered_helpful_delta": _mean([a["judge"]["helpful"] - b["judge"]["helpful"] for a, b in paired]) if paired else None,
        "first_status": dict(Counter(_status_label(r["trace"][0]) if r.get("trace") else r["statuses"][0] for r in with_status)),
        "final_status": dict(Counter(_status_label(r["trace"][-1]) if r.get("trace") else r["statuses"][-1] for r in with_status)),
        "stop_reason": dict(Counter(r["stop_reason"] for r in agent_rows if r.get("stop_reason"))),
    }


def _triggered(r: dict) -> bool:
    if r.get("trace"):
        return any(step.get("next_query") for step in r["trace"])
    return r.get("rounds", 1) > 1


def _status_label(step: dict) -> str:
    return "partial_nr" if step.get("status") == "partial" and step.get("retrievable") is False else step.get("status", "?")


def load_records(path: str | Path) -> list[dict]:
    """读 records.jsonl，同一 (set, qid, system) 取最后一条评审成功的。"""
    return list(RecordStore(Path(path)).records.values())


def _tok(r: dict, key: str) -> float:
    return float((r.get("usage") or {}).get(key, 0))


def compare_tags(records_a: list[dict], records_b: list[dict], systems=SYSTEMS) -> dict:
    """两次运行（两个 tag）之间按 (问题集, 系统) 配对比较：B − A。
    用来做"只改一个配置"的消融，例如写回答开 / 关思考；两次运行配置完全相同时，差值就是重跑噪声。"""
    index = lambda rs: {(r["set"], r["system"], r["qid"]): r for r in rs if r.get("judge")}  # noqa: E731
    a, b = index(records_a), index(records_b)
    out: dict = {}
    for s in SETS:
        for sys_ in systems:
            qs = sorted(q for (ss, sy, q) in a if ss == s and sy == sys_ and (ss, sy, q) in b)
            if not qs:
                continue
            ra, rb = [a[(s, sys_, q)] for q in qs], [b[(s, sys_, q)] for q in qs]
            row: dict = {"n": len(qs)}
            for name, rs in (("a", ra), ("b", rb)):
                secs = sorted(r["seconds"] for r in rs)
                row[name] = {
                    "helpful_mean": _mean([r["judge"]["helpful"] for r in rs]),
                    "refusal_rate": _mean([r["judge"]["refused"] for r in rs]),
                    "ungrounded_answer_rate": _mean([int(not r["judge"]["refused"] and not r["judge"]["grounded"]) for r in rs]),
                    "calls_mean": _mean([_tok(r, "calls") for r in rs]),
                    "prompt_tokens_mean": _mean([_tok(r, "prompt_tokens") for r in rs]),
                    "completion_tokens_mean": _mean([_tok(r, "completion_tokens") for r in rs]),
                    "seconds_p50": secs[len(secs) // 2],
                }
            if len(qs) >= 5:
                tests = {}
                for metric, fn in (("helpful", lambda r: r["judge"]["helpful"]),
                                   ("refused", lambda r: r["judge"]["refused"]),
                                   ("ungrounded", lambda r: int(not r["judge"]["refused"] and not r["judge"]["grounded"]))):
                    tests[metric] = paired_test({q: float(fn(x)) for q, x in zip(qs, ra)},
                                                {q: float(fn(x)) for q, x in zip(qs, rb)}, n_resamples=5000)
                row["tests"] = tests
            out.setdefault(s, {})[sys_] = row
    return out


def estimate(items: list[dict], cfg: dict, done: int = 0) -> dict:
    """不调用大模型，估算调用次数和输入字数（上限）。"""
    g, a = cfg.get("generation", {}) or {}, cfg.get("agent", {}) or {}
    k, chars, rounds = int(g.get("context_top_k", 5)), int(g.get("max_chars_per_doc", 600)), int(a.get("max_rounds", 2))
    ctx = k * chars + 400  # 资料 + 提示词
    n = len(items)
    base_calls = n * 2                                   # 回答 + 评审
    agent_min, agent_max = n * 3, n * (2 * rounds + 3)   # 检查+回答+评审 ~ 每轮检查+改写，再回答+评审
    return {"items": n, "tasks": 2 * n, "done": done,
            "calls_min": base_calls + agent_min, "calls_max": base_calls + agent_max,
            "input_chars_min": (base_calls + agent_min) * ctx, "input_chars_max": (base_calls + agent_max) * ctx}
