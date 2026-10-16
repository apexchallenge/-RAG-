"""大小模型路由（rag/advanced/router.py）的评测：同一批问题，"全部走大模型"（big_only）vs "路由"（router）。

问题组（复用 agent 评测的问题集，另加闲聊组）：
  answerable  dev 里的普通医疗问题 —— 不该被拦下（误判成 chitchat / out_of_scope 等于直接丢答案）；
              分到 simple 的由小模型写回答，看质量掉没掉
  chitchat    eval_sets/chitchat_questions.jsonl：打招呼、闲聊 —— 路由后不检索、不用大模型
  ood         eval_sets/ood_questions.jsonl：知识库里不可能有答案的问题 —— 路由后不检索直接说明超出范围
  hard_ood    eval_sets/ood_hard_questions.jsonl：看似可答的知识库外问题 —— 看路由会不会把它们放给小模型硬答

两种跑法：
  classify   只让小模型分类（不检索、不写回答、不评审）：路由分布、误拦率、分类耗时。小模型放本地 Ollama 时完全免费
  full       两个系统都完整跑一遍 + 盲评（同 agent 评测的 refused / helpful / grounded），统计质量、延迟、大小模型 token
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from .agent_eval import _load_jsonl, build_sets, judge_answer, run_baseline
from .stats import paired_test

SETS = ("answerable", "chitchat", "ood", "hard_ood")
SYSTEMS = ("big_only", "router")
SET_LABELS = {"answerable": "可回答（dev 普通问题）", "chitchat": "闲聊", "ood": "知识库外问题（明显）",
              "hard_ood": "看似可答的知识库外问题"}
# 每组"理想"的路由：answerable 走 simple / complex 都算对；hard_ood 不强求
EXPECTED = {"answerable": {"simple", "complex"}, "chitchat": {"chitchat"}, "ood": {"out_of_scope"}}
BLOCKING = {"chitchat", "out_of_scope"}  # 不检索、不写带资料的回答的路由


def build_router_sets(data_dir, split: str, n_answerable: int, ood_path, n_ood: int, hard_ood_path, n_hard_ood: int,
                      chitchat_path, n_chitchat: int, seed: int = 42) -> list[dict]:
    """[{set, qid, question, exclude}]。可回答 / 知识库外两组的抽样和 agent 评测一致（同 seed 抽到同一批题）。"""
    items = build_sets(data_dir, split, n_answerable, 0, ood_path, n_ood, Path("."), seed, hard_ood_path, n_hard_ood)
    for r in _load_jsonl(chitchat_path)[:n_chitchat] if n_chitchat else []:
        items.append({"set": "chitchat", "qid": r["id"], "question": r["text"], "exclude": []})
    return items


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _p50(xs) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def _tokens(u: dict | None) -> int:
    u = u or {}
    return int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))


# ---------------------------------------------------------------- classify 模式


def run_classify(small_llm, item: dict, kb_description: str, usage: Counter | None = None) -> dict:
    from .advanced.router import CLASSIFY_PROMPT, parse_route

    before = dict(usage or {})
    t0 = time.perf_counter()
    rec = {"set": item["set"], "qid": item["qid"], "question": item["question"]}
    try:
        raw = small_llm(CLASSIFY_PROMPT.format(kb_description=kb_description, question=item["question"]))
        rec.update({"route": parse_route(raw), "raw": raw[-200:]})
    except Exception as e:  # 单条失败不影响整体
        rec.update({"route": None, "error": f"{type(e).__name__}: {e}"[:300]})
    rec["seconds"] = time.perf_counter() - t0
    if usage is not None:
        rec["usage"] = {k: usage.get(k, 0) - before.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens")}
    return rec


def summarize_classify(records: list[dict]) -> dict:
    out: dict = {"by_set": {}, "overall": {}}
    ok = [r for r in records if r.get("route")]
    for s in SETS:
        rows = [r for r in ok if r["set"] == s]
        if not rows:
            continue
        dist = Counter(r["route"] for r in rows)
        row = {"n": len(rows), "routes": dict(dist),
               "blocked_rate": _mean(int(r["route"] in BLOCKING) for r in rows),
               "seconds_p50": _p50(r["seconds"] for r in rows),
               "tokens_mean": _mean(_tokens(r.get("usage")) for r in rows)}
        if s in EXPECTED:
            row["expected_rate"] = _mean(int(r["route"] in EXPECTED[s]) for r in rows)
        out["by_set"][s] = row
    ans = out["by_set"].get("answerable")
    if ans:
        out["overall"]["misblock_rate"] = ans["blocked_rate"]  # 可回答问题被拦下（不检索）的比例
        out["overall"]["simple_rate"] = ans["routes"].get("simple", 0) / ans["n"]
    out["overall"]["n_errors"] = sum(1 for r in records if not r.get("route"))
    return out


# ---------------------------------------------------------------- full 模式


def run_router_task(pipe, small, judge_llm, item: dict, system: str, cfg: dict, max_chars: int = 600) -> dict:
    """跑一条（问题, 系统）+ 评审。usage 分小模型 / 大模型两份记；评审的 token 不算进系统成本。"""
    from .advanced.router import Router

    small_usage: Counter = Counter()
    big_usage: Counter = Counter()
    rec = {"set": item["set"], "qid": item["qid"], "question": item["question"], "system": system}
    try:
        if system == "big_only":
            out = run_baseline(pipe, item, big_usage)
            contexts = out.pop("contexts")
            rec.update({"answer": out["answer"], "route": None, "model": "big", "doc_ids": out["doc_ids"],
                        "seconds": out["seconds"]})
        else:
            router = Router.from_config(pipe, cfg, small=small, small_usage=small_usage, big_usage=big_usage)
            res = router.answer(item["question"], exclude=set(item.get("exclude") or []))
            contexts = [c["text"] for c in res["contexts"]]
            rec.update({"answer": res["answer"], "route": res["route"], "model": res["model"],
                        "doc_ids": [c["id"] for c in res["contexts"]], "timings": res["timings"],
                        "seconds": res["timings"]["total"]})
        verdict, raw = judge_answer(judge_llm, item["question"], contexts, rec["answer"], max_chars)
        rec.update({"usage": {"small": dict(small_usage), "big": dict(big_usage)}, "judge": verdict,
                    "judge_raw": raw[:200]})
    except Exception as e:  # 单条失败不影响整体，下次重跑重试
        rec.update({"error": f"{type(e).__name__}: {e}"[:300], "judge": None})
    return rec


def _quality(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "refusal_rate": _mean(r["judge"]["refused"] for r in rows),
        "helpful_mean": _mean(r["judge"]["helpful"] for r in rows),
        "ungrounded_answer_rate": _mean(int(not r["judge"]["refused"] and not r["judge"]["grounded"]) for r in rows),
        "seconds_p50": _p50(r["seconds"] for r in rows),
        "seconds_mean": _mean(r["seconds"] for r in rows),
        "small_tokens_mean": _mean(_tokens((r.get("usage") or {}).get("small")) for r in rows),
        "big_tokens_mean": _mean(_tokens((r.get("usage") or {}).get("big")) for r in rows),
        "big_calls_mean": _mean(((r.get("usage") or {}).get("big") or {}).get("calls", 0) for r in rows),
    }


def summarize_full(records: list[dict], prices: dict | None = None) -> dict:
    """按问题组汇总两系统，并做 router − big_only 的配对检验；另给整体的大模型 token 节省比例和按路由拆开的质量差。

    prices：{"small": (输入, 输出), "big": (输入, 输出)}，单位 元 / 百万 token；给了就算每题平均成本。
    """
    out: dict = {"by_set": {}, "tests": {}, "by_route": {}, "overall": {}}
    rows = {s: {sys_: {r["qid"]: r for r in records if r["set"] == s and r["system"] == sys_ and r.get("judge")}
                for sys_ in SYSTEMS} for s in SETS}
    for s in SETS:
        if not any(rows[s].values()):
            continue
        out["by_set"][s] = {sys_: _quality(list(rs.values())) for sys_, rs in rows[s].items() if rs}
        if rows[s]["router"]:
            out["by_set"][s]["router"]["routes"] = dict(Counter(r["route"] for r in rows[s]["router"].values()))
        common = sorted(set(rows[s]["big_only"]) & set(rows[s]["router"]))
        if len(common) >= 5:
            tests = {}
            for metric, fn in (("helpful", lambda r: r["judge"]["helpful"]),
                               ("refused", lambda r: r["judge"]["refused"]),
                               ("ungrounded", lambda r: int(not r["judge"]["refused"] and not r["judge"]["grounded"]))):
                tests[metric] = paired_test({q: float(fn(rows[s]["big_only"][q])) for q in common},
                                            {q: float(fn(rows[s]["router"][q])) for q in common}, n_resamples=5000)
            out["tests"][s] = tests
    # 按路由拆开看：分到 simple 的题（小模型写回答）质量掉了多少 —— 只在可回答组上看
    ans = rows["answerable"]
    for route in ("simple", "complex", "chitchat", "out_of_scope"):
        qs = [q for q, r in ans["router"].items() if r["route"] == route and q in ans["big_only"]]
        if qs:
            out["by_route"][route] = {
                "n": len(qs),
                "helpful_router": _mean(ans["router"][q]["judge"]["helpful"] for q in qs),
                "helpful_big_only": _mean(ans["big_only"][q]["judge"]["helpful"] for q in qs),
                "refusal_router": _mean(ans["router"][q]["judge"]["refused"] for q in qs),
            }
    # 整体：两系统都完成的题上，大模型 token 一共省了多少
    pairs = [(rows[s]["big_only"][q], rows[s]["router"][q]) for s in SETS
             for q in set(rows[s]["big_only"]) & set(rows[s]["router"])]
    if pairs:
        big_a = sum(_tokens((a.get("usage") or {}).get("big")) for a, _ in pairs)
        big_b = sum(_tokens((b.get("usage") or {}).get("big")) for _, b in pairs)
        out["overall"] = {
            "n_pairs": len(pairs),
            "big_tokens_big_only": big_a,
            "big_tokens_router": big_b,
            "big_token_saving": (1 - big_b / big_a) if big_a else None,
            "small_tokens_router": sum(_tokens((b.get("usage") or {}).get("small")) for _, b in pairs),
            "seconds_mean_big_only": _mean(a["seconds"] for a, _ in pairs),
            "seconds_mean_router": _mean(b["seconds"] for _, b in pairs),
        }
        if prices:
            out["overall"]["cost_per_q_big_only"] = _mean(cost_of(a, prices) for a, _ in pairs)
            out["overall"]["cost_per_q_router"] = _mean(cost_of(b, prices) for _, b in pairs)
    return out


def cost_of(rec: dict, prices: dict) -> float:
    """一条记录的成本（元），prices 单位 元 / 百万 token。"""
    total = 0.0
    for who in ("small", "big"):
        u = (rec.get("usage") or {}).get(who) or {}
        pin, pout = prices.get(who, (0.0, 0.0))
        total += (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
    return total


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
