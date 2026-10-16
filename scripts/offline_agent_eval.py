"""大模型 API 不可用时的离线评测：仍然跑 baseline / agent / 评审的真实代码，只是大模型调用从缓存回放，缺的由外部模型补上。

检索和重排需要本机的模型和索引，所以拆成两类"缺口"：
  pending_llm.jsonl        还没有回答的提示词（判断、写补检索文字、写回答、评审），交给外部模型回答，回答追加进 llm_cache.jsonl
  pending_retrieval.jsonl  还没有结果的检索请求，在装了模型和索引的机器上用 retrieve 子命令跑，结果追加进 retrieval_cache.jsonl

流程（D = results/agent/<tag>）：
    python scripts/offline_agent_eval.py step --tag T --seed-records results/agent/v3/records.jsonl ...   # 跑到卡住为止，写出缺口
    （外部模型回答 D/pending_llm.jsonl -> 追加到 D/llm_cache.jsonl）
    python scripts/offline_agent_eval.py retrieve --tag T        # 有待检索时，在本机跑
    ……重复 step，直到没有缺口；最后一次 step 写 records.jsonl / report.md（格式和 eval_agent.py 一样）

同一个提示词只回答一次（缓存按提示词内容做键）：baseline 和 agent 拿到完全相同的资料时，回答和评审也完全相同，
两个系统的差别只来自资料和流程，不来自生成的随机性。评审提示词里没有系统名，回答时可以打乱顺序做盲评。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.agent_eval import SETS, SYSTEMS, build_sets, run_task, summarize  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.data import doc_text, load_corpus  # noqa: E402
from rag.generator import build_messages  # noqa: E402
from rag.paths import resolve  # noqa: E402


class NeedInput(BaseException):
    """缺大模型回答或检索结果。继承 BaseException，这样 run_task 里的 except Exception 不会把它当成失败记下来。"""


def key_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def kind_of(prompt: str) -> str:
    if prompt.startswith("判断【参考资料】"):
        return "check"
    if prompt.startswith("用户问题："):
        return "refine"
    if prompt.startswith("你是医疗问答系统的评审员"):
        return "judge"
    if prompt.startswith("[SYSTEM]"):
        return "answer"
    return "other"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def excl_key(exclude) -> str:
    return ",".join(sorted(exclude or []))


class Store:
    def __init__(self, d: Path):
        self.d = d
        self.llm = {r["key"]: r["response"] for r in read_jsonl(d / "llm_cache.jsonl")}
        self.ret = {(r["query"], r["exclude"]): r for r in read_jsonl(d / "retrieval_cache.jsonl")}
        self.pending_llm: dict[str, dict] = {}
        self.pending_ret: dict[str, dict] = {}


class ReplayLLM:
    def __init__(self, store: Store, item: dict):
        self.store, self.item = store, item
        self.last_missing = ""

    def __call__(self, prompt: str) -> str:
        if prompt.startswith("用户问题："):
            m = re.search(r"已经检索到的资料没有覆盖到：(.*)", prompt)
            miss = m.group(1).strip() if m else ""
            self.last_missing = "" if miss == "（未说明）" else miss
        k = key_of(prompt)
        if k in self.store.llm:
            return self.store.llm[k]
        self.store.pending_llm.setdefault(k, {"key": k, "kind": kind_of(prompt), "prompt": prompt})
        raise NeedInput()


class OfflineGenerator:
    available = True

    def __init__(self, store: Store, item: dict, max_chars: int):
        self.llm = ReplayLLM(store, item)
        self.max_chars = max_chars

    def answer(self, question, contexts, usage=None):
        msgs = build_messages(question, contexts, self.max_chars)
        out = self.llm(f"[SYSTEM]\n{msgs[0]['content']}\n\n[USER]\n{msgs[1]['content']}")
        if usage is not None:
            usage["calls"] += 1
        return out

    def as_llm(self, thinking=False, max_tokens=None, temperature=0.0, usage=None):
        def llm(prompt):
            out = self.llm(prompt)
            if usage is not None:
                usage["calls"] += 1
            return out
        return llm


class CachedReranker:
    def __init__(self, pipe):
        self.pipe = pipe

    def rerank(self, queries, candidates):
        entry = self.pipe.last_entry
        if entry is None or entry.get("rank_query") != queries[0]:
            raise RuntimeError(f"没有按 {queries[0][:30]!r} 重排的缓存结果")
        order = {d: i for i, d in enumerate(entry["ranked"])}
        out = []
        for cands in candidates:
            missing = [d for d, _ in cands if d not in order]
            if missing:
                raise RuntimeError(f"重排缓存里缺文档 {missing[:3]}")
            out.append([(d, -float(order[d])) for d, _ in sorted(cands, key=lambda x: order[x[0]])])
        return out


class OfflinePipeline:
    def __init__(self, cfg: dict, docs: dict, store: Store, item: dict):
        self.cfg, self.docs, self.store, self.item = cfg, docs, store, item
        self.generator = OfflineGenerator(store, item, int((cfg.get("generation") or {}).get("max_chars_per_doc", 600)))
        self.reranker = CachedReranker(self)
        self.last_entry = None

    def retrieve(self, q: str) -> dict:
        ek = excl_key(self.item.get("exclude"))
        entry = self.store.ret.get((q, ek))
        if entry is None:
            rq = f"{self.item['question']} {self.generator.llm.last_missing}".strip() if q != self.item["question"] else ""
            rid = key_of(q + "\t" + ek)
            self.store.pending_ret.setdefault(rid, {"id": rid, "query": q, "exclude": ek, "rank_query": rq})
            raise NeedInput()
        self.last_entry = entry
        return {"final": [(d, 1.0 - i / 1000) for i, d in enumerate(entry["ids"])]}


def select_items(cfg: dict, args) -> list[dict]:
    items = build_sets(cfg["data_dir"], args.split, args.base_n, args.base_n_masked, resolve(args.ood_file), args.n_ood,
                       resolve(args.relabel_dir), args.seed, resolve(args.hard_ood_file), args.n_hard_ood)
    want = {"answerable": args.n, "masked": args.n_masked, "ood": args.n_ood, "hard_ood": args.n_hard_ood}
    seen: Counter = Counter()
    out = []
    for it in items:
        if seen[it["set"]] < want[it["set"]]:
            seen[it["set"]] += 1
            out.append(it)
    return out


def seed_retrieval(store: Store, items: list[dict], paths: list[str]) -> int:
    """用之前在线评测的记录补第一轮检索结果（检索配置没变时，第一轮检索结果完全一样）。"""
    by_key = {(it["set"], it["qid"]): it for it in items}
    rows = []
    for p in paths:
        for r in read_jsonl(resolve(p)):
            it = by_key.get((r.get("set"), r.get("qid")))
            if it is None or not r.get("judge"):
                continue
            ids = r["trace"][0]["doc_ids"] if r.get("trace") else r.get("doc_ids")
            k = (it["question"], excl_key(it.get("exclude")))
            if ids and k not in store.ret:
                entry = {"query": it["question"], "exclude": k[1], "ids": ids, "rank_query": "", "ranked": [],
                         "source": str(p)}
                store.ret[k] = entry
                rows.append(entry)
    append_jsonl(store.d / "retrieval_cache.jsonl", rows)
    return len(rows)


def cmd_step(args) -> None:
    cfg = load_config(args.config, args.set)
    cfg.setdefault("generation", {})["model"] = args.model_label
    d = resolve(args.out) / args.tag
    d.mkdir(parents=True, exist_ok=True)
    store = Store(d)
    items = select_items(cfg, args)
    if args.seed_records:
        n = seed_retrieval(store, items, args.seed_records)
        if n:
            print(f"从之前的记录补了 {n} 条第一轮检索结果")
    docs = {x["id"]: x for x in load_corpus(cfg["data_dir"])}
    records, blocked = [], Counter()
    for it in items:
        for sys_ in SYSTEMS:
            pipe = OfflinePipeline(cfg, docs, store, it)
            try:
                rec = run_task(pipe, pipe.generator.llm, it, sys_, int(cfg["generation"].get("max_chars_per_doc", 600)))
            except NeedInput:
                blocked[sys_] += 1
                continue
            records.append(rec)
    for name, rows in (("pending_llm.jsonl", store.pending_llm), ("pending_retrieval.jsonl", store.pending_ret)):
        path = d / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()), encoding="utf-8")
    kinds = Counter(r["kind"] for r in store.pending_llm.values())
    print(f"条目 {len(items)}（{dict(Counter(it['set'] for it in items))}）；完成 {len(records)} / {2 * len(items)}，"
          f"卡住 {dict(blocked)}；待回答提示词 {len(store.pending_llm)} {dict(kinds)}；待检索 {len(store.pending_ret)}")
    errors = [r for r in records if r.get("error") or r.get("judge") is None]
    if errors:
        print(f"[注意] {len(errors)} 条出错或评审解析失败，例如：{errors[0].get('error') or errors[0].get('judge_raw')}")
    if not store.pending_llm and not store.pending_ret:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_agent import render

        with open(d / "records.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = summarize([r for r in records if r.get("judge")])
        md = render(summary, cfg, args.tag, args.split)
        md = md.replace("\n- 评审：", f"\n- **离线评测**：大模型调用全部由 {args.model_label} 完成（判断、补检索文字、写回答、评审），"
                                      "评审提示词不含系统名；同一提示词只回答一次\n- 评审：", 1)
        (d / "report.md").write_text(md, encoding="utf-8")
        with open(d / "report.json", "w", encoding="utf-8") as f:
            json.dump({"tag": args.tag, "split": args.split, "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                       "llm": args.model_label, "counts": dict(Counter(it["set"] for it in items)), "summary": summary},
                      f, ensure_ascii=False, indent=2)
        print(md.split("<details>")[0])
        print(f"已保存：{d / 'report.md'}")


def cmd_retrieve(args) -> None:
    """在装了模型和索引的机器上跑待检索的请求（不调用大模型）。"""
    from rag.pipeline import RAGPipeline

    d = resolve(args.out) / args.tag
    reqs = read_jsonl(d / "pending_retrieval.jsonl")
    done = {(r["query"], r["exclude"]) for r in read_jsonl(d / "retrieval_cache.jsonl")}
    reqs = [r for r in reqs if (r["query"], r["exclude"]) not in done]
    if not reqs:
        print("没有待检索的请求。")
        return
    cfg = load_config(args.config, args.set)
    pipe = RAGPipeline(cfg)
    k = int((cfg.get("generation") or {}).get("context_top_k", 5))
    rows = []
    for i, r in enumerate(reqs, 1):
        ex = set(filter(None, r["exclude"].split(",")))
        ids = [doc for doc, _ in pipe.retrieve(r["query"])["final"] if doc not in ex][:2 * k]
        ranked = []
        if r.get("rank_query") and pipe.reranker is not None and ids:
            cands = [(doc, doc_text(pipe.docs[doc])) for doc in ids]
            ranked = [doc for doc, _ in pipe.reranker.rerank([r["rank_query"]], [cands])[0]]
        rows.append({"query": r["query"], "exclude": r["exclude"], "ids": ids, "rank_query": r.get("rank_query", ""),
                     "ranked": ranked})
        if i % 20 == 0 or i == len(reqs):
            print(f"  {i}/{len(reqs)}")
    append_jsonl(d / "retrieval_cache.jsonl", rows)
    print(f"已追加 {len(rows)} 条到 {d / 'retrieval_cache.jsonl'}")


SEP_KEY, SEP_END = "### KEY: ", "### END"


def cmd_export(args) -> None:
    """把待回答的提示词按类型切成若干批文本文件（顺序打乱），交给外部模型逐条回答。"""
    import random

    d = resolve(args.out) / args.tag
    rows = read_jsonl(d / "pending_llm.jsonl")
    bdir = d / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    random.Random(args.seed).shuffle(rows)
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    made = []
    for kind, rs in sorted(by_kind.items()):
        size = args.batch_size.get(kind, 40)
        for i in range(0, len(rs), size):
            name = f"{args.prefix}{kind}_{i // size + 1:02d}"
            text = "".join(f"{SEP_KEY}{r['key']}\n{r['prompt']}\n{SEP_END}\n\n" for r in rs[i:i + size])
            (bdir / f"{name}.in.txt").write_text(text, encoding="utf-8")
            made.append((name, len(rs[i:i + size])))
    for name, n in made:
        print(f"{bdir / (name + '.in.txt')}\t{n}")


def parse_responses(text: str) -> dict[str, str]:
    """回答文件格式：每条以 "### KEY: <key>" 开头、"### END" 结尾，中间是回答原文。"""
    out = {}
    for m in re.finditer(r"^### KEY: ([0-9a-f]{40})[ \t]*\n(.*?)\n### END[ \t]*$", text, flags=re.S | re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


def cmd_import(args) -> None:
    d = resolve(args.out) / args.tag
    pending = {r["key"]: r for r in read_jsonl(d / "pending_llm.jsonl")}
    have = {r["key"] for r in read_jsonl(d / "llm_cache.jsonl")}
    rows, unknown, empty = [], 0, 0
    for f in sorted((d / "batches").glob("*.out*.txt")):
        for k, resp in parse_responses(f.read_text(encoding="utf-8")).items():
            if k in have:
                continue
            if k not in pending:
                unknown += 1
                continue
            if not resp:
                empty += 1
                continue
            have.add(k)
            rows.append({"key": k, "kind": pending[k]["kind"], "response": resp, "source": f.name})
    append_jsonl(d / "llm_cache.jsonl", rows)
    missing = [k for k in pending if k not in have]
    print(f"导入 {len(rows)} 条；不在待回答列表里的 {unknown} 条，空回答 {empty} 条；仍缺 {len(missing)} 条")


def main() -> None:
    p = argparse.ArgumentParser(description="离线 agent 评测（大模型调用由外部完成）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("step", "retrieve", "export", "import"):
        s = sub.add_parser(name)
        s.add_argument("--config", default="configs/default.yaml")
        s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        s.add_argument("--tag", required=True)
        s.add_argument("--out", default="results/agent")
    s = sub.choices["step"]
    s.add_argument("--split", default="dev")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--n-masked", type=int, default=50)
    s.add_argument("--n-ood", type=int, default=50)
    s.add_argument("--n-hard-ood", type=int, default=30)
    s.add_argument("--base-n", type=int, default=200, help="抽题时按这个规模抽（和之前的在线评测一致），再取前 --n 条")
    s.add_argument("--base-n-masked", type=int, default=100)
    s.add_argument("--ood-file", default="eval_sets/ood_questions.jsonl")
    s.add_argument("--hard-ood-file", default="eval_sets/ood_hard_questions.jsonl")
    s.add_argument("--relabel-dir", default="results/relabel")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--seed-records", nargs="*", default=[], help="之前在线评测的 records.jsonl，用来补第一轮检索结果")
    s.add_argument("--model-label", default="外部模型")
    s = sub.choices["export"]
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--prefix", default="")
    s.add_argument("--batch-size", type=lambda x: dict((k, int(v)) for k, v in (kv.split("=") for kv in x.split(","))),
                   default={"check": 40, "refine": 40, "answer": 40, "judge": 40}, help="例如 check=40,answer=30")
    args = p.parse_args()
    {"step": cmd_step, "retrieve": cmd_retrieve, "export": cmd_export, "import": cmd_import}[args.cmd](args)


if __name__ == "__main__":
    main()
