"""命令行问答：看清每一路召回、融合、重排各自返回了什么，再看大模型的回答。

    python scripts/ask.py --config configs/default.yaml
    python scripts/ask.py -q "宿舍里能用电热毯吗" --config configs/demo.yaml
    python scripts/ask.py --agent -q "宝宝夜里咳嗽怎么办"     # 先检查资料够不够，不够就改写查询再检索，仍不足就拒答
    python scripts/ask.py --router -q "你好"                  # 小模型先分类：闲聊/知识库外不检索，简单题小模型答，复杂题大模型答
不设置 LLM_API_KEY 时只展示检索结果（--agent 需要大模型）；本地 Ollama 可把 generation.base_url 改成 http://localhost:11434/v1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.data import doc_text  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


def snippet(text: str, n: int = 60) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def show(pipe: RAGPipeline, question: str, n: int = 3) -> None:
    res = pipe.ask(question)
    for stage in ("bm25", "dense", "final"):
        name = {"bm25": "BM25", "dense": "向量", "final": "重排后" if pipe.reranker else "融合后"}[stage]
        print(f"\n[{name} 前 {n}]")
        for d, s in res[stage][:n]:
            print(f"  {s:8.3f}  {d}  {snippet(doc_text(pipe.docs[d]))}")
    print("\n[回答]")
    print(res["answer"] if res["answer"] is not None else "（未配置大模型：设置 LLM_API_KEY 或使用本地 Ollama 后可生成回答）")
    print("\n[耗时] " + " | ".join(f"{k} {1000 * v:.0f}ms" for k, v in res["timings"].items()))


def show_agent(pipe: RAGPipeline, question: str, n: int = 3) -> None:
    if not pipe.generator.available:
        print("--agent 需要大模型：请设置 LLM_API_KEY（或写进 .env）")
        return
    before = dict(pipe.generator.usage)
    res = pipe.ask_with_check(question)
    for step in res["trace"]:
        label = {"sufficient": "充分", "partial": "部分", "irrelevant": "无关"}.get(step.get("status"), "部分")
        if label == "部分" and step.get("retrievable") is False:
            label = "部分（查不到）"
        verdict = label if label == "充分" else f"{label}：{step['missing'] or '（未说明）'}"
        print(f"\n[第 {step['round'] + 1} 轮] 查询：{step['query']}")
        for d in step["doc_ids"][:n]:
            print(f"  {d}  {snippet(doc_text(pipe.docs[d]))}")
        print(f"  判断：{verdict}")
        if step.get("next_query"):
            print(f"  -> 补检索用的文本：{step['next_query']}")
        if step.get("added"):
            print(f"  -> 新放进资料的：{', '.join(step['added'])}")
    reasons = {"sufficient": "资料充分", "not_retrievable": "缺的部分知识库里不会有", "context_unchanged": "补检索没带来新资料",
               "duplicate_refine": "补检索文字和之前重复", "no_progress": "补检索后判断和缺的东西都没变", "irrelevant": "资料无关", "max_rounds": "到最大轮数"}
    print(f"\n[停止原因] {reasons.get(res.get('stop_reason'), res.get('stop_reason'))}")
    tag = "（拒答）" if res["refused"] else ("（资料只覆盖一部分）" if res["status"] == "partial" else "")
    print("\n[回答]" + tag)
    print(res["answer"])
    used = {k: pipe.generator.usage[k] - before.get(k, 0) for k in pipe.generator.usage}
    print(f"\n[统计] 检查 {res['rounds']} 次，检索 {len(res['queries'])} 次，大模型调用 {res['llm_calls']} 次，"
          f"token 输入 {used.get('prompt_tokens', 0)} / 输出 {used.get('completion_tokens', 0)}")
    print("[耗时] " + " | ".join(f"{k} {1000 * v:.0f}ms" for k, v in res["timings"].items()))


def show_router(pipe: RAGPipeline, question: str, n: int = 3) -> None:
    from collections import Counter

    from rag.advanced.router import Router

    small_usage, big_usage = Counter(), Counter()
    try:
        router = Router.from_config(pipe, small_usage=small_usage, big_usage=big_usage)
    except RuntimeError as e:
        print(f"--router 需要小模型：{e}")
        return
    if not pipe.generator.available:
        print("--router 需要大模型：请设置 LLM_API_KEY（或写进 .env）")
        return
    res = router.answer(question)
    names = {"chitchat": "闲聊（不检索，小模型回应）", "out_of_scope": "知识库外（不检索，直接说明）",
             "simple": "简单（检索 + 小模型写回答）", "complex": "复杂（检索 + 大模型写回答）"}
    print(f"\n[路由] {names[res['route']]}")
    for c in res["contexts"][:n]:
        print(f"  {c['id']}  {snippet(c['text'])}")
    print("\n[回答]")
    print(res["answer"])
    print(f"\n[token] 小模型 {small_usage.get('prompt_tokens', 0)}+{small_usage.get('completion_tokens', 0)}，"
          f"大模型 {big_usage.get('prompt_tokens', 0)}+{big_usage.get('completion_tokens', 0)}")
    print("[耗时] " + " | ".join(f"{k} {1000 * v:.0f}ms" for k, v in res["timings"].items()))


def main() -> None:
    p = argparse.ArgumentParser(description="RAG 命令行问答")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("-q", "--question", default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--agent", action="store_true", help="用 answer_with_check：判断资料够不够，值得的话补检索，无关才拒答")
    p.add_argument("--router", action="store_true", help="大小模型路由（配置见 router 段，小模型默认本地 Ollama）")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    pipe = RAGPipeline(load_config(args.config, args.set), use_reranker=not args.no_rerank)
    pipe.warmup()
    run = show_agent if args.agent else (show_router if args.router else show)
    if args.question:
        run(pipe, args.question)
        return
    while True:
        try:
            q = input("\n问题（回车退出）> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(pipe, q)


if __name__ == "__main__":
    main()
