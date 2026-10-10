"""命令行问答：看清每一路召回、融合、重排各自返回了什么，再看大模型的回答。

    python scripts/ask.py --config configs/default.yaml
    python scripts/ask.py -q "宿舍里能用电热毯吗" --config configs/demo.yaml
不设置 LLM_API_KEY 时只展示检索结果；本地 Ollama 可把 generation.base_url 改成 http://localhost:11434/v1
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


def main() -> None:
    p = argparse.ArgumentParser(description="RAG 命令行问答")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("-q", "--question", default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    pipe = RAGPipeline(load_config(args.config, args.set), use_reranker=not args.no_rerank)
    if args.question:
        show(pipe, args.question)
        return
    while True:
        try:
            q = input("\n问题（回车退出）> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(pipe, q)


if __name__ == "__main__":
    main()
