"""统一的数据格式（公开数据集和你自己的语料都转成这个格式）：

    corpus.jsonl   每行 {"id": "...", "text": "...", "title": "..."}
    queries.jsonl  每行 {"id": "...", "text": "..."}
    qrels.tsv      表头 query-id<TAB>corpus-id<TAB>score，score>0 表示相关
    splits.json    {"train": [...], "dev": [...], "test": [...]}  查询 id 的划分

划分的规矩：调参（融合权重、top-K 等）只看 dev，最终报告只看 test，微调只用 train。
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig：兼容 Windows 记事本/PowerShell 写出的带 BOM 文件
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_corpus(data_dir: str | Path) -> list[dict]:
    docs = read_jsonl(Path(data_dir) / "corpus.jsonl")
    for d in docs:
        d["id"] = str(d["id"])
        d.setdefault("title", "")
    return docs


def doc_text(doc: dict) -> str:
    """检索和重排时用的文本：有标题就拼在正文前面。"""
    title = (doc.get("title") or "").strip()
    return f"{title}\n{doc['text']}" if title else doc["text"]


def load_queries(data_dir: str | Path) -> dict[str, str]:
    return {str(q["id"]): q["text"] for q in read_jsonl(Path(data_dir) / "queries.jsonl")}


def load_qrels(data_dir: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(Path(data_dir) / "qrels.tsv", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header and len(header) >= 3 and _is_number(header[2]):  # 没有表头：第一行也是数据
            qrels.setdefault(header[0], {})[header[1]] = int(float(header[2]))
        for row in reader:
            if len(row) >= 3:
                qrels.setdefault(row[0], {})[row[1]] = int(float(row[2]))
    return qrels


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def write_qrels(path: str | Path, qrels: dict[str, dict[str, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for qid, docs in qrels.items():
            for did, score in docs.items():
                writer.writerow([qid, did, score])


def make_splits(qids: list[str], n_test: int, n_dev: int, seed: int = 42) -> dict[str, list[str]]:
    """固定随机种子打乱后切成 test / dev / train，保证每次划分一致、可复现。"""
    ids = sorted(qids)
    random.Random(seed).shuffle(ids)
    n_test = min(n_test, len(ids))
    n_dev = min(n_dev, len(ids) - n_test)
    return {
        "test": sorted(ids[:n_test]),
        "dev": sorted(ids[n_test : n_test + n_dev]),
        "train": sorted(ids[n_test + n_dev :]),
    }


def load_split_ids(data_dir: str | Path, split: str) -> list[str] | None:
    """返回某个划分的查询 id；split="all" 时返回 None（表示全部查询）。

    没有 splits.json 时直接报错，而不是悄悄用全部查询——否则"在 dev 上调参"会混进 test 查询。
    """
    if split == "all":
        return None
    path = Path(data_dir) / "splits.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{data_dir} 下没有 splits.json，无法取 {split!r} 划分。"
            f"先运行 python scripts/prepare_data.py --make-splits {data_dir}（或显式使用 --split all）"
        )
    with open(path, encoding="utf-8-sig") as f:
        splits = json.load(f)
    if split not in splits:
        raise KeyError(f"splits.json 里没有 {split!r}，可选: {list(splits)}")
    return [str(x) for x in splits[split]]


def select_queries(
    data_dir: str | Path,
    split: str = "test",
    max_queries: int | None = None,
    seed: int = 42,
) -> tuple[list[str], dict[str, str], dict[str, dict[str, int]]]:
    """取出用于评测的查询：属于指定划分、且至少有一个相关文档。"""
    queries = load_queries(data_dir)
    qrels = load_qrels(data_dir)
    allowed = load_split_ids(data_dir, split)
    qids = sorted(allowed if allowed is not None else queries)
    qids = [q for q in qids if q in queries and any(s > 0 for s in qrels.get(q, {}).values())]
    if max_queries and len(qids) > max_queries:
        qids = sorted(random.Random(seed).sample(qids, max_queries))
    return qids, {q: queries[q] for q in qids}, {q: qrels[q] for q in qids}
