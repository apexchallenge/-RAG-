"""数据准备：把公开数据集或你自己的文档转成统一格式（见 rag/data.py）。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .data import load_qrels, load_queries, make_splits, write_jsonl, write_qrels
from .text import chunk_text


def _pick(row: dict, *names):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    raise KeyError(f"行里找不到字段 {names}，实际字段: {list(row)}")


def _load_hf_config(name: str, config: str, split: str | None):
    from datasets import load_dataset  # 延迟导入

    ds = load_dataset(name, config)
    if split:
        return ds[split]
    if len(ds) == 1:
        return next(iter(ds.values()))
    for s in ("dev", "test", "validation"):
        if s in ds:
            return ds[s]
    raise ValueError(f"{name}/{config} 有多个划分 {list(ds)}，请用 --hf-split 指定")


def prepare_hf(
    name: str,
    out_dir: str | Path,
    n_test: int = 1000,
    n_dev: int = 500,
    seed: int = 42,
    max_corpus: int | None = None,
    hf_split: str | None = None,
) -> dict:
    """下载 MTEB 格式的检索数据集（如 mteb/CmedqaRetrieval：corpus / queries / default 三个子集）。"""
    out = Path(out_dir)
    qrels: dict[str, dict[str, int]] = {}
    for row in _load_hf_config(name, "default", hf_split):
        qid = str(_pick(row, "query-id", "qid"))
        did = str(_pick(row, "corpus-id", "pid"))
        qrels.setdefault(qid, {})[did] = int(_pick(row, "score"))

    queries = [
        {"id": str(_pick(r, "_id", "id")), "text": r["text"]}
        for r in _load_hf_config(name, "queries", hf_split)
    ]
    queries = [q for q in queries if q["id"] in qrels]

    corpus = [
        {"id": str(_pick(r, "_id", "id")), "text": r["text"], "title": r.get("title") or ""}
        for r in _load_hf_config(name, "corpus", hf_split)
    ]
    if max_corpus and len(corpus) > max_corpus:
        # 调试用的小语料：保留所有被标注为相关的文档，再随机补足到 max_corpus 篇
        # 注意：语料变小后指标会虚高，最终报告请用完整语料
        relevant = {d for docs in qrels.values() for d in docs}
        keep = [d for d in corpus if d["id"] in relevant]
        rest = [d for d in corpus if d["id"] not in relevant]
        random.Random(seed).shuffle(rest)
        corpus = keep + rest[: max(0, max_corpus - len(keep))]

    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "queries.jsonl", queries)
    write_qrels(out / "qrels.tsv", {q["id"]: qrels[q["id"]] for q in queries})
    splits = write_splits(out, [q["id"] for q in queries], n_test, n_dev, seed)
    stats = {"docs": len(corpus), "queries": len(queries), **{k: len(v) for k, v in splits.items()}}
    print(f"[prepare] {name} -> {out}: {stats}")
    return stats


def write_splits(out_dir: str | Path, qids: list[str], n_test: int, n_dev: int, seed: int = 42) -> dict:
    splits = make_splits(qids, n_test, n_dev, seed)
    with open(Path(out_dir) / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)
    return splits


def make_splits_for(data_dir: str | Path, n_test: int, n_dev: int, seed: int = 42) -> dict:
    """给已经有 queries.jsonl + qrels.tsv 的数据目录（比如你自己的语料）生成 splits.json。"""
    queries, qrels = load_queries(data_dir), load_qrels(data_dir)
    labeled = [q for q in queries if any(s > 0 for s in qrels.get(q, {}).values())]
    if not labeled:
        raise ValueError(f"{data_dir} 里没有带正例标注的查询")
    if n_test + n_dev > len(labeled):
        print(f"[prepare] 只有 {len(labeled)} 条带标注的查询，test / dev 会被截断，train 可能为空")
    splits = write_splits(data_dir, labeled, n_test, n_dev, seed)
    stats = {k: len(v) for k, v in splits.items()}
    print(f"[prepare] {Path(data_dir) / 'splits.json'}: {stats}")
    return stats


def read_text_file(path: Path) -> str:
    """中文 Windows 上很多 txt 是 GBK 编码，utf-8 读失败时退回 gb18030。"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def prepare_docs(docs_dir: str | Path, out_dir: str | Path, chunk_size: int = 400, overlap: int = 80) -> dict:
    """把一个文件夹里的 .txt / .md 切块成 corpus.jsonl。PDF / Word 请先转成文本再放进来。"""
    docs_dir, out = Path(docs_dir), Path(out_dir)
    files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in (".txt", ".md") and p.is_file())
    if not files:
        raise FileNotFoundError(f"{docs_dir} 下没有 .txt / .md 文件")
    rows = []
    for p in files:
        rel = p.relative_to(docs_dir).as_posix()
        for i, chunk in enumerate(chunk_text(read_text_file(p), chunk_size, overlap)):
            rows.append({"id": f"{rel}#{i:04d}", "text": chunk, "title": p.stem, "source": rel})
    write_jsonl(out / "corpus.jsonl", rows)
    stats = {"files": len(files), "chunks": len(rows)}
    print(f"[prepare] {docs_dir} -> {out / 'corpus.jsonl'}: {stats}")
    print("[prepare] 下一步：写 queries.jsonl 和 qrels.tsv（可以先用大模型生成问题，再人工核对），"
          f"然后运行 python scripts/prepare_data.py --make-splits {out}")
    return stats
