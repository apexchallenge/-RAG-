"""保存 / 读取每条查询的完整排序结果（run 文件）。

有了 run 文件，事后的分析都不用重跑模型：补标注后重算指标、跨两次评测做显著性检验（比如 bge-base vs bge-large）、
统计融合丢失了多少、抽查任意一条查询的前 k 名。

格式：gzip 压缩的 TSV，每行  qid  Q0  doc_id  rank  score  mode
（列和 TREC run 格式一致，只是用制表符分隔，doc_id 里有空格也不会读错）。
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path

from .types import Hits


def write_runs(path: str | Path, qids: list[str], lists: dict[str, list[Hits]], k: int = 100) -> Path:
    """lists: {mode: [每个查询的 Hits]}，和 qids 顺序一致；每个查询只存前 k 条。"""
    path = Path(path)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        for mode, per_query in lists.items():
            if len(per_query) != len(qids):
                raise ValueError(f"{mode} 的结果条数 {len(per_query)} 和查询数 {len(qids)} 不一致")
            for q, hits in zip(qids, per_query):
                for rank, (doc_id, score) in enumerate(hits[:k], start=1):
                    f.write(f"{q}\tQ0\t{doc_id}\t{rank}\t{score:.6g}\t{mode}\n")
    return path


def read_runs(path: str | Path) -> dict[str, dict[str, Hits]]:
    """返回 {mode: {qid: [(doc_id, score), ...] 按 rank 排好}}。"""
    rows: dict[str, dict[str, list[tuple[int, str, float]]]] = defaultdict(lambda: defaultdict(list))
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            q, _, d, rank, score, mode = parts
            rows[mode][q].append((int(rank), d, float(score)))
    return {
        mode: {q: [(d, s) for _, d, s in sorted(items)] for q, items in per_q.items()}
        for mode, per_q in rows.items()
    }
