"""小工具：进度条、计时、文件名。"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager

try:  # tqdm 会随 sentence-transformers 一起装上；没有时退化为普通迭代
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def progress(iterable, desc: str = "", disable: bool = False, total: int | None = None):
    if _tqdm is None or disable:
        return iterable
    return _tqdm(iterable, desc=desc, total=total)


@contextmanager
def timer(store: dict, key: str):
    """with timer(t, "bm25"): ...  结束后 t["bm25"] 为耗时（秒，累加）。"""
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - start


def slugify(name: str) -> str:
    """把模型名/路径变成能当目录名的字符串：BAAI/bge-base-zh-v1.5 -> BAAI__bge-base-zh-v1.5"""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name).strip("_") or "model"
