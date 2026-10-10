"""路径约定：配置文件和脚本参数里的相对路径，一律相对"项目根目录"解析。

这样无论在哪个目录运行脚本（PyCharm 默认在 scripts/ 下运行），读写的都是同一个 data/、indexes/、results/。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str | Path | None) -> Path | None:
    if p is None or p == "":
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def resolve_model(name: str) -> str:
    """模型名既可能是 HF 上的 "BAAI/bge-base-zh-v1.5"，也可能是本地目录（如微调输出 output/bge-ft）。
    只有项目根目录下真的存在这个目录时才当本地路径用，否则原样交给 sentence-transformers 去下载。"""
    p = Path(name)
    if not p.is_absolute() and (ROOT / p).exists():
        return str(ROOT / p)
    return name
