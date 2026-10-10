"""配置加载：YAML 文件 + 命令行覆盖（--set reranker.top_n=20）。

做消融实验时不用改文件，直接在命令行覆盖某个参数即可：
    python scripts/evaluate.py --set reranker.top_n=20 --set retrieval.fusion=weighted
配置里的相对路径（data_dir、index_dir、词典等）一律相对项目根目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import resolve

_PATH_KEYS = (("data_dir",), ("index_dir",), ("bm25", "user_dict"), ("bm25", "stopwords"))


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(resolve(path), encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        set_by_path(cfg, item)
    for keys in _PATH_KEYS:  # 覆盖之后再解析路径，--set data_dir=... 也生效
        node = cfg
        for k in keys[:-1]:
            node = node.get(k) or {}
        if node.get(keys[-1]):
            node[keys[-1]] = str(resolve(node[keys[-1]]))
    return cfg


def set_by_path(cfg: dict, assignment: str) -> None:
    """把 "a.b.c=value" 写进嵌套字典；value 按 YAML 语法解析（数字、true/false、null、列表都行）。"""
    if "=" not in assignment:
        raise ValueError(f"覆盖参数格式应为 key.sub=value，收到: {assignment!r}")
    key, raw = assignment.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"覆盖参数缺少 key: {assignment!r}")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"{key} 的上层 {p} 不是字典，无法覆盖")
    node[parts[-1]] = yaml.safe_load(raw)


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
