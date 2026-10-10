"""中文文本处理：jieba 分词（给 BM25 用）、切块（给你自己的长文档用）。"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

import jieba

jieba.setLogLevel(logging.WARNING)  # 关掉 "Building prefix dict..." 之类的日志

# 至少含一个中文、字母或数字的词才保留（过滤标点、空白）
_KEEP = re.compile(r"[0-9A-Za-z一-鿿]")
# 在中文句末标点、换行，以及英文 ". " 之后断句（零宽断言，标点留在句子末尾；"3.5" 不会被切开）
_SENT_END = re.compile(r"(?<=[。！？!?；;\n])|(?<=\.\s)")


def load_stopwords(path: str | Path) -> set[str]:
    with open(path, encoding="utf-8-sig") as f:
        return {line.strip() for line in f if line.strip()}


def file_sha1(path: str | Path | None) -> str | None:
    """词典/停用词文件的指纹：文件内容一改，BM25 索引就需要重建。"""
    if not path:
        return None
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


class ChineseTokenizer:
    """BM25 用的分词器。

    mode="search"：jieba 搜索引擎模式，长词会再切出短词（"人工智能" -> 人工/智能/人工智能），召回更高；
    mode="accurate"：jieba 精确模式，词更少更准。两种模式可以做消融对比。
    user_dict：领域词典（每行"词 [词频] [词性]"，词频和词性可省略），让"幽门螺杆菌"这类专有名词不被切碎。
    """

    def __init__(self, mode: str = "search", user_dict: str | None = None, stopwords: str | None = None):
        if mode not in ("search", "accurate"):
            raise ValueError(f"jieba_mode 只能是 search 或 accurate，收到 {mode!r}")
        self.mode = mode
        self._jieba = jieba.Tokenizer()  # 独立实例，加载自定义词典不影响全局
        self._jieba.initialize()  # 先加载词典（约 1 秒），免得算进第一条查询的耗时
        if user_dict:
            self._jieba.load_userdict(str(user_dict))
        self.stopwords = load_stopwords(stopwords) if stopwords else set()
        # 这些参数会影响索引里的词，存进索引目录，加载时核对
        self.params = {"mode": mode, "user_dict": file_sha1(user_dict), "stopwords": file_sha1(stopwords)}

    @classmethod
    def from_config(cls, bm25_cfg: dict) -> "ChineseTokenizer":
        return cls(
            mode=bm25_cfg.get("jieba_mode", "search"),
            user_dict=bm25_cfg.get("user_dict"),
            stopwords=bm25_cfg.get("stopwords"),
        )

    def __call__(self, text: str) -> list[str]:
        # NFKC：全角数字/字母转半角（"２８" -> "28"）；再统一小写（BM25 和 bm25 算同一个词）
        text = unicodedata.normalize("NFKC", text).lower()
        words = self._jieba.lcut_for_search(text) if self.mode == "search" else self._jieba.lcut(text)
        out = []
        for w in words:
            w = w.strip()
            if w and _KEEP.search(w) and w not in self.stopwords:
                out.append(w)
        return out


def split_sentences(text: str) -> list[str]:
    """断句。保留原文里的空白和换行（纯空白片段并到上一句末尾），拼回去不会把 "BM25\\nBGE" 粘成一个词，段落也还在。"""
    out: list[str] = []
    for piece in _SENT_END.split(text):
        if piece.strip():
            out.append(piece)
        elif out:
            out[-1] += piece
    return out


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """按句子贪心拼成不超过 chunk_size 字的块；相邻块重叠约 overlap 字，避免答案被切断。

    重叠优先取上一块末尾的整句；一句都放不下时，退而取上一块的最后 overlap 个字。
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("需要 chunk_size > overlap >= 0")
    pieces: list[str] = []
    for s in split_sentences(text):
        while len(s) > chunk_size:  # 超长句子硬切
            pieces.append(s[:chunk_size])
            s = s[chunk_size:]
        if s.strip():
            pieces.append(s)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in pieces:
        if cur and cur_len + len(s) > chunk_size:
            prev = "".join(cur)
            chunks.append(prev)
            keep: list[str] = []
            keep_len = 0
            for t in reversed(cur):
                if keep_len + len(t) > overlap:
                    break
                keep.insert(0, t)
                keep_len += len(t)
            if not keep and overlap > 0:
                keep, keep_len = [prev[-overlap:]], len(prev[-overlap:])
            if keep_len + len(s) > chunk_size:
                keep, keep_len = [], 0
            cur, cur_len = keep, keep_len
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append("".join(cur))
    return [c.strip() for c in chunks if c.strip()]
