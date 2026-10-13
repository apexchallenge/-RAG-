"""大模型补标：给"没被标注过"的 (问题, 回答) 打相关性分，修正标注不完整造成的假负例。

为什么需要：CMedQA 的"正确答案"只是原帖下面医生的回复，别处同样能回答这个问题的回答一律算错，
重排把更贴题的回答排上来反而扣分。补标按 TREC 的 pooling 思路：
  1. 汇总各个方法前 depth 名的结果（候选池），对池里没有标注的文档请大模型打分；
  2. 原标注 ∪ 大模型判为相关的，作为新的 qrels 重算指标；
  3. 抽一批让人盲评（看不到大模型的分），算一致率（Cohen's kappa）—— 没有这一步，补标结果站不住。

注意：
  - 补标只用来"评估"，不能拿来调参（尤其不能在 test 上调）；报告里原始指标和补标指标都要给。
  - 大模型评审本身有偏好（可能和重排模型口味相近），所以人工抽查和一致率必须写进报告。
  - 只对池子里的文档打分，池外的仍当不相关，所以池子要包含所有被比较的方法。
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .utils import progress

PROMPT_VERSION = "v1"
GRADES = (0, 1, 2)

JUDGE_PROMPT = """你是医疗问答检索系统的相关性评审员。请判断【回答】对【用户问题】有多大帮助。

评分标准：
2 = 直接回答了用户问题的核心（用户问的原因、处理办法或具体信息），用户读完能得到想要的答案
1 = 相关，但只回答了一部分、过于笼统，或回答的是相关的另一个方面
0 = 无关、答非所问，或基本只有广告、网页残留和套话

要求：只看内容是否针对这个问题；不评判医学观点是否完全正确；不要因为回答长或短而加减分；忽略"关注""投诉""已帮助用户"等网页残留文字。
只输出一个数字（0、1 或 2），不要解释。

【用户问题】
{query}

【回答】
{doc}"""

LLM = Callable[[str], str]


class EmptyOutput(RuntimeError):
    """大模型返回了空正文（多半是推理型模型的输出上限太小，全花在思考过程上了）。"""


class JudgeAbort(RuntimeError):
    """试跑的几条全部失败：配置有问题，直接停下，免得白白调用上千次。"""


def build_judge_prompt(query: str, doc: str, max_chars: int = 800) -> str:
    return JUDGE_PROMPT.format(query=query.strip(), doc=doc.strip()[:max_chars])


_ONLY = re.compile(r"^\D{0,3}([012])\D{0,3}$")
_LABELED = re.compile(r"(?:评分|分数|得分|结论|答案|输出)\s*[:：为是]?\s*([012])(?!\d)")
_TAIL = re.compile(r"(?<!\d)([012])(?!\d)\D{0,6}$")


def parse_grade(text: str | None) -> int | None:
    """从大模型输出里取分数。正常情况就是一个数字；带解释时取"评分：x"或结尾的数字。取不到返回 None（不猜）。"""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # 推理模型可能把思考过程也输出来
    for pat in (_ONLY, _LABELED, _TAIL):
        m = pat.search(t)
        if m:
            return int(m.group(1))
    return None


def make_llm(gen_cfg: dict, max_tokens: int = 8, timeout: float = 60.0) -> LLM:
    """用 configs 里 generation 段的 OpenAI 兼容接口（DeepSeek / 通义 / 本地 Ollama 都行），temperature=0。"""
    from openai import OpenAI

    from .generator import Generator

    gen = Generator(gen_cfg)
    if not gen.available:
        raise RuntimeError(f"没有可用的大模型：请设置环境变量 {gen_cfg.get('api_key_env', 'LLM_API_KEY')}（或写进 .env）")
    client = OpenAI(base_url=gen.base_url, api_key=gen.api_key, timeout=timeout, max_retries=5)

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=gen.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        if not content.strip():
            # 推理型模型会先把 token 花在思考过程（reasoning_content）上，max_tokens 太小时正文是空的
            thinking = bool(getattr(choice.message, "reasoning_content", None))
            raise EmptyOutput(f"空输出 finish={choice.finish_reason}" + ("，有思考内容" if thinking else ""))
        return content

    return call


class JudgmentCache:
    """打分结果追加写进 jsonl，中断后重跑会跳过已打过的对。换了模型或提示词版本，旧结果自动不用。"""

    def __init__(self, path: str | Path, model: str, prompt_version: str = PROMPT_VERSION):
        self.path = Path(path)
        self.model = model
        self.version = prompt_version
        self.grades: dict[tuple[str, str], int | None] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 上次中断时写了半行
                    if r.get("model") == model and r.get("prompt_version") == prompt_version:
                        self.grades[(r["qid"], r["doc_id"])] = r.get("grade")

    def __contains__(self, key: tuple[str, str]) -> bool:
        return self.grades.get(key) is not None  # 解析失败的（None）下次重试

    def add(self, qid: str, doc_id: str, grade: int | None, raw: str) -> None:
        row = {"qid": qid, "doc_id": doc_id, "grade": grade, "model": self.model,
               "prompt_version": self.version, "raw": raw[:80]}
        with self._lock:
            self.grades[(qid, doc_id)] = grade
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_pairs(
    pairs: list[tuple[str, str]],
    queries: dict[str, str],
    doc_texts: dict[str, str],
    llm: LLM,
    cache: JudgmentCache,
    workers: int = 8,
    max_chars: int = 800,
    show_progress: bool = True,
    probe_n: int = 5,
    max_calls: int | None = None,
) -> dict:
    """给还没打过分的 (qid, doc_id) 打分，结果写进 cache。返回本次的统计。

    断点续跑：每条结果一出来就追加写盘（在工作线程里写，按 Ctrl+C 时已经返回的请求也不会丢）。
    按 Ctrl+C：取消还没发出的请求，只等正在进行的几条（最多 workers 条）返回，然后抛出 KeyboardInterrupt；
    重跑同一命令会跳过已打过分的对。max_calls：本次最多调用多少次，用来分批控制花费。
    """
    todo = [p for p in dict.fromkeys(pairs) if p not in cache]
    n_unique, n_uncached = len(set(pairs)), len(todo)
    if max_calls is not None:
        todo = todo[:max(0, max_calls)]
    stats = Counter()
    lock = threading.Lock()

    def work(pair):
        q, d = pair
        try:
            raw = llm(build_judge_prompt(queries[q], doc_texts[d], max_chars))
            grade = parse_grade(raw)
        except Exception as e:  # 单条失败不影响整体，记成 None，下次重跑会重试
            raw, grade = f"ERROR {type(e).__name__}: {e}"[:80], None
        cache.add(q, d, grade, raw)  # 立刻落盘
        with lock:
            stats["failed" if grade is None else f"grade_{grade}"] += 1
        return raw

    # 先串行试跑几条：全部失败说明配置有问题（key、模型名、推理模型输出上限……），直接停下
    probe, rest = todo[:probe_n], todo[probe_n:]
    probe_raw = [work(pair) for pair in probe]
    if probe and stats["failed"] == len(probe):
        raise JudgeAbort(f"试跑的 {len(probe)} 条全部没取到分数，已停止。模型返回：{probe_raw[:3]}。"
                         "常见原因：推理型模型输出上限太小（加大 --max-tokens，如 1024，或换非推理模型）；"
                         "API key / base_url / 模型名不对。")

    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = [ex.submit(work, p) for p in rest]
        for fut in progress(as_completed(futures), "大模型打分", not show_progress, total=len(futures)):
            fut.result()
    except KeyboardInterrupt:
        # 不加 cancel_futures 的话，退出时会把队列里剩下的几千个请求全部发完（照样计费）
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    stats["requested"] = len(todo)
    stats["cached"] = n_unique - n_uncached
    stats["remaining"] = n_uncached - len(todo)  # 受 max_calls 限制这次没打的
    return dict(stats)


def build_pool(runs: dict[str, dict[str, list[str]]], qids: list[str], modes: list[str], depth: int) -> dict[str, list[str]]:
    """每条查询：各模式前 depth 名的并集（保持首次出现的顺序）。"""
    pool: dict[str, list[str]] = {}
    for q in qids:
        seen: dict[str, None] = {}
        for m in modes:
            for d in runs.get(m, {}).get(q, [])[:depth]:
                seen.setdefault(d, None)
        pool[q] = list(seen)
    return pool


def merge_qrels(
    qrels: dict[str, dict[str, int]],
    grades: dict[tuple[str, str], int | None],
    threshold: int = 2,
) -> dict[str, dict[str, int]]:
    """原标注的正例保持相关；大模型分数 >= threshold 的也算相关（记 1）；判过但不够的记 0（算"已判"）。"""
    merged = {q: dict(v) for q, v in qrels.items()}
    for (q, d), g in grades.items():
        if g is None or q not in merged:
            continue
        if merged[q].get(d, 0) > 0:
            continue  # 原标注优先
        merged[q][d] = 1 if g >= threshold else 0
    return merged


def judged_at_k(run: dict[str, list[str]], judged: dict[str, set[str]], k: int = 10) -> float:
    """前 k 名里有多少比例被判过（原正例或大模型打过分）。补标后接近 100% 才说明池子够深。"""
    vals = [sum(1 for d in docs[:k] if d in judged.get(q, set())) / k for q, docs in run.items() if docs]
    return sum(vals) / len(vals) if vals else 0.0


def cohen_kappa(a: list, b: list) -> float:
    """两组标签的一致性，扣除了"碰巧一致"的部分。>0.6 一般认为较好，>0.8 很好。"""
    if len(a) != len(b) or not a:
        raise ValueError("两组标签长度要相同且非空")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
