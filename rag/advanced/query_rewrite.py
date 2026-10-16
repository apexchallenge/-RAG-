"""查询改写 + 推理（"加点 reasoning"）。

四个由浅入深的功能，每个都要用评测说话（在 dev 上调，在 test 上报告；同时记录多花的延迟和 token）：
  1. multi_query：大模型把问题改写成 3 个不同说法，分别检索后用 RRF 合并 —— 解决口语化、表述不全
  2. hyde：大模型先"假想"一段答案，用它做向量检索（HyDE, Gao et al. 2022）—— 短问题找长文档
  3. decompose：把需要多条证据的问题拆成子问题，逐个检索再汇总（IRCoT 的思路）
  4. answer_with_check：回答前让大模型把资料判成 充分 / 部分 / 部分（查不到）/ 无关；
     只有"部分"（缺的是别的医生回答里可能有的通用知识）才针对缺的那块写一段 HyDE 式口语文字补检索，
     补检索到的新文档保留名额放进资料；资料没变化、补检索文字重复时提前停；最后仍是部分也照常回答，只有"无关"才拒答
     （CRAG 的分档判断 + Self-RAG 的检索后反思，简化版）

llm 统一约定为一个函数 llm(prompt: str) -> str，单测里用假模型替换；真实调用可以包一下 rag.generator 里的 OpenAI 客户端。
"""

from __future__ import annotations

import re
from typing import Callable

from ..data import doc_text
from ..fusion import fuse, rrf_fuse
from ..generator import REFUSAL
from ..types import Hits
from ..utils import timer

LLM = Callable[[str], str]

REWRITE_PROMPT = """请把下面的用户问题改写成 {n} 个意思相同、但措辞不同的检索查询，便于在知识库中搜索。
要求：补全省略的主语和关键词，把口语换成书面语；每行一个，不要编号，不要解释。
用户问题：{question}"""

HYDE_PROMPT = """请针对下面的问题，写一段约 150 字、像知识库原文一样的回答段落（可以不完全准确，只用于检索）。
问题：{question}"""

DECOMPOSE_PROMPT = """下面的问题可能需要多条资料才能回答。请把它拆成最多 {n} 个可以独立检索的子问题，每行一个，不要编号。
如果问题本身已经足够简单，只输出原问题。
问题：{question}"""

CHECK_PROMPT = """判断【参考资料】能不能用来回答【问题】。
参考资料是网上医生对类似问题的回答，通常是通用建议，不会针对提问者的个人情况；不要因为资料不够个性化、不能确诊就判为不够。
问题可能包含好几问，逐个看，再选一档：
充分：资料能针对问题的核心给出有用的回答（可能原因、处理办法或用户问的具体信息），哪怕比较笼统
部分：资料能回答一部分，但某个关键方面没有涉及，而这方面是通用的医学知识，别的医生回答里可能有
部分（查不到）：问题的核心资料能回答，没涉及的只是一些细节，而这些细节是任何医生回答里都不会有的具体信息 ——
提问者本人的检查结果和病情判断、某家医院或机构的安排、具体的价格和政策、用户没说名字的药等
无关：资料对问题的任何一问都没有帮助（只要有一问能用资料回答，就不算无关）；
或者问题的核心本身就是查不到的具体信息（某家医院 / 机构的安排、某个人的账户或记录、"最新""今年""现在"的信息），
资料里只有别的地区、别的机构、过去某个时间的类似内容 —— 不能拿来当成用户问的那一个，判无关
只输出一行："充分"，或 "部分：<没涉及的方面>"，或 "部分（查不到）：<没涉及的方面>"，或 "无关：<原因>"。
【参考资料】
{contexts}
【问题】{question}"""


REFINE_PROMPT = """用户问题：{question}
已经检索到的资料没有覆盖到：{missing}
之前用过的检索查询：{tried}
请写 1 个新的检索查询，用来在网上医疗问答知识库里找到这部分信息。知识库里是普通网友的提问和医生的口语化回答，
所以请像患者提问那样用口语写，一句话，不要堆砌医学术语和关键词，不要和之前的查询重复。
只输出查询本身，不要解释。"""

REFINE_HYDE_PROMPT = """用户问题：{question}
已经检索到的资料没有覆盖到：{missing}
请写一段 100 字左右、像网上医生回答这类问题那样的口语化回答，重点写资料没覆盖到的那部分（可以不完全准确，只用来检索相似的回答）。
不要堆砌关键词，不要分点，只输出这段话。{tried}"""

_TRIED_HINT = "\n之前已经用这些文字检索过，没找到想要的，请换一个角度或说法，不要重复：\n{texts}"

# 判断结果的三档（从差到好）
IRRELEVANT, PARTIAL, SUFFICIENT = "irrelevant", "partial", "sufficient"
_RANK = {IRRELEVANT: 0, PARTIAL: 1, SUFFICIENT: 2}

_PREFIX = re.compile(r"^\s*(?:\d+\s*[\.、\)）:：]|[-•*·]|[（(]\d+[)）])\s*")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_lines(text: str, max_n: int = 3) -> list[str]:
    """把大模型输出的多行文本解析成干净的列表：去掉 "1." "1、" "-" "•" 等前缀、空行和重复项，最多 max_n 条。"""
    out: list[str] = []
    for line in _THINK.sub("", text or "").splitlines():
        line = _PREFIX.sub("", line).strip().strip('"“”')
        if line and line not in out:
            out.append(line)
        if len(out) >= max_n:
            break
    return out


def multi_query(llm: LLM, question: str, n: int = 3) -> list[str]:
    """返回 [原问题] + 最多 n 个改写（原问题放第一个，保证不会比不改写更差太多）。"""
    rewrites = [q for q in parse_lines(llm(REWRITE_PROMPT.format(n=n, question=question)), max_n=n + 1) if q != question]
    return [question] + rewrites[:n]


def hyde(llm: LLM, question: str) -> str:
    """让大模型先写一段"假想答案"，用它去做向量检索（答案和知识库原文的写法更像）。失败时退回原问题。"""
    text = _THINK.sub("", llm(HYDE_PROMPT.format(question=question)) or "").strip()
    return text or question


def decompose(llm: LLM, question: str, n: int = 3) -> list[str]:
    """拆成最多 n 个可以独立检索的子问题；模型认为不用拆时返回 [原问题]。"""
    subs = parse_lines(llm(DECOMPOSE_PROMPT.format(n=n, question=question)), max_n=n)
    return subs or [question]


def retrieve_multi(pipeline, queries: list[str], top_k: int = 100) -> Hits:
    """每个查询分别走 pipeline 的 BM25 + 向量 + 融合（不重排），再用 RRF 合并成一个列表。"""
    r = pipeline.cfg["retrieval"]
    lists: list[Hits] = []
    for q in queries:
        b = pipeline.bm25.search([q], int(r.get("bm25_top_k", 100)))[0]
        d = pipeline.dense.search([q], int(r.get("dense_top_k", 100)))[0]
        lists.append(fuse(r, b, d))
    return rrf_fuse(lists, k=int(r.get("rrf_k", 60)), top_k=top_k)


def parse_verdict(text: str) -> tuple[str, str]:
    """解析 CHECK_PROMPT 的输出：返回 (sufficient / partial / irrelevant, 说明)。
    兼容旧的"不足："写法（当作 partial）；解析不了时按 partial 处理：再查一轮，但不会因此拒答。"""
    t = _THINK.sub("", text or "").strip()
    first = t.splitlines()[0].strip() if t else ""
    status, note, _ = parse_check_detail(text)
    return status, note


_NOT_FOUND = re.compile(r"^[\s*_`]*[（(【\[]?\s*(查不到|无需再查|不用再查|不可检索)\s*[)）】\]]?")


def parse_check_detail(text: str) -> tuple[str, str, bool]:
    """(档位, 说明, 缺的部分值不值得再检索)。"部分（查不到）"返回 (partial, 说明, False)。"""
    t = _THINK.sub("", text or "").strip()
    first = t.splitlines()[0].strip() if t else ""
    m = re.match(r"^[\W_]*(不充分|不足|部分|无关|充分)", first)
    if not m:
        return PARTIAL, "", True
    rest = first[m.end():]
    retrievable = True
    nf = _NOT_FOUND.match(rest)
    if m.group(1) in ("部分", "不足", "不充分") and nf:
        retrievable, rest = False, rest[nf.end():]
    note = re.sub(r"^[\s:：，,。—*_`\-]+", "", rest).strip()
    word = m.group(1)
    if word == "充分":
        return SUFFICIENT, "", True
    if word == "无关":
        return IRRELEVANT, note, True
    return PARTIAL, note, retrievable


def parse_check(text: str) -> tuple[bool, str]:
    """(是否充分, 缺失信息)。只有明确的"充分"才算充分；"不充分"不会被误判成充分。"""
    status, note = parse_verdict(text)
    return status == SUFFICIENT, note


def _format_contexts(texts: list[str], max_chars: int) -> str:
    return "\n\n".join(f"[{i}] {t[:max_chars]}" for i, t in enumerate(texts, start=1))


def _filter(hits: Hits, exclude: set[str] | None) -> Hits:
    return [(d, s) for d, s in hits if not exclude or d not in exclude]


def _bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def similar(a: str, b: str) -> float:
    """两段文字的字符二元组 Jaccard 相似度，用来发现"这轮补检索文字和上一轮几乎一样"。"""
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def _rank(pipeline, query: str, doc_ids: list[str]) -> list[str]:
    """用重排模型按 query 给文档排序；没有重排模型就保持原顺序。"""
    if getattr(pipeline, "reranker", None) is None or len(doc_ids) <= 1:
        return list(doc_ids)
    cands = [(d, doc_text(pipeline.docs[d])) for d in doc_ids]
    return [d for d, _ in pipeline.reranker.rerank([query], [cands])[0]]


def merge_contexts(pipeline, question: str, missing: str, base: list[str], added: list[str], new_ids: list[str],
                   k: int, mode: str = "slots", new_slots: int = 2, min_keep: int = 2) -> tuple[list[str], list[str]]:
    """把补检索到的文档并进资料，返回 (base, added)；最终资料 = base[:k-len(added)] + added。

    mode="slots"（默认）：新文档按"原问题 + 缺失信息"排序，取前 new_slots 条占用固定名额，替换掉原资料里排在最后的；
        至少保留原资料前 min_keep 条。—— 补检索就是冲着缺的那块去的，用原问题去比，新文档很难排进前 k。
    mode="rerank"：旧做法，新旧合在一起按原问题重排取前 k（留着做消融对比）。
    """
    current = base[:k - len(added)] + added
    fresh = [d for d in new_ids if d not in current]
    if mode == "rerank":
        merged = _rank(pipeline, question, list(dict.fromkeys(current + fresh)))[:k]
        return merged, []
    room = max(0, k - min(min_keep, k - 1) - len(added))  # k 很小时至少留 1 个名额
    take = _rank(pipeline, f"{question} {missing}".strip(), fresh)[:min(new_slots, room)]
    return base, added + take


def refine_query(llm: LLM, question: str, missing: str, tried: list[str], mode: str = "hyde") -> str:
    """根据"还缺什么"生成下一轮检索用的文本。

    mode="hyde"（默认）：写一段像医生回答的口语化文字去检索 —— 知识库本身就是医生的回答，答案风格的文字更容易匹配上；
    mode="query"：写一个口语化的检索查询。
    写不出或和之前重复时，退回"原问题 + 缺失信息"。
    """
    if mode == "hyde":
        prev = [x for x in tried if x != question]
        hint = _TRIED_HINT.format(texts="\n".join(f"- {x[:150]}" for x in prev)) if prev else ""
        raw = _THINK.sub("", llm(REFINE_HYDE_PROMPT.format(question=question, missing=missing or "（未说明）", tried=hint)) or "")
        q = " ".join(raw.split()).strip()
    else:
        lines = parse_lines(llm(REFINE_PROMPT.format(question=question, missing=missing or "（未说明）",
                                                     tried="；".join(tried))), max_n=1)
        q = lines[0] if lines else ""
    if not q or q in tried:
        q = f"{question} {missing}".strip()
    return q


def _looks_refused(answer: str) -> bool:
    """回答只是一句"根据现有资料无法回答"（而不是"无法确诊，但资料提示……"这种部分回答）。"""
    a = answer.strip()
    return a.startswith(REFUSAL) and len(a) <= len(REFUSAL) + 30


def answer_with_check(
    pipeline,
    llm: LLM,
    question: str,
    max_rounds: int = 2,
    *,
    answer_fn: Callable[[str, list[str]], str] | None = None,
    k: int | None = None,
    exclude: set[str] | None = None,
    max_chars: int | None = None,
    refine: str = "hyde",
    max_irrelevant_retries: int = 1,
    merge: str = "slots",
    new_slots: int = 2,
    dup_threshold: float = 0.6,
    no_progress_threshold: float = 0.5,
) -> dict:
    """检索 -> 判断资料 -> 值得的话补检索 -> 回答或拒答。

    - 充分：直接回答。
    - 部分（查不到）：缺的是个人检查结果、具体机构安排这类任何资料都不会有的信息，不再检索，直接回答（回答会说明缺什么）。
    - 部分：根据"没覆盖到什么"写一段检索文本（默认 HyDE）再检索，新文档按 merge 规则并进资料（默认保留名额），最多 max_rounds 轮；
            最后仍是部分也照常回答，不硬拒答。
    - 无关：最多再试 max_irrelevant_retries 轮（可能只是检索没找对）；仍无关就拒答，不再调用写回答的大模型。
    最终用判断最好的那一轮资料（补检索换进来的文档可能让资料变差）；只有每一轮都判"无关"才拒答。
    提前停止（stop_reason）：资料这一轮没有任何变化（再判断一次结果也一样）、补检索文字和之前的几乎一样、
    补检索之后判断结果和缺的东西都没变（no_progress：说明这块知识库里大概率没有，再查也是白花调用）。
    判断始终针对原问题。exclude：评测用，屏蔽这些文档，模拟"知识库里没有答案"。
    """
    gcfg = pipeline.cfg.get("generation", {}) or {}
    k = int(k or gcfg.get("context_top_k", 5))
    max_chars = int(max_chars or gcfg.get("max_chars_per_doc", 600))
    answer_fn = answer_fn or pipeline.generator.answer
    t: dict[str, float] = {}
    queries = [question]
    trace: list[dict] = []
    llm_calls = 0
    irrelevant_retries = 0
    stop_reason = "max_rounds"

    with timer(t, "retrieve"):
        hits = _filter(pipeline.retrieve(question)["final"], exclude)
    base, added = [d for d, _ in hits[:k]], []
    ctx_ids = list(base)
    status, note = PARTIAL, ""
    for r in range(max_rounds + 1):
        texts = [doc_text(pipeline.docs[d]) for d in ctx_ids]
        with timer(t, "check"):
            raw = llm(CHECK_PROMPT.format(contexts=_format_contexts(texts, max_chars), question=question))
        llm_calls += 1
        status, note, retrievable = parse_check_detail(raw)
        step = {"round": r, "query": queries[-1], "doc_ids": list(ctx_ids), "check": raw.strip()[:200],
                "status": status, "sufficient": status == SUFFICIENT, "retrievable": retrievable, "missing": note}
        trace.append(step)
        if status == SUFFICIENT:
            stop_reason = "sufficient"
            break
        if status == PARTIAL and not retrievable:
            stop_reason = "not_retrievable"
            break
        prev = trace[-2] if len(trace) >= 2 else None
        if prev is not None and prev["status"] == status and similar(prev["missing"], note) >= no_progress_threshold:
            stop_reason = "no_progress"
            break
        if r == max_rounds:
            break
        if status == IRRELEVANT:
            if irrelevant_retries >= max_irrelevant_retries:
                stop_reason = "irrelevant"
                break
            irrelevant_retries += 1
        with timer(t, "refine"):
            new_q = refine_query(llm, question, note, queries, mode=refine)
        llm_calls += 1
        step["next_query"] = new_q
        if any(similar(new_q, q) >= dup_threshold for q in queries[1:]):
            stop_reason = "duplicate_refine"
            break
        queries.append(new_q)
        with timer(t, "retrieve"):
            new_hits = _filter(pipeline.retrieve(new_q)["final"], exclude)
        with timer(t, "merge"):
            base, added = merge_contexts(pipeline, question, note, base, added, [d for d, _ in new_hits[:2 * k]], k,
                                         mode=merge, new_slots=new_slots)
        new_ctx = base[:k - len(added)] + added
        step["added"] = [d for d in new_ctx if d not in ctx_ids]
        if new_ctx == ctx_ids:
            stop_reason = "context_unchanged"
            break
        ctx_ids = new_ctx

    # 补检索换进来的文档也可能让资料变差（例如"部分"之后反而判成"无关"）：最终用判断最好的那一轮资料，
    # 同样好的取最后一轮（资料更全）；只有每一轮都判"无关"才拒答
    best = max(range(len(trace)), key=lambda i: (_RANK[trace[i]["status"]], i))
    reverted_to = None
    if _RANK[trace[best]["status"]] > _RANK[status]:
        reverted_to = best
        ctx_ids, status, note = list(trace[best]["doc_ids"]), trace[best]["status"], trace[best]["missing"]
    texts = [doc_text(pipeline.docs[d]) for d in ctx_ids]
    if status == IRRELEVANT:
        answer = REFUSAL + (f"。{note}" if note else "。")
        refused = True
    else:
        with timer(t, "answer"):
            answer = answer_fn(question, texts)
        llm_calls += 1
        refused = _looks_refused(answer)
    statuses = [s["status"] for s in trace]
    return {
        "question": question,
        "answer": answer,
        "refused": refused,
        "status": status,
        "statuses": statuses,
        "improved": _RANK[status] > _RANK[statuses[0]],
        "reverted_to_round": reverted_to,
        "context_changed": ctx_ids != trace[0]["doc_ids"],
        "stop_reason": stop_reason,
        "sufficient": status == SUFFICIENT,
        "rounds": len(trace),
        "queries": queries,
        "trace": trace,
        "contexts": [{"id": d, "text": tx} for d, tx in zip(ctx_ids, texts)],
        "timings": t,
        "llm_calls": llm_calls,
    }
