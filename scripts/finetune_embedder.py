"""BGE 领域微调 —— 即 "engineering 层的 fine-tune"。

前提：已经有未微调模型的基线结果，且 splits.json 里有 train 划分（prepare_data.py 自动生成）。
先装训练依赖：pip install "sentence-transformers[train]"（会装上 datasets 和 accelerate）。

    python scripts/finetune_embedder.py --dry-run                       # 只构造训练对、做泄漏检查，不加载模型
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/finetune_embedder.py              # 挖难负例 + 训练，存到 output/bge-ft
    # 训练完：用微调模型重建向量索引，在 test 上和原始模型比（向量放在 indexes/<数据集>/dense/output__bge-ft/，不覆盖原来的）
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/build_index.py --only dense --set embedder.model=output/bge-ft
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/evaluate.py --modes dense --set embedder.model=output/bge-ft --tag ft
    python scripts/compare_runs.py ...                                   # 和原始 base 的 dense 结果做配对检验

步骤
1. 构造训练对：train 划分里的每个查询 + qrels 里的每个正例文档 -> (anchor=查询, positive=正例文本)。
   泄漏检查：train 查询的文本不能和 dev / test 查询重复；正例文档同时是 dev / test 查询正例的训练对默认去掉
   （CMedQA 里有 59 对：模型会把这篇文档往 train 查询的方向拉，而 test 查询问的是同一件事，test 指标会虚高）。
2. 挖难负例（sentence_transformers.util.mine_hard_negatives，output_format="n-tuple"）：
   - range_min=10：跳过最像的前 10 个，它们很可能是"没标注出来的正例"（假负例）——补标实验证明 CMedQA 标注严重不全
   - relative_margin=0.05：负例的相似度必须比正例低 5% 以上，进一步过滤假负例
   - 挖完再去掉"恰好是这个查询其他正例"的负例（一个查询平均 1.86 个正例，每行只带一个）
   - --cross-encoder 传重排模型时，用它的打分代替向量相似度来过滤（更稳，但慢）
3. 训练：SentenceTransformerTrainer + MultipleNegativesRankingLoss（InfoNCE：批内负例 + 难负例）
   - learning_rate 2e-5，num_train_epochs 2，warmup_ratio 0.1，fp16（GPU 支持时 bf16）
   - 默认 CachedMultipleNegativesRankingLoss：batch 可以开大（对比学习里负例越多越好），显存只按 mini_batch 算
   - batch_sampler=NO_DUPLICATES：避免同一批出现重复文本变成假负例
   - 训练时查询（anchor）和推理时一样加 BGE 查询指令：prompts={"anchor": 指令}
   - evaluator：dev 查询 + 抽样语料上的 InformationRetrievalEvaluator，按 nDCG@10 选最好的 checkpoint
     （抽样语料 = dev 全部正例 + 随机 2 万篇，绝对值偏高，只用来选 checkpoint；最终数字用 evaluate.py 在全量语料上跑）
4. 保存到 output/bge-ft（含 finetune_meta.json：参数、训练对统计、训练前后 dev 指标）。
5. 写进简历的那张表：原始 base / 原始 large / 微调 base 的 test nDCG@10、Recall@10、编码速度。

注意数据泄漏：test 划分的查询绝不能进训练集；调参（学习率、轮数、负例数）只看 dev。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.data import doc_text, load_corpus, load_qrels, load_queries, load_split_ids  # noqa: E402
from rag.paths import resolve, resolve_model  # noqa: E402


def build_pair_lists(data_dir: str, split: str = "train", exclude_eval_docs: bool = True,
                     eval_splits: tuple[str, ...] = ("dev", "test")) -> tuple[dict, dict]:
    """返回 ({"anchor": [...], "positive": [...], "qid": [...], "doc_id": [...]}, 统计)。不依赖 datasets，方便单测。"""
    queries, qrels = load_queries(data_dir), load_qrels(data_dir)
    docs = {d["id"]: d for d in load_corpus(data_dir)}
    train_ids = load_split_ids(data_dir, split) or []
    eval_ids: set[str] = set()
    for s in eval_splits:
        try:
            eval_ids |= set(load_split_ids(data_dir, s) or [])
        except KeyError:
            pass
    overlap = set(train_ids) & eval_ids
    if overlap:
        raise ValueError(f"{split} 划分和 {eval_splits} 有 {len(overlap)} 个相同的查询 id，划分有问题")
    eval_texts = {queries[q].strip() for q in eval_ids if q in queries}
    eval_docs = {d for q in eval_ids for d, s in qrels.get(q, {}).items() if s > 0}
    out = {"anchor": [], "positive": [], "qid": [], "doc_id": []}
    stats = {"queries": 0, "pairs": 0, "dup_query_text": 0, "eval_doc_pairs_dropped": 0, "missing_docs": 0}
    for q in sorted(train_ids):
        if q not in queries:
            continue
        if queries[q].strip() in eval_texts:  # 查询文本和 dev / test 重复：直接丢掉
            stats["dup_query_text"] += 1
            continue
        used = False
        for d, s in sorted(qrels.get(q, {}).items()):
            if s <= 0:
                continue
            if d not in docs:
                stats["missing_docs"] += 1
                continue
            if exclude_eval_docs and d in eval_docs:
                stats["eval_doc_pairs_dropped"] += 1
                continue
            out["anchor"].append(queries[q])
            out["positive"].append(doc_text(docs[d]))
            out["qid"].append(q)
            out["doc_id"].append(d)
            used = True
        stats["queries"] += int(used)
    stats["pairs"] = len(out["anchor"])
    return out, stats


def build_pairs(data_dir: str, split: str = "train", exclude_eval_docs: bool = True):
    """返回 (datasets.Dataset[anchor, positive], 统计, 每行对应的 qid 列表)。"""
    from datasets import Dataset

    lists, stats = build_pair_lists(data_dir, split, exclude_eval_docs)
    return Dataset.from_dict({"anchor": lists["anchor"], "positive": lists["positive"]}), stats, lists["qid"]


def drop_known_positive_negatives(rows: list[dict], positives_by_anchor: dict[str, set[str]]) -> tuple[list[dict], int]:
    """去掉负例里"其实是这个查询另一个正例"的行。rows 是 n-tuple 格式（anchor / positive / negative_1..n）。"""
    kept, dropped = [], 0
    for r in rows:
        pos = positives_by_anchor.get(r["anchor"], set())
        if any(v in pos for k, v in r.items() if k.startswith("negative")):
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def mine_negatives(pairs, model, corpus_texts: list[str], query_prompt: str, num_negatives: int = 5,
                   range_min: int = 10, range_max: int = 50, relative_margin: float = 0.05, cross_encoder=None,
                   batch_size: int = 128):
    """调用 sentence_transformers.util.mine_hard_negatives（output_format="n-tuple"），返回带 negative_1..n 列的数据集。"""
    from sentence_transformers.util import mine_hard_negatives

    try:
        import faiss  # noqa: F401
        use_faiss = True
    except ImportError:
        use_faiss = False
    return mine_hard_negatives(
        pairs, model, corpus=corpus_texts, cross_encoder=cross_encoder,
        range_min=range_min, range_max=range_max, relative_margin=relative_margin,
        num_negatives=num_negatives, sampling_strategy="random", query_prompt=query_prompt,
        batch_size=batch_size, use_faiss=use_faiss, output_format="n-tuple", verbose=True,
    )


def dev_evaluator(data_dir: str, query_prompt: str, eval_corpus: int = 20000, seed: int = 42, split: str = "dev"):
    """dev 查询 + 抽样语料（dev 全部正例 + 随机 eval_corpus 篇）上的检索评测器，用来选 checkpoint。"""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries, qrels = load_queries(data_dir), load_qrels(data_dir)
    docs = {d["id"]: d for d in load_corpus(data_dir)}
    ids = [q for q in (load_split_ids(data_dir, split) or []) if q in queries and any(s > 0 for s in qrels.get(q, {}).values())]
    relevant = {q: {d for d, s in qrels[q].items() if s > 0 and d in docs} for q in ids}
    keep = set().union(*relevant.values()) if relevant else set()
    rest = sorted(set(docs) - keep)
    random.Random(seed).shuffle(rest)
    keep |= set(rest[: max(0, eval_corpus)])
    return InformationRetrievalEvaluator(
        queries={q: queries[q] for q in ids}, corpus={d: doc_text(docs[d]) for d in keep}, relevant_docs=relevant,
        name=split, query_prompt=query_prompt or None, ndcg_at_k=[10], accuracy_at_k=[1, 10], precision_recall_at_k=[10, 100],
        mrr_at_k=[10], map_at_k=[100], batch_size=128, show_progress_bar=False,
    )


def train(model_name: str, train_ds, dev_eval, out_dir: str, query_prompt: str = "", epochs: float = 2,
          lr: float = 2e-5, batch_size: int = 64, mini_batch_size: int = 16, loss_name: str = "cmnrl",
          max_seq_length: int | None = None, seed: int = 42, model=None) -> dict:
    """SentenceTransformerTrainer + (Cached)MultipleNegativesRankingLoss，训练完把 dev 上最好的 checkpoint 保存到 out_dir。"""
    import torch
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    model = model or SentenceTransformer(model_name)
    if max_seq_length:
        model.max_seq_length = int(max_seq_length)
    if loss_name == "cmnrl":
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    else:
        loss = MultipleNegativesRankingLoss(model)
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    eval_steps = max(10, int(steps_per_epoch * epochs) // 8)  # 整个训练评 8 次左右
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    metric = f"eval_{dev_eval.name}_cosine_ndcg@10"
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(out_dir) / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=cuda and not bf16,
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"anchor": query_prompt} if query_prompt else None,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=True,
        logging_steps=max(1, eval_steps // 2),
        report_to="none",
        seed=seed,
    )
    before = dev_eval(model)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss, evaluator=dev_eval)
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    after = dev_eval(model)  # load_best_model_at_end：这里是最好的 checkpoint
    model.save_pretrained(str(out_dir))
    key = f"{dev_eval.name}_cosine_ndcg@10"
    return {"dev_before": before, "dev_after": after, "dev_ndcg10_before": before.get(key),
            "dev_ndcg10_after": after.get(key), "train_seconds": seconds, "steps_per_epoch": steps_per_epoch,
            "eval_steps": eval_steps, "best_checkpoint": trainer.state.best_model_checkpoint}


def main() -> None:
    p = argparse.ArgumentParser(description="BGE 领域微调（train 划分 + 难负例 + InfoNCE）")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--model", default=None, help="底座模型，默认读 embedder.model")
    p.add_argument("--out", default="output/bge-ft")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mini-batch-size", type=int, default=16, help="cmnrl 每次前向的条数，显存不够就调小")
    p.add_argument("--loss", choices=["cmnrl", "mnrl"], default="cmnrl")
    p.add_argument("--num-negatives", type=int, default=5)
    p.add_argument("--range-min", type=int, default=10)
    p.add_argument("--range-max", type=int, default=50)
    p.add_argument("--relative-margin", type=float, default=0.05)
    p.add_argument("--cross-encoder", default=None, help="用重排模型过滤假负例，例如 BAAI/bge-reranker-v2-m3（慢）")
    p.add_argument("--no-negatives", action="store_true", help="不挖难负例，只用批内负例（消融用）")
    p.add_argument("--keep-eval-docs", action="store_true", help="不去掉正例同时是 dev/test 正例的训练对（不建议）")
    p.add_argument("--eval-corpus", type=int, default=20000, help="dev 评测器的抽样语料大小（另加 dev 全部正例）")
    p.add_argument("--max-seq-length", type=int, default=None, help="默认读 embedder.max_seq_length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="只构造训练对、做泄漏检查")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    data_dir = cfg["data_dir"]
    ecfg = cfg["embedder"]
    model_name = args.model or ecfg["model"]
    query_prompt = ecfg.get("query_instruction") or ""
    lists, stats = build_pair_lists(data_dir, "train", exclude_eval_docs=not args.keep_eval_docs)
    print(f"[pairs] {stats}")
    if not stats["pairs"]:
        raise SystemExit("train 划分里没有训练对：先用 prepare_data.py 生成带 train 的 splits.json")
    if args.dry_run:
        return

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    model = SentenceTransformer(resolve_model(model_name))
    msl = args.max_seq_length or ecfg.get("max_seq_length")
    if msl:
        model.max_seq_length = int(msl)
    pairs = Dataset.from_dict({"anchor": lists["anchor"], "positive": lists["positive"]})
    mine_stats: dict = {}
    if args.no_negatives:
        train_ds = pairs
    else:
        corpus_texts = [doc_text(d) for d in load_corpus(data_dir)]
        ce = None
        if args.cross_encoder:
            from sentence_transformers import CrossEncoder

            ce = CrossEncoder(resolve_model(args.cross_encoder), max_length=512)
        t0 = time.perf_counter()
        mined = mine_negatives(pairs, model, corpus_texts, query_prompt, args.num_negatives, args.range_min,
                               args.range_max, args.relative_margin, ce)
        positives: dict[str, set[str]] = {}
        for a, pos in zip(lists["anchor"], lists["positive"]):
            positives.setdefault(a, set()).add(pos)
        rows, dropped = drop_known_positive_negatives(mined.to_list(), positives)
        train_ds = Dataset.from_list(rows)
        mine_stats = {"mined_rows": len(mined), "dropped_known_positive": dropped, "train_rows": len(train_ds),
                      "mine_seconds": time.perf_counter() - t0}
        print(f"[negatives] {mine_stats}")
        with open(out / "train_mined.jsonl", "w", encoding="utf-8") as f:  # 抽样人工检查负例质量
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dev_eval = dev_evaluator(data_dir, query_prompt, args.eval_corpus, args.seed)
    result = train(model_name, train_ds, dev_eval, str(out), query_prompt, args.epochs, args.lr, args.batch_size,
                   args.mini_batch_size, args.loss, msl, args.seed, model=model)
    meta = {"base_model": model_name, "args": vars(args), "pairs": stats, "negatives": mine_stats,
            **{k: v for k, v in result.items() if k not in ("dev_before", "dev_after")},
            "dev_before": result["dev_before"], "dev_after": result["dev_after"],
            "note": "dev 指标在抽样语料上，只用来选 checkpoint；最终用 build_index + evaluate.py 在全量语料、test 划分上比较"}
    with open(out / "finetune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    b, a = result["dev_ndcg10_before"], result["dev_ndcg10_after"]
    print(f"\n[done] dev(抽样语料) nDCG@10：{b:.4f} -> {a:.4f}；模型已保存到 {out}")
    rel = Path(args.out).as_posix()
    print("下一步（全量语料上的真实数字）：\n"
          f"  python scripts/build_index.py --only dense --set embedder.model={rel}\n"
          f"  python scripts/evaluate.py --split dev --modes dense --set embedder.model={rel} --tag ft_dev\n"
          f"  # dev 上确认有提升后，test 只跑一次：\n"
          f"  python scripts/evaluate.py --modes dense,dense_rerank --set embedder.model={rel} --tag ft_test")


if __name__ == "__main__":
    main()
