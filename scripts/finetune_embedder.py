"""TODO（你来实现）：BGE 领域微调 —— 即 "engineering 层的 fine-tune"。

前提：已经有未微调模型的基线结果，且 splits.json 里有 train 划分（prepare_data.py 自动生成）。
先装训练依赖：pip install "sentence-transformers[train]"（会装上 datasets 和 accelerate）。

步骤
1. 构造训练对：train 划分里的每个查询 + qrels 里的正例文档
       -> datasets.Dataset.from_dict({"anchor": [...查询], "positive": [...正例文本]})
   自有语料没有标注时：让大模型为每个块生成 1~3 个问题（合成数据），抽样 100 条人工检查质量。
2. 挖难负例（sentence-transformers 自带）：
       from sentence_transformers.util import mine_hard_negatives
       mine_hard_negatives(dataset, model, corpus=<全部文档文本>, range_min=10, range_max=50,
                           relative_margin=0.05, num_negatives=5, sampling_strategy="random",
                           query_prompt=<BGE 查询指令>, use_faiss=True, output_format="n-tuple")
   - output_format="n-tuple" 得到 anchor / positive / negative_1..5 列；默认 "triplet" 是每个负例一行
   - range_min=10：跳过最像的前 10 个，它们很可能是"没标注出来的正例"（假负例）
   - relative_margin=0.05：负例的相似度必须比正例低 5% 以上，进一步过滤假负例
   - 更稳的做法：再传 cross_encoder=重排模型，用它的打分过滤
3. 训练：SentenceTransformerTrainer + losses.MultipleNegativesRankingLoss（InfoNCE：批内负例 + 难负例）
   - learning_rate 1e-5 ~ 2e-5，num_train_epochs 1 ~ 3，warmup_ratio 0.1，fp16=True（或 bf16）
   - batch 尽量大（对比学习里负例越多越好）；显存不够换 CachedMultipleNegativesRankingLoss
   - batch_sampler=BatchSamplers.NO_DUPLICATES：避免同一批出现重复文本变成假负例
     （from sentence_transformers.training_args import BatchSamplers）
   - evaluator：InformationRetrievalEvaluator 用 dev 划分，按 nDCG@10 选最好的 checkpoint
   - 训练时 anchor（查询）要和推理时一样加 BGE 查询指令（或者两边都不加），
     可以用 SentenceTransformerTrainingArguments(prompts={"anchor": 指令})
4. 保存到 output/bge-ft；把 configs 里 embedder.model 改成这个路径，然后
       python scripts/build_index.py --only dense && python scripts/evaluate.py      # 只看 test！
5. 写进简历的那张表：原始 base / 原始 large / 微调 base 的 test nDCG@10、Recall@10、编码速度。

注意数据泄漏：test 划分的查询绝不能进训练集；调参只看 dev。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_pairs(data_dir: str, split: str = "train"):
    """返回 datasets.Dataset，列为 anchor / positive。提示：rag.data 里的 load_split_ids / load_queries / load_qrels。"""
    raise NotImplementedError("TODO: 构造 (查询, 正例) 训练对")


def mine_negatives(pairs, model, corpus_texts: list[str], query_prompt: str):
    """调用 sentence_transformers.util.mine_hard_negatives（output_format="n-tuple"），返回带 negative_1..n 列的数据集。"""
    raise NotImplementedError("TODO: 挖难负例")


def train(model_name: str, train_ds, dev_evaluator, out_dir: str):
    """SentenceTransformerTrainer + MultipleNegativesRankingLoss，训练完保存到 out_dir。"""
    raise NotImplementedError("TODO: 训练")


def main() -> None:
    p = argparse.ArgumentParser(description="BGE 领域微调（待实现）")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="output/bge-ft")
    p.parse_args()
    raise SystemExit("finetune_embedder.py 还没实现：按文件开头的步骤完成 build_pairs / mine_negatives / train")


if __name__ == "__main__":
    main()
