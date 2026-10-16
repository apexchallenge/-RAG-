# agent 评测 v2_nothink

- 数据：`D:\PersonalProject\rag-zh-starter\data\cmedqa`，划分 `dev`；大模型 `deepseek-flash`；写回答 thinking=None；agent 判断步骤 thinking=False，max_rounds=2，refine=hyde，max_irrelevant_retries=1
- baseline：检索一次取前 k 条直接回答（提示词允许部分回答和拒答）；agent：answer_with_check（资料判成 充分/部分/无关；部分就补检索再回答，无关才拒答）
- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)，见 rag/agent_eval.py

## 可回答（dev 普通问题）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 200 | 4.0% | 1.72 | 75.5% | 100.0% | 0.0% | 1.00 | 1.0 | 2296 | 11.8 |
| agent | 200 | 6.0% | 1.70 | 75.5% | 100.0% | 0.0% | 1.74 | 3.4 | 3390 | 15.0 |

再检索有没有起作用：触发补检索 41.0%（82 条）；其中判断变好 19.5%；这些题上 agent − baseline 的 helpful -0.06。
第一轮判断：充分 118、部分 75、无关 7；最后判断：充分 134、部分 58、无关 8

agent − baseline（配对检验）：

- helpful：-0.02（95% CI -0.08 ~ +0.04，p=0.6063，不显著；胜/负/平 13/15/172）
- 拒答率：+2.00（95% CI -1.50 ~ +5.50，p=0.3967，不显著；胜/负/平 8/4/188）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/200）（×100）

## 知识库外问题（明显）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 50 | 82.0% | 0.28 | 10.0% | 100.0% | 0.0% | 1.00 | 1.0 | 1397 | 4.6 |
| agent | 50 | 94.0% | 0.10 | 4.0% | 100.0% | 0.0% | 2.18 | 3.5 | 2413 | 6.9 |

再检索有没有起作用：触发补检索 98.0%（49 条）；其中判断变好 6.1%；这些题上 agent − baseline 的 helpful -0.18。
第一轮判断：无关 42、部分 7、充分 1；最后判断：无关 41、部分 8、充分 1

agent − baseline（配对检验）：

- helpful：-0.18（95% CI -0.34 ~ -0.06，p=0.0258，显著；胜/负/平 0/6/44）
- 拒答率：+12.00（95% CI +4.00 ~ +22.00，p=0.0258，显著；胜/负/平 6/0/44）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/50）（×100）

## 看似可答的知识库外问题

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 60.0% | 0.53 | 13.3% | 100.0% | 0.0% | 1.00 | 1.0 | 1297 | 5.5 |
| agent | 30 | 63.3% | 0.50 | 13.3% | 100.0% | 0.0% | 2.43 | 4.5 | 2421 | 12.8 |

再检索有没有起作用：触发补检索 90.0%（27 条）；其中判断变好 0.0%；这些题上 agent − baseline 的 helpful -0.07。
第一轮判断：部分 16、无关 11、充分 3；最后判断：部分 15、无关 12、充分 3

agent − baseline（配对检验）：

- helpful：-0.03（95% CI -0.13 ~ +0.07，p=1.0000，不显著；胜/负/平 1/2/27）
- 拒答率：+3.33（95% CI +0.00 ~ +10.00，p=1.0000，不显著；胜/负/平 1/0/29）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/30）（×100）

## 屏蔽相关文档

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 100 | 22.0% | 1.26 | 48.0% | 100.0% | 0.0% | 1.00 | 1.0 | 2180 | 10.9 |
| agent | 100 | 20.0% | 1.32 | 52.0% | 100.0% | 0.0% | 2.18 | 4.2 | 3559 | 16.4 |

再检索有没有起作用：触发补检索 66.0%（66 条）；其中判断变好 18.2%；这些题上 agent − baseline 的 helpful +0.11。
第一轮判断：部分 49、充分 34、无关 17；最后判断：部分 45、充分 42、无关 13

agent − baseline（配对检验）：

- helpful：+0.06（95% CI -0.05 ~ +0.17，p=0.3685，不显著；胜/负/平 14/10/76）
- 拒答率：-2.00（95% CI -9.00 ~ +5.00，p=0.7904，不显著；胜/负/平 6/8/86）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/100）（×100）

怎么读：可回答组希望 helpful 不降、拒答率不明显升高；知识库外 / 屏蔽组希望拒答率高、无依据回答率低；看似可答组最看重无依据回答率（拿沾边资料答一部分、并说明缺什么，也算合格）。
再检索那一行：判断变好的比例低、helpful 差值接近 0，说明多花的调用没换来东西。
耗时含排队（检索和重排在 GPU 上串行），只适合同一次运行内两系统之间比较。

<details><summary>完整配置</summary>

```yaml
data_dir: D:\PersonalProject\rag-zh-starter\data\cmedqa
index_dir: D:\PersonalProject\rag-zh-starter\indexes\cmedqa
bm25:
  jieba_mode: search
  user_dict: null
  stopwords: null
  k1: 1.2
  b: 0.75
embedder:
  model: BAAI/bge-base-zh-v1.5
  query_instruction: 为这个句子生成表示以用于检索相关文章：
  query_prompt_name: null
  max_seq_length: 512
  batch_size: 128
  fp16: true
  device: null
faiss:
  index_type: flat
  hnsw_m: 32
  hnsw_ef_construction: 200
  hnsw_ef_search: 128
  ivf_nlist: 1024
  ivf_nprobe: 32
  pq_m: 64
  pq_nbits: 8
retrieval:
  bm25_top_k: 100
  dense_top_k: 100
  fusion: rrf
  rrf_k: 10
  rrf_use_weights: true
  weights:
    bm25: 0.1
    dense: 0.9
  fused_top_k: 100
reranker:
  enabled: true
  model: BAAI/bge-reranker-v2-m3
  top_n: 50
  max_length: 512
  batch_size: 64
  fp16: true
  device: null
generation:
  base_url: https://api.deepseek.com
  model: deepseek-flash
  api_key_env: LLM_API_KEY
  context_top_k: 5
  max_chars_per_doc: 600
  temperature: 0.2
  thinking: null
agent:
  max_rounds: 2
  refine: hyde
  max_irrelevant_retries: 1
  thinking: false
  max_tokens: 1024
eval:
  split: test
  max_queries: null
  recall_ks:
  - 5
  - 10
  - 20
  - 100
  seed: 42

```
</details>