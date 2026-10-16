# agent 评测 v2_think

- 数据：`D:\PersonalProject\rag-zh-starter\data\cmedqa`，划分 `dev`；大模型 `deepseek-flash`；写回答 thinking=None；agent 判断步骤 thinking=True，max_rounds=2，refine=hyde，max_irrelevant_retries=1
- baseline：检索一次取前 k 条直接回答（提示词允许部分回答和拒答）；agent：answer_with_check（资料判成 充分/部分/无关；部分就补检索再回答，无关才拒答）
- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)，见 rag/agent_eval.py

## 可回答（dev 普通问题）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 0.0% | 1.70 | 70.0% | 100.0% | 0.0% | 1.00 | 1.0 | 2322 | 12.0 |
| agent | 20 | 5.0% | 1.75 | 80.0% | 100.0% | 0.0% | 2.10 | 4.2 | 5867 | 31.5 |

再检索有没有起作用：触发补检索 60.0%（12 条）；其中判断变好 16.7%；这些题上 agent − baseline 的 helpful +0.00。
第一轮判断：部分 10、充分 8、无关 2；最后判断：部分 10、充分 9、无关 1

agent − baseline（配对检验）：

- helpful：+0.05（95% CI -0.15 ~ +0.25，p=1.0000，不显著；胜/负/平 3/2/15）
- 拒答率：+5.00（95% CI +0.00 ~ +15.00，p=1.0000，不显著；胜/负/平 1/0/19）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/20）（×100）

## 看似可答的知识库外问题

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 60.0% | 0.53 | 13.3% | 100.0% | 0.0% | 1.00 | 1.0 | 1314 | 6.3 |
| agent | 30 | 60.0% | 0.60 | 20.0% | 100.0% | 0.0% | 2.50 | 4.6 | 5270 | 34.2 |

再检索有没有起作用：触发补检索 90.0%（27 条）；其中判断变好 3.7%；这些题上 agent − baseline 的 helpful -0.07。
第一轮判断：部分 18、无关 9、充分 3；最后判断：部分 15、无关 11、充分 4

agent − baseline（配对检验）：

- helpful：+0.07（95% CI -0.13 ~ +0.30，p=0.7826，不显著；胜/负/平 3/3/24）
- 拒答率：+0.00（95% CI -16.67 ~ +16.67，p=1.0000，不显著；胜/负/平 3/3/24）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/30）（×100）

## 屏蔽相关文档

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 10 | 10.0% | 1.60 | 70.0% | 100.0% | 0.0% | 1.00 | 1.0 | 2634 | 13.8 |
| agent | 10 | 0.0% | 1.90 | 90.0% | 100.0% | 0.0% | 2.10 | 4.2 | 6649 | 36.6 |

再检索有没有起作用：触发补检索 60.0%（6 条）；其中判断变好 33.3%；这些题上 agent − baseline 的 helpful +0.50。
第一轮判断：部分 5、充分 4、无关 1；最后判断：充分 5、部分 5

agent − baseline（配对检验）：

- helpful：+0.30（95% CI +0.00 ~ +0.70，p=0.4927，不显著；胜/负/平 2/0/8）
- 拒答率：-10.00（95% CI -30.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/1/9）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/10）（×100）

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
  thinking: true
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