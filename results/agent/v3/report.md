# agent 评测 v3

- 数据：`D:\PersonalProject\rag-zh-starter\data\cmedqa`，划分 `dev`；大模型 `deepseek-flash`；写回答 thinking=None；agent 判断步骤 thinking=False，max_rounds=2，refine=hyde，merge=slots，new_slots=2，max_irrelevant_retries=1
- baseline：检索一次取前 k 条直接回答（提示词允许部分回答和拒答）；agent：answer_with_check（资料判成 充分/部分/部分（查不到）/无关；只有'部分'才补检索，新文档保留名额；无关才拒答）
- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)，见 rag/agent_eval.py

## 可回答（dev 普通问题）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 200 | 4.0% | 1.72 | 75.5% | 100.0% | 0.0% | 1.00 | 1.0 | 2296 | 11.8 |
| agent | 200 | 5.0% | 1.74 | 78.5% | 100.0% | 0.0% | 2.10 | 4.2 | 4118 | 20.0 |

再检索有没有起作用：触发补检索 59.5%（119 条）；其中资料真的变了 100.0%、判断变好 13.4%；这些题上 agent − baseline 的 helpful +0.02。
第一轮判断：部分 118、充分 57、部分（查不到） 24、无关 1；最后判断：部分 95、充分 73、部分（查不到） 31、无关 1
循环停止原因：到最大轮数 95、充分 73、缺的查不到 31、无关 1

agent − baseline（配对检验）：

- helpful：+0.02（95% CI -0.04 ~ +0.09，p=0.6465，不显著；胜/负/平 20/16/164）
- 拒答率：+1.00（95% CI -1.50 ~ +3.50，p=0.6999，不显著；胜/负/平 4/2/194）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/200）（×100）

## 知识库外问题（明显）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 50 | 82.0% | 0.28 | 10.0% | 100.0% | 0.0% | 1.00 | 1.0 | 1397 | 4.6 |
| agent | 50 | 86.0% | 0.20 | 6.0% | 100.0% | 0.0% | 1.20 | 2.3 | 2459 | 7.1 |

再检索有没有起作用：触发补检索 18.0%（9 条）；其中资料真的变了 100.0%、判断变好 11.1%；这些题上 agent − baseline 的 helpful -0.11。
第一轮判断：部分（查不到） 40、无关 5、部分 4、充分 1；最后判断：部分（查不到） 44、无关 4、部分 1、充分 1
循环停止原因：缺的查不到 44、无关 4、到最大轮数 1、充分 1

agent − baseline（配对检验）：

- helpful：-0.08（95% CI -0.20 ~ +0.02，p=0.2981，不显著；胜/负/平 1/4/45）
- 拒答率：+4.00（95% CI -4.00 ~ +12.00，p=0.6099，不显著；胜/负/平 3/1/46）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/50）（×100）

## 看似可答的知识库外问题

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 60.0% | 0.53 | 13.3% | 100.0% | 0.0% | 1.00 | 1.0 | 1297 | 5.5 |
| agent | 30 | 63.3% | 0.57 | 20.0% | 100.0% | 0.0% | 2.27 | 4.4 | 3055 | 17.7 |

再检索有没有起作用：触发补检索 70.0%（21 条）；其中资料真的变了 100.0%、判断变好 0.0%；这些题上 agent − baseline 的 helpful +0.05。
第一轮判断：部分 19、部分（查不到） 9、无关 2；最后判断：部分 15、部分（查不到） 12、无关 3
循环停止原因：到最大轮数 16、缺的查不到 12、无关 2

agent − baseline（配对检验）：

- helpful：+0.03（95% CI -0.13 ~ +0.20，p=1.0000，不显著；胜/负/平 4/3/23）
- 拒答率：+3.33（95% CI -6.67 ~ +13.33，p=1.0000，不显著；胜/负/平 2/1/27）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/30）（×100）

## 屏蔽相关文档

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 100 | 22.0% | 1.26 | 48.0% | 100.0% | 0.0% | 1.00 | 1.0 | 2180 | 10.9 |
| agent | 100 | 12.0% | 1.38 | 50.0% | 100.0% | 0.0% | 2.44 | 4.9 | 4354 | 21.7 |

再检索有没有起作用：触发补检索 76.0%（76 条）；其中资料真的变了 100.0%、判断变好 13.2%；这些题上 agent − baseline 的 helpful +0.17。
第一轮判断：部分 75、部分（查不到） 16、充分 8、无关 1；最后判断：部分 63、部分（查不到） 19、充分 17、无关 1
循环停止原因：到最大轮数 64、缺的查不到 19、充分 17

agent − baseline（配对检验）：

- helpful：+0.12（95% CI -0.01 ~ +0.25，p=0.1036，不显著；胜/负/平 24/13/63）
- 拒答率：-10.00（95% CI -18.00 ~ -2.00，p=0.0340，显著；胜/负/平 4/14/82）（×100）
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
  merge: slots
  new_slots: 2
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