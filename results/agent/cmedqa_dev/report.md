# agent 评测 cmedqa_dev

- 数据：`D:\PersonalProject\rag-zh-starter\data\cmedqa`，划分 `dev`；大模型 `deepseek-flash`；写回答 thinking=None；agent 判断步骤 thinking=False，max_rounds=2
- baseline：检索一次取前 k 条直接回答（提示词允许拒答）；agent：answer_with_check（不足就改写查询再检索，仍不足就拒答）
- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)，见 rag/agent_eval.py

## 可回答（dev 普通问题）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 20.0% | 1.55 | 75.0% | 100.0% | 0.0% | 1.00 | 1.0 | 2113 | 10.5 |
| agent | 20 | 60.0% | 0.80 | 40.0% | 100.0% | 0.0% | 2.25 | 3.9 | 2261 | 11.3 |

agent − baseline（配对检验）：

- helpful：-0.75（95% CI -1.15 ~ -0.35，p=0.0074，显著；胜/负/平 0/8/12）
- 拒答率：+40.00（95% CI +20.00 ~ +60.00，p=0.0074，显著；胜/负/平 8/0/12）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/20）（×100）

## 知识库外问题

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 10 | 100.0% | 0.00 | 0.0% | — | 0.0% | 1.00 | 1.0 | 985 | 3.6 |
| agent | 10 | 100.0% | 0.00 | 0.0% | — | 0.0% | 3.00 | 5.0 | 2137 | 9.3 |

agent − baseline（配对检验）：

- helpful：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/10）
- 拒答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/10）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/10）（×100）

## 屏蔽相关文档

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 10 | 40.0% | 1.20 | 60.0% | 100.0% | 0.0% | 1.00 | 1.0 | 2064 | 12.0 |
| agent | 10 | 80.0% | 0.40 | 20.0% | 100.0% | 0.0% | 2.70 | 4.6 | 2105 | 10.4 |

agent − baseline（配对检验）：

- helpful：-0.80（95% CI -1.60 ~ +0.00，p=0.2204，不显著；胜/负/平 1/5/4）
- 拒答率：+40.00（95% CI +0.00 ~ +80.00，p=0.2204，不显著；胜/负/平 5/1/4）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/10）（×100）

怎么读：可回答组希望 helpful 不降、拒答率不明显升高；知识库外 / 屏蔽组希望拒答率高、无依据回答率低。
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