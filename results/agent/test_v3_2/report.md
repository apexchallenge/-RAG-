# agent 评测 test_v3_2

- 数据：`/mnt/user-data/uploads/PersonalProject/rag-zh-starter/data/cmedqa`，划分 `test`；大模型 `Claude（claude-opus-5-5，人工会话代跑）`；写回答 thinking=None；agent 判断步骤 thinking=False，max_rounds=2，refine=hyde，merge=slots，new_slots=2，max_irrelevant_retries=1
- baseline：检索一次取前 k 条直接回答（提示词允许部分回答和拒答）；agent：answer_with_check（资料判成 充分/部分/部分（查不到）/无关；只有'部分'才补检索，新文档保留名额；无关才拒答）
- **离线评测**：大模型调用全部由 Claude（claude-opus-5-5，人工会话代跑） 完成（判断、补检索文字、写回答、评审），评审提示词不含系统名；同一提示词只回答一次
- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)，见 rag/agent_eval.py

## 可回答（dev 普通问题）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 200 | 2.0% | 1.69 | 71.0% | 98.5% | 1.5% | 1.00 | 1.0 | 0 | 0.0 |
| agent | 200 | 4.5% | 1.72 | 76.0% | 99.0% | 1.0% | 1.35 | 2.7 | 0 | 0.0 |

再检索有没有起作用：触发补检索 24.5%（49 条）；其中资料真的变了 100.0%、判断变好 40.8%；这些题上 agent − baseline 的 helpful +0.10。
第一轮判断：充分 145、部分 42、无关 7、部分（查不到） 6；最后判断：充分 163、部分 24、无关 7、部分（查不到） 6
循环停止原因：充分 163、到最大轮数 18、补检索后没进展 9、缺的查不到 6、无关 4

agent − baseline（配对检验）：

- helpful：+0.03（95% CI -0.03 ~ +0.07，p=0.3973，不显著；胜/负/平 13/7/180）
- 拒答率：+2.50（95% CI +0.50 ~ +5.00，p=0.0562，不显著；胜/负/平 5/0/195）（×100）
- 无依据回答率：-0.50（95% CI -1.50 ~ +0.00，p=1.0000，不显著；胜/负/平 0/1/199）（×100）

## 知识库外问题（明显）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 50 | 76.0% | 0.24 | 0.0% | 91.7% | 2.0% | 1.00 | 1.0 | 0 | 0.0 |
| agent | 50 | 98.0% | 0.02 | 0.0% | 100.0% | 0.0% | 2.00 | 3.0 | 0 | 0.0 |

再检索有没有起作用：触发补检索 100.0%（50 条）；其中资料真的变了 100.0%、判断变好 2.0%；这些题上 agent − baseline 的 helpful -0.22。
第一轮判断：无关 50；最后判断：无关 49、部分（查不到） 1
循环停止原因：无关 32、补检索后没进展 17、缺的查不到 1

agent − baseline（配对检验）：

- helpful：-0.22（95% CI -0.34 ~ -0.12，p=0.0016，显著；胜/负/平 0/11/39）
- 拒答率：+22.00（95% CI +12.00 ~ +34.00，p=0.0016，显著；胜/负/平 11/0/39）（×100）
- 无依据回答率：-2.00（95% CI -6.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/1/49）（×100）

## 看似可答的知识库外问题

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 96.7% | 0.03 | 0.0% | 100.0% | 0.0% | 1.00 | 1.0 | 0 | 0.0 |
| agent | 30 | 93.3% | 0.07 | 0.0% | 100.0% | 0.0% | 2.10 | 3.3 | 0 | 0.0 |

再检索有没有起作用：触发补检索 100.0%（30 条）；其中资料真的变了 100.0%、判断变好 6.7%；这些题上 agent − baseline 的 helpful +0.03。
第一轮判断：无关 29、部分 1；最后判断：无关 29、部分 1
循环停止原因：无关 23、补检索后没进展 5、到最大轮数 2

agent − baseline（配对检验）：

- helpful：+0.03（95% CI +0.00 ~ +0.10，p=1.0000，不显著；胜/负/平 1/0/29）
- 拒答率：-3.33（95% CI -10.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/1/29）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/30）（×100）

怎么读：可回答组希望 helpful 不降、拒答率不明显升高；知识库外 / 屏蔽组希望拒答率高、无依据回答率低；看似可答组最看重无依据回答率（拿沾边资料答一部分、并说明缺什么，也算合格）。
再检索那一行：判断变好的比例低、helpful 差值接近 0，说明多花的调用没换来东西。
耗时含排队（检索和重排在 GPU 上串行），只适合同一次运行内两系统之间比较。

<details><summary>完整配置</summary>

```yaml
data_dir: /mnt/user-data/uploads/PersonalProject/rag-zh-starter/data/cmedqa
index_dir: /home/claude/work/rag-zh-starter/indexes/cmedqa
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
  model: Claude（claude-opus-5-5，人工会话代跑）
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