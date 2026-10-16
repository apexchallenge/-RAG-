# agent 评测 v3_2_claude

- 数据：`/mnt/user-data/uploads/PersonalProject/rag-zh-starter/data/cmedqa`，划分 `dev`；大模型 `Claude（claude-opus-5-5，人工会话代跑）`；写回答 thinking=None；agent 判断步骤 thinking=False，max_rounds=2，refine=hyde，merge=slots，new_slots=2，max_irrelevant_retries=1
- baseline：检索一次取前 k 条直接回答（提示词允许部分回答和拒答）；agent：answer_with_check（资料判成 充分/部分/部分（查不到）/无关；只有'部分'才补检索，新文档保留名额；无关才拒答）
- **离线评测**：大模型调用全部由 Claude（claude-opus-5-5，人工会话代跑） 完成（判断、补检索文字、写回答、评审），评审提示词不含系统名；同一提示词只回答一次
- 评审：大模型盲评 refused / helpful(0-2) / grounded(0/1)，见 rag/agent_eval.py

## 可回答（dev 普通问题）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 100 | 3.0% | 1.69 | 72.0% | 100.0% | 0.0% | 1.00 | 1.0 | 0 | 0.0 |
| agent | 100 | 5.0% | 1.70 | 75.0% | 100.0% | 0.0% | 1.38 | 2.7 | 0 | 0.0 |

再检索有没有起作用：触发补检索 25.0%（25 条）；其中资料真的变了 100.0%、判断变好 48.0%；这些题上 agent − baseline 的 helpful +0.04。
第一轮判断：充分 73、部分 22、无关 3、部分（查不到） 2；最后判断：充分 84、部分 9、无关 5、部分（查不到） 2
循环停止原因：充分 84、到最大轮数 9、补检索后没进展 3、无关 2、缺的查不到 2

agent − baseline（配对检验）：

- helpful：+0.01（95% CI -0.04 ~ +0.06，p=1.0000，不显著；胜/负/平 4/3/93）
- 拒答率：+2.00（95% CI +0.00 ~ +5.00，p=0.4951，不显著；胜/负/平 2/0/98）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/100）（×100）

## 知识库外问题（明显）

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 50 | 84.0% | 0.18 | 2.0% | 100.0% | 0.0% | 1.00 | 1.0 | 0 | 0.0 |
| agent | 50 | 100.0% | 0.00 | 0.0% | — | 0.0% | 2.00 | 3.0 | 0 | 0.0 |

再检索有没有起作用：触发补检索 100.0%（50 条）；其中资料真的变了 100.0%、判断变好 0.0%；这些题上 agent − baseline 的 helpful -0.18。
第一轮判断：无关 50；最后判断：无关 50
循环停止原因：无关 36、补检索后没进展 14

agent − baseline（配对检验）：

- helpful：-0.18（95% CI -0.30 ~ -0.08，p=0.0088，显著；胜/负/平 0/8/42）
- 拒答率：+16.00（95% CI +6.00 ~ +26.00，p=0.0088，显著；胜/负/平 8/0/42）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/50）（×100）

## 看似可答的知识库外问题

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 63.3% | 0.43 | 6.7% | 100.0% | 0.0% | 1.00 | 1.0 | 0 | 0.0 |
| agent | 30 | 56.7% | 0.53 | 10.0% | 100.0% | 0.0% | 2.33 | 4.1 | 0 | 0.0 |

再检索有没有起作用：触发补检索 96.7%（29 条）；其中资料真的变了 100.0%、判断变好 6.9%；这些题上 agent − baseline 的 helpful +0.10。
第一轮判断：无关 19、部分 10、充分 1；最后判断：无关 19、部分 10、充分 1
循环停止原因：无关 13、到最大轮数 8、补检索后没进展 8、充分 1

agent − baseline（配对检验）：

- helpful：+0.10（95% CI -0.03 ~ +0.23，p=0.3761，不显著；胜/负/平 4/1/25）
- 拒答率：-6.67（95% CI -20.00 ~ +6.67，p=0.6253，不显著；胜/负/平 1/3/26）（×100）
- 无依据回答率：+0.00（95% CI +0.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/0/30）（×100）

## 屏蔽相关文档

| 系统 | 条数 | 拒答率 | helpful 均值 | helpful=2 比例 | 回答有依据率 | 无依据回答率 | 检查轮数 | 大模型调用 | token | 耗时 P50(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 50 | 10.0% | 1.32 | 42.0% | 97.8% | 2.0% | 1.00 | 1.0 | 0 | 0.0 |
| agent | 50 | 12.0% | 1.36 | 48.0% | 100.0% | 0.0% | 1.76 | 3.5 | 0 | 0.0 |

再检索有没有起作用：触发补检索 54.0%（27 条）；其中资料真的变了 100.0%、判断变好 37.0%；这些题上 agent − baseline 的 helpful +0.07。
第一轮判断：充分 23、部分 21、无关 6；最后判断：充分 30、部分 15、无关 4、部分（查不到） 1
循环停止原因：充分 30、到最大轮数 10、补检索后没进展 6、无关 3、缺的查不到 1

agent − baseline（配对检验）：

- helpful：+0.04（95% CI -0.08 ~ +0.16，p=0.7489，不显著；胜/负/平 6/4/40）
- 拒答率：+2.00（95% CI -6.00 ~ +10.00，p=1.0000，不显著；胜/负/平 3/2/45）（×100）
- 无依据回答率：-2.00（95% CI -6.00 ~ +0.00，p=1.0000，不显著；胜/负/平 0/1/49）（×100）

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