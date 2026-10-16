# 项目交接笔记（给新对话 / 新会话先读）

> 新开对话时先说："先读项目根目录的 PROJECT_NOTES.md 和 README.md，再继续。"
> 本文件记录到 2026-10-15（下午）为止的状态、约定和待办；详细实验数据以 README 和 `results/` 为准。

## 1. 项目目标与使用者偏好

- 目标：做成可写进简历的"AI agent / RAG"项目，重点是**知道项目里真实存在哪些问题、每项技术为什么用、怎么验证**，不堆关键词。
- 交流用中文；结论要有数据支撑，问题要如实写（README 里有完整迭代记录）。
- 运行环境：Windows，用 **bash**（Git Bash）。本地模型已下载，跑检索类命令时加离线变量：
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ...`
- 大模型：配置里是 DeepSeek（`deepseek-flash`，默认开思考；北京时间工作日 9–12、14–18 是高峰价，其余半价）。
  **目前 API 没有额度**，需要大模型的评测改用 `scripts/offline_agent_eval.py` 离线跑（见第 4 节）。

## 2. 代码地图（只列关键文件）

| 文件 | 作用 |
| --- | --- |
| `rag/pipeline.py` | 检索链路：BM25 + BGE/Faiss → 加权 RRF → bge-reranker；`ask_with_check` 调 agent |
| `rag/generator.py` | 写回答（带引用；能答的部分照答、完全无关才拒答），`as_llm()` 给判断步骤用 |
| `rag/advanced/query_rewrite.py` | agent `answer_with_check`（v3.2 定版）+ multi_query / hyde / decompose（已实现未评测） |
| `rag/advanced/router.py` + `rag/router_eval.py` + `scripts/eval_router.py` | 大小模型路由（10-15 实现，**未跑数**）：小模型分 闲聊/简单/复杂/知识库外；`ask.py --router` |
| `rag/advanced/hierarchical.py` + `rag/hier_eval.py` + `scripts/eval_hierarchical.py` | 父子分块（10-15 实现，**未跑数**）：子块索引在 `indexes/cmedqa_pc1200_300_50/` |
| `scripts/finetune_embedder.py` | BGE 微调（10-15 实现，**未跑数**）：train 4.7k 对 + 难负例 + CachedMNRL；默认去掉 59 对正例是 dev/test 正例的训练对 |
| `scripts/bench_faiss.py` | Faiss 对比，10-15 起结果存 `results/faiss_*.md/.json`（含 CPU 信息和"召回≥95% 时最快设置"） |
| `rag/evaluation.py`、`rag/stats.py` | 检索消融、失败分类、融合调参、配对置换检验 + bootstrap CI |
| `rag/judge.py`、`scripts/relabel.py` | 大模型补标（候选池、打分缓存、合并标注、kappa） |
| `rag/agent_eval.py`、`scripts/eval_agent.py` | agent vs baseline 评测（四组问题、盲评、"再检索有没有起作用"） |
| `scripts/compare_agent_runs.py` | 两次评测（两个 tag）之间的配对比较 |
| `scripts/offline_agent_eval.py` | API 不可用时的离线评测：提示词缓存回放 + 外部模型补答 + 本机跑检索 |
| `eval_sets/` | 知识库外问题集；`*_test.jsonl` 是 test 留出集（迭代时没用过） |
| `configs/default.yaml` | `agent:` 段：max_rounds 2、refine hyde、merge slots、new_slots 2、max_irrelevant_retries 1、thinking false |

测试：`python -m unittest discover -s tests`（agent 相关在 `tests/test_agent.py`；新增 test_router / test_hierarchical / test_finetune / test_bench_faiss）。
10-15 的改动是在云端写的，云端装不上 jieba / bm25s / faiss，用临时替身跑过单测（新加的全部通过）；**回到本机先跑一遍全部单测**。

## 3. 关键结果（详见 README "二、实验结论"）

检索（dev，原始标注）：Dense nDCG@10 41.0；等权 RRF 30.8 → 调参后 41.1（融合丢失率 5.6% → 0.6%）；BM25 无增益。
补标（dev，1.1 万对，约 30 元）：Hit@10 62% → 85%（严格）；Dense+Rerank 相对 Dense 由 −1.72 变为 +0.60（严格，不显著）/ +2.33（宽松，p<0.001）；人工 kappa 0.49 / 0.57 / 0.67。
重排 top_n 50 → 20：重排延迟约 170ms → 70ms，dev 指标不降。

agent 迭代（详见 README "四、8. agent 评测 → 迭代记录"）：

| 版本 | 大模型 | 规模 | 要点 |
| --- | --- | --- | --- |
| v1 | DeepSeek | dev 20 | 过度拒答（60%），helpful 显著下降 |
| v2 | DeepSeek | dev 380 | 拒答修好；明显知识库外拒答 82% → 94%（显著）；补检索新文档挤不进资料 |
| v3 | DeepSeek | dev 380（复用 v2 baseline） | 保留名额生效、屏蔽组拒答 22% → 12%（显著）；但"部分（查不到）"挤掉"无关"，把关失效 |
| v3.1 | Claude 离线 | dev 230 | 把关找回（84% → 100%）；新问题：补检索后资料变差被硬拒答 |
| v3.2（定版） | Claude 离线（缓存重放） | dev 230 | 取判断最好的一轮资料；可回答 helpful 1.69 / 1.70 |
| v3.2 test | Claude 离线 | test 280 | 可回答 1.69 / 1.72（持平）；明显知识库外拒答 76% → 98%（p=0.002）；新写的"看似可答"组无区分度（两边约 95% 拒答） |

稳定结论：agent 不提升可回答问题质量（缺的多是知识库里没有的个人化细节）；补检索只在知识库更深处有相关文档时有用；
真正价值是"无关就拒答"的把关，其价值取决于写回答的模型有多谨慎；判断步骤开思考无收益、耗时约 2 倍；
评审的 grounded 维度饱和（查不出"资料不适用于用户情况"）。
注意：DeepSeek 批次和 Claude 离线批次之间只能比各自批次内 baseline vs agent，不能直接比绝对数字。

结果目录：`results/agent/{cmedqa_dev(v1), v2_nothink, v2_think, v3, v3_1_claude, v3_2_claude, test_v3_2}/`，`results/relabel/`。

## 4. 离线评测流程（API 没额度时）

```bash
# 云端/任意机器：跑到卡住，写出缺口（--seed-records 可以用之前在线评测的第一轮检索结果）
python scripts/offline_agent_eval.py step --tag T [--split test ...] [--seed-records results/agent/v3/records.jsonl]
python scripts/offline_agent_eval.py export --tag T     # 待回答提示词按类型打乱切批 -> results/agent/T/batches/*.in.txt
#   外部模型（例如子代理）逐条回答，写 batches/*.out.txt，格式：### KEY: <key>\n<回答>\n### END
python scripts/offline_agent_eval.py import --tag T
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/offline_agent_eval.py retrieve --tag T   # 有待检索时在本机跑
#   重复 step → export/import/retrieve，直到没有缺口；最后一次 step 写 records.jsonl / report.md
```

同一提示词只回答一次（按内容做缓存键）；只改 agent 收尾逻辑时，把 `llm_cache.jsonl` / `retrieval_cache.jsonl` 拷到新 tag 重放即可。
一轮 agent 评测通常需要本机跑 1~3 次 retrieve（第一轮检索 + 最多两轮补检索）。

## 5. 待办（按优先级）

1. **本机跑全部单测**：`python -m unittest discover -s tests -t . -v`（10-15 新代码没在真 jieba / faiss 上跑过）。
2. **检索 test 数字**：本机跑 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/evaluate.py`（默认 test），把 README 检索部分"test 待补"换成真实数字，简历里检索数字也换成 test。
3. **Faiss 索引对比**：代码已改为保存结果，直接跑 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/bench_faiss.py`，把 md 里的表和"最快设置"写进 README。
4. **跑三个进阶功能的数字**（建议顺序，前两项不花 API 钱）：
   - 父子分块：`eval_hierarchical.py stats` → `build` → `eval --split dev`；先看 stats 判断 CMedQA 上值不值得（多数文档只有一个父块的话，结论就是"短文档语料用不上"，也是可写的结论）
   - 路由分类：qwen3:4b 已装好、已调通（10-15 晚 `results/router/try2/`：3 题全部分对，但开着思考，每题 14~80 秒、输出 1000~3000 token，
     比直接用大模型还慢）→ 先 `eval_router.py probe` 找关思考的写法写进 `router.small` → 再 `eval_router.py classify`；
     误拦率高就先改 `CLASSIFY_PROMPT` / `router.kb_description` 再跑
   - 微调：`finetune_embedder.py --dry-run` → 训练 → `build_index --only dense --set embedder.model=output/bge-ft` → dev 上 evaluate；dev 有提升再 test 跑一次
   - 路由完整对比 `eval_router.py full` 需要大模型 API 额度（小模型本地免费）
5. **评审加"资料是否适用"维度**：用现有回答重新评审即可，不用重新生成。
6. "看似可答"的 test 留出集区分度不够（新药名在语料里完全搜不到），如要重做，应选语料里有沾边资料的主题。
7. MARL 不建议写进简历。

## 6. 当前简历条目（2026-10-15 版）

2026.09—2026.10｜中文医疗 RAG 问答与 Agent 评测系统｜大模型应用开发／算法评测（时间按实际改）
• 项目背景：针对中文医疗问答场景，基于 CMedQA（约 10 万篇文档）开发检索增强生成（RAG）系统，并建立可复现的评测流程，用来验证不同检索方法和 Agent 设计的实际效果。
• 检索系统：基于 jieba + BM25 关键词召回、BGE embedding 向量召回、Faiss 向量索引（支持 Flat / HNSW / IVF / IVF-PQ）构建混合检索，通过加权 RRF 融合与 bge-reranker 交叉编码器 rerank 排序；对召回、融合、重排、送入大模型各阶段的 topK 做消融调参，重排 top_n 从 50 降到 20 后重排延迟由约 170ms 降至约 70ms，验证集指标不降。
• Reasoning 与 Agent：设计 Agentic RAG 流程，由大模型对检索资料做"充分 / 部分 / 无关"推理判断，资料部分缺失时用 HyDE 生成口语化检索文本补检索，资料无关时拒答，并记录每轮判断依据以便追溯；经 5 轮迭代修复过度拒答、上下文合并、拒答把关失效等设计缺陷。
• 评测与问题诊断：设计检索失败分类，定位融合缺陷，nDCG@10 从 30.8 提升至 41.1；发现数据集标注不全，搭建"大模型补标 + 人工盲评"流程，补标 1.1 万对，修正了"rerank 使效果变差"的错误结论（Hit@10 由 62% 修正为 85%，rerank 增益 +2.3，p<0.001）。
• 项目成果：在 test 留出集上，Agent 将知识库外问题的拒答率从 76% 提升至 98%（p=0.002），同时可回答问题的质量与直接回答持平；配套实现配对显著性检验、断点续跑、成本统计和离线回放评测工具。

进阶三项跑出数字之前**不要写进简历**；跑完后按"做法 + 对比数字 + 结论（包括没提升的）"补一条。

面试要能讲清：agent 为什么不提升可回答问题质量；test 结果由另一大模型离线代跑、盲评但可能偏宽；把关价值取决于写回答模型的谨慎程度；grounded 评审维度的盲区。
