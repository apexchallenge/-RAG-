"""进阶部分。每个文件开头写了思路和评测方法；单测在 tests/test_advanced.py、test_router.py、test_hierarchical.py、test_finetune.py。

    hierarchical.py   检索层级化：父子分块（small-to-big），评测 scripts/eval_hierarchical.py
    query_rewrite.py  reasoning：多查询改写 / HyDE / 问题分解 / 证据充分性检查（已实现，answer_with_check 即 agent 模式）
    router.py         分小模型：小模型做意图分类和路由，大模型只负责难题，评测 scripts/eval_router.py
    ../../scripts/finetune_embedder.py   engineering 层的 fine-tune：BGE 领域微调
"""
