"""显著性检验：两个方法在同一批查询上的指标差，是真提升还是随机波动？

为什么要做：bge-base 和 bge-large 的 nDCG@10 只差 0.9 个点，1000 条查询上这种差距很可能是噪声。
做法（IR 领域常用，见 Smucker et al. 2007）：
  - 配对随机化检验：如果两个方法没有差别，那每条查询上"谁是 A 谁是 B"可以随便交换。
    随机翻转每条查询差值的正负号很多次，看有多大比例的随机结果比实际差距还极端 -> p 值。
  - 配对 bootstrap：有放回地重抽查询，得到平均差值的 95% 置信区间。
p < 0.05 且置信区间不跨 0，才说"显著提升"。同时比较很多组时注意多重比较（可以用更严的阈值）。
"""

from __future__ import annotations

import numpy as np


def paired_test(
    base: dict[str, float],
    other: dict[str, float],
    n_resamples: int = 10000,
    seed: int = 42,
    chunk: int = 1000,
) -> dict:
    """base / other: {query_id: 指标值}，只比较两边都有的查询。返回 other - base 的平均差、95% CI、p 值、胜负数。"""
    qs = sorted(set(base) & set(other))
    if not qs:
        raise ValueError("两组结果没有共同的查询，无法做配对检验")
    x = np.array([other[q] - base[q] for q in qs], dtype=np.float64)
    delta = float(x.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    boots: list[np.ndarray] = []
    done = 0
    while done < n_resamples:  # 分块算，1000 条查询 x 10000 次也只占几十 MB 内存
        m = min(chunk, n_resamples - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, len(x)))
        extreme += int(np.sum(np.abs((signs * x).mean(axis=1)) >= abs(delta) - 1e-12))
        idx = rng.integers(0, len(x), size=(m, len(x)))
        boots.append(x[idx].mean(axis=1))
        done += m
    lo, hi = np.percentile(np.concatenate(boots), [2.5, 97.5])
    return {
        "n": len(qs),
        "delta": delta,
        "ci95": [float(lo), float(hi)],
        "p": (extreme + 1) / (n_resamples + 1),  # +1：把实际观测本身算进去，p 值不会是 0
        "wins": int((x > 0).sum()),
        "losses": int((x < 0).sum()),
        "ties": int((x == 0).sum()),
    }


def format_test(t: dict, scale: float = 100.0) -> str:
    sig = "显著" if t["p"] < 0.05 and (t["ci95"][0] > 0 or t["ci95"][1] < 0) else "不显著"
    return (f"{scale * t['delta']:+.2f}（95% CI {scale * t['ci95'][0]:+.2f} ~ {scale * t['ci95'][1]:+.2f}，"
            f"p={t['p']:.4f}，{sig}；胜/负/平 {t['wins']}/{t['losses']}/{t['ties']}）")
