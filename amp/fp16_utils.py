"""半精度模拟的底层工具。

关键约定：
- 模型主权重、优化器状态一律 float32；每步前向/反向前把权重就近转成 float16。
- 矩阵乘模拟张量核：输入先转 float16，乘积累加在 float32 里做，结果再转回 float16。
- 支持两种舍入模式：near-else-even（numpy 默认就近偶舍入）和 stochastic（随机舍入）。
- 随机舍入按元素的真实值在相邻两个可表示数之间按距离成比例取整，用独立的随机种子，
  这样实验可复现。
"""

from __future__ import annotations

import numpy as np

# float16 的关键边界（6.10e-5 是最小正规格化数，5.96e-8 是最小正次规格化数）
FP16_MAX = np.finfo(np.float16).max          # 65504.0
FP16_MIN_NORMAL = np.float16(2.0 ** -14)     # 6.1035e-05
FP16_MIN_SUBNORMAL = np.float16(2.0 ** -24)  # 5.9605e-08（float16 最小的正数）

_RNG_DTYPE = np.float32


class CastContext:
    """舍入模式 + 随机数发生器的上下文。

    mode == "near"  ：就近偶舍入（numpy astype 的默认行为）。
    mode == "stoch" ：随机舍入，rng 决定每个元素向上还是向下取整。
    """

    def __init__(self, mode: str = "near", seed: int | None = None):
        assert mode in ("near", "stoch")
        self.mode = mode
        # 独立 RNG，和数据打乱用的 RNG 分开，保证两条训练路径可逐元素复现
        self.rng = np.random.default_rng(seed)


def stochastic_cast(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """把 float32 数组随机舍入成 float16。

    对每个 x，取相邻的两个 float16 可表示数 lo <= x < hi，以
    P(取 hi) = (x - lo) / (hi - lo) 的概率向上取整，否则向下取整。
    期望无偏：E[round(x)] = x，这正是随机舍入能"攒住"小更新的原因。
    超出 float16 范围的值按就近规则溢出成 inf；非有限值原样保留。
    """
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore"):
        r = x.astype(np.float16)  # 先拿就近偶舍入的结果（同时处理溢出 -> inf）

    finite = np.isfinite(x)
    too_big = finite & (np.abs(x) > FP16_MAX)
    # 溢出的元素 astype 已经给了 +/-inf，直接保持；非有限值原样保留
    work_mask = finite & ~too_big
    if not np.any(work_mask):
        return r

    xw = x[work_mask]
    rw = r[work_mask]
    rf = rw.astype(np.float32)
    down = np.nextafter(rw, np.float16(-np.inf)).astype(np.float32)
    up = np.nextafter(rw, np.float16(np.inf)).astype(np.float32)
    # 以真实值 x 锚定相邻可表示数：x 恰好等于 r 时 lo==hi==r（gap=0，保持不动）
    lo = np.where(rf <= xw, rf, down)
    hi = np.where(rf >= xw, rf, up)

    # x 恰好等于某个可表示数时 lo == hi（gap 为 0），无需掷骰子
    gap = hi - lo
    roll = rng.random(size=xw.shape, dtype=_RNG_DTYPE)
    rand_up = gap > 0
    take_hi = np.zeros_like(xw, dtype=bool)
    take_hi[rand_up] = roll[rand_up] * gap[rand_up] < (xw[rand_up] - lo[rand_up])

    res = np.where(take_hi, hi, lo).astype(np.float16)
    r[work_mask] = res
    return r


def cast_fp16(x: np.ndarray, ctx: CastContext) -> np.ndarray:
    """按上下文指定的舍入模式把数组转成 float16。"""
    x = np.asarray(x)
    if x.dtype == np.float16:
        return x
    if ctx.mode == "near":
        return x.astype(np.float16)
    return stochastic_cast(x, ctx.rng)


def cast_fp32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def hp_matmul(a: np.ndarray, b: np.ndarray, half: bool, ctx: CastContext) -> np.ndarray:
    """模拟半精度矩阵乘。

    half=True 时两个操作数先按 ctx 的舍入模式转 float16（模拟张量核的半精度输入），
    乘积累加在 float32 中完成（避免 numpy 部分平台 fp16 dot 的中间精度差异），
    输出再按 ctx 舍入回 float16。
    half=False 时全程 float32，作为全单精度对照路径。
    """
    if half:
        a16 = cast_fp16(a, ctx)
        b16 = cast_fp16(b, ctx)
        acc = np.dot(a16.astype(np.float32), b16.astype(np.float32))
        return cast_fp16(acc, ctx)
    return np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
