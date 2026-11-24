"""自造图像分类数据：12x12 单通道图上的三类几何图形。

类别：0=实心方块，1=横线，2=竖线。叠加小幅平移、高斯噪声，保证线性不可分
但小 CNN 能稳定学出来。所有随机量走显式种子，保证三组对照实验用同一批数据。
"""

from __future__ import annotations

import numpy as np

IMAGE = 12
N_CLASSES = 3


def _draw_square(rng, img):
    s = rng.integers(3, 6)          # 边长 3~5
    r0 = int(rng.integers(0, IMAGE - s))
    c0 = int(rng.integers(0, IMAGE - s))
    img[r0:r0 + s, c0:c0 + s] += 1.0


def _draw_hline(rng, img):
    s = rng.integers(4, 8)
    r0 = int(rng.integers(1, IMAGE - 1))
    c0 = int(rng.integers(0, IMAGE - s))
    img[r0, c0:c0 + s] += 1.0


def _draw_vline(rng, img):
    s = rng.integers(4, 8)
    r0 = int(rng.integers(0, IMAGE - s))
    c0 = int(rng.integers(1, IMAGE - 1))
    img[r0:r0 + s, c0] += 1.0


_DRAW = [_draw_square, _draw_hline, _draw_vline]


def make_dataset(n_per_class: int = 300, seed: int = 42,
                 noise: float = 0.12):
    rng = np.random.default_rng(seed)
    n = n_per_class * N_CLASSES
    x = np.zeros((n, 1, IMAGE, IMAGE), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for cls in range(N_CLASSES):
        for i in range(n_per_class):
            idx = cls * n_per_class + i
            _DRAW[cls](rng, x[idx, 0])
            y[idx] = cls
    x += rng.normal(0.0, noise, size=x.shape).astype(np.float32)
    # 固定归一化（全数据集统计量，避免泄露到 float16 动态范围之外）
    x = (x - x.mean()) / (x.std() + 1e-8)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def iter_minibatches(x, y, batch_size, rng, shuffle=True):
    idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for start in range(0, len(x) - batch_size + 1, batch_size):
        sel = idx[start:start + batch_size]
        yield x[sel], y[sel]


def make_overflow_batch(batch_size: int = 48, magnitude: float = 1e6):
    """构造让半精度反向必然溢出的批次。

    输入幅度远超 float16 上界 65504：conv 线性变换后的激活直接变 inf，
    反向梯度随之出现 inf。正常数据永远不会长这样，只用于验收测试和网页演示。
    """
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=(batch_size, 1, IMAGE, IMAGE)).astype(np.float32)
    x *= np.float32(magnitude)
    y = rng.integers(0, N_CLASSES, size=batch_size).astype(np.int64)
    return x, y
