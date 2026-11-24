"""SGD + momentum，全部状态 float32（需求 5、6）。

- 主权重 W32 永远 float32：即便单次更新量小到在 float16 里会被舍掉，
  float32 主副本也能逐次累加不丢失；
- 动量缓冲 v 是 float32；
- 权重衰减在 float32 里直接作用于主权重；
- 每步训练前由 model.sync_weights 从主副本重新转 float16 工作副本。
"""

from __future__ import annotations

import numpy as np


class SGDMomentum:
    def __init__(self, master_params: dict, lr: float, momentum: float = 0.9,
                 weight_decay: float = 1e-4):
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.wd = float(weight_decay)
        self.master = master_params
        # 动量状态按主参数同形 float32 初始化
        self.velocity = {
            lname: {k: np.zeros_like(p, dtype=np.float32)
                    for k, p in plist.items()}
            for lname, plist in master_params.items()
        }

    def step(self, grads32: dict):
        """grads32: {layer_name: {param_name: fp32 已还原尺度的梯度}}。"""
        stats = {"updates_eaten_in_fp16": 0, "updates_total": 0}
        for lname, plist in self.master.items():
            g_list = grads32[lname]
            for k, w in plist.items():
                g = g_list[k].astype(np.float32)
                # 兜底防护：非有限梯度绝不允许污染主副本（正常流程已在 trainer 跳过）
                if not np.isfinite(g).all():
                    continue
                # 权重衰减（L2），float32
                if self.wd:
                    g = g + self.wd * w
                # 动量更新，float32
                v = self.velocity[lname][k]
                v[:] = self.momentum * v - self.lr * g
                # 统计：这个更新量若直接加到 float16 权重上，有多少元素会被舍成 0
                delta16 = v.astype(np.float16)
                eaten = int(np.count_nonzero(v) - np.count_nonzero(delta16))
                stats["updates_eaten_in_fp16"] += eaten
                stats["updates_total"] += int(v.size)
                # 主副本累加，小更新不丢
                w += v
        return stats
