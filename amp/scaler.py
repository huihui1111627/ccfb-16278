"""动态损失缩放器（dynamic loss scaling）。

规则（对应需求 2、3、8）：
- 反向开始前把损失乘 scale，梯度算完再除回去；
- 任何梯度出现 inf/nan（或前向激活溢出）→ 整步跳过，scale 减半；
- 连续 growth_interval 步没溢出 → scale 翻倍；
- scale 限制在 [min_scale, max_scale]；
- fix=True 时完全冻结 scale，用于网页上手动固定对照实验。
"""

from __future__ import annotations


class DynamicLossScaler:
    def __init__(
        self,
        init_scale: float = 128.0,
        min_scale: float = 1.0,
        max_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 32,
        fixed: bool = False,
    ):
        self.scale = float(init_scale)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.fixed = bool(fixed)
        self.good_steps = 0
        # 累计统计，供网页展示
        self.total_overflow = 0
        self.total_growth = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grad(self, g):
        """把缩放后的梯度除回真实尺度。"""
        return g / self.scale

    def update_after_step(self, overflow: bool) -> str:
        """根据本步是否溢出更新 scale。

        返回本步发生的事件："overflow" / "growth" / "hold"。
        fixed 模式下只统计不调整。
        """
        if overflow:
            self.total_overflow += 1
            self.good_steps = 0
            if not self.fixed:
                self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            return "overflow"

        self.good_steps += 1
        if not self.fixed and self.good_steps >= self.growth_interval:
            new_scale = min(self.max_scale, self.scale * self.growth_factor)
            if new_scale != self.scale:
                self.scale = new_scale
                self.total_growth += 1
            self.good_steps = 0
            return "growth"
        return "hold"

    def snapshot(self) -> dict:
        return {
            "scale": self.scale,
            "fixed": self.fixed,
            "good_steps": self.good_steps,
            "total_overflow": self.total_overflow,
            "total_growth": self.total_growth,
        }
