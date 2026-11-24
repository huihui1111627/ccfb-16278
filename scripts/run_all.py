"""跑三组固定种子对照实验 + 一个固定 scale 演示。

1. full_fp32   ：全程单精度，基准收敛曲线；
2. mixed_near  ：混合精度 + 就近偶舍入 + 动态损失缩放；
3. mixed_stoch ：混合精度 + 随机舍入 + 动态损失缩放；
4. fixed_scale ：混合精度 + 网页手动固定 scale（演示不做动态调整会怎样）。

同种子 => 模型初始化、数据顺序逐批完全一致，只差精度路径。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.data import make_dataset, make_overflow_batch
from amp.trainer import TrainConfig, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")

STEPS = int(os.environ.get("AMP_STEPS", "300"))


def main():
    x, y = make_dataset(n_per_class=300, seed=42)
    n_val = 240
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]
    overflow_batch = make_overflow_batch(batch_size=48, magnitude=1e6)
    # 训练中段注入一次必然溢出的批次，演示跳步 + scale 减半 + 参数不动
    inject = (120,)

    base = dict(
        steps=STEPS, batch_size=48, lr=0.05, momentum=0.9,
        weight_decay=1e-4, seed=2026, eval_every=10,
        inject_overflow_at=inject,
    )
    scale_common = dict(
        init_scale=128.0, min_scale=1.0, max_scale=65536.0,
        growth_interval=32,
    )

    runs = [
        TrainConfig(run_name="full_fp32", out_dir=os.path.join(RUNS, "full_fp32"),
                    half=False, rounding="near", **base, **scale_common),
        TrainConfig(run_name="mixed_near", out_dir=os.path.join(RUNS, "mixed_near"),
                    half=True, rounding="near", **base, **scale_common),
        TrainConfig(run_name="mixed_stoch",
                    out_dir=os.path.join(RUNS, "mixed_stoch"),
                    half=True, rounding="stoch", **base, **scale_common),
        TrainConfig(run_name="fixed_scale",
                    out_dir=os.path.join(RUNS, "fixed_scale"),
                    half=True, rounding="near", fixed_scale=True,
                    **base, **scale_common),
        # 故意固定在 scale=1（不做任何损失缩放）：下溢显著恶化，供网页对照
        TrainConfig(run_name="fixed_low_scale",
                    out_dir=os.path.join(RUNS, "fixed_low_scale"),
                    half=True, rounding="near", fixed_scale=True,
                    init_scale=1.0, min_scale=1.0, max_scale=1.0,
                    growth_interval=10 ** 9, **base),
    ]

    results = {}
    for cfg in runs:
        print(f"\n===== {cfg.run_name} ({'fp16' if cfg.half else 'fp32'}, "
              f"rounding={cfg.rounding}, fixed={cfg.fixed_scale}) =====")
        out = train(cfg, x_train, y_train, x_val, y_val,
                    overflow_batch=overflow_batch)
        results[cfg.run_name] = out["summary"]
        print("summary:", out["summary"])

    print("\n===== 对照汇总 =====")
    for name, s in results.items():
        print(f"{name:14s} final_loss={s['final_loss']:.4f} "
              f"acc32={s['final_eval']['acc_master_fp32']:.3f} "
              f"skipped={s['n_skipped']} final_scale={s['final_scale']:.1f}")


if __name__ == "__main__":
    main()
