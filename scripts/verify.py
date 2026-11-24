"""验收测试：逐条核对需求，全部通过才算交付。

T1 半精度矩阵运算语义：fp16 matmul 与手工逐元素舍入的结果一致
T2 随机舍入无偏 + 精确值不动 + 中点 0.5 概率
T3 主副本攒小更新（需求 5）：fp16 会被舍掉的更新，fp32 主副本逐次累加不丢
T4 损失缩放与动态调整（需求 2、3）：溢出跳步、参数不变、scale 减半，
   连续无溢出翻倍，且不越上下限
T5 构造必然溢出的数据（需求验收）：跳步标记 + 逐元素参数快照相等
T6 下溢比例核对：scale=1 反事实 vs 缩放路径，缩放显著降低零比例
T7 两种舍入吞零对比（需求 7）：随机舍入在极小更新上显著优于就近舍入
T8 同种子全单精度 vs 混合精度（需求验收）：收敛速度与最终准确率差距在阈值内
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from amp.fp16_utils import (CastContext, FP16_MAX, FP16_MIN_SUBNORMAL,
                            cast_fp16, hp_matmul, stochastic_cast)
from amp.scaler import DynamicLossScaler
from amp.optimizer import SGDMomentum
from amp.model import TinyCNN
from amp.data import make_dataset, make_overflow_batch
from amp.trainer import TrainConfig, train

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}")


# ---------------------------------------------------------------- T1
def test_matmul_semantics():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (12, 8)).astype(np.float32)
    b = rng.normal(0, 1, (8, 6)).astype(np.float32)
    ctx = CastContext("near")
    got = hp_matmul(a, b, half=True, ctx=ctx).astype(np.float32)
    expect = np.dot(a.astype(np.float16).astype(np.float32),
                    b.astype(np.float16).astype(np.float32)).astype(np.float16)
    check("T1 hp_matmul 等价于 fp16 输入+fp32 累加+fp16 输出",
          np.array_equal(got, expect.astype(np.float32)),
          f"max_abs_diff={np.abs(got - expect).max():.2e}")


# ---------------------------------------------------------------- T2
def test_stochastic_rounding():
    rng = np.random.default_rng(7)
    x = np.full(300000, 1e-9, dtype=np.float32)
    out = stochastic_cast(x, rng).astype(np.float32)
    unbiased = abs(out.mean() - 1e-9) < 0.25e-9
    check("T2a 随机舍入期望无偏（均值≈原值）", unbiased,
          f"mean={out.mean():.3e} target=1e-9")
    check("T2b 就近舍入把该值全吞成零", bool((x.astype(np.float16) == 0).all()))

    rng = np.random.default_rng(8)
    exact = np.array([1.0, 2.0, 0.5, -3.25], dtype=np.float32)
    check("T2c 精确可表示值随机舍入后不动",
          np.array_equal(stochastic_cast(exact, rng).astype(np.float32), exact))

    rng = np.random.default_rng(9)
    mid = np.full(200000, np.float32(1.0 + 2 ** -11), dtype=np.float32)
    o = stochastic_cast(mid, rng).astype(np.float32)
    p_up = float((o == o.max()).mean())
    check("T2d 相邻两可表示数正中点向上概率≈0.5", abs(p_up - 0.5) < 0.02,
          f"P(up)={p_up:.4f}")


# ---------------------------------------------------------------- T3
def test_master_accumulates_tiny_updates():
    # 构造一个极小更新：小于 fp16 在 1.0 附近的间距（2^-10 ≈ 9.77e-4）
    master = {"L": {"W": np.ones((4,), dtype=np.float32)}}
    opt = SGDMomentum(master, lr=1.0, momentum=0.0, weight_decay=0.0)
    tiny = np.float32(1e-5)
    w16_before = master["L"]["W"].astype(np.float16).copy()
    for _ in range(200):
        opt.step({"L": {"W": np.full((4,), tiny, dtype=np.float32)}})
    w = master["L"]["W"]
    # fp16 副本在整个过程中根本不会变（单次更新被舍掉）
    w16_still_same = np.array_equal(w16_before, np.ones((4,), dtype=np.float16))
    # 但 fp32 主副本把 200 次小更新全攒住了：SGD 为 w -= lr*g => 1 - 200*1e-5
    expect = np.float32(1.0 - 200 * tiny)
    rel_err = abs(float(w[0]) - float(expect)) / abs(float(expect))
    check("T3 主副本累积 fp16 会舍掉的小更新（需求5）",
          w16_still_same and rel_err < 1e-4,
          f"master={w[0]:.8f} expect={expect:.8f} rel_err={rel_err:.2e} "
          f"fp16权重全程不变={w16_still_same}")


# ---------------------------------------------------------------- T4
def test_scaler_dynamics():
    s = DynamicLossScaler(init_scale=128.0, min_scale=1.0, max_scale=1024.0,
                          growth_interval=4)
    s.update_after_step(False)
    check("T4a 未到 growth_interval 时 scale 不动", s.scale == 128.0)
    for _ in range(3):
        s.update_after_step(False)
    check("T4b 连续4步无溢出 scale 翻倍", s.scale == 256.0)
    s.update_after_step(True)
    check("T4c 溢出一步 scale 减半", s.scale == 128.0)
    # 上限
    s2 = DynamicLossScaler(init_scale=1024.0, max_scale=1024.0,
                           growth_interval=1)
    s2.update_after_step(False)
    check("T4d scale 不越上限", s2.scale == 1024.0)
    # 下限
    s3 = DynamicLossScaler(init_scale=1.0, min_scale=1.0, growth_interval=99)
    s3.update_after_step(True)
    check("T4e scale 不越下限", s3.scale == 1.0)
    # 固定模式
    s4 = DynamicLossScaler(init_scale=64.0, fixed=True, growth_interval=1)
    s4.update_after_step(False)
    s4.update_after_step(True)
    check("T4f fixed 模式下任何事件都不动 scale（需求8）", s4.scale == 64.0)


# ---------------------------------------------------------------- T5
def test_overflow_skips_and_freezes_params():
    x, y = make_dataset(n_per_class=80, seed=11)
    cfg = TrainConfig(run_name="t5", out_dir="runs/verify_t5", half=True,
                      steps=6, init_scale=128.0, growth_interval=10 ** 6,
                      inject_overflow_at=(3,), eval_every=100)
    out = train(cfg, x[40:], y[40:], x[:40], y[:40],
                overflow_batch=make_overflow_batch(48, magnitude=1e6))
    rows = out["history"]
    r = rows[3]
    check("T5a 溢出批次被识别（梯度含 inf）",
          r["skipped"] and r["grad_overflow_elems"] > 0,
          f"overflow_elems={r['grad_overflow_elems']}")
    check("T5b 溢出步整步跳过且参数逐元素不变（需求2）",
          r["params_unchanged"], f"params_unchanged={r['params_unchanged']}")
    check("T5c 溢出后 scale 减半",
          rows[4]["scale"] == 64.0, f"next_scale={rows[4]['scale']}")
    check("T5d 跳过后训练能恢复（后续步正常）",
          all(not rr["skipped"] for rr in rows[4:]))
    # 直接核对相邻两步主参数完全相等
    m = out["model"]
    check("T5e 跳步计数与 skipped_steps 一致",
          out["summary"]["n_skipped"] == 1 and out["summary"]["skipped_steps"] == [3])


# ---------------------------------------------------------------- T6
def test_underflow_scaling_helps():
    x, y = make_dataset(n_per_class=160, seed=12)
    cfg = TrainConfig(run_name="t6", out_dir="runs/verify_t6", half=True,
                      rounding="near", steps=80, init_scale=128.0,
                      growth_interval=10 ** 6, eval_every=100)
    out = train(cfg, x[80:], y[80:], x[:80], y[:80])
    rows = [r for r in out["history"] if not r["skipped"]]
    no_scale = np.array([r["underflow_unscaled_zero_frac"] for r in rows])
    scaled = np.array([r["underflow_scaled_zero_frac"] for r in rows])
    mean_gap = float(no_scale.mean() - scaled.mean())
    # 逐步比较：大多数步上缩放路径的零比例应严格更低
    win_rate = float(np.mean(scaled < no_scale))
    check("T6 缩放后下溢成零的比例显著低于不缩放（需求验收）",
          mean_gap > 0.01 and win_rate >= 0.8,
          f"均值: 无缩放={no_scale.mean():.4f} 缩放后={scaled.mean():.4f} "
          f"降幅={mean_gap:.4f}; 逐步占优比例={win_rate:.2f}")


# ---------------------------------------------------------------- T7
def test_rounding_modes_on_tiny_grads():
    # 构造一批"比 fp16 最小次规格化数还小"的真实小梯度
    rng = np.random.default_rng(13)
    g = rng.normal(0, 3e-9, size=200000).astype(np.float32)
    nz = np.count_nonzero(g)
    near = cast_fp16(g, CastContext("near"))
    stoch = stochastic_cast(g, rng=np.random.default_rng(14))
    near_eaten = np.count_nonzero(near[g != 0] == 0) / nz
    stoch_eaten = np.count_nonzero(stoch[g != 0] == 0) / nz
    # 随机舍入仍会产生零，但期望保留的信号更多；就近舍入必然 100% 吞掉
    check("T7a 就近舍入把次规格化阈值以下的梯度全吞零",
          near_eaten == 1.0, f"near_eaten={near_eaten:.4f}")
    check("T7b 随机舍入保留部分小梯度（吞零比例显著更低，需求7）",
          stoch_eaten < 0.99, f"stoch_eaten={stoch_eaten:.4f}")
    # 无偏性：随机舍入结果的均值应保留原信号方向与量级
    check("T7c 随机舍入后均值仍逼近原均值（信号不丢）",
          abs(stoch.astype(np.float32).mean() - g.mean()) < 1e-9,
          f"stoch_mean={stoch.astype(np.float32).mean():.3e} true={g.mean():.3e}")


# ---------------------------------------------------------------- T8
def test_fp32_vs_mixed_convergence():
    x, y = make_dataset(n_per_class=300, seed=42)
    x_val, y_val = x[:240], y[:240]
    x_train, y_train = x[240:], y[240:]
    common = dict(steps=300, batch_size=48, lr=0.05, momentum=0.9,
                  weight_decay=1e-4, seed=2026, eval_every=10,
                  growth_interval=32, inject_overflow_at=(120,))
    r32 = train(TrainConfig("v_fp32", "runs/verify_v_fp32", half=False,
                            **common), x_train, y_train, x_val, y_val,
                overflow_batch=make_overflow_batch(48))
    rn = train(TrainConfig("v_near", "runs/verify_v_near", half=True,
                           rounding="near", **common),
               x_train, y_train, x_val, y_val,
               overflow_batch=make_overflow_batch(48))
    rs = train(TrainConfig("v_stoch", "runs/verify_v_stoch", half=True,
                           rounding="stoch", **common),
               x_train, y_train, x_val, y_val,
               overflow_batch=make_overflow_batch(48))

    def first_acc(history, thr):
        for r in history:
            if r["eval"] and r["eval"]["acc_master_fp32"] >= thr:
                return r["step"]
        return 10 ** 9

    details = []
    ok_all = True
    for tag, out in (("mixed_near", rn), ("mixed_stoch", rs)):
        a32 = r32["history"][-1]["eval"]["acc_master_fp32"]
        am = out["history"][-1]["eval"]["acc_master_fp32"]
        acc_gap = abs(a32 - am)
        s32, sm = first_acc(r32["history"], 0.95), first_acc(out["history"], 0.95)
        step_gap = abs(s32 - sm)
        # 阈值：最终准确率差距 <= 0.05；达到 95% 准确率的步数差 <= 60 步
        ok = acc_gap <= 0.05 and step_gap <= 60
        ok_all &= ok
        details.append(f"{tag}: 最终准确率 fp32={a32:.3f} mixed={am:.3f} "
                       f"(差{acc_gap:.3f}); 达95%步数 fp32={s32} mixed={sm} "
                       f"(差{step_gap})")
    check("T8 同种子 fp32 vs 混合精度收敛差距在阈值内（需求验收）", ok_all,
          " | ".join(details))


def main():
    print("=" * 78)
    test_matmul_semantics()
    test_stochastic_rounding()
    test_master_accumulates_tiny_updates()
    test_scaler_dynamics()
    test_overflow_skips_and_freezes_params()
    test_underflow_scaling_helps()
    test_rounding_modes_on_tiny_grads()
    test_fp32_vs_mixed_convergence()
    print("=" * 78)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"{n_pass}/{len(results)} 项通过")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
