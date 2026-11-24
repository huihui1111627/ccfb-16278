"""训练主循环：全单精度 / 混合精度两条路径共用一套代码。

每步流程（混合精度）：
  1. 读网页控制文件（可随时固定 scale，需求 8）；
  2. float32 主副本 -> float16 工作副本（需求 1）；
  3. fp16 前向，记录每层激活 min/max 与 fp16 余量（需求 4）；
  4. 损失乘 scale 后再反向（需求 2），矩阵运算全程 fp16 模拟；
  5. 检查 inf/nan：任一梯度非有限 -> 整步跳过、参数不动、scale 减半；
  6. 否则梯度除回尺度转 fp32，统计缩放前后下溢比例与两种舍入的吞零比例；
  7. fp32 动量优化器更新主副本（小更新在主副本攒住，需求 5、6）；
  8. 连续 growth_interval 步无溢出则 scale 翻倍（需求 3）。
所有指标按步追加写入 metrics.jsonl，并原子更新 state.json 供网页读取。
"""

from __future__ import annotations

import json
import os
import dataclasses

import numpy as np

from .fp16_utils import CastContext, FP16_MAX, FP16_MIN_SUBNORMAL, cast_fp16
from .model import TinyCNN
from .optimizer import SGDMomentum
from .scaler import DynamicLossScaler
from . import data as datamod


@dataclasses.dataclass
class TrainConfig:
    run_name: str
    out_dir: str
    half: bool
    rounding: str = "near"            # near | stoch
    seed: int = 2026
    steps: int = 300
    batch_size: int = 48
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    init_scale: float = 128.0
    min_scale: float = 1.0
    max_scale: float = 65536.0
    growth_interval: int = 32
    fixed_scale: bool = False
    eval_every: int = 10
    grad_clip: float = 100.0        # unscaled 梯度的全局范数裁剪阈值
    inject_overflow_at: tuple = ()    # 在这些步注入必然溢出的批次


def _atomic_write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # allow_nan=False 配合 _json_safe：Python 默认会写 NaN/Infinity，
        # 那不是合法 JSON，浏览器 fetch().json() 会直接抛错
        json.dump(_json_safe(obj), f, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def _json_safe(obj):
    """递归把 NaN/Inf 替换成 None，输出严格合法 JSON。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # 非有限值检查必须在具体类型分支之前（np.nan 同时是 Python float）
    if isinstance(obj, (int, np.integer)):
        return obj
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def _has_nonfinite(arrays) -> bool:
    return any(not np.isfinite(a.astype(np.float32)).all() for a in arrays)


def _fp16_zero_stats(g32: np.ndarray, step: int):
    """对同一份 fp32 梯度分别用就近/随机舍入 cast 到 fp16，统计被吞成零的比例。

    随机舍入的对照 RNG 按步独立播种，不干扰训练主 RNG，保证可复现。
    """
    near_ctx = CastContext("near")
    stoch_ctx = CastContext("stoch", seed=100003 + step)
    g_near = cast_fp16(g32, near_ctx)
    g_stoch = cast_fp16(g32, stoch_ctx)
    total = g32.size
    nz32 = np.count_nonzero(g32)
    near_zero = total - np.count_nonzero(g_near)
    stoch_zero = total - np.count_nonzero(g_stoch)
    # 只看 fp32 中非零的元素，两种舍入分别把多少个吞成了零
    mask = g32 != 0
    near_eaten = int(np.count_nonzero(g_near[mask] == 0))
    stoch_eaten = int(np.count_nonzero(g_stoch[mask] == 0))
    return {
        "total": int(total),
        "nonzero_fp32": int(nz32),
        "near_zero_frac": float(near_zero / total),
        "stoch_zero_frac": float(stoch_zero / total),
        # 只在 fp32 里非零的元素中统计"被吃掉"的比例（需求 7）
        "near_eaten_frac": float(near_eaten / max(nz32, 1)),
        "stoch_eaten_frac": float(stoch_eaten / max(nz32, 1)),
        "subnormal_frac": float(
            np.count_nonzero((np.abs(g32) > 0) &
                             (np.abs(g32) < float(FP16_MIN_SUBNORMAL))) / total),
    }


def _relative_param_gap(model: TinyCNN) -> float:
    """float32 主副本与重新 cast 出的 fp16 副本之间的相对 L2 差距。"""
    num, den = 0.0, 0.0
    ctx = CastContext("near")
    for layer in model.trainable_layers():
        for k, w32 in layer.params32.items():
            w16 = cast_fp16(w32, ctx).astype(np.float32)
            num += float(np.sum((w32 - w16) ** 2))
            den += float(np.sum(w32 ** 2))
    return float(np.sqrt(num / (den + 1e-30)))


def train(cfg: TrainConfig, x_train, y_train, x_val, y_val,
          overflow_batch=None, log_every: int = 1) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    state_path = os.path.join(cfg.out_dir, "state.json")
    control_path = os.path.join(cfg.out_dir, "control.json")
    # 初始控制文件：只在不存在时写入，保留网页/外部预先下发的控制指令
    if not os.path.exists(control_path):
        _atomic_write_json(control_path, {"fixed": cfg.fixed_scale, "scale": None})

    # 固定种子：模型初始化、数据顺序、随机舍入三条 RNG 分开，三组实验完全对齐
    init_rng = np.random.default_rng(cfg.seed)
    batch_rng = np.random.default_rng(cfg.seed + 1)
    cast_ctx = CastContext(cfg.rounding, seed=cfg.seed + 2)
    # fp32 对照路径同样保留一个 cast 上下文（只用于指标统计）
    model = TinyCNN(image=datamod.IMAGE, n_classes=datamod.N_CLASSES, rng=init_rng)
    optimizer = SGDMomentum(model.master_params(), lr=cfg.lr,
                            momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    # 全单精度路径不存在下溢问题，损失缩放恒为 1 且冻结；
    # 注入的极端批次在 fp32 下也不会溢出（fp32 上界 3.4e38），可正常跳过该批数据
    if cfg.half:
        scaler = DynamicLossScaler(
            init_scale=cfg.init_scale, min_scale=cfg.min_scale,
            max_scale=cfg.max_scale, growth_interval=cfg.growth_interval,
            fixed=cfg.fixed_scale)
    else:
        scaler = DynamicLossScaler(
            init_scale=1.0, min_scale=1.0, max_scale=1.0,
            growth_interval=10 ** 9, fixed=True)

    batch_iter = datamod.iter_minibatches(x_train, y_train, cfg.batch_size,
                                          batch_rng)
    history = []
    skipped_steps = []

    def next_batch(step):
        if overflow_batch is not None and step in set(cfg.inject_overflow_at):
            return overflow_batch
        nonlocal batch_iter
        try:
            return next(batch_iter)
        except StopIteration:
            batch_iter = datamod.iter_minibatches(x_train, y_train,
                                                  cfg.batch_size, batch_rng)
            return next(batch_iter)

    def evaluate():
        # 主副本 fp32 评估 + 当前工作精度评估，两条曲线对照
        model.sync_weights(CastContext("near"), half=False)
        acc32 = float(np.mean(model.predict(x_val) == y_val))
        if cfg.half:
            model.sync_weights(cast_ctx, half=True)
            accw = float(np.mean(model.predict(x_val) == y_val))
        else:
            accw = acc32
        return {"acc_master_fp32": acc32, "acc_working": accw}

    for step in range(cfg.steps):
        # ---- 网页控制：随时固定/解冻 scale（需求 8），fp32 路径不参与 ----
        if cfg.half:
            try:
                with open(control_path, encoding="utf-8") as f:
                    control = json.load(f)
                scaler.fixed = bool(control.get("fixed", scaler.fixed))
                manual = control.get("scale")
                if manual is not None and float(manual) > 0:
                    scaler.scale = float(manual)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        xb, yb = next_batch(step)
        model.sync_weights(cast_ctx, half=cfg.half)
        x_work = cast_fp16(xb, cast_ctx) if cfg.half else xb.astype(np.float32)

        # ---- fp16/fp32 前向 ----
        loss, logits = model.forward(x_work, yb)
        act_stats = model.activation_stats()
        forward_overflow = _has_nonfinite([logits]) or any(
            a["has_nonfinite"] for a in act_stats)

        # ---- 缩放后的反向 ----
        scale_before = scaler.scale
        scaled_loss = scaler.scale_loss(loss)
        model.backward(scale_before, half=cfg.half, ctx=cast_ctx)

        # 收集工作精度梯度
        work_grads = {}
        grad_arrays = []
        for layer in model.trainable_layers():
            work_grads[layer.name] = layer.master_grads()
            grad_arrays.extend(work_grads[layer.name].values())

        grad_overflow = forward_overflow or _has_nonfinite(grad_arrays)
        overflow_elems = int(sum(
            int((~np.isfinite(g.astype(np.float32))).sum()) for g in grad_arrays))
        grad_total = int(sum(g.size for g in grad_arrays))

        rec = {
            "step": step,
            "mode": "mixed_fp16" if cfg.half else "full_fp32",
            "rounding": cfg.rounding,
            "loss": float(loss),
            "scaled_loss": float(scaled_loss),
            "scale": scale_before,
            "scaler_fixed": scaler.fixed,
            "injected_overflow": step in set(cfg.inject_overflow_at),
            "grad_total_elems": grad_total,
            "grad_overflow_elems": overflow_elems,
            "grad_overflow_ratio": overflow_elems / max(grad_total, 1),
            "activations": act_stats,
        }

        if grad_overflow:
            # ---- 需求 2：整步跳过，参数一个元素都不许动 ----
            before = {n: {k: v.copy() for k, v in p.items()}
                      for n, p in model.master_params().items()}
            event = scaler.update_after_step(True)
            after = model.master_params()
            unchanged = all(
                np.array_equal(before[n][k], after[n][k])
                for n in before for k in before[n])
            rec.update({
                "event": event, "skipped": True, "params_unchanged": bool(unchanged),
                "underflow_scaled_zero_frac": None,
                "underflow_unscaled_zero_frac": None,
                "counterfactual_overflow": None,
                "near_zero_frac": None, "stoch_zero_frac": None,
                "near_eaten_frac": None, "stoch_eaten_frac": None,
                "subnormal_frac": None, "grad_norm_scaled": None,
                "grad_norm_unscaled": None, "grad_clipped": None,
                "eaten_update_frac": None,
                "param_gap": _relative_param_gap(model),
            })
            skipped_steps.append(step)
        else:
            # ---- 梯度还原尺度到 fp32 ----
            grads32 = {}
            scaled_concat, unscaled_concat = [], []
            for lname, gd in work_grads.items():
                grads32[lname] = {}
                for k, g in gd.items():
                    g32s = g.astype(np.float32)              # 缩放态（fp16 模拟结果）
                    g32u = scaler.unscale_grad(g32s)         # 还原真实尺度
                    grads32[lname][k] = g32u
                    scaled_concat.append(g32s.ravel())
                    unscaled_concat.append(g32u.ravel())
            scaled_all = np.concatenate(scaled_concat)
            unscaled_all = np.concatenate(unscaled_concat)

            # 缩放态：当前工作精度下实际出现的零比例
            scaled_zero_frac = float(
                (scaled_all.size - np.count_nonzero(scaled_all)) / scaled_all.size)

            # 受控反事实（需求验收）：同一前向图，scale=1 再跑一次 fp16 反向。
            # 这才是"完全不做损失缩放"时会真实发生的下溢，与上面缩放路径只差一个系数。
            if cfg.half:
                model.backward(1.0, half=True, ctx=cast_ctx)
                cf_arrays = [g.astype(np.float32).ravel()
                             for layer in model.trainable_layers()
                             for g in layer.master_grads().values()]
                cf_all = np.concatenate(cf_arrays)
                unscaled_zero_frac = float(
                    (cf_all.size - np.count_nonzero(cf_all)) / cf_all.size)
                cf_overflow = bool(~np.isfinite(cf_all).all())
            else:
                unscaled_zero_frac = 0.0
                cf_overflow = False

            # 需求 7：对"真实尺度（未缩放）梯度"做就近 vs 随机舍入的吞零对照
            zs = _fp16_zero_stats(unscaled_all, step)

            # 全局范数裁剪（作用在还原尺度后的 fp32 梯度上；fp16 路径里损失缩放
            # 先把小梯度抬到可表示区间，裁剪在这里统一完成，等价于真实 AMP 流程）
            gnorm = float(np.linalg.norm(unscaled_all))
            clip_coef = min(1.0, cfg.grad_clip / (gnorm + 1e-12))
            clipped = clip_coef < 1.0
            if clipped:
                for lname in grads32:
                    for k in grads32[lname]:
                        grads32[lname][k] = grads32[lname][k] * clip_coef

            opt_stats = optimizer.step(grads32)
            event = scaler.update_after_step(False)

            rec.update({
                "event": event,
                "skipped": False,
                "params_unchanged": False,
                "underflow_scaled_zero_frac": scaled_zero_frac,
                "underflow_unscaled_zero_frac": unscaled_zero_frac,
                "counterfactual_overflow": cf_overflow,
                "near_zero_frac": zs["near_zero_frac"],
                "stoch_zero_frac": zs["stoch_zero_frac"],
                "near_eaten_frac": zs["near_eaten_frac"],
                "stoch_eaten_frac": zs["stoch_eaten_frac"],
                "subnormal_frac": zs["subnormal_frac"],
                "grad_norm_scaled": float(np.linalg.norm(scaled_all)),
                "grad_norm_unscaled": gnorm,
                "grad_clipped": clipped,
                "eaten_update_frac": (
                    opt_stats["updates_eaten_in_fp16"] /
                    max(opt_stats["updates_total"], 1)),
                "param_gap": _relative_param_gap(model),
            })

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            rec["eval"] = evaluate()
        else:
            rec["eval"] = None
        rec["scaler"] = scaler.snapshot()
        rec["fp16_max"] = float(FP16_MAX)
        rec["fp16_min_subnormal"] = float(FP16_MIN_SUBNORMAL)

        history.append(rec)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(rec), ensure_ascii=False,
                               allow_nan=False) + "\n")
        _atomic_write_json(state_path, {
            "config": dataclasses.asdict(cfg) | {"inject_overflow_at":
                                                 list(cfg.inject_overflow_at)},
            "latest": rec,
            "skipped_steps": skipped_steps,
            "n_steps": len(history),
        })
        if step % log_every == 0:
            tag = "SKIP" if rec["skipped"] else "ok"
            print(f"[{cfg.run_name}] step {step:4d} {tag} "
                  f"loss={rec['loss']:.4f} scale={rec['scale']:.1f} "
                  f"event={rec['event']}")

    final_eval = evaluate()
    summary = {
        "run_name": cfg.run_name,
        "half": cfg.half,
        "rounding": cfg.rounding,
        "steps": cfg.steps,
        "skipped_steps": skipped_steps,
        "n_skipped": len(skipped_steps),
        "final_eval": final_eval,
        "final_loss": history[-1]["loss"],
        "final_scale": scaler.scale,
    }
    _atomic_write_json(os.path.join(cfg.out_dir, "summary.json"), summary)
    return {"history": history, "summary": summary, "model": model}
