"""手写小卷积网络，所有矩阵运算走 fp16_utils.hp_matmul。

数据布局 NCHW。半精度路径下：
- 权重主副本 W32/b32 永远 float32，每步 sync 出一份 float16 工作副本 W/b；
- 激活、梯度在 half 模式下都是 float16；
- softmax/交叉熵在 float32 里算（损失不进半精度图），反向的上游梯度在这里
  乘上损失缩放系数后再 cast 回工作精度。
"""

from __future__ import annotations

import numpy as np

from .fp16_utils import cast_fp16, hp_matmul


# ---------------------------------------------------------------- im2col 工具

def im2col(x: np.ndarray, kh: int, kw: int, stride: int,
           pad: int = 0) -> np.ndarray:
    """x: (B, C, H, W) -> cols: (B, C*kh*kw, L)，L = 输出格点数。"""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    b, c, h, w = x.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    cols = np.empty((b, c * kh * kw, oh * ow), dtype=x.dtype)
    idx = 0
    for i in range(kh):
        for j in range(kw):
            patch = x[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
            cols[:, idx * c:(idx + 1) * c, :] = patch.reshape(b, c, oh * ow)
            idx += 1
    return cols, (oh, ow)


def col2im(cols: np.ndarray, x_shape, kh: int, kw: int, stride: int,
           pad: int = 0) -> np.ndarray:
    """im2col 的逆（梯度累加到对应感受野位置），最后裁掉 padding 区域。"""
    b, c, h, w = x_shape
    hp, wp = h + 2 * pad, w + 2 * pad
    oh = (hp - kh) // stride + 1
    ow = (wp - kw) // stride + 1
    dx = np.zeros((b, c, hp, wp), dtype=cols.dtype)
    idx = 0
    for i in range(kh):
        for j in range(kw):
            block = cols[:, idx * c:(idx + 1) * c, :].reshape(b, c, oh, ow)
            dx[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride] += block
            idx += 1
    if pad:
        dx = dx[:, :, pad:pad + h, pad:pad + w]
    return dx


# --------------------------------------------------------------------- 层基类

class Layer:
    name = "layer"

    def __init__(self):
        self.half = True
        self.ctx = None
        self.params32: dict[str, np.ndarray] = {}
        self.working: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.out = None
        self.x_in = None

    def sync_weights(self, ctx, half: bool):
        """每步训练前：从 float32 主副本生成当前精度的工作副本。"""
        self.ctx, self.half = ctx, half
        self.working = {}
        for k, p in self.params32.items():
            self.working[k] = cast_fp16(p, ctx) if half else p.astype(np.float32)

    def forward(self, x):  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def backward(self, dout):  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def master_grads(self) -> dict[str, np.ndarray]:
        """把工作精度梯度取回 float32（缩放还原由 trainer 统一做）。"""
        return {k: g.astype(np.float32) for k, g in self.grads.items()}


class Conv2d(Layer):
    def __init__(self, in_ch, out_ch, ksize, stride=1, rng=None):
        super().__init__()
        self.name = f"conv{out_ch}"
        rng = rng or np.random.default_rng()
        bound = np.sqrt(2.0 / (in_ch * ksize * ksize))  # He 初始化
        self.params32 = {
            "W": (rng.standard_normal((out_ch, in_ch * ksize * ksize)) * bound)
            .astype(np.float32),
            "b": np.zeros(out_ch, dtype=np.float32),
        }
        self.oc, self.ic, self.k, self.s = out_ch, in_ch, ksize, stride
        self.pad = ksize // 2  # same padding，池化前特征图尺寸不变

    def forward(self, x):
        self.x_in = x
        W, b = self.working["W"], self.working["b"]
        col, (oh, ow) = im2col(x, self.k, self.k, self.s, self.pad)
        bsz, _, L = col.shape
        out = np.empty((bsz, self.oc, L), dtype=W.dtype)
        for n in range(bsz):  # 每个样本走一次半精度矩阵乘
            out[n] = hp_matmul(W, col[n], self.half, self.ctx)
        out = out + b[None, :, None]
        self.out = out.reshape(bsz, self.oc, oh, ow)
        self._col = col
        return self.out

    def backward(self, dout):
        bsz = dout.shape[0]
        dcol_flat = dout.reshape(bsz, self.oc, -1)
        W = self.working["W"]
        dW = np.zeros_like(W, dtype=W.dtype)
        dcols = np.empty((bsz, W.shape[1], dcol_flat.shape[2]), dtype=W.dtype)
        for n in range(bsz):
            # dW += dcol @ xcol^T，dxcol = W^T @ dcol
            dW = dW + hp_matmul(dcol_flat[n], self._col[n].swapaxes(0, 1),
                                self.half, self.ctx)
            dcols[n] = hp_matmul(W.swapaxes(0, 1), dcol_flat[n],
                                 self.half, self.ctx)
        self.grads = {
            "W": dW,
            "b": dcol_flat.sum(axis=(0, 2)).astype(W.dtype),
        }
        return col2im(dcols, self.x_in.shape, self.k, self.k, self.s, self.pad)


class Linear(Layer):
    def __init__(self, n_in, n_out, rng=None):
        super().__init__()
        self.name = f"fc{n_out}"
        rng = rng or np.random.default_rng()
        bound = np.sqrt(2.0 / n_in)
        self.params32 = {
            "W": (rng.standard_normal((n_in, n_out)) * bound).astype(np.float32),
            "b": np.zeros(n_out, dtype=np.float32),
        }

    def forward(self, x):
        self.x_in = x
        W, b = self.working["W"], self.working["b"]
        self.out = hp_matmul(x, W, self.half, self.ctx) + b
        return self.out

    def backward(self, dout):
        x = self.x_in
        self.grads = {
            "W": hp_matmul(x.swapaxes(0, 1), dout, self.half, self.ctx),
            "b": dout.sum(axis=0).astype(dout.dtype),
        }
        return hp_matmul(dout, self.working["W"].swapaxes(0, 1),
                         self.half, self.ctx)


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.name = "relu"

    def forward(self, x):
        self.x_in = x
        self.out = np.maximum(x, 0)
        return self.out

    def backward(self, dout):
        return dout * (self.x_in > 0)


class MaxPool2d(Layer):
    def __init__(self, ksize=2, stride=2):
        super().__init__()
        self.name = "pool"
        self.k, self.s = ksize, stride

    def forward(self, x):
        self.x_in = x
        b, c, h, w = x.shape
        oh, ow = (h - self.k) // self.s + 1, (w - self.k) // self.s + 1
        windows = np.empty((b, c, oh, ow, self.k * self.k), dtype=x.dtype)
        idx = 0
        for i in range(self.k):
            for j in range(self.k):
                windows[..., idx] = x[:, :, i:i + self.s * oh:self.s,
                                      j:j + self.s * ow:self.s]
                idx += 1
        self.arg = np.argmax(windows, axis=-1)
        self.out = np.max(windows, axis=-1)
        self._shape = (b, c, oh, ow)
        return self.out

    def backward(self, dout):
        b, c, oh, ow = self._shape
        dwin = np.zeros((b, c, oh, ow, self.k * self.k), dtype=dout.dtype)
        # 把梯度放到 argmax 对应的位置
        oi, oj = np.indices((oh, ow))
        dwin[np.arange(b)[:, None, None, None],
             np.arange(c)[None, :, None, None],
             oi[None, None], oj[None, None], self.arg] = dout
        dx = np.zeros_like(self.x_in)
        idx = 0
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :, i:i + self.s * oh:self.s,
                   j:j + self.s * ow:self.s] += dwin[..., idx]
                idx += 1
        return dx


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.name = "flatten"

    def forward(self, x):
        self.x_in = x
        b = x.shape[0]
        self.out = x.reshape(b, -1)
        return self.out

    def backward(self, dout):
        return dout.reshape(self.x_in.shape)


class SoftmaxCrossEntropy:
    """softmax + 交叉熵合一，永远在 float32 计算（数值稳定 + 损失不进 fp16）。"""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.probs = None
        self.targets = None
        self.bsz = 0

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        z = logits.astype(np.float32)
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        self.probs = ez / ez.sum(axis=1, keepdims=True)
        self.targets = targets
        self.bsz = z.shape[0]
        eps = 1e-12
        return float(-np.log(self.probs[np.arange(self.bsz), targets] + eps).mean())

    def backward(self, scale: float, half: bool, ctx) -> np.ndarray:
        """上游梯度 = (probs - onehot)/N，在这里乘损失缩放，再 cast 回工作精度。"""
        dz = self.probs.copy()
        dz[np.arange(self.bsz), self.targets] -= 1.0
        dz = (dz / self.bsz) * np.float32(scale)
        return cast_fp16(dz, ctx) if half else dz.astype(np.float32)


# --------------------------------------------------------------------- 组装模型

class TinyCNN:
    """小卷积网络：conv3x3 -> relu -> pool2x2 -> conv3x3 -> relu -> pool -> fc。"""

    def __init__(self, in_ch=1, n_classes=3, image=12, rng=None):
        rng = rng or np.random.default_rng()
        h = image // 4  # 两次 2x2 池化
        self.layers = [
            Conv2d(in_ch, 8, 3, stride=1, rng=rng),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(8, 16, 3, stride=1, rng=rng),
            ReLU(),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(16 * h * h, 32, rng=rng),
            ReLU(),
            Linear(32, n_classes, rng=rng),
        ]
        for i, layer in enumerate(self.layers):
            layer.name = f"{i:02d}_{layer.name}"
        self.loss_fn = SoftmaxCrossEntropy(n_classes)

    # ---- 参数主副本（全 float32）----
    def master_params(self) -> dict[str, dict[str, np.ndarray]]:
        return {l.name: dict(l.params32) for l in self.layers if l.params32}

    def sync_weights(self, ctx, half: bool):
        for l in self.layers:
            if l.params32:
                l.sync_weights(ctx, half)

    def forward(self, x, targets):
        for l in self.layers:
            x = l.forward(x)
        return self.loss_fn.forward(x, targets), x

    def backward(self, scale: float, half: bool, ctx):
        dout = self.loss_fn.backward(scale, half, ctx)
        for l in reversed(self.layers):
            dout = l.backward(dout)

    def trainable_layers(self) -> list[Layer]:
        return [l for l in self.layers if l.params32]

    def predict(self, x) -> np.ndarray:
        for l in self.layers:
            x = l.forward(x)
        return np.argmax(x.astype(np.float32), axis=1)

    def activation_stats(self) -> list[dict]:
        """每层激活的 min/max/|max| 以及与 float16 上界的余量（需求 4）。"""
        from .fp16_utils import FP16_MAX
        stats = []
        for l in self.layers:
            a = l.out
            if a is None:
                continue
            af = a.astype(np.float32)
            amin, amax = float(af.min()), float(af.max())
            stats.append({
                "layer": l.name,
                "dtype": str(a.dtype),
                "min": amin,
                "max": amax,
                "absmax": float(np.abs(af).max()),
                "headroom_to_fp16_max": float(FP16_MAX - np.abs(af).max()),
                "has_nonfinite": bool(~np.isfinite(af).all()),
            })
        return stats
