"""快速冒烟测试：fp32/fp16 两条前向反向通路 + 溢出批次。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from amp.model import TinyCNN
from amp.fp16_utils import CastContext
from amp.data import make_dataset, make_overflow_batch

x, y = make_dataset(n_per_class=20, seed=1)
xb, yb = x[:16], y[:16]
for half in (False, True):
    m = TinyCNN(rng=np.random.default_rng(0))
    ctx = CastContext("near", seed=5)
    m.sync_weights(ctx, half=half)
    xw = xb.astype(np.float16) if half else xb.astype(np.float32)
    loss, logits = m.forward(xw, yb)
    m.backward(128.0, half=half, ctx=ctx)
    gn = 0.0
    for l in m.trainable_layers():
        for k, g in l.master_grads().items():
            gn += float((g.astype(np.float32) ** 2).sum())
    print("half=", half, "loss=", round(loss, 4),
          "finite_logits=", bool(np.isfinite(logits).all()),
          "grad_norm=", round(gn ** 0.5, 3))

m = TinyCNN(rng=np.random.default_rng(0))
ctx = CastContext("near", seed=5)
m.sync_weights(ctx, half=True)
ox, oy = make_overflow_batch(16)
loss, logits = m.forward(ox.astype(np.float16), oy)
m.backward(128.0, half=True, ctx=ctx)
bad = sum((~np.isfinite(g.astype(np.float32))).sum()
          for l in m.trainable_layers() for g in l.master_grads().values())
print("overflow batch: finite_logits=", bool(np.isfinite(logits).all()),
      "nonfinite_grad_elems=", bad)
