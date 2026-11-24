"""实时控制演示：训练进行到第 20 步时，通过网页 API 固定 scale=32。

验证需求 8：训练器每步读取 control.json，网页下发的固定指令在下一步立即生效，
之后 scale 不再增长、溢出也不会减半。运行前先启动 dashboard：
    python -I scripts/dashboard.py --port 8011
"""

import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.data import make_dataset
from amp.trainer import TrainConfig, train

PORT = int(os.environ.get("DASH_PORT", "8011"))
BASE = f"http://127.0.0.1:{PORT}"


def fix_at_step(target_step=20, fixed_scale=32.0):
    def worker():
        try:
            print(f"[demo] 控制线程启动，等待训练到达 step {target_step} ...", flush=True)
            while True:
                try:
                    with urllib.request.urlopen(
                            f"{BASE}/api/state?run=live_demo", timeout=2) as r:
                        st = json.load(r)
                    if st["n_steps"] >= target_step:
                        break
                except Exception:
                    pass  # state 尚未写出
                time.sleep(0.05)  # 必须在 try 外，避免异常时忙等占 GIL
            req = urllib.request.Request(
                f"{BASE}/api/control?run=live_demo",
                data=json.dumps({"fixed": True, "scale": fixed_scale}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            resp = urllib.request.urlopen(req, timeout=2).read()
            print(f"[demo] 已通过网页 API 固定 scale={fixed_scale}，响应 {resp}",
                  flush=True)
        except Exception as exc:
            print("[demo] 控制线程异常（dashboard 是否启动？）:", repr(exc), flush=True)
    threading.Thread(target=worker, daemon=True).start()


def main():
    x, y = make_dataset(n_per_class=120, seed=5)
    cfg = TrainConfig("live_demo", "runs/live_demo", half=True, rounding="near",
                      steps=60, init_scale=128.0, growth_interval=10,
                      eval_every=100)
    fix_at_step(20, 32.0)
    out = train(cfg, x[60:], y[60:], x[:60], y[:60])
    rows = out["history"]
    print("\n固定前后对照：")
    for r in rows[18:26]:
        print(f"  step {r['step']:3d} scale={r['scale']:8.1f} "
              f"fixed={r['scaler_fixed']} event={r['event']}")
    # 控制通过轮询异步下发，动态找到第一个生效步，再校验其后全部冻结
    first_fixed = next((r["step"] for r in rows if r["scaler_fixed"]), None)
    after = [r for r in rows if r["step"] >= first_fixed] if first_fixed is not None else []
    ok = (first_fixed is not None
          and all(r["scale"] == 32.0 and r["scaler_fixed"] for r in after))
    print(f"\n首个生效步={first_fixed}；生效后 scale 恒为 32: {ok}；"
          f"生效后增长事件数: {sum(1 for r in after if r['event'] == 'growth')}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
