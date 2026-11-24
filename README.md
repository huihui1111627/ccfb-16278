# 混合精度训练框架（numpy float16 模拟）

用纯 Python + numpy 模拟老显卡上的半精度训练：手写小卷积网络、自造图像数据、
动态损失缩放、就近/随机两种舍入，配一个零第三方依赖的网页看板。

## 目录结构

```
amp/                     框架本体
  fp16_utils.py          float16 转换、就近/随机舍入、半精度矩阵乘模拟
  scaler.py              动态损失缩放（溢出减半、连续无溢出翻倍、上下限、固定模式）
  model.py               手写 CNN（im2col 卷积/全连接/ReLU/池化）+ softmax 交叉熵
  optimizer.py           float32 SGD+动量，权重衰减与动量状态全在单精度
  data.py                三类几何图形合成数据 + 必然溢出的测试批次
  trainer.py             训练主循环、跳步、下溢统计、控制文件、指标落盘
scripts/
  run_all.py             四组对照实验（全单精度 / 混合就近 / 混合随机 / 固定scale）
  verify.py              22 项验收测试（T1~T8）
  live_demo.py           训练中通过网页 API 实时固定 scale 的演示
  dashboard.py           看板 HTTP 服务（标准库）
  smoke.py               前向反向冒烟测试
web/                     看板前端（原生 HTML/JS/Canvas，无 CDN）
runs/<实验名>/
  metrics.jsonl          逐步全部指标（严格合法 JSON，NaN 输出为 null）
  state.json             最新状态快照
  control.json           网页下发的 scale 控制（训练器每步读取）
  summary.json           实验汇总
```

## 快速开始

```bash
# 1. 四组对照训练（约 10 秒，固定种子，结果可复现）
python3 -I scripts/run_all.py

# 2. 启动看板
python3 -I scripts/dashboard.py --port 8000
# 浏览器打开 http://127.0.0.1:8000/

# 3. 22 项验收
python3 -I scripts/verify.py

# 4. 实时控制演示（先启动看板，再跑）
python3 -I scripts/live_demo.py
```

`-I` 为隔离模式，避免当前目录下的同名模块（如伪造的 `json.py`）被导入。
步数可用环境变量调整：`AMP_STEPS=60 python3 -I scripts/run_all.py`。

## 需求与实现对照

| 需求 | 实现 |
| --- | --- |
| 1. float32 主副本，每步转 float16 前反向，更新加在主副本 | `Layer.params32` + `sync_weights()`；优化器只更新主副本 |
| 2. 损失乘 scale 反向、除回；inf/nan 整步跳过、scale 减半 | `scaler.py` + `trainer.py` 溢出分支，逐元素快照校验参数不动 |
| 3. 连续 N 步无溢出翻倍，带上下限 | `DynamicLossScaler(growth_interval, min/max_scale)` |
| 4. 每层激活 min/max 与 fp16 范围余量 | `TinyCNN.activation_stats()`，看板激活表 |
| 5. 小更新在 fp16 会被舍掉，主副本攒住 | 优化器全 float32 累加；验收 T3 用 200 次 1e-5 更新验证 |
| 6. 权重衰减、动量状态在 float32 | `SGDMomentum`，velocity 与 wd 均 float32 |
| 7. 就近/随机舍入可切换，对比小梯度吞零 | `stochastic_cast` 无偏随机舍入；逐步两种舍入零比例对照 |
| 8. 网页手动固定 scale | `control.json` + `POST /api/control`，训练器每步读取 |

## 看板指标

每步损失缩放系数、梯度溢出/下溢元素比例、跳过步数、fp16/fp32 参数相对 L2 差距、
损失曲线（可叠加全单精度对照）、准确率、两种舍入吞零曲线、每层激活范围表、
缩放控制按钮与事件流。

## 关键设计说明

- **半精度矩阵乘**：操作数先按选定舍入模式转 float16，乘积累加在 float32（模拟张量核），
  输出再舍回 float16，见 `hp_matmul`。
- **下溢对照口径**：每步用同一份前向图、scale=1 再跑一次 fp16 反向作为"不缩放反事实"，
  与实际缩放路径只差一个系数，隔离损失缩放的真实作用。
- **全单精度路径不做损失缩放**（scale 恒为 1）：fp32 无下溢问题，放大反而引入风险；
  两条路径共用同一套梯度范数裁剪，保证注入极端批次时 fp32 也能稳定恢复。
- **严格 JSON**：跳步步的 NaN/Inf 落盘前统一转 `null`，浏览器可直接解析。
