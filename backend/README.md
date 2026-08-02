# backend：crystalvol 算法后端

后端是参数、推理和公制换算的唯一所有者。前端只通过 `app/backend_interface.py` 调用它；命令行入口是 `backend/run.py`。

## 目录和职责

- `crystalvol/preprocess.py`：低光增强、高光抑制和去噪；
- `crystalvol/localize.py`：用显著性和中心先验定位最大晶体，自动适配小目标与大目标；
- `crystalvol/edges.py`：Canny、LSD、PiDiNet/HED 和自动择优；缺少深度权重时回退 Canny；
- `crystalvol/segmentation.py`：可选 YOLO-World/SAM2 分割；
- `crystalvol/silhouette.py`：边缘/分割掩膜转晶体剪影，抑制高光、黑条和台面倒影；
- `crystalvol/wireframe.py`：剪影宽度剖面拟合“长方体 + 四棱锥”；
- `crystalvol/stage1.py`：单帧处理、跨帧共识、可视化和 JSON 输出；
- `crystalvol/stage2.py` / `metric.py`：尺度锚点和相机外参公制换算；
- `crystalvol/camera_parameters.py`：统一 JSON schema、加载优先级、外参选择和针孔换算；
- `crystalvol/physical_constraints.py`：1–70 cm 长度及实时生长质量控制；
- `crystalvol/session.py`：实时多拍增量会话。

## 后端命令

### 第一阶段

```bash
uv run python backend/run.py stage1 <图片|目录|视频> \
  --output-dir data/results/stage1 \
  --clean-output \
  --device auto \
  --edge-backend auto \
  --num-frames 7
```

第一阶段不要求相机参数，输出像素域几何和所有诊断图。目录输入会把图片按文件名排序；视频按时间区间均匀抽帧。

### 第二阶段

使用外参：

```bash
uv run python backend/run.py stage2 \
  --stage1-geometry data/results/stage1/geometry/standard_geometry_pixel.json \
  --camera-parameters params/camera_parameters.json \
  --mode auto \
  --output-dir data/results/stage2
```

只有内参时使用尺度锚点：

```bash
uv run python backend/run.py stage2 \
  --stage1-geometry data/results/stage1/geometry/standard_geometry_pixel.json \
  --mode scale_anchor \
  --scale-reference-edge length \
  --scale-reference-value 10 \
  --metric-length-unit cm \
  --output-dir data/results/stage2
```

`auto` 优先使用完整的尺度锚点，否则尝试外参多视角模式。当前 `extrinsic_multiview` 只保留输入校验和清晰错误提示；单张外参本身可由 `calibration extrinsics` 生成并供单目针孔换算使用。

### 完整流程

```bash
uv run python backend/run.py full <输入> \
  --camera-parameters params/camera_parameters.json \
  --scale-reference-edge length \
  --scale-reference-value 10 \
  --metric-length-unit cm \
  --output-dir data/results/stage1 \
  --stage2-output-dir data/results/stage2
```

完整选项使用 `uv run python backend/run.py stage1 -h`、`stage2 -h`、`full -h` 查看。CLI 失败时返回状态码 2，并给出可操作的错误摘要。

## 关键参数

### 预处理和定位

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--lowlight` | `auto` | 按平均亮度自动决定是否增强；`on/off` 可强制开关。 |
| `--gamma` | `0.55` | 小于 1 会提亮暗部；过小会放大噪声。 |
| `--specular-percentile` | `99` | 把亮度最高的一部分当作镜面高光并降低其边缘影响。 |
| `--roi-pad` | `0.6` | 晶体显著块外扩比例；晶体被截断时增大。 |
| `--min-roi-side-ratio` | `0.12` | 小晶体的最小上下文比例。 |
| `--no-localize` | 关闭 | 仅适合晶体占画面大部分时使用。 |
| `--no-sam2` | 关闭 | 禁用 SAM2，完全使用传统边缘剪影，适合无权重/快速测试。 |
| `--edge-backend` | `auto` | 自动比较深度边缘与 Canny；`canny` 最省资源、最容易复现。 |

### 线框和晶体形状

不再提供用户逐次调整的固定 `depth_ratio`。后端内部的 `shape_prior_min_depth_ratio=0.55` 和 `shape_prior_max_depth_ratio=1.25` 只定义安全先验范围；实际比例根据总高/可见长度和屋顶占比连续计算。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `pyramid_top_width_ratio` | `0.7` | 顶部宽度低于满宽的比例时认为存在尖顶。 |
| `pyramid_min_taper_ratio` | `0.06` | 渐扩段至少占总高度的比例。 |
| `min_pyramid_fraction` | `0.15` | 看不到明显尖顶时的最小屋顶高度，避免模型退化为平顶。 |
| `core_percentile` | `40` | 从背光光晕收紧到亮核；值越大越紧，透明晶体不建议盲目增大。 |
| `min-visible-ratio` | `0.5` | 关键线段得到边缘/剪影支持的最低比例。 |
| `min-coverage-ratio` | `0.06` | 剪影占 ROI 的最低比例，过滤碎噪声。 |

每帧 JSON 的 `geometry_px` 会包含 `depth_ratio_estimate`、`vertical_to_length_ratio` 和 `shape_prior_confidence`，用于判断形状先验是否可靠。单目深度不是直接观测量，多视角和尺度锚点仍是提升绝对精度的主要方式。

## 输出结构

典型 `data/results/stage1/`：

```text
inputs/      输入帧副本
enhanced/    低光/高光处理结果
edges/       边缘证据图
masks/       晶体剪影
contours/    线框图
overlays/    汇总叠加图
geometry/    standard_geometry_pixel.json/.obj/预览图
stage1_result.json
```

逐帧异常写入 `failed_frames` 和 `warnings`；如果只有部分帧失败，其他帧仍参与共识；全部帧失败才终止任务。第二阶段写入 `stage2_metric.json`，其中包含 `physical_constraints`：长边必须在 1–70 cm 内，实时模式还会记录与上一有效帧的增长比。

## 性能和长时间运行

- 图像先按 `--max-input-side` 限制最长边，避免超大图造成内存峰值；
- 深度边缘有 `--deep-input-max-side`；
- SAM2、Torch 和后端推理只在 worker 中初始化，不阻塞 Qt 事件循环；
- 失败帧隔离、摄像头资源 `finally` 释放、实时会话结束时清空生长基准；
- 线上采集建议保留 `data/results` 的 JSON 与诊断图一段时间，再由外部归档策略清理。
