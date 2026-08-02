# 透明晶体体积估计

这是一个完全模块化的 PyQt6 桌面工程：前端负责交互，后端负责参数读取和晶体重建，`calibration/` 是独立的 OpenCV 相机标定子项目。当前版本不读取旧的 `camera_params.txt` 或旧的标定脚本。

## 顶层结构

业务运行时只保留五个顶层目录：

```text
pyqt6_UI/
├── app/          PyQt6 前端
├── backend/      crystalvol 算法后端、默认参数、Git LFS 权重
├── calibration/  相机标定命令行子项目
├── params/       用户标定后的参数（默认运行时优先读取）
└── data/         原始数据、标定图、临时文件和运行结果（不进 Git）
```

每个目录的详细说明：

- [app/README.md](app/README.md)：界面、后台线程、摄像头和错误恢复。
- [backend/README.md](backend/README.md)：算法流水线、后端 CLI、输出和性能参数。
- [calibration/README.md](calibration/README.md)：棋盘格、内参和单图外参命令。
- [params/README.md](params/README.md)：统一相机参数 JSON schema 和加载优先级。
- [data/README.md](data/README.md)：原始数据与结果的目录约定。

## 安装与运行

项目使用 uv 管理 Python 3.13 环境。模型权重由 Git LFS 管理，首次克隆前先初始化 LFS：

```bash
git lfs install
git clone <仓库地址>
cd pyqt6_UI
```

macOS、Linux：

```bash
SAM2_BUILD_CUDA=0 uv sync
uv run python main.py
```

Windows PowerShell：

```powershell
$env:SAM2_BUILD_CUDA = "0"
uv sync
uv run python main.py
```

如果权重显示为几百字节的文本指针，说明 LFS 没有拉取成功：运行 `git lfs pull`，或从已有机器复制 `backend/weights/` 下的四个权重文件。权重较大但属于模型资产，继续由 Git LFS 跟踪；原始视频、图片和结果不应放入 Git。

## 相机参数统一规则

前端不解析相机文件。`app/backend_interface.py` 只调用后端 `crystalvol.camera_parameters`，后端按以下顺序选取参数：

1. 调用命令或前端显式传入的 JSON 路径；
2. 环境变量 `CRYSTAL_CAMERA_PARAMETERS`；
3. 项目参数 `params/camera_parameters.json`；
4. 后端默认参数 `backend/crystalvol/defaults/camera_parameters.json`。

因此，标定完成后把输出放在 `params/camera_parameters.json` 即可让前端和后端共同使用；也可以用显式路径临时测试。只有确认稳定的参数才使用 `--update-default` 写入后端默认方案。统一格式是当前版本唯一支持的 JSON schema，不能再传旧 TXT 文件。

## 相机标定快速开始

下面的命令在项目根目录执行。也可以把 `uv run` 换成已激活虚拟环境中的 `python`。

### 1. 生成棋盘格

`chessboard` 的 `--pattern-size` 是 OpenCV 检测用的**内角点数（列 x 行）**，不是黑白方格数。`--square-size` 是实际方格边长，单位由 `--unit` 指定：

```bash
uv run python -m calibration board \
  --type chessboard \
  --pattern-size 9x6 \
  --square-size 30 \
  --unit mm \
  --dpi 300 \
  --output data/calibration/chessboard.png
```

命令同时生成同名 `.json` 元数据。除棋盘格外，还支持 `charuco`、`circles_grid` 和 `asymmetric_circles_grid`；全部参数见 [calibration/README.md](calibration/README.md)。

### 2. 用图片目录标定内参

目录内图片应为同一块标定板在不同位置、距离和角度的拍摄，建议至少 15 张，覆盖画面四角和不同倾角；所有图片必须分辨率和宽高比一致。

```bash
uv run python -m calibration intrinsics data/calibration/intrinsics \
  --type chessboard \
  --pattern-size 9x6 \
  --square-size 30 \
  --unit mm \
  --output params/camera_parameters.json \
  --debug-dir data/calibration/intrinsics_debug
```

程序使用 OpenCV `calibrateCameraExtended`，默认按单视图重投影误差剔除明显异常图，输出内参、畸变、每张图的外参、重投影误差和接受/剔除列表。

### 3. 用单张图片求外参

内参已经生成后，用一张能完整检测标定板的图片求该拍摄位置的外参：

```bash
uv run python -m calibration extrinsics \
  --image data/calibration/pose/center.png \
  --parameters params/camera_parameters.json \
  --type chessboard \
  --pattern-size 9x6 \
  --square-size 30 \
  --unit mm \
  --pose-method iterative \
  --object-center 0 0 0 \
  --debug-output data/calibration/pose/center_axes.png \
  --output params/camera_parameters_with_extrinsic.json
```

`--object-center X Y Z` 表示“标定板坐标原点到晶体中心”的偏移，单位与标定板相同；如果已知相机到晶体中心距离，可加 `--expected-distance` 和 `--distance-tolerance` 做几何校验。`--pose-method` 可选 `iterative`、`ippe`、`ransac`，默认 iterative，并默认使用 `solvePnPRefineLM` 做位姿细化。若希望保留输入文件已有外参，再加 `--append`。

确认输出无误后，可显式安装为后端默认参数：

```bash
uv run python -m calibration install-default \
  --parameters params/camera_parameters_with_extrinsic.json
```

也可以在 `intrinsics` 或 `extrinsics` 命令末尾直接使用 `--update-default`，但这会覆盖 `backend/crystalvol/defaults/camera_parameters.json`，应在确认重投影误差和距离校验通过后再使用。实现对齐 OpenCV 官方 calib3d/ArUco 流程，参考：[calib3d](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html) 和 [ArUco calibration](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html)。

## 后端独立运行

后端输入可以是单张图片、图片目录或视频。第一阶段输出像素域剪影、线框和跨帧共识：

```bash
uv run python backend/run.py stage1 /path/to/input \
  --device auto \
  --edge-backend auto \
  --num-frames 7 \
  --output-dir data/results/stage1 \
  --clean-output
```

只有在需要绝对尺寸时才运行第二阶段。若已有外参，会自动从参数文件取相机到目标的距离；若只有内参，推荐用一条实际可测的边作尺度锚点：

```bash
uv run python backend/run.py full /path/to/input \
  --camera-parameters params/camera_parameters.json \
  --scale-reference-edge length \
  --scale-reference-value 10 \
  --metric-length-unit cm \
  --output-dir data/results/stage1 \
  --stage2-output-dir data/results/stage2 \
  --clean-output
```

`--camera-parameters` 可省略，后端会按统一优先级读取。完整参数和默认值见 [backend/README.md](backend/README.md)。

## 单目重建与物理约束

模型仍以“长方体 + 四棱锥”为可解释几何先验，但不再要求用户为每种晶体手工指定一个固定深度比例：

- 剪影宽度剖面自动检测 apex、肩线、底部和屋顶高度；
- 侧向深度比例根据总高/可见长度、屋顶比例自适应，在 tall、balanced、boxy 形状间连续变化；
- 输出 `depth_ratio_estimate` 和 `shape_prior_confidence`，明确区分观测值与单目先验；
- 视频、图片目录和实时多拍会用跨帧共识降低反光、遮挡和单帧分割错误的影响。

单目图像无法从数学上唯一恢复被遮挡的深度，所以这个深度仍是可解释先验；要提高绝对精度，应使用多个角度、可靠外参或尺度锚点。后端还会把晶体长边限制在 **1–70 cm**；实时序列中单帧突然收缩超过 25% 会标记为异常，不会让异常帧成为下一帧的生长基准，但原始结果仍保留供诊断。

体积公式为：`V = L × W × (Hb + Hp / 3)`，其中 `L/W` 是长方体底面两边，`Hb` 是柱体高度，`Hp` 是四棱锥高度。

## 工程运行约定

- 前端耗时任务全部在 `QThread` 中运行；主线程只负责界面和信号，关闭窗口时会等待后台任务安全结束。
- 原始输入面板只显示压缩预览：静态图最长边限制为 1280 px，视频预览限制为 15 FPS/1280 px，实时预览限制为 640 px；算法后端仍按自己的输入缩放参数处理。
- 单帧处理失败会记录到结果 JSON 并继续处理其他帧；全部失败才返回清晰错误。
- 后端目录和实时会话采用流式读取，帧产物写盘后释放大数组；临时结果、摄像头、模型缓存和 Qt 视频定时器都有退出清理路径。
- CLI 对可恢复错误返回非零状态和简短中文提示，不打印不可操作的长 traceback。
- `data/` 除 README 外全部被 `.gitignore` 忽略，包含原始视频、标定图片、调试图、临时文件和结果。
- `backend/weights/` 继续由 Git LFS 跟踪；不要把权重复制到 `data/` 或提交普通 Git blob。

提交前建议运行：

```bash
git lfs ls-files
git status --short
uv run python -m compileall -q app backend calibration
```
