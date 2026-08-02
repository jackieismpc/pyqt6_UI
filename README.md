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
- [calibration/README.md](calibration/README.md)：ChArUco、内参和单图外参命令。
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

### 1. 生成 ChArUco 标定板

项目默认使用 OpenCV 官方 ChArUco 示例：7 行、5 列方格，30 mm 方格边长、15 mm marker、`DICT_5X5_100` 字典。项目命令统一使用**列 x 行**，因此官方示例写成 `5x7`；也可以使用无歧义的 `--columns 5 --rows 7`。ChArUco 的尺寸是方格数，不是内角点数：

```bash
uv run python -m calibration board \
  --type charuco \
  --columns 5 \
  --rows 7 \
  --square-size 30 \
  --marker-length 15 \
  --dictionary DICT_5X5_100 \
  --unit mm \
  --paper a4 \
  --orientation portrait \
  --dpi 300 \
  --output data/calibration/charuco_a4.svg
```

命令同时生成同名 `.json` 元数据。A4 页面是 210×297 mm 竖版，标定板本体是 150×210 mm；打印时必须选择“实际大小/100%”，关闭“适应页面”。SVG 是打印主文件，PNG 仅用于预览或不支持 SVG 的打印流程。若确实要使用棋盘格，必须显式加入 `--type chessboard`；全部参数见 [calibration/README.md](calibration/README.md)。

### 2. 用图片目录标定内参

目录内图片应为同一块标定板在不同位置、距离和角度的拍摄，建议至少 15 张，覆盖画面四角和不同倾角；所有图片必须分辨率和宽高比一致。

```bash
uv run python -m calibration intrinsics data/calibration/intrinsics \
  --type charuco \
  --columns 5 \
  --rows 7 \
  --square-size 30 \
  --marker-length 15 \
  --dictionary DICT_5X5_100 \
  --unit mm \
  --output params/camera_parameters.json \
  --debug-dir data/calibration/intrinsics_debug
```

程序使用 OpenCV `CharucoDetector`、`Board.matchImagePoints` 和 `calibrateCameraExtended`，默认按单视图重投影误差剔除明显异常图，输出内参、畸变、每张图的外参、重投影误差和接受/剔除列表。

如果提示有效标定图为 0，先检查生成板时的列、行、方格边长、marker 边长和字典是否完全一致。旋转同一块板不需要交换列行。旧版项目生成的 `7x5 + marker-length=22` 图片不能与新的官方 `5x7 + marker-length=15` 规格混用，应重新打印和拍摄。

命令顺序只能保证计算流程正确，不能保证输入数据一定能得到准确参数。必须同时检查有效图片数量、画面覆盖、角点检测调试图、`reprojection_error_px`、`per_view_errors_px` 和剔除列表；重投影误差约 10 px 的结果不能安装为默认参数。

### 标定拍摄要求和相机状态

- 使用与生产运行完全相同的分辨率、宽高比、ROI/crop、像素 binning 和镜头安装方式；改变这些项目后要重新标定或换用对应参数文件。
- 固定焦距/变焦位置、对焦位置、镜头与相机的机械连接；自动对焦必须关闭或锁定。变焦和对焦位置变化会改变内参，不能共用一份参数。
- 光圈通常不会直接改变理想针孔模型的内参，但会改变清晰度、景深，并可能引起 focus breathing。工程上应固定为生产光圈；如果更换光圈后重新对焦或镜头光学组件发生移动，应重新验证或标定。
- 自动曝光、自动增益、自动白平衡通常不会改变几何内参，但曝光过暗、过曝、噪声或运动模糊会降低角点精度。工业环境建议固定曝光/增益/白平衡；允许自动曝光时应限制范围并保证每帧板面清晰。
- 不要只在一个距离、一个画面中心位置拍摄。建议 20–30 张，覆盖中心、四角和边缘，并包含近/中/远距离以及不同水平和俯仰倾角；标定板完整可见、平整、无反光和运动模糊。近景可让板覆盖约 40%–80% 的图像高度，远景也必须保留足够大的 marker。
- 标定板应固定在刚性平面上。A4 是打印页面尺寸，不要求相机画面中的标定板占满画面；要求的是方格物理尺寸正确、每张照片中清晰可见。

### 3. 用单张图片求外参

内参已经生成后，用一张能完整检测标定板的图片求该拍摄位置的外参：

```bash
uv run python -m calibration extrinsics \
  --image data/calibration/pose/center.png \
  --parameters params/camera_parameters.json \
  --type charuco \
  --columns 5 \
  --rows 7 \
  --square-size 30 \
  --marker-length 15 \
  --dictionary DICT_5X5_100 \
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

也可以在 `intrinsics` 或 `extrinsics` 命令末尾直接使用 `--update-default`，但这会覆盖 `backend/crystalvol/defaults/camera_parameters.json`，应在确认重投影误差和距离校验通过后再使用。实现对齐 OpenCV 官方流程，参考：[标定板生成](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)、[ChArUco 检测](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html) 和 [calib3d](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。

## 后端独立运行

后端输入可以是单张图片、图片目录或视频。第一阶段输出像素域剪影、线框和跨帧共识：

```bash
uv run python backend/run.py stage1 /path/to/input \
  --device auto \
  --edge-backend auto \
  --edge-candidates auto \
  --candidate-top-k 3 \
  --num-frames 7 \
  --output-dir data/results/stage1 \
  --clean-output
```

`auto` 会按帧比较 `pidinet+canny` 与 `canny`；如果希望扩大候选池，可使用
`--edge-candidates canny,pidinet+canny,hed+canny,lsd`。第一阶段会保存每帧候选的
评分明细和跨帧聚合候选，第二阶段会再次结合尺度锚点、体积范围和 1–70 cm 物理约束
复评，而不是盲目接受第一阶段的单一结果。

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

第二阶段结果中的 `candidate_selection` 包含最终候选、综合置信度、第一名与第二名的
分数差和 `ambiguous` 标记。分数差低于 `--selection-margin-threshold` 时仍会返回
最佳结果，但会明确提示候选接近，需要人工或更多视角复核。

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
