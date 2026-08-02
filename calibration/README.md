# calibration：OpenCV 相机标定子项目

这是一个可独立运行的 Python 子项目，入口为：

```bash
uv run python -m calibration <command> ...
```

它只输出当前版本统一的 `camera_parameters.json`，不兼容旧的 TXT 参数或旧的后端工具脚本。项目默认使用 OpenCV 官方 ChArUco 示例：7 行、5 列方格，30 mm 方格、15 mm marker、`DICT_5X5_100`。实现使用 OpenCV calib3d 和现代 ArUco/ChArUco API，生成板固定为 modern pattern，OpenCV 版本由根目录 `pyproject.toml` 管理。

## 命令概览

```text
board            生成打印用标定板图片和元数据
intrinsics       从图片目录标定内参，并输出统一 JSON
extrinsics       用内参对一张图片求外参，并输出统一 JSON
install-default  显式复制一份统一 JSON 到后端默认参数位置
```

每个命令都支持 `-h` 查看实际参数：

```bash
uv run python -m calibration board -h
uv run python -m calibration intrinsics -h
uv run python -m calibration extrinsics -h
```

## 1. 生成标定板

### ChArUco（项目默认）

```bash
uv run python -m calibration board \
  --type charuco \
  --columns 5 \
  --rows 7 \
  --square-size 30 \
  --marker-length 15 \
  --dictionary DICT_5X5_100 \
  --unit mm \
  --dpi 300 \
  --paper a4 \
  --orientation portrait \
  --output data/calibration/charuco_a4.svg
```

参数：

- `--type charuco`：ArUco marker 与棋盘格角点组合的标定板；
- `--columns 5 --rows 7`：方格列数和行数，顺序与 OpenCV `CharucoBoard((columns, rows), ...)` 一致；也可以写成 `--pattern-size 5x7`；
- `--square-size 30`：一个方格的物理边长；
- `--marker-length 15`：ArUco marker 边长，必须小于方格边长；
- `--dictionary DICT_5X5_100`：marker 使用的 OpenCV 预定义字典；
- `--unit mm|cm|m`：所有物理长度的单位；
- `--dpi`：生成图片的打印分辨率，默认 300；
- `--paper a4`：输出 A4 页面；当前只支持 A4；
- `--orientation portrait`：A4 竖版，官方打印方案使用该方向；
- `--margin-mm`：标定板图案外附加白边，不改变 150×210 mm 的图案尺寸，默认 0；
- `--output`：推荐输出 SVG；也支持 PNG/JPG，同时生成同名 `.json` 元数据。

官方样例的页面尺寸是 A4 210×297 mm，板本体是 5×7 个方格，即 150×210 mm。板居中后左右约 30 mm、上下约 43.5 mm。打印 SVG 或带 DPI 的 PNG 时必须选择“实际大小/100%”，关闭“适应页面”；打印后应实测方格边长为 30 mm。项目不根据图片是横向还是竖向自动交换列行，旋转同一块板时仍使用原始规格。

OpenCV 曾经存在 legacy ChArUco 模板。当前项目明确使用 modern pattern，不设置 `setLegacyPattern(True)`；生成、检测、内参和外参必须使用同一套板参数和同一版本的 OpenCV。

### 棋盘格（显式选择）

```bash
uv run python -m calibration board \
  --type chessboard \
  --pattern-size 9x6 \
  --square-size 30 \
  --unit mm \
  --output data/calibration/chessboard.png
```

棋盘格的 `pattern-size` 表示内角点数量（列 x 行），不是黑白方格数量。只有命令中显式指定 `--type chessboard` 时才使用棋盘格流程。

### 圆点板

```bash
uv run python -m calibration board \
  --type circles_grid \
  --pattern-size 7x5 \
  --circle-distance 30 \
  --unit mm \
  --output data/calibration/circles.png
```

非对称圆点板只需把类型和点阵尺寸换成对应规格：

```bash
uv run python -m calibration board --type circles_grid --pattern-size 7x5 --circle-distance 30 --unit mm --output data/calibration/circles.png
uv run python -m calibration board --type asymmetric_circles_grid --pattern-size 4x11 --circle-distance 20 --unit mm --output data/calibration/asymmetric.png
```

## 2. 内参标定

把同一块板从不同角度、距离和画面位置拍摄成一个目录。推荐 15–30 张，必须保证所有图像尺寸和宽高比一致；少量无法检测的图片会进入 `detection_rejected_images`，有效图片不足时命令失败。

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

内参参数：

- `image_dir`：图片目录；`--recursive` 允许递归读取子目录；
- `--model standard|rational`：标准径向/切向模型，或增加高阶径向畸变；
- `--fix-aspect-ratio`：固定 `fx/fy` 比例；只有明确知道像素非方形或已有可靠初值时使用；
- `--zero-tangent-dist`：固定切向畸变为 0，镜头确实满足该假设时使用；
- `--fix-principal-point`：固定主点为初始估计值；通常不要启用；
- `--max-view-error`：单视图重投影误差阈值，默认 2 px；
- `--max-rounds`：异常图剔除迭代次数，默认 3；
- `--min-views`：最少有效视图，默认 5；建议实际拍摄数量远大于它；
- `--max-iterations` / `--epsilon`：OpenCV 优化终止条件；
- `--no-reject-outliers`：关闭异常视图剔除，只有在数据已人工清理时使用；
- `--debug-dir`：输出每张检测结果图，便于确认角点顺序和漏检；
- `--update-default`：成功后覆盖后端内置默认参数，谨慎使用。

默认 ChArUco 流程使用 `CharucoDetector`、`Board.matchImagePoints` 和 `calibrateCameraExtended`；`calibration` 负责检测和 schema 封装，数值优化仍由 OpenCV 完成。棋盘格模式才使用 `findChessboardCornersSB`。

如果有效图片为 0，先检查列数、行数、方格边长、marker 边长和字典是否与打印板完全一致。旋转同一块板不需要交换列行。旧版 `7x5 + marker-length=22` 图片不能用于新的官方 `5x7 + marker-length=15` 方案，应重新打印和拍摄。

标定命令成功不代表样本质量一定足够。应检查输出中的 `calibration.reprojection_error_px`、`per_view_errors_px`、`detection_rejected_images` 和 `outlier_rejected_images`。通常应重新拍摄覆盖画面中心、四角、不同距离和倾角的清晰图片；标定板应占画面较大区域且保持平整。重投影误差约 10 px 的结果不建议安装为后端默认参数。

## 3. 单图外参

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
  --expected-distance 800 \
  --distance-tolerance 40 \
  --debug-output data/calibration/pose/center_axes.png \
  --output params/camera_parameters_with_extrinsic.json
```

参数：

- `--image`：一张完整、清晰、分辨率比例与内参相符的标定板图片；
- `--parameters`：前一步生成的统一 JSON；
- `--pose-method iterative`：常规 `solvePnP`；`ippe` 适合平面目标，`ransac` 适合含少量错误点的图；
- `--no-refine-pose`：关闭 `solvePnPRefineLM`，一般保持默认开启；
- `--object-center X Y Z`：标定板坐标原点到实际晶体中心的偏移，单位同 `--unit`；
- `--expected-distance`：相机到晶体中心的期望距离，作为校验而不是强行修改位姿；
- `--distance-tolerance`：距离允许误差，未填写时默认期望距离的 5%，至少 1 个单位；
- `--append`：保留输入 JSON 中已有外参并追加当前结果；否则只输出当前这组外参；
- `--debug-output`：画出坐标轴的检查图；
- `--update-default`：同时覆盖后端内置默认参数。

输出外参使用 `coordinate_convention: object_to_camera`，即 OpenCV 的 `X_camera = R * X_object + t`。`translation_vector` 的单位就是标定板长度单位，后端加载后统一换算为米；`distance_to_object_center` 已考虑 `--object-center`。

## 输出 schema

```json
{
  "schema_version": 1,
  "camera": {
    "image_width": 1920,
    "image_height": 1080,
    "camera_matrix": [[...], [...], [...]],
    "distortion_coeffs": [...],
    "distortion_model": "opencv_radtan"
  },
  "extrinsics": [{
    "id": "center",
    "rotation_matrix": [[...], [...], [...]],
    "translation_vector": [...],
    "translation_unit": "mm",
    "coordinate_convention": "object_to_camera"
  }],
  "calibration": {"reprojection_error_px": 0.4}
}
```

不要手动把不同单位或不同分辨率的参数拼在一起。后端会验证矩阵尺寸、有限数值、焦距、外参坐标约定和单位；不合法文件会在启动或 CLI 开始时明确失败。

官方参考：[OpenCV 标定板生成](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)、[ChArUco 检测](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)、[CharucoBoard API](https://docs.opencv.org/4.x/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html)。
