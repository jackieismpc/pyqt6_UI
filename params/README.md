# params：项目运行参数

这个目录保存用户确认过的相机内外参。默认文件名是：

```text
params/camera_parameters.json
```

目录本身应该提交，但真实标定参数是否提交由项目协作策略决定；如果参数包含现场信息，可仅在部署机生成。`data/` 中的标定原图和调试图始终不跟踪。

## 读取优先级

所有前端和后端都由 `backend/crystalvol/camera_parameters.py` 读取，优先级固定为：

1. 调用方显式路径（例如 `--camera-parameters /path/to/file.json`）；
2. 环境变量 `CRYSTAL_CAMERA_PARAMETERS`；
3. `params/camera_parameters.json`；
4. `backend/crystalvol/defaults/camera_parameters.json`。

显式路径不存在或格式错误时不会偷偷回退；没有显式路径时，项目参数无效才会由前端启动层回退到后端默认方案。这样部署和诊断都能知道实际使用了哪份参数。

## 生成和更新

推荐由标定子项目生成：

```bash
uv run python -m calibration intrinsics data/calibration/intrinsics \
  --columns 5 --rows 7 \
  --square-size 30 --marker-length 15 \
  --dictionary DICT_5X5_100 \
  --output params/camera_parameters.json

uv run python -m calibration extrinsics \
  --image data/calibration/pose/center.png \
  --parameters params/camera_parameters.json \
  --columns 5 --rows 7 \
  --square-size 30 --marker-length 15 \
  --dictionary DICT_5X5_100 \
  --output params/camera_parameters_with_extrinsic.json
```

确认新方案稳定后，可替换项目参数文件，或显式执行：

```bash
uv run python -m calibration install-default \
  --parameters params/camera_parameters_with_extrinsic.json
```

`install-default` 会验证 JSON 后再写入 `backend/crystalvol/defaults/camera_parameters.json`。后端默认文件是代码仓库的一部分，更新它必须和本次标定的板规格、分辨率、重投影误差一起记录。

## 字段说明

- `schema_version`：当前为 `1`；
- `camera.image_width/image_height`：标定图像尺寸；运行图像可以同比例缩放，不能改变宽高比；
- `camera.camera_matrix`：3x3 的 `K`，顺序为 `fx, 0, cx / 0, fy, cy / 0, 0, 1`；
- `camera.distortion_coeffs`：OpenCV 畸变系数；
- `camera.distortion_model`：当前为 `opencv_radtan`；
- `extrinsics`：外参数组，可存多张板位姿，前端用 `extrinsic_index` 选择；
- `extrinsics.rotation_matrix`：目标坐标到相机坐标的旋转矩阵；
- `extrinsics.translation_vector`：目标坐标原点在相机坐标中的平移，单位由 `translation_unit` 给出；
- `extrinsics.coordinate_convention`：必须是 `object_to_camera`；
- `extrinsics.distance_to_object_center`：可选，标定板原点偏移到晶体中心后计算出的距离；
- `calibration`：重投影误差、标定板规格、接受/剔除视图等审计信息。

当前项目官方 ChArUco 标定板规格为 5 列×7 行、方格 30 mm、marker 15 mm、`DICT_5X5_100`，打印页为 A4 竖版。参数文件中的 `calibration.pattern` 必须与实际打印并拍摄的标定板一致；旧板或仅旋转图片都不能通过交换参数来“修正”。

内参不是曝光参数：自动曝光、增益和白平衡通常可以变化，但必须保证图像清晰、不过曝、不欠曝。焦距/变焦、对焦位置、分辨率、ROI、数字裁剪或 binning 改变后，不能继续无条件使用同一份内参。光圈建议固定在生产值；如果改光圈引起重新对焦或镜头组件移动，应重新验证。

后端公制换算统一把外参距离转成米，再把结果输出为 cm 和 cm³；不允许前端另行读取或修改这些字段。
