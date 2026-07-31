# -*- coding: utf-8 -*-
"""
crystalvol —— 透明晶体体积估计核心包。

整体分两阶段：

- 第一阶段（像素域 / stage1）：
  从视频、图片目录或单张图片中提取「最大的那个透明晶体」的轮廓边缘，
  拟合预定义几何体「长方体 + 四棱锥（屋顶形）」，输出像素域标准几何与线框可视化。

- 第二阶段（公制域 / stage2）：
  借助相机标定参数，把第一阶段的像素几何恢复到真实尺度，输出真实边长与真实体积（m^3）。

模块职责一览：

- config           所有可调参数的数据类（Stage1Config / Stage2Config / 各子配置）
- logging_utils    统一日志输出
- device           计算设备自动选择（CUDA / MPS / CPU）
- io               输入读取（单图/目录/视频抽帧）与输出目录、文件命名规范
- calibration      相机内参 JSON 读取与按分辨率缩放
- geometry         标准几何（长方体+四棱锥）顶点/棱/体积公式、像素几何聚合
- metric           像素 <-> 公制换算（尺度锚点）
- preprocess       低光增强、高光/镜面抑制、暗部遮挡处理（针对透明晶体拍摄难点）
- edges            边缘提取：传统 Canny/LSD + 深度 PiDiNet/HED，融合与自动择优
- segmentation     YOLO-World + SAM2 分割（只保留最大晶体）
- silhouette       由融合边缘/掩膜提取「最大连通域」晶体剪影
- wireframe        长方体+四棱锥线框拟合（主线族/角点/几何硬约束 + 兜底）
- visualize        统一可视化：轮廓图 / overlay / 三维几何预览
- stage1           第一阶段编排
- stage2           第二阶段框架（scale_anchor 可运行；extrinsic_multiview 为骨架）
- reference_target 参考板检测→外参（第二阶段 extrinsic_multiview 使用）
- multiview        多视角联合几何拟合后端（第二阶段 extrinsic_multiview 使用）
"""

__version__ = "2.0.0"
