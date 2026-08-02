# -*- coding: utf-8 -*-
"""全流程配置数据类。

所有可调参数集中在这里定义并给出默认值；命令行参数（crystalvol/cli.py）只是
把用户输入映射到这些配置对象。这样「参数含义」只有一处真源，README 也按此说明。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# 默认权重路径：backend/weights/（与代码仓库统一管理，完全离线）
_BACKEND_ROOT = Path(__file__).resolve().parent.parent  # backend/
_DEFAULT_WORLD_MODEL = str(_BACKEND_ROOT / "weights" / "yolov8s-worldv2.pt")
_DEFAULT_SAM2_CKPT = str(_BACKEND_ROOT / "weights" / "sam2.1_hiera_tiny.pt")
_DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_DEFAULT_DEEP_WEIGHTS_DIR = str(_BACKEND_ROOT / "weights")

# YOLO-World 默认类别提示词（透明晶体 / 矿石 / 石英）
DEFAULT_WORLD_CLASSES: List[str] = [
    "crystal",
    "transparent crystal",
    "quartz crystal",
    "mineral",
    "glass block",
]


@dataclass
class PreprocessConfig:
    """低光增强 + 高光/镜面抑制配置。

    针对透明晶体拍摄的三大难点：整体偏暗、强高光反射、暗部遮挡。
    """

    lowlight_mode: str = "auto"      # auto | on | off；auto 时按整图平均亮度自动开启
    lowlight_luma_threshold: float = 90.0   # 平均亮度低于该值判为暗图（auto 触发阈值）
    clahe_clip: float = 3.0          # CLAHE 对比度限制
    clahe_grid: int = 8              # CLAHE 网格尺寸
    gamma: float = 0.55              # gamma 提亮系数（<1 提亮暗部）
    use_msr: bool = False            # 是否叠加多尺度 Retinex（更强但更慢）
    denoise: bool = True             # 增强后是否做一次轻度去噪（抑制水珠/散景噪点）

    specular_percentile: float = 99.0   # 高光分位数：高于该分位视为镜面高光
    specular_min_value: int = 240       # 高光绝对亮度下限
    specular_dilate: int = 9            # 高光掩膜膨胀核，用于抑制高光附近的伪边缘


@dataclass
class LocalizeConfig:
    """晶体定位（显著性 ROI）配置。

    目的：先在整幅里把「最大晶体」定位出来并裁一个自适应 ROI，再在 ROI 内做
    边缘/分割/线框。这样从占画面 1% 的小晶体到占大半幅的大晶体都能处理，
    而不是在整幅上用同一套绝对阈值（小晶体会被黑条/暗角/反光淹没）。

    显著性 = 边缘密度 × 亮度（背光晶体更亮）× 中心权重，并抑制：
    - 竖直黑色遮挡条（低亮度大块）
    - 暗角/边框
    - 纯镜面高光（亮但内部结构少）
    """

    enable: bool = True
    edge_density_ksize_ratio: float = 0.03   # 边缘密度滤波核相对短边比例
    center_weight: float = 0.35              # 中心权重强度（0 关闭；越大越偏向画面中央）
    border_margin_ratio: float = 0.04        # 边框忽略比例（抑制暗角/边缘伪显著）
    specular_suppress: float = 0.7           # 纯高光区显著性压制强度（0~1）
    saliency_std_k: float = 1.8              # 阈值 = mean + k*std
    min_blob_area_ratio: float = 0.0004      # 有效晶体块最小面积占比（滤噪点）
    max_blob_area_ratio: float = 0.75        # 超过则视为背景/黑条，不作为晶体
    max_aspect_ratio: float = 6.0            # 长宽比过大（细长黑条）判为非晶体
    roi_pad_ratio: float = 0.6               # ROI 相对晶体块外扩比例
    min_roi_side_ratio: float = 0.12         # ROI 最短边相对整图短边的下限（给小晶体足够上下文）
    fullframe_area_ratio: float = 0.45       # 晶体块超过该占比 -> 直接用整幅（大晶体）
    large_regime_area_ratio: float = 0.06    # 显著性总面积超过该占比判为「大晶体」：改用大核闭运算合并碎块
    # （实测：小晶体视频帧总显著面积≈0.03-0.04，大/低对比晶体 test.jpg≈0.095，取 0.06 分界）


@dataclass
class EdgeConfig:
    """边缘提取配置。

    backend 可选：
    - auto    ：同时尝试 canny 与 pidinet(+canny)，按线框质量自动择优（推荐）
    - pidinet ：深度边缘 PiDiNet（对反光/暗图鲁棒，需权重，可自动下载）
    - hed     ：深度边缘 HED（备选深度模型）
    - canny   ：仅传统 Canny（完全离线、零下载）
    - lsd     ：仅直线段检测（主要用于直棱线）
    """

    backend: str = "auto"
    fuse_canny: bool = True          # 深度边缘是否与 Canny 融合（按位或）
    deep_threshold: float = 0.12     # 深度边缘概率图二值化阈值（0~1）
    deep_input_max_side: int = 1024  # 深度模型推理前缩放的最长边（CPU 提速）
    deep_repo: str = _DEFAULT_DEEP_WEIGHTS_DIR  # PiDiNet/HED 权重本地目录（weights/）；首次需放入文件
    canny_low: int = 24
    canny_high: int = 72
    device: str = "auto"


@dataclass
class SegmentationConfig:
    """YOLO-World + SAM2 分割配置（只保留最大晶体）。"""

    enable: bool = True              # 关闭时纯靠边缘提取剪影
    use_yolo: bool = False           # 是否用 YOLO-World 选框（需联网下载 CLIP，透明物体常检不到；
                                     # 默认 False：直接用中心先验框 + 中心正点提示 SAM2，更稳更省依赖）
    world_model_path: str = _DEFAULT_WORLD_MODEL
    sam2_checkpoint_path: str = _DEFAULT_SAM2_CKPT
    sam2_config_path: str = _DEFAULT_SAM2_CFG
    world_classes: List[str] = field(default_factory=lambda: list(DEFAULT_WORLD_CLASSES))
    world_conf: float = 0.05         # YOLO-World 置信度阈值（透明物体低置信，放宽）
    world_iou: float = 0.5
    box_expand_ratio: float = 0.08
    center_fallback_ratio: float = 0.6   # YOLO 无框时的中心先验框比例（透明物体常检不到）
    min_mask_area_ratio: float = 0.01    # 掩膜最小面积占比
    morphology_kernel: int = 7
    device: str = "auto"


@dataclass
class WireframeConfig:
    """长方体 + 四棱锥线框拟合配置。"""

    # 主线（Hough）提取
    hough_threshold: int = 80
    min_line_length_ratio: float = 0.06  # 直线最小长度相对短边比例
    max_line_gap_ratio: float = 0.01
    # 线族聚类
    vertical_angle_tol: float = 18.0     # 竖直棱线角度容差（deg）
    horizontal_angle_tol: float = 22.0   # 水平棱线角度容差（deg）
    # 单视角深度启发式：width = depth_ratio * length（无法观测侧向深度时的默认比例）
    depth_ratio: float = 0.9             # 宽度/长度比；典型晶体接近正方形截面（0.8–1.0）
    # 四棱锥屋顶检测
    pyramid_top_width_ratio: float = 0.7  # 顶部宽度 < 该比例×满宽 才认为有屋顶（越小越严格）
    pyramid_min_taper_ratio: float = 0.06 # 渐缩段高度 ≥ 该比例×总高 才认为有屋顶
    min_pyramid_fraction: float = 0.15     # 未检测到屋顶时的最小棱锥高度比例（保底，避免平顶）
    # 亮核 + 边缘紧化：把真实晶体从背光光晕里收紧
    core_percentile: float = 40.0        # 前景内亮度分位，高于它视为晶体亮核（越大收得越紧；
                                        # 透明晶体边缘偏暗，40 比 55 保留更多本体像素）
    # 采用门槛
    min_visible_ratio: float = 0.5       # 达到该可见比例才判为 fit_ready
    min_coverage_ratio: float = 0.06     # 剪影相对 ROI 面积的最小占比（滤掉碎块）
    # 兜底：即使门槛未过，也强制输出当前最优结果（三张图一定有）
    force_emit_best: bool = True


@dataclass
class MetricAnchorConfig:
    """尺度锚点：用一条已知真实边长把像素几何换算到公制（第二阶段模式 A）。"""

    scale_reference_edge: Optional[str] = None   # length|width|body_height|pyramid_height|total_height
    scale_reference_value: Optional[float] = None
    metric_length_unit: str = "cm"
    # 可选真值，用于误差对比
    gt_length: Optional[float] = None
    gt_width: Optional[float] = None
    gt_body_height: Optional[float] = None
    gt_pyramid_height: Optional[float] = None


@dataclass
class Stage1Config:
    """第一阶段（像素域）总配置。"""

    input_path: str
    output_dir: str = "outputs/stage1"
    clean_output: bool = False
    device: str = "auto"

    # 输入处理
    max_input_side: int = 2304       # 超大图先等比缩放到该最长边再处理
    num_frames: int = 7              # 视频均匀抽帧数
    frame_start_ratio: float = 0.0
    frame_end_ratio: float = 1.0

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    localize: LocalizeConfig = field(default_factory=LocalizeConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    wireframe: WireframeConfig = field(default_factory=WireframeConfig)
    metric_anchor: MetricAnchorConfig = field(default_factory=MetricAnchorConfig)


@dataclass
class Stage2Config:
    """第二阶段（公制域）总配置。"""

    stage1_geometry_json: str        # 第一阶段产物 standard_geometry_pixel.json
    camera_calibration: str          # 相机内参 JSON（当前已有）
    output_dir: str = "outputs/stage2"
    mode: str = "auto"               # auto | scale_anchor | extrinsic_multiview

    metric_anchor: MetricAnchorConfig = field(default_factory=MetricAnchorConfig)

    # extrinsic_multiview 模式所需（当前缺外参，暂不可用）
    turntable_config: Optional[str] = None
    angles_file: Optional[str] = None
    expected_volume_min_m3: float = 0.1
    expected_volume_max_m3: float = 1.5
