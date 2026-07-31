"""
数据模型定义模块。

本模块用 dataclass 描述后端产物 stage1_result.json 反序列化后的内存结构，
并提供基于经验公式的置信度估计（用于 UI 展示，不代表最终算法置信度）。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameResult:
    """单帧的处理结果，对应 stage1_result.json 中 frames 列表的一个元素。"""

    name: str  # 帧名，如 "frame_01"，对应 json 中 frames[i].name
    backend: str  # 该帧使用的边缘检测后端，对应 frames[i].backend
    fit_ready: bool  # 该帧结构证据是否足够用于拟合，对应 frames[i].fit_ready
    visible_ratio: float  # 关键边可见比例(0~1)，越高越可信，对应 frames[i].visible_ratio
    coverage_ratio: float  # 覆盖率，对应 frames[i].coverage_ratio
    volume_px3: float  # 该帧估计的像素域体积，对应 frames[i].geometry_px.volume_px3
    geometry: dict  # 该帧完整的几何字典，对应 frames[i].geometry_px（保留全部键）
    warnings: list  # 告警信息列表，对应 frames[i].warnings
    images: dict  # 该帧关联的产物图片绝对路径字典
    # images 的键含义：
    #   'raw'      -> inputs/<name>.png              原始输入帧
    #   'enhanced' -> enhanced/<name>_enhanced.png    低光增强
    #   'edges'    -> edges/<name>_edge.png           边缘证据图
    #   'mask'     -> masks/<name>_mask.png           剪影掩膜
    #   'overlay'  -> overlays/<name>_overlay.png     线框叠加回增强图
    #   'contour'  -> contours/<name>_wireframe.png   轮廓提取图
    # 若对应产物文件不存在，则该键的值为 None。


@dataclass
class Stage1Result:
    """stage1 阶段的汇总结果，对应 stage1_result.json 顶层结构。"""

    input: str  # 输入路径描述，对应顶层 input
    frame_count: int  # 参与处理的总帧数，对应顶层 frame_count
    fit_ready_count: int  # 结构证据足够的帧数，对应顶层 fit_ready_count
    edge_backend: str  # 边缘检测后端名称，对应顶层 edge_backend
    consensus_frames: list  # 达成共识的帧名列表，对应顶层 consensus_frames
    consensus_frame_count: int  # 达成共识的帧数，对应顶层 consensus_frame_count
    representative_frame: str  # 代表帧名，对应顶层 representative_frame
    aggregate_volume_px3: float  # 汇总(共识)像素域体积，对应顶层 geometry_px.volume_px3
    aggregate_geometry: dict  # 汇总几何字典，对应顶层 geometry_px（保留全部键）
    metric: Optional[dict]  # 公制结果（真实体积等），未接入时为 None，对应顶层 metric
    geometry_preview: Optional[str]  # 三维几何重建预览图绝对路径
    # (geometry/standard_geometry_pixel_preview.png)，该图是汇总单图，所有帧共用
    frames: list = field(default_factory=list)  # FrameResult 列表，对应顶层 frames，顺序与 json 一致


def frame_confidence(fr: FrameResult) -> tuple:
    """
    计算单帧置信度。

    经验公式：score = 0.6 * visible_ratio + (0.4 if fit_ready else 0.0)
    返回 (标签, 百分比)，标签为 "高"/"中"/"低"。
    """
    score = 0.6 * fr.visible_ratio + (0.4 if fr.fit_ready else 0.0)
    pct = round(score * 100)
    if pct >= 66:
        label = "高"
    elif pct >= 40:
        label = "中"
    else:
        label = "低"
    return label, pct


def aggregate_confidence(res: Stage1Result) -> tuple:
    """
    计算整体（汇总）置信度。

    经验公式：
        fit_frac = fit_ready_count / max(frame_count, 1)
        rep = 代表帧的 visible_ratio（找不到代表帧时取 0）
        cons_frac = consensus_frame_count / max(frame_count, 1)
        score = 0.4 * fit_frac + 0.4 * rep + 0.2 * cons_frac
    返回 (标签, 百分比)。
    """
    fit_frac = res.fit_ready_count / max(res.frame_count, 1)

    rep = 0.0
    for fr in res.frames:
        if fr.name == res.representative_frame:
            rep = fr.visible_ratio
            break

    cons_frac = res.consensus_frame_count / max(res.frame_count, 1)

    score = 0.4 * fit_frac + 0.4 * rep + 0.2 * cons_frac
    pct = round(score * 100)
    if pct >= 66:
        label = "高"
    elif pct >= 40:
        label = "中"
    else:
        label = "低"
    return label, pct


def confidence_color(label: str) -> str:
    """根据置信度标签返回浅色主题下的徽章底色（十六进制颜色字符串）。"""
    mapping = {
        "高": "#34c759",  # Apple 系统绿
        "中": "#ff9500",  # Apple 系统橙
        "低": "#ff3b30",  # Apple 系统红
    }
    return mapping.get(label, "#86868b")
