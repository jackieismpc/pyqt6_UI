"""
相机参数配置模块。

解析 camera_params.txt 文件，提供 CameraConfig 数据类和
pinhole 模型换算函数（像素 → 公制）。

假定：晶体置于标定板坐标系原点附近，外参 t 的模 |t| 为相机到晶体的近似距离。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExtrinsicSet:
    """一组外参（旋转矩阵 R + 平移向量 t，OpenCV 约定：X_cam = R @ X_world + t）。"""

    index: int
    r: list[list[float]]  # 3×3
    t: list[float]        # 3×1，单位 mm


@dataclass
class CameraParams:
    """从 camera_params.txt 解析出的完整参数。"""

    k: list[list[float]]           # 3×3 内参矩阵
    distortion: list[float]         # [k1,k2,p1,p2,k3]
    extrinsics: list[ExtrinsicSet]  # 12 组外参
    reprojection_error: float


@dataclass
class CameraConfig:
    """传递给后端/UI 的相机运行配置。"""

    mode: str = "monocular"            # monocular | binocular
    extrinsic_index: int = 0           # 选哪组外参（0-based）
    scale_anchor_value: Optional[float] = None  # 可选的尺度锚点校正值（cm）
    scale_anchor_edge: str = "length"          # 锚点对应的边


from .platform_utils import default_camera_params_path

# ---- 解析 camera_params.txt ----


def parse_camera_params(path: str | Path | None = None) -> CameraParams:
    """解析 camera_params.txt 并返回 CameraParams。"""
    filepath = Path(path).expanduser() if path else default_camera_params_path()
    text = filepath.read_text(encoding="utf-8")

    # 重投影误差
    reproj_match = re.search(r"重投影误差:\s*([\d.]+)", text)
    reproj = float(reproj_match.group(1)) if reproj_match else 0.0

    # 内参矩阵 K —— 逐行匹配
    k_match = re.search(
        r"内参矩阵 K.*?"
        r"\[\[([\d\s.\-]+)\]\s*\n\s*"
        r"\[([\d\s.\-]+)\]\s*\n\s*"
        r"\[([\d\s.\-]+)\]\]",
        text, re.DOTALL,
    )
    if not k_match:
        raise ValueError("未找到相机内参矩阵 K")
    k_rows = [
        [float(x) for x in row.split() if x]
        for row in k_match.groups()
    ]

    # 畸变系数
    dist_match = re.search(
        r"畸变系数.*?\n\[\[([\d.\-\s]+?)\]\]", text
    )
    dist = [0.0] * 5
    if dist_match:
        dist_vals = [float(x) for x in dist_match.group(1).split() if x]
        for i, v in enumerate(dist_vals[:5]):
            dist[i] = v

    # 外参：按标定图分块，逐块提取 R(3行) + t
    extrinsics: list[ExtrinsicSet] = []
    blocks = re.split(r"第\s*\d+\s*张标定图", text)[1:]
    for idx, block in enumerate(blocks):
        r_match = re.search(
            r"旋转矩阵 R.*?"
            r"\[\[([\d\s.\-]+)\]\s*\n\s*"
            r"\[([\d\s.\-]+)\]\s*\n\s*"
            r"\[([\d\s.\-]+)\]\]",
            block, re.DOTALL,
        )
        t_match = re.search(
            r"平移向量 t.*?\n\[\[([\d\s.\-]+?)\]\]", block
        )
        if not r_match or not t_match:
            continue

        r_mat = [
            [float(x) for x in row.split() if x]
            for row in r_match.groups()
        ]
        t_flat = [float(x) for x in t_match.group(1).split() if x]

        extrinsics.append(ExtrinsicSet(index=idx, r=r_mat, t=t_flat))

    return CameraParams(
        k=k_rows,
        distortion=dist,
        extrinsics=extrinsics,
        reprojection_error=reproj,
    )


# ---- Pinhole 模型换算 ----
def pinhole_pixel_to_cm(
    pixel_geometry: dict[str, float],
    camera_params: CameraParams,
    extrinsic_index: int = 0,
) -> dict[str, object]:
    """用选定外参的 |t| 做 pinhole 近似，把像素几何换算为 cm 制。

    换算公式：
        real_mm = pixel * |t| / fx
        real_cm = real_mm / 10

    假定外参 t 单位是 mm，晶体位于世界坐标系原点附近。
    """
    if not camera_params.extrinsics:
        return {}

    ext = camera_params.extrinsics[
        min(extrinsic_index, len(camera_params.extrinsics) - 1)
    ]
    fx = camera_params.k[0][0]
    distance_mm = math.sqrt(sum(v * v for v in ext.t))

    if fx <= 0 or distance_mm <= 0:
        return {}

    # mm_per_pixel = distance_mm / fx
    mm_per_px = distance_mm / fx
    cm_per_px = mm_per_px / 10.0

    length_cm = pixel_geometry.get("length_px", 0.0) * cm_per_px
    width_cm = pixel_geometry.get("width_px", 0.0) * cm_per_px
    body_cm = pixel_geometry.get("body_height_px", 0.0) * cm_per_px
    pyramid_cm = pixel_geometry.get("pyramid_height_px", 0.0) * cm_per_px
    total_cm = body_cm + pyramid_cm
    volume_cm3 = length_cm * width_cm * (body_cm + pyramid_cm / 3.0)

    return {
        "volume": volume_cm3,
        "volume_m3": volume_cm3 * 1e-6,
        "unit": "cm³",
        "dimensions_cm": {
            "length": length_cm,
            "width": width_cm,
            "body_height": body_cm,
            "pyramid_height": pyramid_cm,
            "total_height": total_cm,
        },
        "scale_info": {
            "method": "pinhole",
            "distance_mm": distance_mm,
            "focal_length_px": fx,
            "cm_per_px": cm_per_px,
            "extrinsic_index": ext.index,
        },
    }


def apply_scale_anchor_correction(
    metric: dict[str, object],
    edge: str,
    real_value_cm: float,
) -> dict[str, object]:
    """在 pinhole 结果上叠加尺度锚点校正。

    用已知真实边长重新计算尺度因子，覆盖 pinhole 估计。
    """
    dims = metric.get("dimensions_cm", {})
    if not isinstance(dims, dict) or edge not in dims:
        return metric

    estimated_val = float(dims[edge])
    if estimated_val <= 1e-6:
        return metric

    correction = real_value_cm / estimated_val
    corrected_dims = {k: v * correction for k, v in dims.items()}

    L = corrected_dims["length"]
    W = corrected_dims["width"]
    Hb = corrected_dims["body_height"]
    Hp = corrected_dims["pyramid_height"]
    volume_cm3 = L * W * (Hb + Hp / 3.0)

    return {
        **metric,
        "volume": volume_cm3,
        "volume_m3": volume_cm3 * 1e-6,
        "dimensions_cm": corrected_dims,
        "scale_info": {
            **(metric.get("scale_info", {})),
            "corrected_by": "scale_anchor",
            "anchor_edge": edge,
            "anchor_value_cm": real_value_cm,
            "correction_factor": correction,
        },
    }
