# -*- coding: utf-8 -*-
"""像素几何 <-> 公制几何换算（尺度锚点法）。

单目单视角无法直接得到绝对尺度，必须提供一个「尺度锚点」：
已知某一条边的真实长度（如 length = 40 cm），即可得到 单位/像素 比例，
再把其余像素尺寸线性换算成真实尺寸，并按体积公式得到真实体积。

这是第二阶段「模式 A：scale_anchor」的核心，只依赖内参无关的一条真值边长，
因此在只有内参、尚无外参时也能运行。
"""

from __future__ import annotations

from typing import Dict, Optional

from .calibration import length_unit_to_meter
from .geometry import compute_volume, volume_breakdown

# 尺度锚点边名 -> 像素参数键
_EDGE_TO_PX_KEY = {
    "length": "length_px",
    "width": "width_px",
    "body_height": "body_height_px",
    "pyramid_height": "pyramid_height_px",
    "total_height": "total_height_px",
}


def convert_pixel_to_metric(
    geometry_params_px: Dict[str, float],
    scale_reference_edge: str,
    scale_reference_value: float,
    metric_length_unit: str = "cm",
    ground_truth: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """把像素几何换算到公制。

    参数
    ----
    geometry_params_px : 含 length_px/width_px/body_height_px/pyramid_height_px/total_height_px
    scale_reference_edge : 用哪条边做锚点（length/width/body_height/pyramid_height/total_height）
    scale_reference_value : 该边的真实长度（单位由 metric_length_unit 指定）
    metric_length_unit : 真实长度单位（mm/cm/m）
    ground_truth : 可选真值 {length,width,body_height,pyramid_height}，用于误差对比

    返回
    ----
    含公制尺寸、volume_m3、尺度信息与误差对比的字典。
    """
    px_key = _EDGE_TO_PX_KEY.get(scale_reference_edge)
    if px_key is None:
        raise ValueError(f"未知的尺度锚点边: {scale_reference_edge}")
    reference_px = float(geometry_params_px.get(px_key, 0.0))
    if reference_px <= 1e-6:
        raise ValueError(f"尺度锚点像素长度无效（{px_key}={reference_px}），无法换算。")

    unit_to_meter = length_unit_to_meter(metric_length_unit)
    unit_per_px = float(scale_reference_value) / reference_px  # 真实单位/像素

    params_unit = {
        "length": float(geometry_params_px.get("length_px", 0.0)) * unit_per_px,
        "width": float(geometry_params_px.get("width_px", 0.0)) * unit_per_px,
        "body_height": float(geometry_params_px.get("body_height_px", 0.0)) * unit_per_px,
        "pyramid_height": float(geometry_params_px.get("pyramid_height_px", 0.0)) * unit_per_px,
    }
    params_unit["total_height"] = params_unit["body_height"] + params_unit["pyramid_height"]
    params_m = {k: v * unit_to_meter for k, v in params_unit.items()}

    volume_unit3 = compute_volume(
        params_unit["length"], params_unit["width"],
        params_unit["body_height"], params_unit["pyramid_height"],
    )
    volume_m3 = compute_volume(
        params_m["length"], params_m["width"],
        params_m["body_height"], params_m["pyramid_height"],
    )

    result: Dict[str, object] = {
        "units": metric_length_unit,
        "scale_reference": {
            "edge": scale_reference_edge,
            "real_value": float(scale_reference_value),
            "pixel_value": reference_px,
            "unit_per_px": unit_per_px,
        },
        "geometry_params_unit": {f"{k}_{metric_length_unit}": v for k, v in params_unit.items()},
        "geometry_params_m": {f"{k}_m": v * unit_to_meter for k, v in params_unit.items()},
        "volume_unit3": volume_unit3,
        "volume_m3": volume_m3,
        "volume_breakdown_m3": volume_breakdown(
            params_m["length"], params_m["width"],
            params_m["body_height"], params_m["pyramid_height"],
        ),
    }

    if ground_truth:
        gt = {k: float(v) for k, v in ground_truth.items() if v is not None}
        errors = {}
        for key, gt_value in gt.items():
            est = float(params_unit.get(key, 0.0))
            abs_err = est - gt_value
            errors[key] = {
                "estimated": est,
                "ground_truth": gt_value,
                "abs_error": abs_err,
                "rel_error_pct": abs(abs_err) / max(abs(gt_value), 1e-9) * 100.0,
            }
        result["comparison"] = {"dimension_errors": errors}
        if {"length", "width", "body_height", "pyramid_height"}.issubset(gt):
            gt_vol_unit3 = compute_volume(gt["length"], gt["width"], gt["body_height"], gt["pyramid_height"])
            gt_vol_m3 = gt_vol_unit3 * (unit_to_meter ** 3)
            result["comparison"]["volume"] = {
                "estimated_m3": volume_m3,
                "ground_truth_m3": gt_vol_m3,
                "abs_error_m3": volume_m3 - gt_vol_m3,
                "rel_error_pct": abs(volume_m3 - gt_vol_m3) / max(abs(gt_vol_m3), 1e-12) * 100.0,
            }
    return result
