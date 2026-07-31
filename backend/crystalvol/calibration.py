# -*- coding: utf-8 -*-
"""相机内参读取与按分辨率缩放。

内参 JSON 约定格式：
{
  "image_width": 5120,
  "image_height": 5120,
  "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
  "distortion_coeffs": [k1,k2,p1,p2,k3]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# 长度单位 -> 米
LENGTH_UNIT_TO_METER = {
    "mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3,
    "cm": 1e-2, "centimeter": 1e-2, "centimeters": 1e-2,
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
}


def length_unit_to_meter(unit_name: str) -> float:
    """把长度单位换算成米的比例因子。"""
    key = (unit_name or "").strip().lower()
    if key not in LENGTH_UNIT_TO_METER:
        supported = ", ".join(sorted(set(LENGTH_UNIT_TO_METER)))
        raise ValueError(f"不支持的长度单位: {unit_name}；当前支持: {supported}")
    return float(LENGTH_UNIT_TO_METER[key])


@dataclass
class CameraCalibration:
    """相机内参与标定分辨率。"""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    image_width: int
    image_height: int
    source_path: str


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    """读取内参 JSON。"""
    file_path = Path(path).expanduser().resolve()
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    matrix = np.array(payload["camera_matrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.array(
        payload.get("distortion_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64
    ).reshape(-1)
    return CameraCalibration(
        camera_matrix=matrix,
        distortion_coeffs=distortion,
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        source_path=str(file_path),
    )


def resolve_camera_matrix_for_image(
    calibration: CameraCalibration, image_width: int, image_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """按当前图像尺寸等比缩放内参，便于复用同一相机的一次标定结果。

    只允许等比缩放；若长宽比不一致（裁剪/ROI/binning），应重新标定。
    """
    calibrated_ratio = float(calibration.image_width) / max(float(calibration.image_height), 1.0)
    runtime_ratio = float(image_width) / max(float(image_height), 1.0)
    if abs(calibrated_ratio - runtime_ratio) > 1e-3:
        raise RuntimeError(
            "当前图像长宽比与标定分辨率不一致，疑似裁剪或不同成像模式，"
            "不能直接复用内参，请重新标定。"
        )
    scale_x = float(image_width) / float(calibration.image_width)
    scale_y = float(image_height) / float(calibration.image_height)
    matrix = calibration.camera_matrix.copy().astype(np.float64)
    matrix[0, 0] *= scale_x
    matrix[0, 2] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[1, 2] *= scale_y
    return matrix, calibration.distortion_coeffs.copy().astype(np.float64)


def save_camera_calibration(
    path: str | Path,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
    reprojection_error: Optional[float] = None,
) -> Path:
    """把内参标定结果写成统一 JSON（供 tools/calibrate_camera.py 使用）。"""
    payload = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
        "distortion_coeffs": np.asarray(distortion_coeffs, dtype=float).reshape(-1).tolist(),
    }
    if reprojection_error is not None:
        payload["reprojection_error"] = float(reprojection_error)
    output_path = Path(path).expanduser().resolve()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
