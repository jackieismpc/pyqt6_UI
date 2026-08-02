# -*- coding: utf-8 -*-
"""标定子项目使用的相机参数 JSON schema。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def load_parameters(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"不支持的相机参数 schema_version: {payload.get('schema_version')}")
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("参数文件缺少 camera 对象")
    matrix = np.asarray(camera.get("camera_matrix"), dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("camera.camera_matrix 必须是 3x3 矩阵")
    if int(camera.get("image_width", 0)) <= 0 or int(camera.get("image_height", 0)) <= 0:
        raise ValueError("标定图像尺寸必须为正数")
    return payload


def save_parameters(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    load_parameters(output_path)
    return output_path


def make_intrinsics_payload(
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    image_size: tuple[int, int],
    extrinsics: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "camera": {
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
            "camera_matrix": np.asarray(camera_matrix, dtype=float).reshape(3, 3).tolist(),
            "distortion_coeffs": np.asarray(distortion_coeffs, dtype=float).reshape(-1).tolist(),
            "distortion_model": "opencv_radtan",
        },
        "extrinsics": extrinsics,
        "calibration": calibration,
    }
