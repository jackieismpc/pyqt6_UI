# -*- coding: utf-8 -*-
"""统一的相机参数模型、加载优先级和像素到公制换算。

参数文件只支持当前版本的 ``camera_parameters.json`` schema。文件优先级为：

1. 调用方显式传入的路径；
2. ``CRYSTAL_CAMERA_PARAMETERS`` 环境变量；
3. 项目根目录 ``params/camera_parameters.json``；
4. 后端内置默认文件 ``crystalvol/defaults/camera_parameters.json``。

前端不直接解析参数文件，所有矩阵、外参和尺度换算都通过本模块完成。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMETERS_PATH = Path(__file__).resolve().parent / "defaults" / "camera_parameters.json"
PARAMETERS_ENV = "CRYSTAL_CAMERA_PARAMETERS"

LENGTH_UNIT_TO_METER = {
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
}


def length_unit_to_meter(unit_name: str) -> float:
    key = (unit_name or "").strip().lower()
    if key not in LENGTH_UNIT_TO_METER:
        supported = ", ".join(sorted(LENGTH_UNIT_TO_METER))
        raise ValueError(f"不支持的长度单位: {unit_name}；当前支持: {supported}")
    return float(LENGTH_UNIT_TO_METER[key])


def _as_matrix(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} 必须是 {shape[0]}x{shape[1]} 矩阵，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限数值")
    return array


def _as_vector(value: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} 必须包含 {size} 个数值，实际为 {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限数值")
    return array


@dataclass(frozen=True)
class ExtrinsicParameters:
    """一组目标坐标系到相机坐标系的外参。"""

    identifier: str
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    translation_unit: str
    coordinate_convention: str
    source_image: Optional[str] = None
    distance_to_object_center: Optional[float] = None

    @property
    def translation_m(self) -> np.ndarray:
        return self.translation_vector * length_unit_to_meter(self.translation_unit)

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.translation_m))

    @property
    def camera_center_m(self) -> np.ndarray:
        """相机中心在目标坐标系中的位置。"""
        return -self.rotation_matrix.T @ self.translation_m


@dataclass(frozen=True)
class CameraParameters:
    """统一的相机内外参对象。"""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    image_width: int
    image_height: int
    distortion_model: str
    extrinsics: tuple[ExtrinsicParameters, ...]
    reprojection_error: Optional[float]
    source_path: str
    payload: dict[str, Any]

    def select_extrinsic(self, index: int = 0) -> ExtrinsicParameters:
        if not self.extrinsics:
            raise RuntimeError("当前相机参数没有可用外参，请先完成单图外参标定。")
        if index < 0 or index >= len(self.extrinsics):
            raise IndexError(f"外参索引超出范围: {index}，当前共有 {len(self.extrinsics)} 组外参")
        return self.extrinsics[index]


# 与旧的 Stage2 内部命名保持一致，但不再接受旧文件格式。
CameraCalibration = CameraParameters


def resolve_camera_parameters_path(
    explicit_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """按统一优先级解析参数文件路径。"""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"显式指定的相机参数文件不存在: {path}")
        return path

    env_path = os.environ.get(PARAMETERS_ENV)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{PARAMETERS_ENV} 指向的文件不存在: {path}")
        return path

    root = Path(project_root).expanduser().resolve() if project_root else PROJECT_ROOT
    project_path = root / "params" / "camera_parameters.json"
    if project_path.is_file():
        return project_path
    if DEFAULT_PARAMETERS_PATH.is_file():
        return DEFAULT_PARAMETERS_PATH
    raise FileNotFoundError(
        "没有找到相机参数：请提供显式路径、设置 CRYSTAL_CAMERA_PARAMETERS，"
        "或创建 params/camera_parameters.json。"
    )


def _parse_extrinsics(payload: dict[str, Any]) -> tuple[ExtrinsicParameters, ...]:
    values = payload.get("extrinsics", [])
    if not isinstance(values, list):
        raise ValueError("extrinsics 必须是数组")
    result: list[ExtrinsicParameters] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"extrinsics[{index}] 必须是对象")
        unit = str(item.get("translation_unit", "mm"))
        length_unit_to_meter(unit)
        convention = str(item.get("coordinate_convention", "object_to_camera"))
        if convention != "object_to_camera":
            raise ValueError(
                f"extrinsics[{index}] 使用了不支持的坐标约定: {convention}"
            )
        result.append(
            ExtrinsicParameters(
                identifier=str(item.get("id", f"view-{index + 1:02d}")),
                source_image=item.get("source_image"),
                rotation_matrix=_as_matrix(
                    item.get("rotation_matrix"),
                    f"extrinsics[{index}].rotation_matrix",
                    (3, 3),
                ),
                translation_vector=_as_vector(
                    item.get("translation_vector"),
                    f"extrinsics[{index}].translation_vector",
                    3,
                ),
                translation_unit=unit,
                coordinate_convention=convention,
                distance_to_object_center=(
                    float(item["distance_to_object_center"])
                    if item.get("distance_to_object_center") is not None
                    else None
                ),
            )
        )
    return tuple(result)


def load_camera_parameters(
    path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> CameraParameters:
    file_path = resolve_camera_parameters_path(path, project_root)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"不支持的相机参数 schema_version: {payload.get('schema_version')}")

    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("相机参数缺少 camera 对象")
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    if width <= 0 or height <= 0:
        raise ValueError("标定图像尺寸必须为正数")
    matrix = _as_matrix(camera["camera_matrix"], "camera.camera_matrix", (3, 3))
    distortion = _as_vector(
        camera.get("distortion_coeffs", [0, 0, 0, 0, 0]),
        "camera.distortion_coeffs",
        len(camera.get("distortion_coeffs", [0, 0, 0, 0, 0])),
    )
    if matrix[2, 2] == 0 or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera_matrix 的焦距和齐次项必须为正数")

    calibration = payload.get("calibration", {})
    reprojection_error = calibration.get("reprojection_error_px")
    return CameraParameters(
        camera_matrix=matrix,
        distortion_coeffs=distortion,
        image_width=width,
        image_height=height,
        distortion_model=str(camera.get("distortion_model", "opencv_radtan")),
        extrinsics=_parse_extrinsics(payload),
        reprojection_error=(float(reprojection_error) if reprojection_error is not None else None),
        source_path=str(file_path),
        payload=payload,
    )


def load_camera_calibration(path: str | Path | None = None) -> CameraParameters:
    return load_camera_parameters(path)


def resolve_camera_matrix_for_image(
    calibration: CameraParameters,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按分辨率缩放内参；不同宽高比视为裁剪或成像模式变化。"""
    calibrated_ratio = calibration.image_width / max(calibration.image_height, 1)
    runtime_ratio = int(image_width) / max(int(image_height), 1)
    if abs(calibrated_ratio - runtime_ratio) > 1e-3:
        raise RuntimeError(
            "当前图像宽高比与标定分辨率不一致，疑似裁剪或不同成像模式，不能直接复用内参。"
        )
    scale_x = float(image_width) / calibration.image_width
    scale_y = float(image_height) / calibration.image_height
    matrix = calibration.camera_matrix.copy()
    matrix[0, 0] *= scale_x
    matrix[0, 2] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[1, 2] *= scale_y
    return matrix, calibration.distortion_coeffs.copy()


def save_camera_parameters(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    load_camera_parameters(output_path)
    return output_path


def save_camera_calibration(
    path: str | Path,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
    reprojection_error: Optional[float] = None,
    extrinsics: Optional[list[dict[str, Any]]] = None,
    calibration: Optional[dict[str, Any]] = None,
) -> Path:
    metadata = dict(calibration or {})
    if reprojection_error is not None:
        metadata["reprojection_error_px"] = float(reprojection_error)
    payload = {
        "schema_version": 1,
        "camera": {
            "image_width": int(image_width),
            "image_height": int(image_height),
            "camera_matrix": np.asarray(camera_matrix, dtype=float).reshape(3, 3).tolist(),
            "distortion_coeffs": np.asarray(distortion_coeffs, dtype=float).reshape(-1).tolist(),
            "distortion_model": "opencv_radtan",
        },
        "extrinsics": list(extrinsics or []),
        "calibration": metadata,
    }
    return save_camera_parameters(path, payload)


def pinhole_pixel_to_cm(
    pixel_geometry: dict[str, float],
    camera_params: CameraParameters,
    extrinsic_index: int = 0,
) -> dict[str, object]:
    """用选定外参的目标距离做针孔换算，返回 cm 制结果。"""
    ext = camera_params.select_extrinsic(extrinsic_index)
    distance_m = ext.distance_m
    fx = float(camera_params.camera_matrix[0, 0])
    fy = float(camera_params.camera_matrix[1, 1])
    if distance_m <= 0 or fx <= 0 or fy <= 0:
        return {}

    cm_per_px_x = distance_m * 100.0 / fx
    cm_per_px_y = distance_m * 100.0 / fy
    length_cm = float(pixel_geometry.get("length_px", 0.0)) * cm_per_px_x
    width_cm = float(pixel_geometry.get("width_px", 0.0)) * cm_per_px_x
    body_cm = float(pixel_geometry.get("body_height_px", 0.0)) * cm_per_px_y
    pyramid_cm = float(pixel_geometry.get("pyramid_height_px", 0.0)) * cm_per_px_y
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
            "distance_m": distance_m,
            "focal_length_px": {"fx": fx, "fy": fy},
            "cm_per_px": {"x": cm_per_px_x, "y": cm_per_px_y},
            "extrinsic_id": ext.identifier,
            "extrinsic_index": extrinsic_index,
        },
    }


def apply_scale_anchor_correction(
    metric: dict[str, object],
    edge: str,
    real_value_cm: float,
) -> dict[str, object]:
    dims = metric.get("dimensions_cm", {})
    if not isinstance(dims, dict) or edge not in dims:
        return metric
    estimated_value = float(dims[edge])
    if estimated_value <= 1e-6:
        return metric
    correction = float(real_value_cm) / estimated_value
    corrected_dims = {key: float(value) * correction for key, value in dims.items()}
    length = corrected_dims["length"]
    width = corrected_dims["width"]
    body_height = corrected_dims["body_height"]
    pyramid_height = corrected_dims["pyramid_height"]
    volume_cm3 = length * width * (body_height + pyramid_height / 3.0)
    return {
        **metric,
        "volume": volume_cm3,
        "volume_m3": volume_cm3 * 1e-6,
        "dimensions_cm": corrected_dims,
        "scale_info": {
            **(metric.get("scale_info", {})),
            "corrected_by": "scale_anchor",
            "anchor_edge": edge,
            "anchor_value_cm": float(real_value_cm),
            "correction_factor": correction,
        },
    }


def camera_parameters_summary(parameters: CameraParameters) -> dict[str, object]:
    return {
        "source_path": parameters.source_path,
        "image_size": [parameters.image_width, parameters.image_height],
        "distortion_model": parameters.distortion_model,
        "reprojection_error_px": parameters.reprojection_error,
        "extrinsics": [
            {
                "id": item.identifier,
                "source_image": item.source_image,
                "distance_m": item.distance_m,
            }
            for item in parameters.extrinsics
        ],
    }
