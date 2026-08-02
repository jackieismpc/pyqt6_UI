# -*- coding: utf-8 -*-
"""使用 OpenCV 官方 calib3d API 标定相机内参。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .patterns import BoardSpec, PatternDetection, detect_pattern
from .schema import make_intrinsics_payload, save_parameters


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CalibrationView:
    path: Path
    detection: PatternDetection


def collect_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"标定图片目录不存在: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"目录中没有支持的标定图片: {root}")
    return paths


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return image


def detect_views(paths: list[Path], spec: BoardSpec) -> tuple[list[CalibrationView], tuple[int, int], list[str]]:
    views: list[CalibrationView] = []
    rejected: list[str] = []
    image_size: tuple[int, int] | None = None
    for path in paths:
        image = _read_image(path)
        size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise RuntimeError(
                f"标定图片分辨率不一致：{path.name} 为 {size}，期望 {image_size}。"
            )
        detection = detect_pattern(image, spec)
        if detection is None:
            rejected.append(str(path))
            continue
        views.append(CalibrationView(path, detection))
    if image_size is None:
        raise RuntimeError("没有可读取的标定图片")
    return views, image_size, rejected


def _calibration_flags(
    model: str,
    fix_aspect_ratio: bool,
    zero_tangent_dist: bool,
    fix_principal_point: bool,
) -> int:
    flags = 0
    if model == "rational":
        flags |= cv2.CALIB_RATIONAL_MODEL
    if fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    return flags


def _run_calibration(
    views: list[CalibrationView],
    image_size: tuple[int, int],
    flags: int,
    max_iterations: int,
    epsilon: float,
):
    object_points = [view.detection.object_points.astype(np.float32) for view in views]
    image_points = [view.detection.image_points.astype(np.float32) for view in views]
    camera_matrix = None
    if flags & (cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT):
        camera_matrix = cv2.initCameraMatrix2D(object_points, image_points, image_size, 0)
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
    result = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        None,
        flags=flags,
        criteria=criteria,
    )
    reprojection_error, matrix, distortion, rvecs, tvecs, _, _, per_view_errors = result
    return (
        float(reprojection_error),
        matrix,
        distortion,
        rvecs,
        tvecs,
        np.asarray(per_view_errors, dtype=float).reshape(-1),
    )


def _rotation_matrix(rvec: np.ndarray) -> list[list[float]]:
    matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return matrix.astype(float).tolist()


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法保存调试图: {path}")
    path.write_bytes(encoded.tobytes())


def calibrate_intrinsics(
    image_dir: str | Path,
    spec: BoardSpec,
    output: str | Path,
    *,
    recursive: bool = False,
    model: str = "standard",
    fix_aspect_ratio: bool = False,
    zero_tangent_dist: bool = False,
    fix_principal_point: bool = False,
    reject_outliers: bool = True,
    max_view_error: float = 2.0,
    max_rounds: int = 3,
    min_views: int = 5,
    max_iterations: int = 100,
    epsilon: float = 1e-7,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    if model not in {"standard", "rational"}:
        raise ValueError("model 只支持 standard 或 rational")
    if min_views < 3:
        raise ValueError("min-views 不能小于 3")
    paths = collect_images(image_dir, recursive=recursive)
    views, image_size, detection_rejected = detect_views(paths, spec)
    if len(views) < min_views:
        raise RuntimeError(f"有效标定图只有 {len(views)} 张，至少需要 {min_views} 张")

    if debug_dir:
        debug_root = Path(debug_dir).expanduser().resolve()
        for view in views:
            _write_image(debug_root / view.path.name, view.detection.debug_image)

    flags = _calibration_flags(model, fix_aspect_ratio, zero_tangent_dist, fix_principal_point)
    rejected_outliers: list[str] = []
    calibration_result = None
    for round_index in range(max_rounds + 1):
        calibration_result = _run_calibration(
            views, image_size, flags, max_iterations=max_iterations, epsilon=epsilon
        )
        errors = calibration_result[-1]
        if not reject_outliers or round_index >= max_rounds:
            break
        bad_indices = [index for index, error in enumerate(errors) if float(error) > max_view_error]
        if not bad_indices:
            break
        if len(views) - len(bad_indices) < min_views:
            break
        bad_set = set(bad_indices)
        rejected_outliers.extend(str(views[index].path) for index in bad_indices)
        views = [view for index, view in enumerate(views) if index not in bad_set]

    if calibration_result is None:
        raise RuntimeError("内参标定没有生成结果")
    reprojection_error, matrix, distortion, rvecs, tvecs, per_view_errors = calibration_result
    extrinsics: list[dict[str, Any]] = []
    for index, (view, rvec, tvec, error) in enumerate(zip(views, rvecs, tvecs, per_view_errors)):
        extrinsics.append(
            {
                "id": f"view-{index + 1:02d}",
                "source_image": str(view.path),
                "rotation_matrix": _rotation_matrix(rvec),
                "rotation_vector": np.asarray(rvec, dtype=float).reshape(-1).tolist(),
                "translation_vector": np.asarray(tvec, dtype=float).reshape(-1).tolist(),
                "translation_unit": spec.length_unit,
                "coordinate_convention": "object_to_camera",
                "reprojection_error_px": float(error),
                "detected_point_count": int(view.detection.image_points.shape[0]),
            }
        )

    calibration_metadata = {
        "method": "opencv_calibrateCameraExtended",
        "model": model,
        "pattern": {
            "type": spec.pattern_type,
            "pattern_size": list(spec.pattern_size),
            "square_size": spec.square_size,
            "circle_distance": spec.circle_distance,
            "marker_length": spec.marker_length,
            "dictionary": spec.dictionary,
            "length_unit": spec.length_unit,
        },
        "reprojection_error_px": reprojection_error,
        "per_view_errors_px": [float(value) for value in per_view_errors],
        "accepted_images": [str(view.path) for view in views],
        "detection_rejected_images": detection_rejected,
        "outlier_rejected_images": rejected_outliers,
        "flags": int(flags),
    }
    payload = make_intrinsics_payload(
        matrix, distortion, image_size, extrinsics, calibration_metadata
    )
    save_parameters(output, payload)
    return payload
