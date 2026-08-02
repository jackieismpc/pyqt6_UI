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
    try:
        with path.open("rb") as handle:
            encoded = np.frombuffer(handle.read(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, MemoryError, cv2.error) as exc:
        raise RuntimeError(f"无法读取图片: {path} ({exc})") from exc
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return image


def detect_views(
    paths: list[Path],
    spec: BoardSpec,
    include_debug: bool = False,
) -> tuple[list[CalibrationView], tuple[int, int], list[str]]:
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
        detection = detect_pattern(image, spec, include_debug=include_debug)
        if detection is None:
            rejected.append(str(path))
            continue
        views.append(CalibrationView(path, detection))
    if image_size is None:
        raise RuntimeError("没有可读取的标定图片")
    return views, image_size, rejected


def _swapped_charuco_spec(spec: BoardSpec) -> BoardSpec | None:
    """返回交换列/行后的 ChArUco 规格，用于提示板型方向写反。"""
    if spec.pattern_type != "charuco" or spec.pattern_size[0] == spec.pattern_size[1]:
        return None
    return BoardSpec(
        pattern_type=spec.pattern_type,
        pattern_size=(spec.pattern_size[1], spec.pattern_size[0]),
        square_size=spec.square_size,
        circle_distance=spec.circle_distance,
        marker_length=spec.marker_length,
        dictionary=spec.dictionary,
        length_unit=spec.length_unit,
    )


def _detection_failure_message(
    spec: BoardSpec,
    paths: list[Path],
    views: list[CalibrationView],
    rejected: list[str],
    min_views: int,
) -> str:
    columns, rows = spec.pattern_size
    message = (
        f"有效标定图只有 {len(views)}/{len(paths)} 张，至少需要 {min_views} 张；"
        f"当前规格为 {spec.pattern_type} {columns}x{rows}"
    )
    if rejected:
        sample = ", ".join(Path(path).name for path in rejected[:3])
        message += f"，未检测到 {len(rejected)} 张（示例：{sample}）"
    if spec.pattern_type == "charuco":
        message += "。ChArUco 的 pattern-size 表示方格数，顺序是列x行"
    return message


def _calibration_flags(
    model: str,
    fix_aspect_ratio: bool,
    zero_tangent_dist: bool,
    fix_principal_point: bool,
    fix_focal_length: bool,
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
    if fix_focal_length:
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    return flags


def _physical_initial_camera_matrix(
    image_size: tuple[int, int],
    focal_length_mm: float,
    pixel_size_um: float,
) -> tuple[np.ndarray, float]:
    """Convert an approximate physical focal length into an OpenCV initial K."""
    if focal_length_mm <= 0:
        raise ValueError("focal-length-mm 必须为正数")
    if pixel_size_um <= 0:
        raise ValueError("pixel-size-um 必须为正数")
    focal_length_px = float(focal_length_mm) * 1000.0 / float(pixel_size_um)
    width, height = image_size
    matrix = np.array(
        [
            [focal_length_px, 0.0, (width - 1) / 2.0],
            [0.0, focal_length_px, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix, focal_length_px


def _run_calibration(
    views: list[CalibrationView],
    image_size: tuple[int, int],
    flags: int,
    max_iterations: int,
    epsilon: float,
    initial_camera_matrix: np.ndarray | None = None,
):
    object_points = [view.detection.object_points.astype(np.float32) for view in views]
    image_points = [view.detection.image_points.astype(np.float32) for view in views]
    camera_matrix = None
    if initial_camera_matrix is not None:
        camera_matrix = np.asarray(initial_camera_matrix, dtype=np.float64).copy()
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    elif flags & (cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT):
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
    focal_length_mm: float | None = None,
    pixel_size_um: float | None = None,
    fix_focal_length: bool = False,
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
    if (focal_length_mm is None) != (pixel_size_um is None):
        raise ValueError("--focal-length-mm 和 --pixel-size-um 必须同时提供")
    if fix_focal_length and focal_length_mm is None:
        raise ValueError("固定物理焦距时必须同时提供 focal-length-mm 和 pixel-size-um")
    if min_views < 3:
        raise ValueError("min-views 不能小于 3")
    paths = collect_images(image_dir, recursive=recursive)
    views, image_size, detection_rejected = detect_views(
        paths, spec, include_debug=debug_dir is not None
    )
    if len(views) < min_views:
        message = _detection_failure_message(spec, paths, views, detection_rejected, min_views)
        swapped_spec = _swapped_charuco_spec(spec)
        if swapped_spec is not None and not views:
            swapped_views, _, _ = detect_views(paths, swapped_spec, include_debug=False)
            if swapped_views:
                swapped_columns, swapped_rows = swapped_spec.pattern_size
                message += (
                    f"；检测到 {len(swapped_views)} 张图片与交换后的规格 "
                    f"{swapped_columns}x{swapped_rows} 匹配，请检查并显式使用 "
                    f"--pattern-size {swapped_columns}x{swapped_rows}"
                )
        raise RuntimeError(message)

    if debug_dir:
        debug_root = Path(debug_dir).expanduser().resolve()
        for view in views:
            if view.detection.debug_image is not None:
                _write_image(debug_root / view.path.name, view.detection.debug_image)
                # 调试图已经落盘，不要在标定优化期间继续保留每张原尺寸图像。
                view.detection.debug_image = None

    flags = _calibration_flags(
        model,
        fix_aspect_ratio,
        zero_tangent_dist,
        fix_principal_point,
        fix_focal_length,
    )
    initial_camera_matrix = None
    initial_focal_length_px = None
    if focal_length_mm is not None and pixel_size_um is not None:
        initial_camera_matrix, initial_focal_length_px = _physical_initial_camera_matrix(
            image_size, focal_length_mm, pixel_size_um
        )
    rejected_outliers: list[str] = []
    calibration_result = None
    for round_index in range(max_rounds + 1):
        calibration_result = _run_calibration(
            views,
            image_size,
            flags,
            max_iterations=max_iterations,
            epsilon=epsilon,
            initial_camera_matrix=initial_camera_matrix,
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
        "method": (
            "opencv_CharucoDetector+Board.matchImagePoints+calibrateCameraExtended"
            if spec.pattern_type == "charuco"
            else "opencv_calibrateCameraExtended"
        ),
        "model": model,
        "pattern": {
            "type": spec.pattern_type,
            "pattern_size": list(spec.pattern_size),
            "square_size": spec.square_size,
            "circle_distance": spec.circle_distance,
            "marker_length": spec.marker_length,
            "dictionary": spec.dictionary,
            "length_unit": spec.length_unit,
            "charuco_pattern": "modern" if spec.pattern_type == "charuco" else None,
            "opencv_size_order": "columns(x) x rows(y)",
            "charuco_interpolation": (
                "homography_without_intrinsics" if spec.pattern_type == "charuco" else None
            ),
            "marker_corner_refinement": "none" if spec.pattern_type == "charuco" else None,
        },
        "reprojection_error_px": reprojection_error,
        "per_view_errors_px": [float(value) for value in per_view_errors],
        "accepted_images": [str(view.path) for view in views],
        "detection_rejected_images": detection_rejected,
        "outlier_rejected_images": rejected_outliers,
        "flags": int(flags),
    }
    if focal_length_mm is not None and pixel_size_um is not None:
        calibration_metadata["focal_length_constraint"] = {
            "physical_focal_length_mm": float(focal_length_mm),
            "pixel_size_um": float(pixel_size_um),
            "initial_focal_length_px": float(initial_focal_length_px),
            "mode": "fixed" if fix_focal_length else "initial_guess",
            "estimated_focal_length_mm": {
                "fx": float(matrix[0, 0]) * float(pixel_size_um) / 1000.0,
                "fy": float(matrix[1, 1]) * float(pixel_size_um) / 1000.0,
            },
        }
    payload = make_intrinsics_payload(
        matrix, distortion, image_size, extrinsics, calibration_metadata
    )
    save_parameters(output, payload)
    return payload
