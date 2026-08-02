# -*- coding: utf-8 -*-
"""使用已标定内参对单张标定板图片求外参。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .patterns import BoardSpec, detect_pattern
from .schema import load_parameters, save_parameters


def _camera_matrix_for_image(payload: dict[str, Any], image_size: tuple[int, int]):
    camera = payload["camera"]
    calibrated_width = int(camera["image_width"])
    calibrated_height = int(camera["image_height"])
    width, height = image_size
    if abs(calibrated_width / calibrated_height - width / height) > 1e-3:
        raise RuntimeError(
            f"图片宽高比 {width}x{height} 与内参标定尺寸 "
            f"{calibrated_width}x{calibrated_height} 不一致"
        )
    matrix = np.asarray(camera["camera_matrix"], dtype=np.float64).reshape(3, 3).copy()
    matrix[0, 0] *= width / calibrated_width
    matrix[0, 2] *= width / calibrated_width
    matrix[1, 1] *= height / calibrated_height
    matrix[1, 2] *= height / calibrated_height
    distortion = np.asarray(camera.get("distortion_coeffs", []), dtype=np.float64).reshape(-1, 1)
    return matrix, distortion


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法保存调试图: {path}")
    path.write_bytes(encoded.tobytes())


def _rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return matrix


def _reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points.astype(np.float64), rvec, tvec, camera_matrix, distortion
    )
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def calibrate_extrinsic(
    image_path: str | Path,
    parameters_path: str | Path,
    spec: BoardSpec,
    output: str | Path,
    *,
    pose_method: str = "iterative",
    refine_pose: bool = True,
    object_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    expected_distance: float | None = None,
    distance_tolerance: float | None = None,
    append: bool = False,
    debug_output: str | Path | None = None,
) -> dict[str, Any]:
    image_file = Path(image_path).expanduser().resolve()
    try:
        with image_file.open("rb") as handle:
            encoded = np.frombuffer(handle.read(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, MemoryError, cv2.error) as exc:
        raise RuntimeError(f"无法读取外参图片: {image_file} ({exc})") from exc
    if image is None:
        raise RuntimeError(f"无法读取外参图片: {image_file}")
    payload = load_parameters(parameters_path)
    camera_matrix, distortion = _camera_matrix_for_image(
        payload, (int(image.shape[1]), int(image.shape[0]))
    )
    detection = detect_pattern(
        image,
        spec,
        camera_matrix=camera_matrix,
        distortion=distortion,
    )
    if detection is None:
        raise RuntimeError("外参图片中没有检测到完整的标定板")
    object_points = detection.object_points.astype(np.float64)
    image_points = detection.image_points.astype(np.float64)

    method = pose_method.lower()
    flags = {
        "iterative": cv2.SOLVEPNP_ITERATIVE,
        "ippe": cv2.SOLVEPNP_IPPE,
    }
    if method not in {"iterative", "ippe", "ransac"}:
        raise ValueError("pose-method 只支持 iterative、ippe、ransac")
    if method == "ransac":
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=4.0,
            confidence=0.995,
            iterationsCount=200,
        )
        inlier_count = int(len(inliers)) if inliers is not None else 0
    else:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=flags[method],
        )
        inlier_count = int(len(object_points))
    if not success:
        raise RuntimeError("solvePnP 无法求出有效外参")

    if refine_pose and hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rvec,
            tvec,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-10),
        )

    rotation = _rotation_matrix(rvec)
    center_offset = np.asarray(object_center, dtype=np.float64).reshape(3)
    camera_center_target = -rotation.T @ tvec.reshape(3)
    object_center_camera = rotation @ center_offset + tvec.reshape(3)
    object_center_distance = float(np.linalg.norm(object_center_camera))
    reprojection_error = _reprojection_error(
        object_points, image_points, camera_matrix, distortion, rvec, tvec
    )
    distance_error = None
    if expected_distance is not None:
        distance_error = object_center_distance - float(expected_distance)
        tolerance = float(distance_tolerance) if distance_tolerance is not None else max(1.0, abs(float(expected_distance)) * 0.05)
        if abs(distance_error) > tolerance:
            raise RuntimeError(
                f"外参距离校验失败：估计 {object_center_distance:.3f} {spec.length_unit}，"
                f"期望 {expected_distance:.3f}，允许误差 {tolerance:.3f}"
            )

    debug = detection.debug_image
    axis_length = max(float(spec.square_size or spec.circle_distance or 1.0) * 2.0, 1.0)
    cv2.drawFrameAxes(
        debug,
        camera_matrix,
        distortion,
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        axis_length,
        2,
    )
    if debug_output:
        _write_image(Path(debug_output).expanduser().resolve(), debug)

    extrinsic = {
        "id": image_file.stem,
        "source_image": str(image_file),
        "rotation_matrix": rotation.astype(float).tolist(),
        "rotation_vector": np.asarray(rvec, dtype=float).reshape(-1).tolist(),
        "translation_vector": np.asarray(tvec, dtype=float).reshape(-1).tolist(),
        "translation_unit": spec.length_unit,
        "coordinate_convention": "object_to_camera",
        "camera_center": camera_center_target.astype(float).tolist(),
        "object_center_offset": center_offset.astype(float).tolist(),
        "distance_to_object_center": object_center_distance,
        "reprojection_error_px": reprojection_error,
        "detected_point_count": int(len(object_points)),
        "inlier_point_count": inlier_count,
    }
    output_payload = dict(payload)
    output_payload["extrinsics"] = list(payload.get("extrinsics", [])) if append else []
    output_payload["extrinsics"].append(extrinsic)
    output_payload["calibration"] = {
        **dict(payload.get("calibration", {})),
        "last_extrinsic_calibration": {
            "method": method,
            "target_type": spec.pattern_type,
            "pattern_size": list(spec.pattern_size),
            "square_size": spec.square_size,
            "marker_length": spec.marker_length,
            "dictionary": spec.dictionary,
            "charuco_pattern": "modern" if spec.pattern_type == "charuco" else None,
            "charuco_interpolation": (
                "pose_reprojection_with_intrinsics" if spec.pattern_type == "charuco" else None
            ),
            "marker_corner_refinement": "none" if spec.pattern_type == "charuco" else None,
            "reprojection_error_px": reprojection_error,
            "expected_distance": expected_distance,
            "distance_error": distance_error,
        },
    }
    save_parameters(output, output_payload)
    return output_payload
