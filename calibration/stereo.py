"""双相机内参固定的立体标定。

左右目录按同名文件配对，例如 ``0001.png`` 对 ``0001.png``。先分别完成
左右相机内参标定，再用同一块标定板的同步图像估计左相机到右相机的 R/T。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .intrinsics import collect_images
from .patterns import BoardSpec, detect_pattern
from .schema import load_parameters, save_parameters


def _pair_images(left_dir: str | Path, right_dir: str | Path) -> list[tuple[Path, Path]]:
    left = {path.name: path for path in collect_images(left_dir, recursive=False)}
    right = {path.name: path for path in collect_images(right_dir, recursive=False)}
    names = sorted(set(left) & set(right))
    if not names:
        raise RuntimeError("左右标定目录没有同名图片，无法配对。")
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))
    if missing_left or missing_right:
        raise RuntimeError(
            "左右标定图片集合不一致；请确保两侧使用完全相同的文件名。"
            f" 右侧多出 {missing_left[:3]}，左侧多出 {missing_right[:3]}"
        )
    return [(left[name], right[name]) for name in names]


def _read_image(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取标定图片: {path}")
    return image


def _aligned_points(left, right):
    """返回一对观测中真正对应的 object/image points。"""
    if left.point_ids is None or right.point_ids is None:
        if len(left.image_points) != len(right.image_points):
            return None
        return left.object_points, left.image_points, right.image_points

    left_map = {int(point_id): index for index, point_id in enumerate(left.point_ids)}
    right_map = {int(point_id): index for index, point_id in enumerate(right.point_ids)}
    common = [point_id for point_id in left.point_ids if int(point_id) in right_map]
    if len(common) < 6:
        return None
    left_indices = [left_map[int(point_id)] for point_id in common]
    right_indices = [right_map[int(point_id)] for point_id in common]
    return (
        left.object_points[left_indices],
        left.image_points[left_indices],
        right.image_points[right_indices],
    )


def calibrate_stereo(
    left_dir: str | Path,
    right_dir: str | Path,
    left_parameters: str | Path,
    right_parameters: str | Path,
    spec: BoardSpec,
    output: str | Path,
    min_views: int = 5,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    """使用已标定内参完成双目标定并保存统一 JSON。"""
    left_payload = load_parameters(left_parameters)
    right_payload = load_parameters(right_parameters)
    pairs = _pair_images(left_dir, right_dir)

    left_camera = left_payload["camera"]
    right_camera = right_payload["camera"]
    left_size = (int(left_camera["image_width"]), int(left_camera["image_height"]))
    right_size = (int(right_camera["image_width"]), int(right_camera["image_height"]))
    if left_size != right_size:
        raise ValueError(
            f"当前实现要求左右图像尺寸一致，左侧 {left_size}，右侧 {right_size}。"
        )

    camera_matrix_left = np.asarray(left_camera["camera_matrix"], dtype=np.float64)
    distortion_left = np.asarray(left_camera.get("distortion_coeffs", []), dtype=np.float64)
    camera_matrix_right = np.asarray(right_camera["camera_matrix"], dtype=np.float64)
    distortion_right = np.asarray(right_camera.get("distortion_coeffs", []), dtype=np.float64)

    object_points: list[np.ndarray] = []
    image_points_left: list[np.ndarray] = []
    image_points_right: list[np.ndarray] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    debug_path = Path(debug_dir).expanduser().resolve() if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    for left_path, right_path in pairs:
        left_image = _read_image(left_path)
        right_image = _read_image(right_path)
        if (left_image.shape[1], left_image.shape[0]) != left_size:
            rejected.append({"name": left_path.name, "reason": "left_size_mismatch"})
            continue
        if (right_image.shape[1], right_image.shape[0]) != right_size:
            rejected.append({"name": left_path.name, "reason": "right_size_mismatch"})
            continue

        left_detection = detect_pattern(
            left_image, spec, include_debug=True,
            camera_matrix=camera_matrix_left, distortion=distortion_left,
        )
        right_detection = detect_pattern(
            right_image, spec, include_debug=True,
            camera_matrix=camera_matrix_right, distortion=distortion_right,
        )
        if left_detection is None or right_detection is None:
            rejected.append({"name": left_path.name, "reason": "pattern_not_detected"})
            continue
        aligned = _aligned_points(left_detection, right_detection)
        if aligned is None:
            rejected.append({"name": left_path.name, "reason": "insufficient_common_points"})
            continue
        obj, img_left, img_right = aligned
        object_points.append(np.asarray(obj, dtype=np.float32))
        image_points_left.append(np.asarray(img_left, dtype=np.float32))
        image_points_right.append(np.asarray(img_right, dtype=np.float32))
        accepted.append({"name": left_path.name, "point_count": int(len(obj))})

        if debug_path and left_detection.debug_image is not None and right_detection.debug_image is not None:
            cv2.imwrite(str(debug_path / f"{left_path.stem}_left.png"), left_detection.debug_image)
            cv2.imwrite(str(debug_path / f"{left_path.stem}_right.png"), right_detection.debug_image)

    if len(object_points) < max(int(min_views), 3):
        raise RuntimeError(
            f"双目标定有效配对图像只有 {len(object_points)} 张，至少需要 {max(int(min_views), 3)} 张。"
        )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        200,
        1e-8,
    )
    rms, matrix_left, distortion_left, matrix_right, distortion_right, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        camera_matrix_left.copy(),
        distortion_left.copy(),
        camera_matrix_right.copy(),
        distortion_right.copy(),
        left_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    payload = {
        "schema_version": 1,
        # 顶层 camera 保留左相机，因而该文件仍可被现有单相机读取器安全检查。
        "camera": left_payload["camera"],
        "right_camera": right_payload["camera"],
        "extrinsics": left_payload.get("extrinsics", []),
        "stereo": {
            "left_parameters": str(Path(left_parameters).expanduser().resolve()),
            "right_parameters": str(Path(right_parameters).expanduser().resolve()),
            "image_size": list(left_size),
            "rotation_matrix_left_to_right": np.asarray(rotation, dtype=float).tolist(),
            "translation_vector_left_to_right": np.asarray(translation, dtype=float).reshape(-1).tolist(),
            "essential_matrix": np.asarray(essential, dtype=float).tolist(),
            "fundamental_matrix": np.asarray(fundamental, dtype=float).tolist(),
            "translation_unit": spec.length_unit,
            "coordinate_convention": "left_camera_to_right_camera",
            "reprojection_error_px": float(rms),
            "accepted_views": accepted,
            "rejected_views": rejected,
            "pattern": {
                "type": spec.pattern_type,
                "size": list(spec.pattern_size),
                "square_size": spec.square_size,
                "circle_distance": spec.circle_distance,
                "marker_length": spec.marker_length,
                "dictionary": spec.dictionary,
                "length_unit": spec.length_unit,
            },
        },
        "calibration": {
            "type": "stereoCalibrate",
            "flags": ["CALIB_FIX_INTRINSIC"],
            "reprojection_error_px": float(rms),
            "paired_view_count": len(accepted),
            "rejected_view_count": len(rejected),
        },
    }
    save_parameters(output, payload)
    return payload


__all__ = ["calibrate_stereo"]
