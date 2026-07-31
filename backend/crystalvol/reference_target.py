#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reference_targets.py

这个模块负责：
1. 读取参考板/转台配置；
2. 支持几种常见标定板的位姿检测；
3. 返回统一的参考板位姿结果，供后续几何拟合使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import length_unit_to_meter as get_length_unit_to_meter_scale

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None


SUPPORTED_TARGET_TYPES = {
    "chessboard",
    "circles_grid",
    "asymmetric_circles_grid",
    "aruco_single",
    "aruco_board",
    "charuco_board",
    "apriltag",
}


@dataclass
class ReferenceTargetConfig:
    """保存参考板配置与转台几何先验。"""

    target_type: str
    length_unit: str
    mount_mode: str
    object_offset_xyz_m: Tuple[float, float, float]
    angle_offset_deg: float
    pattern_size: Optional[Tuple[int, int]] = None
    square_size_m: Optional[float] = None
    circle_distance_m: Optional[float] = None
    dictionary_name: Optional[str] = None
    marker_length_m: Optional[float] = None
    marker_separation_m: Optional[float] = None
    marker_ids: Optional[List[int]] = None


@dataclass
class ReferencePoseResult:
    """保存一次参考板检测结果。"""

    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    exclusion_polygon: np.ndarray
    debug_image: np.ndarray
    debug_info: Dict[str, object]


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV 不可用，无法检测参考板: {CV2_IMPORT_ERROR}")


def _resolve_aruco_dictionary(name: str):
    _require_cv2()
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 不包含 aruco 模块，请安装 opencv-contrib-python。")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"不支持的字典名称: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _build_planar_grid_points(
    pattern_size: Tuple[int, int],
    step_x: float,
    step_y: float,
    asymmetric: bool,
) -> np.ndarray:
    columns, rows = pattern_size
    points = []
    for row in range(rows):
        for column in range(columns):
            if asymmetric:
                x_value = (2 * column + (row % 2)) * step_x
            else:
                x_value = column * step_x
            points.append([x_value, row * step_y, 0.0])
    return np.array(points, dtype=np.float32)


def load_reference_target_config(path: str | Path) -> ReferenceTargetConfig:
    """读取转台/参考板 JSON 配置。"""

    file_path = Path(path).expanduser().resolve()
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    target_type = str(payload["target_type"]).strip().lower()
    if target_type not in SUPPORTED_TARGET_TYPES:
        raise ValueError(f"不支持的参考板类型: {target_type}")

    unit_name = str(payload.get("length_unit", "m"))
    unit_scale = get_length_unit_to_meter_scale(unit_name)
    object_offset = payload.get("object_offset_xyz", [0.0, 0.0, 0.0])

    marker_ids = payload.get("marker_ids")
    if marker_ids is not None:
        marker_ids = [int(value) for value in marker_ids]

    pattern_size = payload.get("pattern_size")
    if pattern_size is not None:
        pattern_size = (int(pattern_size[0]), int(pattern_size[1]))

    square_size = payload.get("square_size")
    circle_distance = payload.get("circle_distance")
    marker_length = payload.get("marker_length")
    marker_separation = payload.get("marker_separation")

    return ReferenceTargetConfig(
        target_type=target_type,
        length_unit=unit_name,
        mount_mode=str(payload.get("mount_mode", "fixed_world")).strip().lower(),
        object_offset_xyz_m=tuple(float(value) * unit_scale for value in object_offset),
        angle_offset_deg=float(payload.get("angle_offset_deg", 0.0)),
        pattern_size=pattern_size,
        square_size_m=(float(square_size) * unit_scale if square_size is not None else None),
        circle_distance_m=(float(circle_distance) * unit_scale if circle_distance is not None else None),
        dictionary_name=payload.get("dictionary"),
        marker_length_m=(float(marker_length) * unit_scale if marker_length is not None else None),
        marker_separation_m=(float(marker_separation) * unit_scale if marker_separation is not None else None),
        marker_ids=marker_ids,
    )


def save_reference_target_config(path: str | Path, payload: Dict[str, object]) -> Path:
    """把转台/参考板配置保存为 JSON。"""

    output_path = Path(path).expanduser().resolve()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def detect_reference_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    config: ReferenceTargetConfig,
) -> Optional[ReferencePoseResult]:
    """根据配置自动选择参考板检测方式。"""

    _require_cv2()
    method = config.target_type
    if method == "chessboard":
        return _detect_chessboard_pose(image_bgr, camera_matrix, distortion_coeffs, config)
    if method in {"circles_grid", "asymmetric_circles_grid"}:
        return _detect_circle_grid_pose(image_bgr, camera_matrix, distortion_coeffs, config)
    if method in {"aruco_single", "apriltag"}:
        return _detect_single_marker_pose(image_bgr, camera_matrix, distortion_coeffs, config)
    if method == "aruco_board":
        return _detect_aruco_board_pose(image_bgr, camera_matrix, distortion_coeffs, config)
    if method == "charuco_board":
        return _detect_charuco_pose(image_bgr, camera_matrix, distortion_coeffs, config)
    raise ValueError(f"未实现的参考板类型: {method}")


def _draw_pose_axes(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    rotation_vector: np.ndarray,
    translation_vector: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(
            canvas,
            camera_matrix.astype(np.float64),
            distortion_coeffs.astype(np.float64),
            rotation_vector.astype(np.float64).reshape(3, 1),
            translation_vector.astype(np.float64).reshape(3, 1),
            float(axis_length),
            2,
        )
    return canvas


def _build_exclusion_polygon(points_2d: np.ndarray) -> np.ndarray:
    """把参考板检测点转成一个稳定的凸包区域。"""

    points = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float32)
    hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float32)
    return hull


def build_reference_exclusion_mask(
    image_shape: Tuple[int, int],
    polygon: np.ndarray,
    scale_ratio: float = 0.18,
    min_padding_px: int = 12,
) -> np.ndarray:
    """根据参考板多边形生成需要剔除的区域掩膜。"""

    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if polygon.shape[0] < 3:
        return mask

    polygon_int = np.round(polygon).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon_int, 255, lineType=cv2.LINE_AA)

    x_value, y_value, box_w, box_h = cv2.boundingRect(polygon_int.reshape(-1, 1, 2))
    padding = max(int(round(max(box_w, box_h) * float(scale_ratio))), int(min_padding_px))
    kernel_size = max(3, padding * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _detect_chessboard_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    config: ReferenceTargetConfig,
) -> Optional[ReferencePoseResult]:
    if config.pattern_size is None or config.square_size_m is None:
        raise ValueError("棋盘格需要提供 pattern_size 与 square_size。")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, config.pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not found:
        return None

    object_points = _build_planar_grid_points(config.pattern_size, config.square_size_m, config.square_size_m, asymmetric=False)
    success, rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float64),
        corners.reshape(-1, 2).astype(np.float64),
        camera_matrix.astype(np.float64),
        distortion_coeffs.astype(np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    debug_image = image_bgr.copy()
    cv2.drawChessboardCorners(debug_image, config.pattern_size, corners, found)
    debug_image = _draw_pose_axes(
        debug_image,
        camera_matrix,
        distortion_coeffs,
        rvec,
        tvec,
        axis_length=max(config.square_size_m * 2.0, 0.02),
    )
    return ReferencePoseResult(
        rotation_matrix=rotation_matrix.astype(np.float64),
        translation_vector=tvec.reshape(3).astype(np.float64),
        exclusion_polygon=_build_exclusion_polygon(corners.reshape(-1, 2)),
        debug_image=debug_image,
        debug_info={
            "target_type": "chessboard",
            "pattern_size": list(config.pattern_size),
            "corners": corners.reshape(-1, 2).astype(float).tolist(),
        },
    )


def _detect_circle_grid_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    config: ReferenceTargetConfig,
) -> Optional[ReferencePoseResult]:
    if config.pattern_size is None or config.circle_distance_m is None:
        raise ValueError("圆点板需要提供 pattern_size 与 circle_distance。")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID if config.target_type == "asymmetric_circles_grid" else cv2.CALIB_CB_SYMMETRIC_GRID
    found, centers = cv2.findCirclesGrid(gray, config.pattern_size, flags=flags)
    if not found:
        return None

    object_points = _build_planar_grid_points(
        config.pattern_size,
        config.circle_distance_m,
        config.circle_distance_m,
        asymmetric=(config.target_type == "asymmetric_circles_grid"),
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float64),
        centers.reshape(-1, 2).astype(np.float64),
        camera_matrix.astype(np.float64),
        distortion_coeffs.astype(np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    debug_image = image_bgr.copy()
    cv2.drawChessboardCorners(debug_image, config.pattern_size, centers, found)
    debug_image = _draw_pose_axes(
        debug_image,
        camera_matrix,
        distortion_coeffs,
        rvec,
        tvec,
        axis_length=max(config.circle_distance_m * 2.0, 0.02),
    )
    return ReferencePoseResult(
        rotation_matrix=rotation_matrix.astype(np.float64),
        translation_vector=tvec.reshape(3).astype(np.float64),
        exclusion_polygon=_build_exclusion_polygon(centers.reshape(-1, 2)),
        debug_image=debug_image,
        debug_info={
            "target_type": config.target_type,
            "pattern_size": list(config.pattern_size),
            "centers": centers.reshape(-1, 2).astype(float).tolist(),
        },
    )


def _filter_marker_candidates(
    ids: Optional[np.ndarray],
    corners: Sequence[np.ndarray],
    allowed_ids: Optional[Sequence[int]],
) -> Tuple[List[np.ndarray], List[int]]:
    if ids is None or len(ids) == 0:
        return [], []
    allowed = set(int(value) for value in allowed_ids) if allowed_ids else None
    selected_corners: List[np.ndarray] = []
    selected_ids: List[int] = []
    for index, marker_id in enumerate(ids.reshape(-1).astype(int).tolist()):
        if allowed is not None and marker_id not in allowed:
            continue
        selected_corners.append(corners[index])
        selected_ids.append(marker_id)
    return selected_corners, selected_ids


def _detect_single_marker_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    config: ReferenceTargetConfig,
) -> Optional[ReferencePoseResult]:
    if config.dictionary_name is None or config.marker_length_m is None:
        raise ValueError("单个 ArUco/AprilTag 需要提供 dictionary 与 marker_length。")

    dictionary = _resolve_aruco_dictionary(config.dictionary_name)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    chosen_corners, chosen_ids = _filter_marker_candidates(ids, corners, config.marker_ids)
    if config.marker_ids is not None and len(chosen_corners) < 3:
        raise ValueError(
            "ChArUco 不建议只保留少量 marker；marker_ids 过滤后 marker 数不足 3，"
            "请移除 marker_ids 或提供完整参考板上的多个 marker ID。"
        )
    if not chosen_corners:
        return None

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        chosen_corners,
        float(config.marker_length_m),
        camera_matrix.astype(np.float64),
        distortion_coeffs.astype(np.float64),
    )
    if rvecs is None or tvecs is None:
        return None

    areas = [
        float(cv2.contourArea(item.reshape(4, 2).astype(np.float32)))
        for item in chosen_corners
    ]
    best_index = int(np.argmax(np.array(areas, dtype=np.float64)))
    rvec = rvecs[best_index].reshape(3)
    tvec = tvecs[best_index].reshape(3)
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    debug_image = image_bgr.copy()
    cv2.aruco.drawDetectedMarkers(debug_image, chosen_corners, np.array(chosen_ids, dtype=np.int32))
    debug_image = _draw_pose_axes(
        debug_image,
        camera_matrix,
        distortion_coeffs,
        rvec,
        tvec,
        axis_length=max(config.marker_length_m * 0.8, 0.02),
    )
    return ReferencePoseResult(
        rotation_matrix=rotation_matrix.astype(np.float64),
        translation_vector=tvec.astype(np.float64),
        exclusion_polygon=_build_exclusion_polygon(chosen_corners[best_index].reshape(-1, 2)),
        debug_image=debug_image,
        debug_info={
            "target_type": config.target_type,
            "marker_id": int(chosen_ids[best_index]),
            "corners": chosen_corners[best_index].reshape(4, 2).astype(float).tolist(),
        },
    )


def _build_aruco_board(config: ReferenceTargetConfig):
    if config.pattern_size is None or config.marker_length_m is None or config.marker_separation_m is None:
        raise ValueError("ArUco Board 需要提供 pattern_size、marker_length 与 marker_separation。")
    dictionary = _resolve_aruco_dictionary(config.dictionary_name or "DICT_4X4_50")
    markers_x, markers_y = config.pattern_size
    return cv2.aruco.GridBoard(
        (int(markers_x), int(markers_y)),
        float(config.marker_length_m),
        float(config.marker_separation_m),
        dictionary,
    )


def _detect_aruco_board_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    config: ReferenceTargetConfig,
) -> Optional[ReferencePoseResult]:
    board = _build_aruco_board(config)
    dictionary = _resolve_aruco_dictionary(config.dictionary_name or "DICT_4X4_50")
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    chosen_corners, chosen_ids = _filter_marker_candidates(ids, corners, config.marker_ids)
    if not chosen_corners:
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseBoard(
        chosen_corners,
        np.array(chosen_ids, dtype=np.int32),
        board,
        camera_matrix.astype(np.float64),
        distortion_coeffs.astype(np.float64),
        None,
        None,
    )
    if int(ok) <= 0:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    debug_image = image_bgr.copy()
    cv2.aruco.drawDetectedMarkers(debug_image, chosen_corners, np.array(chosen_ids, dtype=np.int32))
    debug_image = _draw_pose_axes(
        debug_image,
        camera_matrix,
        distortion_coeffs,
        rvec.reshape(3),
        tvec.reshape(3),
        axis_length=max(config.marker_length_m * 1.2, 0.02),
    )
    return ReferencePoseResult(
        rotation_matrix=rotation_matrix.astype(np.float64),
        translation_vector=tvec.reshape(3).astype(np.float64),
        exclusion_polygon=_build_exclusion_polygon(
            np.vstack([item.reshape(-1, 2) for item in chosen_corners]).reshape(-1, 2)
        ),
        debug_image=debug_image,
        debug_info={
            "target_type": "aruco_board",
            "marker_ids": [int(value) for value in chosen_ids],
        },
    )


def _build_charuco_board(config: ReferenceTargetConfig):
    if config.pattern_size is None or config.square_size_m is None or config.marker_length_m is None:
        raise ValueError("ChArUco 需要提供 pattern_size、square_size 与 marker_length。")
    dictionary = _resolve_aruco_dictionary(config.dictionary_name or "DICT_4X4_50")
    squares_x, squares_y = config.pattern_size
    return cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(config.square_size_m),
        float(config.marker_length_m),
        dictionary,
    )


def _detect_charuco_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    config: ReferenceTargetConfig,
) -> Optional[ReferencePoseResult]:
    board = _build_charuco_board(config)
    dictionary = _resolve_aruco_dictionary(config.dictionary_name or "DICT_4X4_50")
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    chosen_corners, chosen_ids = _filter_marker_candidates(ids, corners, config.marker_ids)
    if not chosen_corners:
        return None

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        chosen_corners,
        np.array(chosen_ids, dtype=np.int32),
        gray,
        board,
        cameraMatrix=camera_matrix.astype(np.float64),
        distCoeffs=distortion_coeffs.astype(np.float64),
    )
    min_charuco_count = 6
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_charuco_count:
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        camera_matrix.astype(np.float64),
        distortion_coeffs.astype(np.float64),
        None,
        None,
    )
    if not bool(ok):
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    debug_image = image_bgr.copy()
    cv2.aruco.drawDetectedMarkers(debug_image, chosen_corners, np.array(chosen_ids, dtype=np.int32))
    cv2.aruco.drawDetectedCornersCharuco(debug_image, charuco_corners, charuco_ids)
    debug_image = _draw_pose_axes(
        debug_image,
        camera_matrix,
        distortion_coeffs,
        rvec.reshape(3),
        tvec.reshape(3),
        axis_length=max(config.square_size_m * 2.0, 0.02),
    )
    return ReferencePoseResult(
        rotation_matrix=rotation_matrix.astype(np.float64),
        translation_vector=tvec.reshape(3).astype(np.float64),
        exclusion_polygon=_build_exclusion_polygon(
            np.vstack([item.reshape(-1, 2) for item in chosen_corners]).reshape(-1, 2)
        ),
        debug_image=debug_image,
        debug_info={
            "target_type": "charuco_board",
            "charuco_count": int(len(charuco_ids)),
        },
    )
