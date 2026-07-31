#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate_camera.py

第一阶段：相机内参标定脚本。

当前支持：
1. chessboard
2. circles_grid
3. asymmetric_circles_grid
4. charuco_board
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 让脚本能 import crystalvol
from crystalvol.calibration import length_unit_to_meter as get_length_unit_to_meter_scale, save_camera_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="相机内参标定。默认优先使用 ChArUco。")
    parser.add_argument("input_dir", help="标定图像目录。")
    parser.add_argument("--pattern-type", choices=["chessboard", "circles_grid", "asymmetric_circles_grid", "charuco_board"], default="charuco_board", help="标定板类型。默认 charuco_board。")
    parser.add_argument("--pattern-size", default="7x5", help="角点或圆点阵列尺寸，格式如 9x6。ChArUco 默认 7x5。")
    parser.add_argument("--square-size", type=float, default=30.0, help="棋盘格或 ChArUco 方格边长。ChArUco 默认 30 mm。")
    parser.add_argument("--marker-length", type=float, default=15.0, help="ChArUco marker 边长。默认 15 mm。")
    parser.add_argument("--circle-distance", type=float, default=None, help="圆点板相邻圆心距离。")
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="ChArUco 字典。默认 DICT_5X5_100。")
    parser.add_argument("--length-unit", default="mm", help="输入尺寸单位，例如 mm/cm/m。默认 mm。")
    parser.add_argument("--output", required=True, help="输出标定 JSON。")
    return parser.parse_args()


def parse_pattern_size(raw_value: str) -> Tuple[int, int]:
    raw = raw_value.lower().replace(" ", "")
    if "x" not in raw:
        raise RuntimeError("pattern-size 格式错误，应为 9x6 这类写法。")
    left, right = raw.split("x", 1)
    return int(left), int(right)


def collect_images(input_dir: Path) -> List[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [item.resolve() for item in sorted(input_dir.iterdir()) if item.is_file() and item.suffix.lower() in suffixes]
    if not images:
        raise RuntimeError(f"目录中没有标定图像: {input_dir}")
    return images


def build_grid_points(pattern_size: Tuple[int, int], step: float, asymmetric: bool) -> np.ndarray:
    columns, rows = pattern_size
    points = []
    for row in range(rows):
        for column in range(columns):
            if asymmetric:
                x_value = (2 * column + row % 2) * step
            else:
                x_value = column * step
            points.append([x_value, row * step, 0.0])
    return np.array(points, dtype=np.float32)


def calibrate_standard_pattern(image_paths: List[Path], args: argparse.Namespace) -> None:
    pattern_size = parse_pattern_size(args.pattern_size)
    unit_scale = get_length_unit_to_meter_scale(args.length_unit)
    if args.pattern_type == "chessboard":
        object_points_template = build_grid_points(pattern_size, float(args.square_size) * unit_scale, asymmetric=False)
    else:
        if args.circle_distance is None:
            raise RuntimeError("圆点板标定需要提供 --circle-distance。")
        object_points_template = build_grid_points(
            pattern_size,
            float(args.circle_distance) * unit_scale,
            asymmetric=(args.pattern_type == "asymmetric_circles_grid"),
        )

    object_points = []
    image_points = []
    image_size = None
    debug_success = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        if args.pattern_type == "chessboard":
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)
        else:
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID if args.pattern_type == "asymmetric_circles_grid" else cv2.CALIB_CB_SYMMETRIC_GRID
            found, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags)
        if not found:
            continue

        object_points.append(object_points_template.copy())
        image_points.append(corners.reshape(-1, 2).astype(np.float32))
        debug_success += 1

    if len(object_points) < 5 or image_size is None:
        raise RuntimeError("有效标定图不足，至少需要 5 张能识别到参考板的图像。")

    reproj_error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    output_path = save_camera_calibration(
        args.output,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion,
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error=float(reproj_error),
    )
    print(f"[calibrate_camera] 成功视图数: {debug_success}")
    print(f"[calibrate_camera] 重投影误差: {reproj_error:.6f}")
    print(f"[calibrate_camera] 已保存: {output_path}")


def calibrate_charuco(image_paths: List[Path], args: argparse.Namespace) -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 不包含 aruco 模块，请安装 opencv-contrib-python。")
    if args.marker_length is None:
        raise RuntimeError("ChArUco 标定必须提供 --marker-length。")

    pattern_size = parse_pattern_size(args.pattern_size)
    unit_scale = get_length_unit_to_meter_scale(args.length_unit)
    square_length = float(args.square_size) * unit_scale
    marker_length = float(args.marker_length) * unit_scale
    if not hasattr(cv2.aruco, args.dictionary):
        raise RuntimeError(f"不支持的字典: {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    board = cv2.aruco.CharucoBoard(pattern_size, square_length, marker_length, dictionary)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
            continue
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

    if len(all_charuco_corners) < 5 or image_size is None:
        raise RuntimeError("有效 ChArUco 标定图不足，至少需要 5 张。")

    reproj_error, camera_matrix, distortion, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None,
    )
    output_path = save_camera_calibration(
        args.output,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion,
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error=float(reproj_error),
    )
    print(f"[calibrate_camera] 成功视图数: {len(all_charuco_corners)}")
    print(f"[calibrate_camera] 重投影误差: {reproj_error:.6f}")
    print(f"[calibrate_camera] 已保存: {output_path}")


def main() -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV 不可用: {CV2_IMPORT_ERROR}")
    args = parse_args()
    image_paths = collect_images(Path(args.input_dir).expanduser().resolve())
    if args.pattern_type == "charuco_board":
        calibrate_charuco(image_paths, args)
    else:
        calibrate_standard_pattern(image_paths, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[calibrate_camera][ERROR] {exc}")
        raise SystemExit(1)
