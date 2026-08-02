# -*- coding: utf-8 -*-
"""标定板规格、官方 OpenCV 检测和打印图生成辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


PATTERN_TYPES = (
    "chessboard",
    "charuco",
    "circles_grid",
    "asymmetric_circles_grid",
)


def parse_pattern_size(value: str) -> tuple[int, int]:
    raw = value.lower().replace(" ", "")
    if "x" not in raw:
        raise ValueError("pattern-size 必须使用列x行格式，例如 9x6")
    left, right = raw.split("x", 1)
    columns, rows = int(left), int(right)
    if columns < 2 or rows < 2:
        raise ValueError("pattern-size 的列数和行数都必须至少为 2")
    return columns, rows


def unit_to_mm(value: float, unit: str) -> float:
    factors = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    key = unit.strip().lower()
    if key not in factors:
        raise ValueError("unit 只支持 mm、cm、m")
    return float(value) * factors[key]


@dataclass(frozen=True)
class BoardSpec:
    pattern_type: str
    pattern_size: tuple[int, int]
    square_size: Optional[float] = None
    circle_distance: Optional[float] = None
    marker_length: Optional[float] = None
    dictionary: str = "DICT_5X5_100"
    length_unit: str = "mm"

    def validate(self) -> None:
        if self.pattern_type not in PATTERN_TYPES:
            raise ValueError(f"不支持的标定板类型: {self.pattern_type}")
        if self.pattern_type in {"chessboard", "charuco"}:
            if self.square_size is None or self.square_size <= 0:
                raise ValueError(f"{self.pattern_type} 必须提供正数 square-size")
        if self.pattern_type in {"circles_grid", "asymmetric_circles_grid"}:
            if self.circle_distance is None or self.circle_distance <= 0:
                raise ValueError(f"{self.pattern_type} 必须提供正数 circle-distance")
        if self.pattern_type == "charuco":
            if self.marker_length is None or self.marker_length <= 0:
                raise ValueError("charuco 必须提供正数 marker-length")
            if self.marker_length >= float(self.square_size):
                raise ValueError("marker-length 必须小于 square-size")
        unit_to_mm(1.0, self.length_unit)
        if not hasattr(cv2, "aruco") and self.pattern_type == "charuco":
            raise RuntimeError("当前 OpenCV 不包含 aruco 模块，无法使用 ChArUco")


@dataclass
class PatternDetection:
    object_points: np.ndarray
    image_points: np.ndarray
    debug_image: np.ndarray
    debug_info: dict[str, Any]


def _dictionary(name: str):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, name):
        raise ValueError(f"OpenCV 不支持 ArUco 字典: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def build_charuco_board(spec: BoardSpec):
    spec.validate()
    return cv2.aruco.CharucoBoard(
        spec.pattern_size,
        float(spec.square_size),
        float(spec.marker_length),
        _dictionary(spec.dictionary),
    )


def build_object_points(spec: BoardSpec) -> np.ndarray:
    spec.validate()
    columns, rows = spec.pattern_size
    step = spec.square_size if spec.pattern_type == "chessboard" else spec.circle_distance
    if step is None:
        raise ValueError("标定板缺少物理间距")
    points: list[list[float]] = []
    for row in range(rows):
        for column in range(columns):
            if spec.pattern_type == "asymmetric_circles_grid":
                x = (2 * column + row % 2) * float(step)
            else:
                x = column * float(step)
            points.append([x, row * float(step), 0.0])
    return np.asarray(points, dtype=np.float32)


def _draw_chessboard(spec: BoardSpec, dpi: int, margin_mm: float) -> np.ndarray:
    columns, rows = spec.pattern_size
    square_px = max(2, round(unit_to_mm(float(spec.square_size), spec.length_unit) / 25.4 * dpi))
    margin_px = max(0, round(margin_mm / 25.4 * dpi))
    board_columns, board_rows = columns + 1, rows + 1
    image = np.full(
        (board_rows * square_px + 2 * square_px + 2 * margin_px,
         board_columns * square_px + 2 * square_px + 2 * margin_px),
        255,
        dtype=np.uint8,
    )
    origin_x = margin_px + square_px
    origin_y = margin_px + square_px
    for row in range(board_rows):
        for column in range(board_columns):
            if (row + column) % 2 == 0:
                x0 = origin_x + column * square_px
                y0 = origin_y + row * square_px
                image[y0:y0 + square_px, x0:x0 + square_px] = 0
    return image


def _draw_circle_grid(spec: BoardSpec, dpi: int, margin_mm: float) -> np.ndarray:
    columns, rows = spec.pattern_size
    spacing_px = max(2, round(unit_to_mm(float(spec.circle_distance), spec.length_unit) / 25.4 * dpi))
    margin_px = max(spacing_px, round(margin_mm / 25.4 * dpi))
    x_factor = 2 if spec.pattern_type == "asymmetric_circles_grid" else 1
    width = (x_factor * (columns - 1) + 1) * spacing_px + 2 * margin_px
    height = (rows - 1) * spacing_px + 2 * margin_px
    image = np.full((height, width), 255, dtype=np.uint8)
    radius = max(2, round(spacing_px * 0.22))
    for row in range(rows):
        for column in range(columns):
            x = margin_px + (2 * column + row % 2 if x_factor == 2 else column) * spacing_px
            y = margin_px + row * spacing_px
            cv2.circle(image, (x, y), radius, 0, thickness=-1, lineType=cv2.LINE_AA)
    return image


def draw_board(spec: BoardSpec, dpi: int = 300, margin_mm: float = 10.0) -> np.ndarray:
    spec.validate()
    if dpi < 30:
        raise ValueError("dpi 必须至少为 30")
    if spec.pattern_type == "chessboard":
        return _draw_chessboard(spec, dpi, margin_mm)
    if spec.pattern_type in {"circles_grid", "asymmetric_circles_grid"}:
        return _draw_circle_grid(spec, dpi, margin_mm)

    board = build_charuco_board(spec)
    square_mm = unit_to_mm(float(spec.square_size), spec.length_unit)
    width_px = max(100, round(spec.pattern_size[0] * square_mm / 25.4 * dpi))
    height_px = max(100, round(spec.pattern_size[1] * square_mm / 25.4 * dpi))
    margin_px = max(1, round(margin_mm / 25.4 * dpi))
    return board.generateImage(
        (width_px + 2 * margin_px, height_px + 2 * margin_px),
        marginSize=margin_px,
        borderBits=1,
    )


def board_metadata(spec: BoardSpec, dpi: int, margin_mm: float) -> dict[str, Any]:
    return {
        "pattern_type": spec.pattern_type,
        "pattern_size": list(spec.pattern_size),
        "square_size": spec.square_size,
        "circle_distance": spec.circle_distance,
        "marker_length": spec.marker_length,
        "dictionary": spec.dictionary,
        "length_unit": spec.length_unit,
        "dpi": dpi,
        "margin_mm": margin_mm,
        "opencv_pattern_size_meaning": (
            "chessboard uses inner corners (columns x rows); charuco uses squares (columns x rows)"
        ),
    }


def detect_pattern(image: np.ndarray, spec: BoardSpec) -> Optional[PatternDetection]:
    spec.validate()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if spec.pattern_type == "chessboard":
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, spec.pattern_size, flags=flags)
        if not found:
            return None
        image_points = corners.reshape(-1, 2).astype(np.float32)
        debug = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(debug, spec.pattern_size, corners, True)
        return PatternDetection(build_object_points(spec), image_points, debug, {"point_count": len(image_points)})

    if spec.pattern_type in {"circles_grid", "asymmetric_circles_grid"}:
        flags = (
            cv2.CALIB_CB_ASYMMETRIC_GRID
            if spec.pattern_type == "asymmetric_circles_grid"
            else cv2.CALIB_CB_SYMMETRIC_GRID
        )
        found, centers = cv2.findCirclesGrid(gray, spec.pattern_size, flags=flags)
        if not found:
            return None
        image_points = centers.reshape(-1, 2).astype(np.float32)
        debug = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(debug, spec.pattern_size, centers, True)
        return PatternDetection(build_object_points(spec), image_points, debug, {"point_count": len(image_points)})

    board = build_charuco_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
        return None
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    debug = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids.reshape(-1, 1))
    cv2.aruco.drawDetectedCornersCharuco(
        debug,
        np.asarray(charuco_corners).reshape(-1, 1, 2),
        np.asarray(charuco_ids).reshape(-1, 1),
    )
    return PatternDetection(
        object_points,
        image_points,
        debug,
        {"point_count": len(image_points), "marker_count": len(marker_ids) if marker_ids is not None else 0},
    )
