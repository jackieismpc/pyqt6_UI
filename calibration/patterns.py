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

A4_SIZE_MM = (210.0, 297.0)
PAPER_SIZES_MM = {"a4": A4_SIZE_MM}
PAPER_ORIENTATIONS = ("portrait", "landscape")


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


def paper_dimensions_mm(paper: str = "a4", orientation: str = "portrait") -> tuple[float, float]:
    paper_key = paper.strip().lower()
    orientation_key = orientation.strip().lower()
    if paper_key not in PAPER_SIZES_MM:
        raise ValueError(f"不支持的纸张尺寸: {paper}，当前只支持 a4")
    if orientation_key not in PAPER_ORIENTATIONS:
        raise ValueError(f"不支持的纸张方向: {orientation}，可选 portrait 或 landscape")
    width_mm, height_mm = PAPER_SIZES_MM[paper_key]
    if orientation_key == "landscape":
        return height_mm, width_mm
    return width_mm, height_mm


def _mm_to_pixels(value_mm: float, dpi: int) -> int:
    return max(1, round(float(value_mm) / 25.4 * dpi))


def board_dimensions_mm(spec: BoardSpec) -> tuple[float, float]:
    """返回标定图案本体的物理宽高，不含额外白边。"""
    spec.validate()
    columns, rows = spec.pattern_size
    if spec.pattern_type == "charuco":
        step_mm = unit_to_mm(float(spec.square_size), spec.length_unit)
        return columns * step_mm, rows * step_mm
    if spec.pattern_type == "chessboard":
        step_mm = unit_to_mm(float(spec.square_size), spec.length_unit)
        return (columns + 1) * step_mm, (rows + 1) * step_mm
    step_mm = unit_to_mm(float(spec.circle_distance), spec.length_unit)
    if spec.pattern_type == "asymmetric_circles_grid":
        return (2 * (columns - 1) + 1) * step_mm, (rows - 1) * step_mm
    return (columns - 1) * step_mm, (rows - 1) * step_mm


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
    debug_image: Optional[np.ndarray]
    debug_info: dict[str, Any]


def _dictionary(name: str):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, name):
        raise ValueError(f"OpenCV 不支持 ArUco 字典: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def build_charuco_board(spec: BoardSpec):
    spec.validate()
    board = cv2.aruco.CharucoBoard(
        spec.pattern_size,
        float(spec.square_size),
        float(spec.marker_length),
        _dictionary(spec.dictionary),
    )
    # OpenCV 曾经存在 legacy ChArUco 模板。官方当前模板是 modern pattern，
    # 显式设置以避免不同 OpenCV 版本使用不同默认值。
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(False)
    return board


def build_charuco_detector(
    board,
    camera_matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
):
    """构造官方 ChArUco detector，按是否有内参选择角点插值路径。"""
    charuco_parameters = cv2.aruco.CharucoParameters()
    charuco_parameters.tryRefineMarkers = False
    if camera_matrix is not None:
        charuco_parameters.cameraMatrix = np.asarray(camera_matrix, dtype=np.float64)
    if distortion is not None:
        charuco_parameters.distCoeffs = np.asarray(distortion, dtype=np.float64)
    detector_parameters = cv2.aruco.DetectorParameters()
    detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    return cv2.aruco.CharucoDetector(board, charuco_parameters, detector_parameters)


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


def _draw_board_art(spec: BoardSpec, dpi: int, margin_mm: float) -> np.ndarray:
    spec.validate()
    if spec.pattern_type == "chessboard":
        return _draw_chessboard(spec, dpi, margin_mm)
    if spec.pattern_type in {"circles_grid", "asymmetric_circles_grid"}:
        return _draw_circle_grid(spec, dpi, margin_mm)

    board = build_charuco_board(spec)
    square_mm = unit_to_mm(float(spec.square_size), spec.length_unit)
    width_px = max(100, _mm_to_pixels(spec.pattern_size[0] * square_mm, dpi))
    height_px = max(100, _mm_to_pixels(spec.pattern_size[1] * square_mm, dpi))
    margin_px = max(0, _mm_to_pixels(margin_mm, dpi))
    return board.generateImage(
        (width_px + 2 * margin_px, height_px + 2 * margin_px),
        marginSize=margin_px,
        borderBits=1,
    )


def draw_board(
    spec: BoardSpec,
    dpi: int = 300,
    margin_mm: float = 0.0,
    paper: str = "a4",
    orientation: str = "portrait",
) -> np.ndarray:
    spec.validate()
    if dpi < 30:
        raise ValueError("dpi 必须至少为 30")
    if margin_mm < 0:
        raise ValueError("margin-mm 不能为负数")
    page_width_mm, page_height_mm = paper_dimensions_mm(paper, orientation)
    board_width_mm, board_height_mm = board_dimensions_mm(spec)
    rendered_width_mm = board_width_mm + 2.0 * float(margin_mm)
    rendered_height_mm = board_height_mm + 2.0 * float(margin_mm)
    if rendered_width_mm > page_width_mm or rendered_height_mm > page_height_mm:
        raise ValueError(
            f"标定板 {rendered_width_mm:.1f}x{rendered_height_mm:.1f} mm "
            f"无法放入 {paper} {orientation} 页面 {page_width_mm:.1f}x{page_height_mm:.1f} mm"
        )

    art = _draw_board_art(spec, dpi, margin_mm)
    page_width_px = _mm_to_pixels(page_width_mm, dpi)
    page_height_px = _mm_to_pixels(page_height_mm, dpi)
    page = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)
    offset_x = (page_width_px - art.shape[1]) // 2
    offset_y = (page_height_px - art.shape[0]) // 2
    page[offset_y:offset_y + art.shape[0], offset_x:offset_x + art.shape[1]] = art
    return page


def board_metadata(
    spec: BoardSpec,
    dpi: int,
    margin_mm: float,
    paper: str = "a4",
    orientation: str = "portrait",
) -> dict[str, Any]:
    page_width_mm, page_height_mm = paper_dimensions_mm(paper, orientation)
    board_width_mm, board_height_mm = board_dimensions_mm(spec)
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
        "paper": {
            "name": paper.lower(),
            "orientation": orientation.lower(),
            "width_mm": page_width_mm,
            "height_mm": page_height_mm,
            "width_px": _mm_to_pixels(page_width_mm, dpi),
            "height_px": _mm_to_pixels(page_height_mm, dpi),
        },
        "board_size_mm": {
            "width": board_width_mm,
            "height": board_height_mm,
        },
        "print_scale": 1.0,
        "opencv_version": cv2.__version__,
        "charuco_pattern": "modern" if spec.pattern_type == "charuco" else None,
        "opencv_pattern_size_meaning": (
            "chessboard uses inner corners (columns x rows); charuco uses squares (columns x rows); "
            "OpenCV official generator names the same dimensions rows x columns"
        ),
    }


def detect_pattern(
    image: np.ndarray,
    spec: BoardSpec,
    include_debug: bool = True,
    camera_matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
) -> Optional[PatternDetection]:
    spec.validate()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if spec.pattern_type == "chessboard":
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, spec.pattern_size, flags=flags)
        if not found:
            return None
        image_points = corners.reshape(-1, 2).astype(np.float32)
        debug = None
        if include_debug:
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
        debug = None
        if include_debug:
            debug = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(debug, spec.pattern_size, centers, True)
        return PatternDetection(build_object_points(spec), image_points, debug, {"point_count": len(image_points)})

    board = build_charuco_board(spec)
    detector = build_charuco_detector(board, camera_matrix=camera_matrix, distortion=distortion)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
        return None
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    debug = None
    if include_debug:
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
        {
            "point_count": len(image_points),
            "marker_count": len(marker_ids) if marker_ids is not None else 0,
            "interpolation": (
                "pose_reprojection_with_intrinsics"
                if camera_matrix is not None and distortion is not None
                else "homography_without_intrinsics"
            ),
            "marker_corner_refinement": "none",
        },
    )
