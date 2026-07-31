# -*- coding: utf-8 -*-
"""统一可视化：轮廓提取图 / overlay 图 / 三维几何预览图。

对应用户需求的三张图：
- 轮廓提取图 render_contour_image：黑底上画剪影轮廓 + 拟合线框（棱线着色）
- overlay 图  render_overlay：把同一套线框叠加回（增强后的）原图
- 几何体图    render_geometry_preview：标准长方体+四棱锥的三维等距预览
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .geometry import build_vertices, edge_index_pairs
from .wireframe import WireframeResult

# 关键边配色（BGR）
_EDGE_COLORS = {
    ("apex", "shoulder_left"): (0, 165, 255),      # 屋顶棱-橙
    ("apex", "shoulder_right"): (0, 165, 255),
    ("shoulder_left", "shoulder_right"): (255, 0, 255),   # 肩线-品红
    ("shoulder_left", "base_left"): (0, 255, 255),        # 竖直棱-黄
    ("shoulder_right", "base_right"): (0, 255, 255),
    ("base_left", "base_right"): (255, 255, 0),           # 底边-青
}
_POINT_COLOR = (60, 60, 255)


def _draw_wireframe_2d(canvas: np.ndarray, wf: WireframeResult, thickness: int) -> None:
    """在画布上画关键边与关键点。"""
    pts = wf.canonical_points
    for a, b in wf.observed_edges:
        if a in pts and b in pts:
            color = _EDGE_COLORS.get((a, b), (0, 0, 255))
            cv2.line(canvas, _ipt(pts[a]), _ipt(pts[b]), color, thickness, cv2.LINE_AA)
    for name, p in pts.items():
        cv2.circle(canvas, _ipt(p), max(thickness + 2, 4), _POINT_COLOR, -1, cv2.LINE_AA)


def _ipt(p: Tuple[float, float]) -> Tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def _line_thickness(shape) -> int:
    return max(int(round(min(shape[:2]) / 400.0)), 2)


def render_contour_image(image_shape, silhouette_contour: np.ndarray, wf: WireframeResult) -> np.ndarray:
    """轮廓提取图：黑底 + 剪影轮廓（暗绿）+ 拟合线框（彩色）。"""
    h, w = image_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    thickness = _line_thickness(image_shape)
    if silhouette_contour is not None and len(silhouette_contour) >= 3:
        cv2.polylines(canvas, [silhouette_contour.astype(np.int32)], True, (0, 120, 0), thickness, cv2.LINE_AA)
    _draw_wireframe_2d(canvas, wf, thickness)
    _put_label(canvas, f"fit_ready={wf.fit_ready}  visible={wf.visible_ratio:.2f}")
    return canvas


def render_overlay(base_bgr: np.ndarray, silhouette_contour: np.ndarray, wf: WireframeResult) -> np.ndarray:
    """overlay 图：线框叠加回原图。"""
    canvas = base_bgr.copy()
    thickness = _line_thickness(base_bgr.shape)
    if silhouette_contour is not None and len(silhouette_contour) >= 3:
        cv2.polylines(canvas, [silhouette_contour.astype(np.int32)], True, (0, 255, 0), thickness, cv2.LINE_AA)
    _draw_wireframe_2d(canvas, wf, thickness)
    _put_label(canvas, f"L={wf.geometry_px['length_px']:.0f}px Hb={wf.geometry_px['body_height_px']:.0f}px "
                       f"Hp={wf.geometry_px['pyramid_height_px']:.0f}px")
    return canvas


def _put_label(canvas: np.ndarray, text: str) -> None:
    scale = max(min(canvas.shape[:2]) / 1200.0, 0.6)
    cv2.putText(canvas, text, (20, int(40 * scale) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), max(int(2 * scale), 1), cv2.LINE_AA)


def _iso_project(vertices: np.ndarray) -> np.ndarray:
    """把 3D 顶点做等距投影到 2D（z 朝上）。"""
    cos30, sin30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    px = (x - y) * cos30
    py = (x + y) * sin30 - z
    return np.stack([px, py], axis=1)


def render_geometry_preview(geometry_px: Dict[str, float], canvas_size: int = 960) -> np.ndarray:
    """标准长方体+四棱锥三维等距预览，并标注像素尺寸与 volume_px3。"""
    canvas = np.full((canvas_size, canvas_size, 3), 248, dtype=np.uint8)
    length = max(geometry_px.get("length_px", 1.0), 1e-3)
    width = max(geometry_px.get("width_px", 1.0), 1e-3)
    body = max(geometry_px.get("body_height_px", 1.0), 1e-3)
    pyramid = max(geometry_px.get("pyramid_height_px", 1.0), 1e-3)

    vertices = build_vertices(length, width, body, pyramid)
    projected = _iso_project(vertices)
    # 归一化到画布中央
    mins, maxs = projected.min(0), projected.max(0)
    span = float(max((maxs - mins).max(), 1e-3))
    scale = 0.62 * canvas_size / span
    center = (mins + maxs) * 0.5
    pts2d = (projected - center) * scale + np.array([canvas_size * 0.5, canvas_size * 0.55])

    for a, b in edge_index_pairs():
        cv2.line(canvas, _ipt(tuple(pts2d[a])), _ipt(tuple(pts2d[b])), (30, 120, 220), 3, cv2.LINE_AA)
    for p in pts2d:
        cv2.circle(canvas, _ipt(tuple(p)), 5, (0, 0, 200), -1, cv2.LINE_AA)

    lines = [
        "晶体三维模型（像素空间）",
        f"长 L   = {length:.1f} px",
        f"宽 W   = {width:.1f} px",
        f"体高 Hb = {body:.1f} px",
        f"锥高 Hp = {pyramid:.1f} px",
        f"总高     = {geometry_px.get('total_height_px', body + pyramid):.1f} px",
        f"像素体积 = {geometry_px.get('volume_px3', 0.0):.3e} px³",
    ]
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (24, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def write_obj(geometry_px: Dict[str, float], path: Path) -> None:
    """把标准几何写成 .obj 线框，便于外部查看。"""
    vertices = build_vertices(
        max(geometry_px.get("length_px", 1.0), 1e-3),
        max(geometry_px.get("width_px", 1.0), 1e-3),
        max(geometry_px.get("body_height_px", 1.0), 1e-3),
        max(geometry_px.get("pyramid_height_px", 1.0), 1e-3),
    )
    lines: List[str] = ["# standard crystal wireframe (cuboid + pyramid), pixel units"]
    for v in vertices:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    for a, b in edge_index_pairs():
        lines.append(f"l {a + 1} {b + 1}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
