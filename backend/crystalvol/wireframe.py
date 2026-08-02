# -*- coding: utf-8 -*-
"""长方体 + 四棱锥线框拟合（单视角，剪影宽度剖面法）。

透明晶体单视角、强反光、弱棱线的条件下，逐条棱线做 Hough 拟合很不稳。
这里采用更鲁棒的「剪影宽度剖面」方法：

    屋顶（四棱锥）：从顶点 apex 往下，剪影宽度由 0 增大到满宽；
    长方体：满宽区域基本恒定；
    底部：剪影最底一行。

因此：
    apex_y   = 剪影最高行
    shoulder_y = 从顶部往下、宽度首次达到 ~满宽 的行（屋顶与长方体的交界=肩线）
    base_y   = 剪影最低行

由此得到关键点（apex / shoulder_left,right / base_left,right）与像素尺寸：
    length_px         = 长方体区域中位宽度（前向长度）
    body_height_px    = base_y - shoulder_y
    pyramid_height_px = shoulder_y - apex_y
    width_px（侧向深度）：单视角不可直接观测，根据剪影的高宽比和屋顶比例
    在晶体形状先验范围内自适应估计（第二阶段可用尺度锚点或外参进一步修正）

该方法对断裂棱线、反光不敏感，且总能给出 best-effort 结果，保证三张图一定能产出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import WireframeConfig
from .geometry import compute_volume
from .silhouette import SilhouetteResult


@dataclass
class WireframeResult:
    """线框拟合结果。"""

    canonical_points: Dict[str, Tuple[float, float]]  # 2D 关键点
    observed_edges: List[Tuple[str, str]]             # 关键边（用点名表示）
    geometry_px: Dict[str, float]                     # length_px/width_px/... /volume_px3
    fit_ready: bool                                   # 结构是否足够完整，适合进入第二阶段
    visible_ratio: float                              # 关键边被边缘证据支持的比例
    coverage_ratio: float
    depth_source: str                                 # width_px 的来源
    warnings: List[str] = field(default_factory=list)


def _smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    """一维滑动平均，抑制剪影边界锯齿。"""
    if len(values) < window or window < 3:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _row_extents(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐行求剪影左右边界与宽度。空行记为 -1。"""
    h, w = mask.shape[:2]
    left = np.full(h, -1.0, dtype=np.float32)
    right = np.full(h, -1.0, dtype=np.float32)
    width = np.zeros(h, dtype=np.float32)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return left, right, width
    for y in np.unique(ys):
        row_xs = xs[ys == y]
        left[y] = float(row_xs.min())
        right[y] = float(row_xs.max())
        width[y] = right[y] - left[y]
    return left, right, width


def _edge_support(edge_map: np.ndarray, p: Tuple[float, float], q: Tuple[float, float], radius: int = 6) -> float:
    """一条线段被边缘证据支持的比例（沿线采样，看邻域是否有边缘）。"""
    dilated = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius | 1, radius | 1)))
    n = 24
    hits = 0
    h, w = edge_map.shape[:2]
    for t in np.linspace(0.0, 1.0, n):
        x = int(round(p[0] * (1 - t) + q[0] * t))
        y = int(round(p[1] * (1 - t) + q[1] * t))
        if 0 <= x < w and 0 <= y < h and dilated[y, x] > 0:
            hits += 1
    return hits / float(n)


def _adaptive_depth_ratio(
    total_height_px: float,
    length_px: float,
    pyramid_height_px: float,
    cfg: WireframeConfig,
) -> Tuple[float, float, float]:
    """按观测形状选择侧向深度先验。

    单目图像无法从数学上唯一恢复被遮挡的侧向深度，因此这里不再要求用户
    为每个晶体手工输入一个固定比例，而是使用可解释的形状先验：越高瘦的
    剪影通常对应较小的深度比例，越接近方形/盒状的剪影对应较大的比例。
    返回 ``(depth_ratio, vertical_ratio, confidence)``。
    """
    safe_length = max(float(length_px), 1.0)
    vertical_ratio = max(float(total_height_px), 0.0) / safe_length
    min_ratio = min(float(cfg.shape_prior_min_depth_ratio), float(cfg.shape_prior_max_depth_ratio))
    max_ratio = max(float(cfg.shape_prior_min_depth_ratio), float(cfg.shape_prior_max_depth_ratio))

    # 0.8~2.8 覆盖从近方形到明显高瘦的大多数视野比例；超出范围时饱和，
    # 避免少量分割噪声把三维尺寸推到不合理的极端。
    tallness = float(np.clip((vertical_ratio - 0.8) / 2.0, 0.0, 1.0))
    depth_ratio = max_ratio - tallness * (max_ratio - min_ratio)

    roof_fraction = float(np.clip(float(pyramid_height_px) / max(float(total_height_px), 1.0), 0.0, 1.0))
    # 屋顶占比能轻微区分同一高宽比下的盒状与尖顶形状，但只作小幅修正，
    # 不让屋顶检测结果主导不可观测的深度。
    depth_ratio *= 1.0 + 0.08 * (roof_fraction - 0.2)
    depth_ratio = float(np.clip(depth_ratio, min_ratio, max_ratio))

    # 形状越接近先验范围中部，估计越稳定；极端细长/扁平时降低置信度，
    # 供 UI 和后续质量控制识别，而不是静默地把启发式结果当成精确测量。
    confidence = 1.0 - abs(tallness - 0.5) * 0.45
    confidence = float(np.clip(confidence, 0.35, 1.0))
    return depth_ratio, vertical_ratio, confidence


def fit_wireframe(
    silhouette: SilhouetteResult,
    edge_map: np.ndarray,
    cfg: WireframeConfig,
) -> WireframeResult:
    """从剪影 + 边缘拟合长方体+四棱锥线框。"""
    warnings: List[str] = list(silhouette.warnings)
    mask = silhouette.mask
    h, w = mask.shape[:2]

    if np.count_nonzero(mask) == 0:
        warnings.append("剪影为空，无法拟合线框。")
        return WireframeResult({}, [], _empty_geometry(), False, 0.0, 0.0, "none", warnings)

    left, right, width = _row_extents(mask)
    valid_rows = np.where(width > 0)[0]
    apex_y, base_y = int(valid_rows.min()), int(valid_rows.max())

    width_smooth = _smooth(width)
    max_width = float(width_smooth[apex_y:base_y + 1].max())
    if max_width <= 1.0:
        warnings.append("剪影宽度异常，退化为包围盒估计。")
        return _bbox_fallback(mask, edge_map, warnings, cfg)

    total_h = float(base_y - apex_y)
    # 肩线：从顶部往下，宽度首次达到 0.85*满宽
    shoulder_y = apex_y
    for y in range(apex_y, base_y + 1):
        if width_smooth[y] >= 0.85 * max_width:
            shoulder_y = y
            break

    # 自适应判断有没有四棱锥屋顶（晶体是生长过程，早期常是近似纯长方体）：
    # 顶部要足够窄 + 渐扩段占一定高度，才认定存在屋顶；否则按纯长方体 pyramid_height=0。
    taper_h = float(shoulder_y - apex_y)
    top_width = float(width_smooth[apex_y])
    roof_present = (top_width < cfg.pyramid_top_width_ratio * max_width) and (taper_h >= cfg.pyramid_min_taper_ratio * max(total_h, 1.0))
    if not roof_present:
        # 保底：未检测到屋顶时，按 min_pyramid_fraction 强制一个最小棱锥高度
        # 避免模型变成纯平顶长方体（与实际晶体形状相差过大）
        min_hp = float(cfg.min_pyramid_fraction) * total_h
        shoulder_y = int(round(apex_y + min_hp))
        warnings.append(f"未检测到明显四棱锥屋顶，使用最小棱锥高度={min_hp:.1f}px (min_pyramid_fraction={cfg.min_pyramid_fraction})。")

    # 长方体区域中位宽度作为 length_px；中位中心作为体中轴
    body_rows = np.arange(shoulder_y, base_y + 1)
    body_widths = width[body_rows]
    row_centers = (left[body_rows] + right[body_rows]) * 0.5
    valid = body_widths > 0
    length_px = float(np.median(body_widths[valid])) if np.any(valid) else max_width
    body_center_x = float(np.median(row_centers[valid])) if np.any(valid) else w * 0.5
    body_height_px = float(base_y - shoulder_y)
    pyramid_height_px = float(shoulder_y - apex_y)

    # 关键点：以体中轴 + length_px 构造对称的「长方体+四棱锥」标准线框，
    # 而非直接连剪影的噪声极值点（绿色剪影轮廓仍保留观测形状用于对照）。
    half_len = length_px * 0.5
    points: Dict[str, Tuple[float, float]] = {
        "apex": (body_center_x, float(apex_y)),
        "shoulder_left": (body_center_x - half_len, float(shoulder_y)),
        "shoulder_right": (body_center_x + half_len, float(shoulder_y)),
        "base_left": (body_center_x - half_len, float(base_y)),
        "base_right": (body_center_x + half_len, float(base_y)),
    }
    roof_edges = [("apex", "shoulder_left"), ("apex", "shoulder_right")] if roof_present else []
    observed_edges = roof_edges + [
        ("shoulder_left", "shoulder_right"),                     # 肩线/顶边
        ("shoulder_left", "base_left"), ("shoulder_right", "base_right"),  # 竖直棱
        ("base_left", "base_right"),                             # 底边
    ]

    # 关键边可见比例：以「边缘图 ∪ 剪影外轮廓」为证据，衡量标准线框与观测的贴合度
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    evidence = cv2.bitwise_or(edge_map, boundary)
    supports = [_edge_support(evidence, points[a], points[b]) for a, b in observed_edges]
    visible_ratio = float(np.mean([1.0 if s > 0.35 else 0.0 for s in supports]))

    total_height_px = body_height_px + pyramid_height_px
    width_px, vertical_ratio, shape_confidence = _adaptive_depth_ratio(
        total_height_px,
        length_px,
        pyramid_height_px,
        cfg,
    )
    width_px *= length_px
    depth_source = "adaptive_single_view_shape_prior"

    volume_px3 = compute_volume(length_px, width_px, body_height_px, pyramid_height_px)
    geometry_px = {
        "length_px": length_px,
        "width_px": width_px,
        "body_height_px": body_height_px,
        "pyramid_height_px": pyramid_height_px,
        "total_height_px": total_height_px,
        "vertical_to_length_ratio": vertical_ratio,
        "depth_ratio_estimate": width_px / max(length_px, 1.0),
        "shape_prior_confidence": shape_confidence,
        "volume_px3": volume_px3,
    }

    fit_ready = (
        visible_ratio >= cfg.min_visible_ratio
        and silhouette.coverage_ratio >= cfg.min_coverage_ratio
        and body_height_px > 5 and length_px > 5
    )
    if not fit_ready:
        warnings.append(
            f"结构证据偏弱（visible_ratio={visible_ratio:.2f}, coverage={silhouette.coverage_ratio:.3f}），"
            "已按 best-effort 输出，请以边缘证据图判断可信度。"
        )

    return WireframeResult(
        canonical_points=points,
        observed_edges=observed_edges,
        geometry_px=geometry_px,
        fit_ready=bool(fit_ready),
        visible_ratio=visible_ratio,
        coverage_ratio=silhouette.coverage_ratio,
        depth_source=depth_source,
        warnings=warnings,
    )


def _empty_geometry() -> Dict[str, float]:
    return {k: 0.0 for k in
            ("length_px", "width_px", "body_height_px", "pyramid_height_px", "total_height_px",
             "vertical_to_length_ratio", "depth_ratio_estimate", "shape_prior_confidence", "volume_px3")}


def _bbox_fallback(
    mask: np.ndarray,
    edge_map: np.ndarray,
    warnings: List[str],
    cfg: WireframeConfig,
) -> WireframeResult:
    """极端兜底：用外接框 + 自适应形状先验，保证有输出。"""
    ys, xs = np.where(mask > 0)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    length_px = float(x2 - x1)
    total_h = float(y2 - y1)
    pyramid_height_px = 0.3 * total_h
    body_height_px = total_h - pyramid_height_px
    width_ratio, vertical_ratio, shape_confidence = _adaptive_depth_ratio(
        total_h,
        length_px,
        pyramid_height_px,
        cfg,
    )
    width_px = width_ratio * length_px
    apex = ((x1 + x2) * 0.5, float(y1))
    points = {
        "apex": apex,
        "shoulder_left": (float(x1), float(y1 + pyramid_height_px)),
        "shoulder_right": (float(x2), float(y1 + pyramid_height_px)),
        "base_left": (float(x1), float(y2)),
        "base_right": (float(x2), float(y2)),
    }
    edges = [("apex", "shoulder_left"), ("apex", "shoulder_right"), ("shoulder_left", "shoulder_right"),
             ("shoulder_left", "base_left"), ("shoulder_right", "base_right"), ("base_left", "base_right")]
    geometry_px = {
        "length_px": length_px, "width_px": width_px,
        "body_height_px": body_height_px, "pyramid_height_px": pyramid_height_px,
        "total_height_px": total_h,
        "vertical_to_length_ratio": vertical_ratio,
        "depth_ratio_estimate": width_ratio,
        "shape_prior_confidence": shape_confidence,
        "volume_px3": compute_volume(length_px, width_px, body_height_px, pyramid_height_px),
    }
    warnings.append("使用外接框兜底几何。")
    return WireframeResult(points, edges, geometry_px, False, 0.0,
                           float(np.count_nonzero(mask)) / float(mask.size), "bbox_fallback", warnings)
