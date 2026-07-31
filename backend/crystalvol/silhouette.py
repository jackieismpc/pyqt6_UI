# -*- coding: utf-8 -*-
"""由融合边缘（+ 可选 SAM2 掩膜）提取「最大晶体」剪影。

难点：融合边缘里既有晶体棱线，也有桌面、背景反光、水珠等杂边。
策略：
1. 用边缘密度图找到「边缘密集区」作为晶体粗前景（晶体区域棱线最密）；
2. 若有 SAM2 掩膜则并入；
3. 形态学闭运算桥接断棱线 -> 填充最大外轮廓；
4. 只保留「最靠近画面中心的最大连通域」，对应「只取最大的那个晶体」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class SilhouetteResult:
    """剪影提取结果。"""

    mask: np.ndarray                 # uint8 0/255
    contour: np.ndarray              # (N,2) 外轮廓点
    coverage_ratio: float            # 剪影面积 / 整图面积
    warnings: List[str] = field(default_factory=list)


def _edge_density_foreground(edge_map: np.ndarray) -> np.ndarray:
    """边缘密度图 -> 粗前景。晶体区域棱线密集，桌面/背景稀疏。"""
    h, w = edge_map.shape[:2]
    # 用相对短边的核做均值滤波，得到密度
    ksize = max(int(min(h, w) * 0.03) | 1, 15)
    density = cv2.blur((edge_map > 0).astype(np.float32), (ksize, ksize))
    if float(density.max()) <= 1e-6:
        return np.zeros((h, w), dtype=np.uint8)
    norm = density / float(density.max())
    fg = (norm > 0.18).astype(np.uint8) * 255
    return fg


def _tighten_to_core(
    foreground: np.ndarray,
    roi_gray: np.ndarray,
    edge_map: np.ndarray,
    core_percentile: float,
    warnings: List[str],
) -> np.ndarray:
    """把前景从「背光光晕」收紧到「真实晶体」。

    背光小晶体周围常有一圈弥散发光光晕，SAM2/边缘密度会把光晕一起圈进来，
    导致尺寸被撑大数倍。真实晶体相对光晕有两个特征：
    - 亮核：晶体本体通常比周围光晕更亮（取前景内高亮分位为核）；
    - 棱线密集：晶体内部/边界棱线密，光晕平滑（用边缘密度补充）。
    取「亮核 ∪ (前景内高边缘密度)」作为收紧后的晶体区域。
    带安全回退：若收紧后过小（<5% 前景），说明该帧晶体与光晕难分，保持原前景。
    """
    fg = foreground > 0
    fg_area = int(fg.sum())
    if fg_area < 50:
        return foreground

    # 亮核：前景内亮度高分位
    values = roi_gray[fg]
    threshold = float(np.percentile(values, float(core_percentile)))
    bright = ((roi_gray >= threshold) & fg).astype(np.uint8) * 255

    # 高边缘密度区（晶体棱线密集）
    h, w = edge_map.shape[:2]
    ksize = max(int(min(h, w) * 0.04) | 1, 9)
    density = cv2.blur((edge_map > 0).astype(np.float32), (ksize, ksize))
    if float(density.max()) > 1e-6:
        density = density / float(density.max())
    edge_region = ((density > 0.28) & fg).astype(np.uint8) * 255

    core = cv2.bitwise_or(bright, edge_region)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k, iterations=1)
    core = _keep_central_largest(core)

    if int(np.count_nonzero(core)) < 0.05 * fg_area:
        warnings.append("亮核收紧后过小，保持原前景（该帧晶体与光晕难分离）。")
        return foreground
    return core


def _bridge_vertical_bars(mask: np.ndarray) -> np.ndarray:
    """桥接竖直黑色遮挡条造成的纵向缝隙。

    遮挡条会把晶体剪影从中间竖着切开（形成竖直空隙）。用一个横向较宽的核做闭运算，
    把被竖条分开的左右两半重新连起来，同时不过度改变上下轮廓。
    """
    h, w = mask.shape[:2]
    kw = max(int(w * 0.18) | 1, 15)
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal, iterations=1)


def _reject_lower_reflection(mask: np.ndarray, neck_frac: float = 0.45, revive_frac: float = 0.60) -> np.ndarray:
    """裁掉台面镜面倒影。

    晶体立在台面上时，下方常有一段镜像倒影，二者之间有一个「细腰」。
    以逐行像素数近似宽度剖面：从主体宽带往下扫，若出现 width<neck_frac*peak 的细腰，
    且其下方又重新变宽(>revive_frac*peak)，判定为倒影并在细腰处截断。
    仅在明确出现「细腰+再变宽」时才裁，避免误伤本身细长的晶体。

    安全约束：细腰必须位于轮廓下半区（>=45% 位置），
    防止把细长晶体本身的自然收窄误判为台面倒影边界。
    revive_frac 取 0.60（比原始 0.55 略严但仍能捕获典型倒影）。
    """
    rows = (mask > 0).sum(axis=1).astype(np.float32)
    ys = np.where(rows > 0)[0]
    if len(ys) < 10:
        return mask
    top, bottom = int(ys[0]), int(ys[-1])
    peak = float(rows[top:bottom + 1].max())
    if peak <= 1.0:
        return mask
    body_start = top
    for y in range(top, bottom + 1):
        if rows[y] >= 0.5 * peak:
            body_start = y
            break
    total_span = float(bottom - top)
    cut = bottom
    for y in range(body_start, bottom + 1):
        if rows[y] < neck_frac * peak:
            # 安全约束：细腰必须位于下半区（反射在底部，不在晶体中间）
            waist_pos = (y - top) / max(total_span, 1.0)
            if waist_pos < 0.45:
                continue
            if float(rows[y:bottom + 1].max()) >= revive_frac * peak:
                cut = y
                break
    if cut < bottom:
        result = mask.copy()
        result[cut:, :] = 0
        return result
    return mask


def _keep_central_largest(mask: np.ndarray) -> np.ndarray:
    """保留最靠近画面中心的最大连通域（面积 × 中心权重最高者）。"""
    h, w = mask.shape[:2]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if num <= 1:
        return mask
    image_center = np.array([w * 0.5, h * 0.5])
    diagonal = float(np.hypot(w, h))
    best_label, best_score = 1, -1.0
    for label in range(1, num):
        area = float(stats[label, cv2.CC_STAT_AREA])
        distance = float(np.linalg.norm(centroids[label] - image_center))
        center_term = max(0.0, 1.0 - distance / max(diagonal * 0.5, 1.0))
        score = area * (0.5 + 0.5 * center_term)
        if score > best_score:
            best_score, best_label = score, label
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def extract_silhouette(
    edge_map: np.ndarray,
    sam2_mask: Optional[np.ndarray] = None,
    roi_gray: Optional[np.ndarray] = None,
    core_percentile: float = 55.0,
    close_kernel_ratio: float = 0.02,
) -> SilhouetteResult:
    """提取最大晶体剪影。

    roi_gray 提供时，会做「亮核 + 边缘紧化」把真实晶体从背光光晕里收紧。
    """
    warnings: List[str] = []
    h, w = edge_map.shape[:2]

    # SAM2 掩膜若可用且规模合理，作为主前景（比纯边缘密度更贴合晶体主体）；
    # 否则回退到边缘密度前景。
    sam2_area_ratio = 0.0
    if sam2_mask is not None and sam2_mask.size > 0:
        sam2_area_ratio = float(np.count_nonzero(sam2_mask)) / float(sam2_mask.size)
    if 0.03 <= sam2_area_ratio <= 0.7:
        foreground = (sam2_mask > 0).astype(np.uint8) * 255
    else:
        if sam2_mask is not None and sam2_mask.size > 0:
            warnings.append(f"SAM2 掩膜规模异常(area_ratio={sam2_area_ratio:.3f})，改用边缘密度前景。")
        foreground = _edge_density_foreground(edge_map)

    # 亮核 + 边缘紧化：把真实晶体从背光光晕里收紧（需要 ROI 灰度图）
    if roi_gray is not None and roi_gray.shape[:2] == foreground.shape[:2]:
        foreground = _tighten_to_core(foreground, roi_gray, edge_map, core_percentile, warnings)

    # 闭运算桥接断裂棱线
    ksize = max(int(min(h, w) * close_kernel_ratio) | 1, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 先桥接竖直黑条造成的缝隙，再取中心最大连通域，最后裁掉台面倒影
    closed = _bridge_vertical_bars(closed)
    closed = _keep_central_largest(closed)
    closed = _reject_lower_reflection(closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        warnings.append("未能从边缘提取到剪影轮廓。")
        return SilhouetteResult(np.zeros((h, w), np.uint8), np.empty((0, 2), np.float32), 0.0, warnings)

    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    coverage = float(np.count_nonzero(filled)) / float(h * w)

    return SilhouetteResult(
        mask=filled,
        contour=contour.reshape(-1, 2).astype(np.float32),
        coverage_ratio=coverage,
        warnings=warnings,
    )
