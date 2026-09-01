# -*- coding: utf-8 -*-
"""晶体定位：显著性 ROI（覆盖小晶体到大晶体的整个范围）。

为什么需要这一步：
    重构初版直接在整张 2048x2048 上做剪影+线框，只有当晶体占画面 >=6% 时才成立。
    实际视频里晶体常常只占画面 ~1%，且被竖直黑色遮挡条切断、暗角包围、旁边有强反光，
    在整幅上用同一套绝对阈值会把黑条/反光当成主体。

策略（对应用户强调的透明/高光反射/黑色遮挡三大难点）：
    1. 显著性 = 边缘密度（晶体内部棱线密集）× 亮度（背光晶体偏亮）× 中心权重；
    2. 压制纯镜面高光（亮但内部结构少）、竖直黑色遮挡条（低亮度大块）、暗角/边框；
    3. 阈值 + 连通域，按「面积适中 + 紧致 + 显著性高 + 靠中心」挑出最大晶体块；
    4. 外扩成自适应 ROI；晶体块很大时直接用整幅（大晶体场景）。

产出 RoiResult 交给 stage1：在该 ROI 内做后续边缘/SAM2/剪影/线框，再映射回整幅。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import LocalizeConfig


@dataclass
class RoiResult:
    """定位结果。"""

    bbox: Tuple[int, int, int, int]        # x1, y1, x2, y2（整幅坐标）
    center: Tuple[float, float]            # 晶体质心（整幅坐标）
    confidence: float                      # 归一化显著性置信度（0~1）
    scale: str                             # small | medium | large | fullframe
    area_ratio: float                      # 晶体块面积占整幅比例
    saliency: np.ndarray                   # 显著性可视化图（uint8，整幅，供 debug）
    found: bool = True
    warnings: List[str] = field(default_factory=list)
    component_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # 未外扩的晶体块框


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _center_weight_map(shape: Tuple[int, int], strength: float) -> np.ndarray:
    """径向中心权重：中心 1.0，边缘衰减到 (1-strength)。"""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.5, h * 0.5
    dist = np.sqrt(((xs - cx) / (w * 0.5)) ** 2 + ((ys - cy) / (h * 0.5)) ** 2)
    dist = np.clip(dist, 0.0, 1.0)
    return (1.0 - float(strength) * dist).astype(np.float32)


def compute_saliency(
    enhanced_bgr: np.ndarray,
    edge_map: np.ndarray,
    specular_mask: Optional[np.ndarray],
    cfg: LocalizeConfig,
) -> np.ndarray:
    """计算晶体显著性图（float32, 0~1）。"""
    h, w = enhanced_bgr.shape[:2]
    short_side = min(h, w)

    # 1) 边缘密度：晶体区域棱线密集
    ksize = max(int(short_side * cfg.edge_density_ksize_ratio) | 1, 15)
    density = cv2.blur((edge_map > 0).astype(np.float32), (ksize, ksize))
    density = _normalize(density)

    # 2) 亮度：背光透明晶体通常比周围亮（用局部对比而非绝对亮度，抑制大面积泛亮背景）
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (ksize, ksize))
    local_contrast = _normalize(np.clip(gray - local_mean, 0, None))
    brightness = _normalize(gray)
    bright_term = 0.5 * brightness + 0.5 * local_contrast

    saliency = density * (0.35 + 0.65 * bright_term)

    # 3) 压制纯镜面高光（亮但通常内部结构少，且已在预处理里标出）
    if specular_mask is not None and specular_mask.size > 0 and cfg.specular_suppress > 0:
        spec = (specular_mask > 0).astype(np.float32)
        spec = cv2.blur(spec, (ksize, ksize))
        saliency *= (1.0 - float(cfg.specular_suppress) * _normalize(spec))

    # 4) 中心权重
    if cfg.center_weight > 0:
        saliency *= _center_weight_map((h, w), cfg.center_weight)

    # 5) 抑制边框/暗角
    margin = int(short_side * cfg.border_margin_ratio)
    if margin > 0:
        saliency[:margin, :] = 0
        saliency[-margin:, :] = 0
        saliency[:, :margin] = 0
        saliency[:, -margin:] = 0

    return _normalize(saliency)


def _normalised_target_box(
    target_roi: Optional[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """把归一化目标框转成当前分辨率的像素框，并拒绝非法配置。"""
    if target_roi is None or len(target_roi) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in target_roi)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)), float(np.clip(x2, 0.0, 1.0))))
    y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0))))
    if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
        return None
    return (
        int(round(x1 * width)), int(round(y1 * height)),
        int(round(x2 * width)), int(round(y2 * height)),
    )


def _bbox_iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    """计算两个 xyxy 框的 IoU。"""
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    first_area = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
    second_area = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
    return float(intersection / max(first_area + second_area - intersection, 1))


def _center_similarity(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
    width: int,
    height: int,
    max_jump_ratio: float,
) -> float:
    """按图像对角线归一化的中心接近度，超过跳变阈值后归零。"""
    first_center = ((first[0] + first[2]) * 0.5, (first[1] + first[3]) * 0.5)
    second_center = ((second[0] + second[2]) * 0.5, (second[1] + second[3]) * 0.5)
    diagonal = max(float(np.hypot(width, height)), 1.0)
    distance_ratio = float(np.hypot(
        first_center[0] - second_center[0], first_center[1] - second_center[1]
    ) / diagonal)
    limit = max(float(max_jump_ratio), 1e-3)
    return float(np.clip(1.0 - distance_ratio / limit, 0.0, 1.0))


def locate_crystal(
    enhanced_bgr: np.ndarray,
    edge_map: np.ndarray,
    specular_mask: Optional[np.ndarray],
    cfg: LocalizeConfig,
    previous_roi: Optional[RoiResult] = None,
) -> RoiResult:
    """定位最大晶体并给出自适应 ROI。"""
    h, w = enhanced_bgr.shape[:2]
    short_side = min(h, w)
    image_area = float(h * w)
    warnings: List[str] = []
    target_box = _normalised_target_box(cfg.target_roi, w, h)
    if cfg.target_roi is not None and target_box is None:
        warnings.append("target_roi 配置无效，已忽略固定目标框。")

    saliency = compute_saliency(enhanced_bgr, edge_map, specular_mask, cfg)
    saliency_vis = (saliency * 255.0).astype(np.uint8)

    # 阈值化：mean + k*std（对小晶体稳定，只保留强显著性）。大/低对比晶体会碎成很多块，
    # 靠后面「所有有效块的总体外接框」来判断是否铺开一大片，而不是靠单块面积。
    thr = float(saliency.mean() + cfg.saliency_std_k * saliency.std())
    raw = (saliency >= max(thr, 1e-3)).astype(np.uint8) * 255

    # 尺度自适应闭运算：单一固定核无法兼顾「小晶体要紧、大晶体要合并碎块」。
    # 先用小核，看显著性总面积：
    #   - 总面积小 -> 小晶体：保持小核，得到紧凑主块（ROI 贴合）；
    #   - 总面积大 -> 大/低对比晶体：改用大核把散落的棱线/刻面合并成一个大块。
    def _close(mask: np.ndarray, ratio: float) -> np.ndarray:
        k = max(int(short_side * ratio) | 1, 15)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)

    small_closed = _close(raw, 0.02)
    total_ratio = float(np.count_nonzero(small_closed)) / image_area
    if total_ratio >= cfg.large_regime_area_ratio:
        binary = _close(raw, 0.06)   # 大晶体：合并碎块
    else:
        binary = small_closed        # 小晶体：紧凑

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    # 先计算所有候选的基础显著性分数；固定目标框和上一帧轨迹只作为空间约束，
    # 不改动原有显著性图，因此关闭约束时结果与基线一致。
    candidate_records: list[tuple[int, float, Tuple[int, int, int, int], float]] = []
    for label in range(1, num):
        area_ratio = float(stats[label, cv2.CC_STAT_AREA]) / image_area
        if area_ratio < cfg.min_blob_area_ratio:
            continue
        bw = float(stats[label, cv2.CC_STAT_WIDTH])
        bh = float(stats[label, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) / max(min(bw, bh), 1.0) > cfg.max_aspect_ratio:   # 细长黑条/边框
            continue
        extent = float(stats[label, cv2.CC_STAT_AREA]) / max(bw * bh, 1.0)
        comp_saliency = float(saliency[labels == label].mean())
        cx_l, cy_l = centroids[label]
        center_term = 1.0 - float(np.hypot((cx_l - w * 0.5) / (w * 0.5), (cy_l - h * 0.5) / (h * 0.5))) / 1.4142
        # 面积项：大块加分（让大晶体的合并大块胜过角落零星反光），但用 sqrt 抑制过猛
        area_term = min(1.0, float(np.sqrt(area_ratio / 0.25)))
        score = comp_saliency * (0.4 + 0.6 * extent) * (0.5 + 0.5 * max(center_term, 0.0)) * (0.4 + 0.6 * area_term)
        component_box = (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_LEFT] + bw),
            int(stats[label, cv2.CC_STAT_TOP] + bh),
        )
        candidate_records.append((label, float(score), component_box, float(area_ratio)))

    previous_box = None
    if (
        cfg.tracking_enabled and previous_roi is not None and previous_roi.found
        and previous_roi.component_bbox[2] > previous_roi.component_bbox[0]
        and previous_roi.component_bbox[3] > previous_roi.component_bbox[1]
    ):
        previous_box = previous_roi.component_bbox

    scored_records: list[tuple[float, int, Tuple[int, int, int, int]]] = []
    tracked_records: list[tuple[int, float, Tuple[int, int, int, int], float, float]] = []
    for label, base_score, component_box, area_ratio in candidate_records:
        spatial_score = 1.0
        if target_box is not None:
            target_iou = _bbox_iou(component_box, target_box)
            target_center = _center_similarity(component_box, target_box, w, h, 1.0)
            target_affinity = 0.7 * target_iou + 0.3 * target_center
            spatial_score *= 0.2 + 0.8 * target_affinity
        track_affinity = 0.0
        track_iou = 0.0
        if previous_box is not None:
            track_iou = _bbox_iou(component_box, previous_box)
            track_center = _center_similarity(
                component_box, previous_box, w, h, cfg.tracking_max_jump_ratio
            )
            previous_area = max(float(previous_roi.area_ratio), 1e-6)
            area_similarity = float(np.exp(-abs(np.log(max(area_ratio, 1e-6) / previous_area))))
            track_affinity = 0.55 * track_iou + 0.35 * track_center + 0.10 * area_similarity
            # 不同拍摄阶段可能从小晶体切换到大晶体。面积相差过大时，
            # 不把它们误当作同一条轨迹，否则会把目录中的多组采集锁在错误目标上。
            if area_similarity >= 0.20:
                tracked_records.append((label, base_score, component_box, track_affinity, track_iou))
        scored_records.append((base_score * spatial_score, label, component_box))

    # 只有存在“靠近上一帧”的候选时才施加跟踪门控；若目标暂时丢失，保留显著性
    # 定位能力，避免一次遮挡导致轨迹永久锁死。
    if previous_box is not None and tracked_records:
        near = [item for item in tracked_records if item[3] > 0.0 or item[4] > 0.02]
        if near:
            near_labels = {item[0] for item in near}
            tracked_affinity = {item[0]: item[3] for item in tracked_records}
            scored_records = [
                (
                    score * (0.35 + 0.65 * tracked_affinity[label])
                    if label in near_labels else score * 0.12,
                    label,
                    box,
                )
                for score, label, box in scored_records
            ]

    if scored_records:
        best_score, best_label, best_component_box = max(
            scored_records, key=lambda item: item[0]
        )
    else:
        best_label, best_score, best_component_box = -1, -1.0, (0, 0, 0, 0)

    if best_label < 0:
        warnings.append("显著性定位未找到有效晶体块，回退整幅中心区域。")
        if target_box is not None:
            x1, y1, x2, y2 = target_box
            return RoiResult(
                (x1, y1, x2, y2), ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                0.0, "fullframe", 0.0, saliency_vis, found=False,
                warnings=warnings, component_bbox=target_box,
            )
        side = int(short_side * 0.5)
        x1, y1 = (w - side) // 2, (h - side) // 2
        fallback_box = (x1, y1, x1 + side, y1 + side)
        return RoiResult((x1, y1, x1 + side, y1 + side), (w * 0.5, h * 0.5), 0.0,
                         "fullframe", 0.0, saliency_vis, found=False, warnings=warnings,
                         component_bbox=fallback_box)

    x, y, x2_component, y2_component = best_component_box
    bw = x2_component - x
    bh = y2_component - y
    area_ratio = float(stats[best_label, cv2.CC_STAT_AREA]) / image_area
    cx, cy = float(centroids[best_label][0]), float(centroids[best_label][1])

    # 自适应 ROI：外扩 + 最小边限制
    pad_x, pad_y = int(bw * cfg.roi_pad_ratio), int(bh * cfg.roi_pad_ratio)
    x1, y1 = x - pad_x, y - pad_y
    x2, y2 = x + bw + pad_x, y + bh + pad_y
    min_side = int(short_side * cfg.min_roi_side_ratio)
    if (x2 - x1) < min_side:
        grow = (min_side - (x2 - x1)) // 2
        x1, x2 = x1 - grow, x2 + grow
    if (y2 - y1) < min_side:
        grow = (min_side - (y2 - y1)) // 2
        y1, y2 = y1 - grow, y2 + grow
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)

    scale = "small" if area_ratio < 0.02 else ("medium" if area_ratio < 0.15 else "large")
    return RoiResult((x1, y1, x2, y2), (cx, cy), float(min(best_score, 1.0)),
                     scale, area_ratio, saliency_vis, warnings=warnings,
                     component_bbox=(x, y, x + bw, y + bh))
