# -*- coding: utf-8 -*-
"""第一阶段编排（像素域）。

对每一帧：预处理 -> 边缘提取 -> （可选 SAM2）-> 最大晶体剪影 -> 长方体+四棱锥线框拟合。
多帧时对 fit_ready 的帧做中位聚合，得到一套标准像素几何；单图时直接使用该图结果。

产出（命名见 crystalvol/io.py OutputLayout）：
- contours/<name>_wireframe.png   轮廓提取图
- overlays/<name>_overlay.png     overlay 图
- geometry/standard_geometry_pixel_preview.png(.json/.obj)  重建几何体
- edges/ masks/ enhanced/ debug/  过程证据
- stage1_result.json / .txt       汇总
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import Stage1Config
from .camera_parameters import CameraParameters, load_camera_calibration, undistort_image_for_calibration
from .edges import canny_edge_map, compute_edge_map
from .geometry import build_vertices, compute_volume, edge_index_pairs
from .io import VIDEO_EXTENSIONS, InputFrame, OutputLayout, _imwrite, iter_inputs
from .localize import RoiResult, _normalised_target_box, _normalised_target_point, locate_crystal
from .logging_utils import log, section, warn
from .metric import convert_pixel_to_metric
from .preprocess import run_preprocess
from .segmentation import CrystalSegmenter
from .silhouette import extract_silhouette
from .visualize import render_contour_image, render_geometry_preview, render_overlay, write_obj
from .wireframe import WireframeResult, fit_wireframe


class Stage1Cancelled(RuntimeError):
    """任务被外部取消（Web/UI 中断请求）。"""


@dataclass
class FrameOutput:
    """单帧的全部中间结果（几何量以整幅像素为单位，与 ROI 平移无关）。"""

    frame: InputFrame
    enhanced_bgr: np.ndarray | None   # 整幅增强图
    roi: RoiResult                    # 定位得到的 ROI
    roi_bgr: np.ndarray | None        # ROI 裁剪（增强）
    edge_map: np.ndarray | None       # ROI 内融合边缘
    mask: np.ndarray | None            # ROI 内最大晶体剪影
    silhouette_contour: np.ndarray | None  # ROI 坐标下的外轮廓
    wireframe: WireframeResult        # ROI 坐标下的线框（geometry_px 为像素长度）
    edge_backend: str
    sam2_used: bool
    warnings: List[str]
    image_size: tuple[int, int]
    candidate_summaries: List[Dict[str, object]] = field(default_factory=list)
    selected_candidate: str = ""
    selection_confidence: float = 0.0
    selection_margin: float = 0.0


def _score_candidate(wf: WireframeResult) -> tuple:
    """多后端择优排序键：优先 fit_ready，其次可见比例、覆盖率。"""
    return (1 if wf.fit_ready else 0, wf.visible_ratio, wf.coverage_ratio)


def _candidate_specs(cfg: Stage1Config) -> list[str]:
    """返回一帧需要尝试的候选算法名称。

    ``auto`` 保持原有行为：优先比较 pidinet+canny 与 canny。显式候选可以把
    HED、PiDiNet 单独结果和 LSD 加入同一评分池，但不会改变预处理、ROI 或
    SAM2 的共享结果。
    """
    configured = [str(item).strip().lower() for item in cfg.edge.candidate_backends if str(item).strip()]
    if configured:
        return list(dict.fromkeys(configured))
    backend = cfg.edge.backend.strip().lower()
    if backend == "auto":
        return ["pidinet+canny", "canny"]
    if backend in {"pidinet", "hed"} and cfg.edge.fuse_canny:
        return [f"{backend}+canny"]
    return [backend]


def _candidate_edge_config(cfg: Stage1Config, candidate: str):
    """把候选名称转换为边缘后端及是否融合 Canny 的配置。"""
    if candidate.endswith("+canny"):
        backend = candidate.removesuffix("+canny")
        if backend not in {"pidinet", "hed"}:
            raise ValueError(f"不支持的融合候选: {candidate}")
        return backend, replace(cfg.edge, fuse_canny=True)
    if candidate not in {"canny", "pidinet", "hed", "lsd"}:
        raise ValueError(f"不支持的边缘候选: {candidate}")
    # 显式写 pidinet/hed 表示深度边缘单独结果；融合版本必须写 +canny。
    return candidate, replace(cfg.edge, fuse_canny=False)


def _candidate_quality(
    wf: WireframeResult,
    cfg: Stage1Config,
) -> tuple[float, dict[str, float]]:
    """计算可解释的候选自洽度分数（0~1）。

    这不是经过真值数据校准的统计概率，而是将边缘支持、剪影覆盖、形状先验
    和几何有效性组合后的质量分。第二阶段会继续用物理约束复评。
    """
    visible = float(np.clip(wf.visible_ratio, 0.0, 1.0))
    coverage = float(wf.coverage_ratio)
    min_coverage = max(float(cfg.wireframe.min_coverage_ratio), 1e-6)
    if coverage < min_coverage:
        coverage_quality = float(np.clip(coverage / min_coverage, 0.0, 1.0))
    elif coverage <= 0.75:
        coverage_quality = float(np.clip(coverage / max(min_coverage * 3.0, 0.12), 0.0, 1.0))
    else:
        # 接近满幅的候选更容易把背景/黑条当成晶体，保留但降低可信度。
        coverage_quality = float(np.clip(1.0 - (coverage - 0.75) / 0.25, 0.0, 1.0))
    shape = float(np.clip(wf.geometry_px.get("shape_prior_confidence", 0.0), 0.0, 1.0))
    geometry_values = [
        wf.geometry_px.get(key, 0.0)
        for key in ("length_px", "width_px", "body_height_px", "pyramid_height_px", "volume_px3")
    ]
    geometry_valid = 1.0 if all(np.isfinite(float(value)) and float(value) > 0 for value in geometry_values) else 0.0
    ready = 1.0 if wf.fit_ready else 0.0
    breakdown = {
        "edge_support": visible,
        "coverage_quality": coverage_quality,
        "shape_prior": shape,
        "geometry_validity": geometry_valid,
        "fit_ready": ready,
    }
    score = (
        0.35 * visible
        + 0.15 * coverage_quality
        + 0.20 * shape
        + 0.15 * geometry_valid
        + 0.15 * ready
    )
    return float(np.clip(score, 0.0, 1.0)), breakdown


def _candidate_summary(
    candidate: str,
    edge_backend: str,
    wf: WireframeResult,
    score: float,
    breakdown: dict[str, float],
) -> Dict[str, object]:
    return {
        "candidate": candidate,
        "backend_used": edge_backend,
        "fallback": edge_backend == "canny(fallback)",
        "status": "ok",
        "score": score,
        "confidence": score,
        "score_breakdown": breakdown,
        "fit_ready": bool(wf.fit_ready),
        "visible_ratio": float(wf.visible_ratio),
        "coverage_ratio": float(wf.coverage_ratio),
        "geometry_px": {key: float(value) for key, value in wf.geometry_px.items()},
        "warnings": list(wf.warnings),
    }


def _shift_wireframe(wf: WireframeResult, dx: float, dy: float) -> WireframeResult:
    """把线框关键点平移到整幅坐标（几何量不变）。"""
    shifted = {name: (p[0] + dx, p[1] + dy) for name, p in wf.canonical_points.items()}
    return WireframeResult(
        canonical_points=shifted, observed_edges=wf.observed_edges, geometry_px=wf.geometry_px,
        fit_ready=wf.fit_ready, visible_ratio=wf.visible_ratio, coverage_ratio=wf.coverage_ratio,
        depth_source=wf.depth_source, warnings=wf.warnings,
    )


def _release_candidate_buffers(out: FrameOutput) -> None:
    """释放未入选候选的大数组；共享的输入帧和 ROI 不在这里清除。"""
    out.enhanced_bgr = None
    out.roi_bgr = None
    out.edge_map = None
    out.mask = None
    out.silhouette_contour = None


def _process_frame(
    frame: InputFrame,
    cfg: Stage1Config,
    segmenter: Optional[CrystalSegmenter],
    previous_roi: Optional[RoiResult] = None,
    calibration: Optional[CameraParameters] = None,
) -> FrameOutput:
    """处理单帧：预处理 -> 显著性定位 ROI -> ROI 内(SAM2/边缘/剪影/线框)。"""
    if frame.image_bgr is None:
        raise ValueError(f"[{frame.name}] 输入图像缓冲为空。")
    image_bgr = frame.image_bgr
    calibration_warning = None
    if cfg.undistort and calibration is not None:
        try:
            image_bgr = undistort_image_for_calibration(image_bgr, calibration)
            frame.image_bgr = image_bgr
        except RuntimeError as exc:
            # 宽高比不一致通常意味着裁剪/采集模式改变；保留像素域流程，
            # 但把问题写入帧告警，避免一帧异常拖垮整个目录处理。
            calibration_warning = f"去畸变跳过：{exc}"
            warn(f"[{frame.name}] {calibration_warning}")
    pre = run_preprocess(image_bgr, cfg.preprocess)
    h, w = pre.enhanced_bgr.shape[:2]

    # 1) 显著性定位：先在整幅上用一张廉价 Canny 找到最大晶体，裁自适应 ROI
    if cfg.localize.enable:
        coarse_edge = canny_edge_map(pre.enhanced_bgr, cfg.edge)
        roi = locate_crystal(
            pre.enhanced_bgr, coarse_edge, pre.specular_mask, cfg.localize,
            previous_roi=previous_roi,
        )
    else:
        roi = RoiResult((0, 0, w, h), (w * 0.5, h * 0.5), 1.0, "fullframe", 1.0,
                        np.zeros((h, w), np.uint8), found=True,
                        component_bbox=(0, 0, w, h))
    x1, y1, x2, y2 = roi.bbox
    roi_bgr = pre.enhanced_bgr[y1:y2, x1:x2]
    specular_roi = pre.specular_mask[y1:y2, x1:x2]

    # 2) ROI 内 SAM2（已修复 Mac 上失效问题；用晶体质心作正点提示）
    sam2_mask = None
    sam2_used = False
    if segmenter is not None:
        try:
            point = (roi.center[0] - x1, roi.center[1] - y1)
            # 自动显著性块可能只覆盖晶体的一条亮棱；在这类透明/强反光图上，
            # 直接把它当作 SAM2 box prompt 会把掩膜锁在局部。只有用户明确给出
            # 固定目标框时才信任该框，默认继续使用质心点提示以保持基线稳定性。
            prompt_box = None
            if cfg.localize.preselection_enabled and cfg.localize.preselection_roi is not None:
                manual_box = _normalised_target_box(cfg.localize.preselection_roi, w, h)
                manual_point = _normalised_target_point(cfg.localize.preselection_point, w, h)
                if manual_box is not None:
                    prompt_box = (
                        max(manual_box[0] - x1, 0), max(manual_box[1] - y1, 0),
                        min(manual_box[2] - x1, x2 - x1), min(manual_box[3] - y1, y2 - y1),
                    )
                    if manual_point is not None:
                        point = (manual_point[0] - x1, manual_point[1] - y1)
            elif cfg.localize.target_roi is not None:
                component_box = roi.component_bbox
                prompt_box = (
                    max(component_box[0] - x1, 0), max(component_box[1] - y1, 0),
                    min(component_box[2] - x1, x2 - x1), min(component_box[3] - y1, y2 - y1),
                )
                if prompt_box[2] - prompt_box[0] < 3 or prompt_box[3] - prompt_box[1] < 3:
                    prompt_box = None
            sam2_mask = segmenter.segment(
                roi_bgr, prompt_box=prompt_box, positive_point=point
            ).mask
            sam2_used = True
        except Exception as exc:
            warn(f"[{frame.name}] SAM2 分割失败，转纯边缘剪影：{exc}")

    # 3) ROI 内多后端边缘 + 剪影 + 线框，按线框质量择优
    candidates_to_try = _candidate_specs(cfg)
    # 亮核收紧只针对「小晶体被背光光晕撑大」这个问题；大/满幅晶体本身就占大片，
    # 收紧会把它缩到最亮的一个刻面，因此大晶体不做亮核收紧。
    tighten_gray = None if roi.scale in ("large", "fullframe") else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    best: Optional[FrameOutput] = None
    best_summary: Dict[str, object] | None = None
    candidate_summaries: list[Dict[str, object]] = []
    for candidate_name in candidates_to_try:
        try:
            backend, edge_cfg = _candidate_edge_config(cfg, candidate_name)
            edge_result = compute_edge_map(roi_bgr, specular_roi, edge_cfg, backend=backend)
            sil = extract_silhouette(
                edge_result.edge_map, sam2_mask,
                roi_gray=tighten_gray, core_percentile=cfg.wireframe.core_percentile,
            )
            wf = fit_wireframe(sil, edge_result.edge_map, cfg.wireframe)
            candidate = FrameOutput(
                frame=frame, enhanced_bgr=pre.enhanced_bgr, roi=roi, roi_bgr=roi_bgr,
                edge_map=edge_result.edge_map, mask=sil.mask, silhouette_contour=sil.contour,
                wireframe=wf, edge_backend=edge_result.backend_used, sam2_used=sam2_used,
                warnings=(list(roi.warnings) + list(edge_result.warnings) + list(wf.warnings)
                          + ([calibration_warning] if calibration_warning else [])),
                image_size=(w, h),
            )
            score, breakdown = _candidate_quality(wf, cfg)
            summary = _candidate_summary(candidate_name, edge_result.backend_used, wf, score, breakdown)
            candidate_summaries.append(summary)
            if best is None or float(summary["score"]) > float(best_summary["score"]):
                if best is not None:
                    _release_candidate_buffers(best)
                best = candidate
                best_summary = summary
            else:
                _release_candidate_buffers(candidate)
        except Exception as exc:  # noqa: BLE001
            candidate_summaries.append({
                "candidate": candidate_name,
                "status": "failed",
                "score": 0.0,
                "confidence": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            })
    if best is None:
        details = "；".join(
            f"{item['candidate']}: {item.get('error', 'unknown error')}"
            for item in candidate_summaries
        )
        raise RuntimeError(f"[{frame.name}] 没有可用的边缘候选。{details}")
    successful = sorted(
        (item for item in candidate_summaries if item.get("status") == "ok"),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    top_k = max(int(cfg.candidate_top_k), 1)
    retained = successful[:top_k]
    for rank, item in enumerate(retained, 1):
        item["rank"] = rank
        item["selected"] = str(item["candidate"]) == str(best_summary["candidate"])
    # 保留失败候选的错误摘要，便于判断是算法没得分还是模型/权重不可用。
    retained.extend(item for item in candidate_summaries if item.get("status") == "failed")
    second_score = float(successful[1]["score"]) if len(successful) > 1 else 0.0
    best.selected_candidate = str(best_summary["candidate"])
    best.selection_confidence = float(best_summary["score"])
    best.selection_margin = float(best_summary["score"]) - second_score
    best.candidate_summaries = retained
    log(f"[{frame.name}] roi={roi.scale}({roi.area_ratio*100:.2f}%) sam2={sam2_used} "
        f"backend={best.edge_backend} candidate={best.selected_candidate} "
        f"confidence={best.selection_confidence:.3f} margin={best.selection_margin:.3f} "
        f"fit_ready={best.wireframe.fit_ready} "
        f"visible={best.wireframe.visible_ratio:.2f} coverage={best.wireframe.coverage_ratio:.3f} "
        f"L={best.wireframe.geometry_px['length_px']:.0f} Hb={best.wireframe.geometry_px['body_height_px']:.0f} "
        f"Hp={best.wireframe.geometry_px['pyramid_height_px']:.0f} lowlight={pre.applied_lowlight}(luma={pre.mean_luma:.1f})")
    return best


def _frame_weight(f: FrameOutput) -> float:
    """帧质量权重：可见比例 + 覆盖率（都做温和下限，避免 0 权重）。"""
    return (0.5 + f.wireframe.visible_ratio) * (0.5 + min(f.wireframe.coverage_ratio, 0.5))


def _select_consensus_pool(frame_outputs: List[FrameOutput]) -> List[FrameOutput]:
    """挑选参与联合拟合的帧（质量加权共识，而非只用 fit_ready 单帧）。

    一段视频是同一个晶体、同一尺寸的多帧采样，所以这里是「跨帧共识出一个晶体」。
    放宽到「证据尚可」的帧一起参与，再靠加权 + 离群剔除得到稳健结果；
    帧间差异大时（有的帧几乎看不见），低质量帧权重低、离群帧被剔除。"""
    cand = [f for f in frame_outputs
            if f.wireframe.visible_ratio >= 0.33
            and f.wireframe.geometry_px["length_px"] > 5
            and 0.03 <= f.wireframe.coverage_ratio <= 0.9]
    if cand:
        return cand
    # 全部偏弱：退而取质量最高的前 3 帧
    return sorted(frame_outputs, key=lambda f: _score_candidate(f.wireframe), reverse=True)[:3]

def _robust_values(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按 MAD 剔除离群值，返回 (保留值, 保留权重)。"""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-6
    keep = np.abs(values - median) <= 3.5 * mad
    if keep.sum() >= 1:
        return values[keep], weights[keep]
    return values, weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / max(np.sum(weights), 1e-9))


def _volume_uncertainty_pct(depth_source: str, confidence: float) -> float:
    """由深度来源与形状先验置信度估计体积相对误差（透明启发式，用于 UI 标注）。

    体积 V = L×W×(Hb+Hp/3)，其中 W（单目深度）是不确定性的主要来源：
    多视角投影范围是观测约束，误差带更小；纯单视角形状先验默认至少 ±20%。
    """
    conf = float(np.clip(confidence, 0.0, 1.0))
    if depth_source == "cross_frame_projection_range":
        return round(10.0 + (1.0 - conf) * 15.0, 1)
    return round(20.0 + (1.0 - conf) * 40.0, 1)


def _consolidate_geometry(pool: List[FrameOutput], cfg: Stage1Config) -> Dict[str, float]:
    """把多帧鲁棒共识成单一晶体几何。

    顺序：先按 MAD 剔除离群帧并做质量加权，合并观测到的 length/body/pyramid；
    深度方向 width 优先用「跨帧投影宽度范围」约束——多视角拍摄时，投影宽度的
    上/下分位观测分别约束 L/W；否则回退每帧形状先验的比例（W = L × 比例，
    保证 V=L×W×(Hb+Hp/3) 内部一致，而不是四个参数各自独立平均后比例失真）。
    """
    weights_all = np.array([_frame_weight(f) for f in pool], dtype=np.float64)

    def robust(key: str) -> float:
        vals = np.array([f.wireframe.geometry_px[key] for f in pool], dtype=np.float64)
        kept, kept_w = _robust_values(vals, weights_all)
        return _weighted_mean(kept, kept_w)

    length = robust("length_px")
    body = robust("body_height_px")
    pyramid = robust("pyramid_height_px")

    # 跨帧投影宽度约束：多视角下 length_px 的观测范围同时编码 L 与 W。
    length_vals = np.array(
        [f.wireframe.geometry_px["length_px"] for f in pool], dtype=np.float64
    )
    kept_len, kept_len_w = _robust_values(length_vals, weights_all)
    spread = (
        (float(kept_len.max()) - float(kept_len.min())) / max(float(np.median(kept_len)), 1e-9)
        if len(kept_len) >= 2 else 0.0
    )
    threshold = max(float(cfg.cross_frame_depth_min_spread), 1e-3)
    use_cross_frame = len(kept_len) >= 2 and spread >= threshold
    if use_cross_frame:
        # 投影宽度上分位 ≈ 正对视图（L），下分位 ≈ 侧向视图（W）。
        med_len = float(np.median(kept_len))
        hi = kept_len >= med_len
        lo = kept_len <= med_len
        length = _weighted_mean(kept_len[hi], kept_len_w[hi])
        width = _weighted_mean(kept_len[lo], kept_len_w[lo])
        width = min(width, length)  # 防御：退化解
        depth_source = "cross_frame_projection_range"
        confidence = float(np.clip(0.55 + spread, 0.6, 1.0))
    else:
        # 回退形状先验：比例来自每帧自适应先验，W = L × 比例（一致性修正）。
        ratio = robust("depth_ratio_estimate")
        width = length * ratio
        depth_source = "adaptive_single_view_shape_prior"
        confidence = robust("shape_prior_confidence")

    return {
        "length_px": length, "width_px": width,
        "body_height_px": body, "pyramid_height_px": pyramid,
        "total_height_px": body + pyramid,
        "volume_px3": compute_volume(length, width, body, pyramid),
        "depth_source": depth_source,
        "volume_uncertainty_pct": _volume_uncertainty_pct(depth_source, confidence),
    }


def _consolidate_candidate_records(records: list[Dict[str, object]]) -> Dict[str, float]:
    """把同一候选算法在多帧上的几何摘要做鲁棒聚合。"""
    if not records:
        raise ValueError("候选记录为空，无法聚合几何")
    weights_all = np.asarray([
        (0.5 + float(record.get("visible_ratio", 0.0)))
        * (0.5 + min(float(record.get("coverage_ratio", 0.0)), 0.5))
        * max(float(record.get("score", 0.0)), 0.1)
        for record in records
    ], dtype=np.float64)
    def robust(key: str) -> float:
        values = np.asarray([
            float(dict(record.get("geometry_px", {})).get(key, 0.0))
            for record in records
        ], dtype=np.float64)
        kept, kept_w = _robust_values(values, weights_all)
        return _weighted_mean(kept, kept_w)

    # W 由 L × 深度比例推导，避免候选几何内部比例失真（与主聚合一致）。
    length = robust("length_px")
    body = robust("body_height_px")
    pyramid = robust("pyramid_height_px")
    ratio = robust("depth_ratio_estimate")
    width = length * ratio
    return {
        "length_px": length,
        "width_px": width,
        "body_height_px": body,
        "pyramid_height_px": pyramid,
        "total_height_px": body + pyramid,
        "volume_px3": compute_volume(length, width, body, pyramid),
    }


def _candidate_geometry_options(pool: List[FrameOutput]) -> list[Dict[str, object]]:
    """生成第二阶段可复评的候选几何集合。

    ``per_frame_ensemble`` 是第一阶段逐帧择优后的实际结果；其余条目是同一
    算法候选跨帧聚合后的备选结果。这样第二阶段可以用物理约束重新改判，且
    不需要重新加载原始图片或模型。
    """
    groups: dict[str, list[Dict[str, object]]] = {"per_frame_ensemble": []}
    for frame_output in pool:
        selected = next(
            (
                item for item in frame_output.candidate_summaries
                if item.get("status") == "ok" and item.get("selected")
            ),
            None,
        )
        if selected is None:
            selected = {
                "candidate": frame_output.selected_candidate or frame_output.edge_backend,
                "score": frame_output.selection_confidence,
                "visible_ratio": frame_output.wireframe.visible_ratio,
                "coverage_ratio": frame_output.wireframe.coverage_ratio,
                "geometry_px": frame_output.wireframe.geometry_px,
            }
        groups["per_frame_ensemble"].append(dict(selected))
        for item in frame_output.candidate_summaries:
            if item.get("status") != "ok":
                continue
            candidate_name = str(item.get("candidate", "unknown"))
            if item.get("fallback") and str(item.get("backend_used", "")).startswith("canny"):
                # 深度权重不可用时的结果与 Canny 是同一张证据图，不要在第二阶段
                # 把它重复计算成两个互相竞争的候选。
                candidate_name = "canny"
            groups.setdefault(candidate_name, []).append(item)

    options: list[Dict[str, object]] = []
    for candidate_name, records in groups.items():
        geometry = _consolidate_candidate_records(records)
        scores = np.asarray([float(item.get("score", 0.0)) for item in records], dtype=np.float64)
        options.append({
            "candidate": candidate_name,
            "stage1_score": float(np.mean(scores)) if len(scores) else 0.0,
            "stage1_score_std": float(np.std(scores)) if len(scores) else 0.0,
            "frame_count": len(records),
            "geometry_params_px": {
                key: float(value) for key, value in geometry.items() if key != "volume_px3"
            },
            "volume_px3": float(geometry["volume_px3"]),
        })
    return sorted(options, key=lambda item: float(item["stage1_score"]), reverse=True)


def run_stage1(
    cfg: Stage1Config,
    progress_callback: Optional[Callable[[int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, object]:
    """第一阶段主入口。

    progress_callback: 每成功处理一帧后调用，参数为已处理帧数（供 Web 进度轮询）。
    should_cancel:     返回 True 时在下一帧开始前抛 Stage1Cancelled，任务可安全中断。
    """
    section("第一阶段：像素域轮廓与线框重建")
    frames = iter_inputs(
        cfg.input_path, cfg.num_frames, cfg.frame_start_ratio, cfg.frame_end_ratio, cfg.max_input_side,
    )
    log(f"开始流式读取输入（输入: {cfg.input_path}）")

    layout = OutputLayout(Path(cfg.output_dir).expanduser().resolve()).prepare(clean=cfg.clean_output)
    log(f"输出目录: {layout.root}")

    segmenter = build_segmenter(cfg)

    calibration: Optional[CameraParameters] = None
    if cfg.undistort:
        try:
            calibration = load_camera_calibration(cfg.camera_parameters)
            log(f"启用相机去畸变：{calibration.source_path}")
        except Exception as exc:  # noqa: BLE001
            warn(f"相机参数加载失败，跳过去畸变并继续像素域处理：{exc}")

    frame_outputs: List[FrameOutput] = []
    failed_frames: list[dict[str, str]] = []
    processed_count = 0
    previous_roi: Optional[RoiResult] = None
    input_file = Path(cfg.input_path).expanduser()
    tracking_active = bool(
        cfg.localize.tracking_enabled
        and (
            cfg.localize.preselection_enabled
            or cfg.localize.tracking_force
            or (input_file.is_file() and input_file.suffix.lower() in VIDEO_EXTENSIONS)
        )
    )
    if cfg.localize.tracking_enabled and not tracking_active:
        log("图片目录默认逐帧定位；如需跨图跟踪请显式设置 target_tracking。")
    try:
        for frame in frames:
            if should_cancel is not None and should_cancel():
                raise Stage1Cancelled("任务已被取消")
            try:
                out = _process_frame(
                    frame, cfg, segmenter,
                    previous_roi=previous_roi if tracking_active else None,
                    calibration=calibration,
                )
                write_frame_products(layout, out)
                _release_frame_buffers(out)
            except Exception as exc:  # noqa: BLE001
                # 单帧异常不应让整批任务崩溃；保留错误并继续处理后续视图。
                failed_frames.append({"name": frame.name, "error": f"{type(exc).__name__}: {exc}"})
                warn(f"[{frame.name}] 单帧处理失败，跳过并继续：{exc}")
                frame.image_bgr = None
                continue
            frame_outputs.append(out)
            previous_roi = out.roi if tracking_active else None
            processed_count += 1
            if progress_callback is not None:
                progress_callback(processed_count)
    except Exception:
        if segmenter is not None:
            segmenter.close()
        raise

    if not frame_outputs:
        details = "；".join(f"{item['name']}: {item['error']}" for item in failed_frames)
        if segmenter is not None:
            segmenter.close()
        raise RuntimeError(f"所有输入帧均处理失败。{details}")

    try:
        summary = finalize_stage1(cfg, layout, frame_outputs)
    finally:
        if segmenter is not None:
            segmenter.close()
    summary["failed_frames"] = failed_frames
    layout.result_json().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"第一阶段完成：fit_ready {summary['fit_ready_count']}/{processed_count} 帧")
    g = summary["geometry_px"]
    log(f"像素几何：length={g['length_px']:.1f}px width={g['width_px']:.1f}px "
        f"body_h={g['body_height_px']:.1f}px pyramid_h={g['pyramid_height_px']:.1f}px "
        f"volume_px3={g['volume_px3']:.3e}")
    return summary


def _release_frame_buffers(out: FrameOutput) -> None:
    """释放单帧推理的大数组，只保留共识和结果 JSON 所需元数据。"""
    out.frame.image_bgr = None
    out.enhanced_bgr = None
    out.roi_bgr = None
    out.edge_map = None
    out.mask = None
    out.silhouette_contour = None
    # saliency 是整幅可视化图，已写入 debug 后不再需要常驻内存。
    out.roi.saliency = np.empty((0, 0), dtype=np.uint8)


def build_segmenter(cfg: Stage1Config) -> Optional[CrystalSegmenter]:
    """按配置构造 SAM2 分割前端；初始化失败时返回 None（退化为纯边缘剪影）。

    单独抽出以便增量会话（session.py）复用同一个常驻分割器，避免每张照片重新加载模型。
    """
    if not cfg.segmentation.enable:
        return None
    try:
        return CrystalSegmenter(cfg.segmentation)
    except Exception as exc:  # noqa: BLE001
        warn(f"SAM2 分割前端初始化失败，转为纯边缘剪影：{exc}")
        return None


def write_frame_products(layout: OutputLayout, out: FrameOutput) -> None:
    """把单帧的全部产物图落盘（inputs/enhanced/edges/masks/contours/overlays/debug）。

    批处理与增量会话共用，保证两条路径产物命名与内容完全一致。
    """
    frame = out.frame
    x1, y1, x2, y2 = out.roi.bbox
    # 映射回整幅坐标用于 overlay
    wf_full = _shift_wireframe(out.wireframe, x1, y1)
    if out.silhouette_contour is None:
        raise RuntimeError(f"帧 {frame.name} 的轮廓缓冲已释放，不能重复写入产物")
    contour_full = (out.silhouette_contour + np.array([x1, y1], np.float32)
                    if out.silhouette_contour.size else out.silhouette_contour)
    overlay = render_overlay(out.enhanced_bgr, contour_full, wf_full)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), max(int(min(overlay.shape[:2]) / 500), 2))
    if out.roi.mode == "manual_anchor":
        ax1, ay1, ax2, ay2 = out.roi.anchor_bbox
        if ax2 > ax1 and ay2 > ay1:
            # 青色=人工首帧锚点，黄色=当前处理搜索窗；线框仍由 render_overlay 绘制。
            cv2.rectangle(
                overlay, (ax1, ay1), (ax2, ay2), (255, 255, 0),
                max(int(min(overlay.shape[:2]) / 700), 2),
            )
    # 落盘每帧产物（轮廓提取图在 ROI 上渲染 -> 小晶体自动放大）
    def write_required(path: Path, image: np.ndarray | None) -> None:
        if image is None or not _imwrite(str(path), image):
            raise RuntimeError(f"无法写入图像产物（磁盘空间、权限或编码失败）: {path}")

    if frame.image_bgr is None or out.roi_bgr is None or out.enhanced_bgr is None:
        raise RuntimeError(f"帧 {frame.name} 的图像缓冲已释放，不能写入产物")
    write_required(layout.input_frame(frame.name), frame.image_bgr)
    write_required(layout.enhanced(frame.name), out.enhanced_bgr)
    write_required(layout.edge(frame.name), out.edge_map)
    write_required(layout.mask(frame.name), out.mask)
    write_required(
        layout.contour(frame.name),
        render_contour_image(out.roi_bgr.shape, out.silhouette_contour, out.wireframe),
    )
    write_required(layout.overlay(frame.name), overlay)
    write_required(layout.debug(frame.name, "saliency"), out.roi.saliency)
    write_required(layout.debug(frame.name, "roi"), out.roi_bgr)


def finalize_stage1(cfg: Stage1Config, layout: OutputLayout,
                    frame_outputs: List[FrameOutput]) -> Dict[str, object]:
    """对已处理好的一组帧做跨帧联合拟合、渲染几何、写汇总 JSON，返回 summary。

    批处理跑完全部帧后调用一次；增量会话每加入一张照片后调用一次（对当前累积的
    全部帧重新联合拟合），因此「累积 + 重新联合拟合」的增量语义完全落在这里。
    """
    # 跨帧联合拟合出「一个」晶体几何
    pool = _select_consensus_pool(frame_outputs)
    geometry_px = _consolidate_geometry(pool, cfg)
    candidate_options = _candidate_geometry_options(pool)
    pool_names = [f.frame.name for f in pool]
    representative = max(pool, key=lambda f: _score_candidate(f.wireframe))
    log(f"联合拟合使用 {len(pool)}/{len(frame_outputs)} 帧：{pool_names}；代表帧={representative.frame.name}")

    vertices = build_vertices(
        max(geometry_px["length_px"], 1e-3), max(geometry_px["width_px"], 1e-3),
        max(geometry_px["body_height_px"], 1e-3), max(geometry_px["pyramid_height_px"], 1e-3),
    )
    if not _imwrite(str(layout.geometry_preview()), render_geometry_preview(geometry_px)):
        raise RuntimeError(f"无法写入几何预览: {layout.geometry_preview()}")
    write_obj(geometry_px, layout.geometry_obj())

    # 代表帧的三张图另存到输出根目录，方便直接查看「这一个晶体」的结果
    import shutil as _shutil
    _shutil.copyfile(layout.contour(representative.frame.name), layout.root / "crystal_contour.png")
    _shutil.copyfile(layout.overlay(representative.frame.name), layout.root / "crystal_overlay.png")

    fit_ready_count = sum(1 for f in frame_outputs if f.wireframe.fit_ready)
    image_sizes = {tuple(f.image_size) for f in frame_outputs if f.image_size[0] > 0 and f.image_size[1] > 0}
    processing_image_size = list(next(iter(image_sizes))) if len(image_sizes) == 1 else None
    # depth_source / volume_uncertainty_pct 是字符串/元数据，不能放进 stage2 的
    # 纯 float geometry_params_px；作为顶层字段供 UI 与诊断使用。
    geometry_payload = {
        "units": "pixel",
        "geometry_params_px": {k: v for k, v in geometry_px.items() if k not in ("volume_px3", "depth_source")},
        "volume_px3": geometry_px["volume_px3"],
        "selected_candidate": "per_frame_ensemble",
        "candidate_geometries": candidate_options,
        "depth_estimation_source": geometry_px.get("depth_source", "adaptive_single_view_shape_prior"),
        "volume_uncertainty_pct": geometry_px.get("volume_uncertainty_pct", 0.0),
        "vertices_px": vertices.astype(float).tolist(),
        "edge_index_pairs": [list(e) for e in edge_index_pairs()],
        "note": "volume_px3 仅供第一阶段相对比较；真实体积需第二阶段用尺度锚点或外参恢复。",
    }
    layout.geometry_json().write_text(json.dumps(geometry_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可选：一阶段就地做尺度锚点换算（若用户提供了一条真实边长）
    metric_payload = None
    anchor = cfg.metric_anchor
    if anchor.scale_reference_edge and anchor.scale_reference_value:
        gt = {k: v for k, v in {
            "length": anchor.gt_length, "width": anchor.gt_width,
            "body_height": anchor.gt_body_height, "pyramid_height": anchor.gt_pyramid_height,
        }.items() if v is not None}
        metric_payload = convert_pixel_to_metric(
            geometry_px, anchor.scale_reference_edge, anchor.scale_reference_value,
            anchor.metric_length_unit, gt or None,
        )
        (layout.root / "geometry" / "standard_geometry_metric.json").write_text(
            json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input": cfg.input_path,
        "output_dir": str(layout.root),
        "frame_count": len(frame_outputs),
        "fit_ready_count": fit_ready_count,
        "edge_backend": frame_outputs[0].edge_backend if frame_outputs else cfg.edge.backend,
        "selected_candidate": "per_frame_ensemble",
        "candidate_selection": {
            "candidates": candidate_options,
            "selection_margin": (
                float(candidate_options[0]["stage1_score"])
                - float(candidate_options[1]["stage1_score"])
                if len(candidate_options) > 1 else 1.0
            ),
        },
        "consensus_frames": pool_names,
        "consensus_frame_count": len(pool_names),
        "representative_frame": representative.frame.name,
        "processing_image_size": processing_image_size,
        "undistort_requested": bool(cfg.undistort),
        "preselection": {
            "enabled": bool(cfg.localize.preselection_enabled),
            "roi_norm": (
                list(cfg.localize.preselection_roi)
                if cfg.localize.preselection_roi is not None else None
            ),
            "search_margin": float(cfg.localize.preselection_search_margin),
            "max_jump_ratio": float(cfg.localize.preselection_max_jump_ratio),
        },
        "geometry_px": geometry_px,
        "metric": metric_payload,
        "frames": [
            {
                "name": f.frame.name,
                "processing_image_size": list(f.image_size),
                "backend": f.edge_backend,
                "roi_bbox": list(f.roi.bbox),
                "localization_mode": f.roi.mode,
                "roi_anchor_bbox": list(f.roi.anchor_bbox),
                "roi_search_bbox": list(f.roi.search_bbox),
                "roi_component_bbox": list(f.roi.component_bbox),
                "roi_scale": f.roi.scale,
                "area_ratio": f.roi.area_ratio,
                "sam2_used": f.sam2_used,
                "fit_ready": f.wireframe.fit_ready,
                "selected_candidate": f.selected_candidate or f.edge_backend,
                "selection_confidence": f.selection_confidence,
                "selection_margin": f.selection_margin,
                "candidate_scores": f.candidate_summaries,
                "visible_ratio": f.wireframe.visible_ratio,
                "coverage_ratio": f.wireframe.coverage_ratio,
                "geometry_px": f.wireframe.geometry_px,
                "warnings": f.warnings,
            }
            for f in frame_outputs
        ],
    }
    layout.result_json().write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_result_txt(layout, summary)
    return summary


def _write_result_txt(layout: OutputLayout, summary: Dict[str, object]) -> None:
    g = summary["geometry_px"]
    lines = [
        "透明晶体第一阶段（像素域）结果",
        f"输入: {summary['input']}",
        f"帧数: {summary['frame_count']}  fit_ready: {summary['fit_ready_count']}",
        f"边缘后端: {summary['edge_backend']}",
        f"候选策略: {summary.get('selected_candidate', 'legacy')}",
        "",
        "标准像素几何（长方体 + 四棱锥）:",
        f"  length_px         = {g['length_px']:.2f}",
        f"  width_px          = {g['width_px']:.2f}  (单视角深度为启发式估计)",
        f"  body_height_px    = {g['body_height_px']:.2f}",
        f"  pyramid_height_px = {g['pyramid_height_px']:.2f}",
        f"  total_height_px   = {g['total_height_px']:.2f}",
        f"  volume_px3        = {g['volume_px3']:.4e}",
    ]
    if summary.get("metric"):
        m = summary["metric"]
        lines += ["", f"尺度锚点换算体积: {m['volume_m3']:.6e} m^3（单位锚点 {m['scale_reference']['edge']}）"]
    layout.result_txt().write_text("\n".join(lines), encoding="utf-8")
