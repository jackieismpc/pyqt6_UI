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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import Stage1Config
from .edges import canny_edge_map, compute_edge_map
from .geometry import build_vertices, edge_index_pairs
from .io import InputFrame, OutputLayout, _imwrite, iter_inputs
from .localize import RoiResult, locate_crystal
from .logging_utils import log, section, warn
from .metric import convert_pixel_to_metric
from .preprocess import run_preprocess
from .segmentation import CrystalSegmenter
from .silhouette import extract_silhouette
from .visualize import render_contour_image, render_geometry_preview, render_overlay, write_obj
from .wireframe import WireframeResult, fit_wireframe


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


def _score_candidate(wf: WireframeResult) -> tuple:
    """多后端择优排序键：优先 fit_ready，其次可见比例、覆盖率。"""
    return (1 if wf.fit_ready else 0, wf.visible_ratio, wf.coverage_ratio)


def _shift_wireframe(wf: WireframeResult, dx: float, dy: float) -> WireframeResult:
    """把线框关键点平移到整幅坐标（几何量不变）。"""
    shifted = {name: (p[0] + dx, p[1] + dy) for name, p in wf.canonical_points.items()}
    return WireframeResult(
        canonical_points=shifted, observed_edges=wf.observed_edges, geometry_px=wf.geometry_px,
        fit_ready=wf.fit_ready, visible_ratio=wf.visible_ratio, coverage_ratio=wf.coverage_ratio,
        depth_source=wf.depth_source, warnings=wf.warnings,
    )


def _process_frame(frame: InputFrame, cfg: Stage1Config, segmenter: Optional[CrystalSegmenter]) -> FrameOutput:
    """处理单帧：预处理 -> 显著性定位 ROI -> ROI 内(SAM2/边缘/剪影/线框)。"""
    pre = run_preprocess(frame.image_bgr, cfg.preprocess)
    h, w = pre.enhanced_bgr.shape[:2]

    # 1) 显著性定位：先在整幅上用一张廉价 Canny 找到最大晶体，裁自适应 ROI
    if cfg.localize.enable:
        coarse_edge = canny_edge_map(pre.enhanced_bgr, cfg.edge)
        roi = locate_crystal(pre.enhanced_bgr, coarse_edge, pre.specular_mask, cfg.localize)
    else:
        roi = RoiResult((0, 0, w, h), (w * 0.5, h * 0.5), 1.0, "fullframe", 1.0,
                        np.zeros((h, w), np.uint8), found=True)
    x1, y1, x2, y2 = roi.bbox
    roi_bgr = pre.enhanced_bgr[y1:y2, x1:x2]
    specular_roi = pre.specular_mask[y1:y2, x1:x2]

    # 2) ROI 内 SAM2（已修复 Mac 上失效问题；用晶体质心作正点提示）
    sam2_mask = None
    sam2_used = False
    if segmenter is not None:
        try:
            point = (roi.center[0] - x1, roi.center[1] - y1)
            sam2_mask = segmenter.segment(roi_bgr, positive_point=point).mask
            sam2_used = True
        except Exception as exc:
            warn(f"[{frame.name}] SAM2 分割失败，转纯边缘剪影：{exc}")

    # 3) ROI 内多后端边缘 + 剪影 + 线框，按线框质量择优
    backends_to_try = ["pidinet", "canny"] if cfg.edge.backend == "auto" else [cfg.edge.backend]
    # 亮核收紧只针对「小晶体被背光光晕撑大」这个问题；大/满幅晶体本身就占大片，
    # 收紧会把它缩到最亮的一个刻面，因此大晶体不做亮核收紧。
    tighten_gray = None if roi.scale in ("large", "fullframe") else cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    best: Optional[FrameOutput] = None
    for backend in backends_to_try:
        edge_result = compute_edge_map(roi_bgr, specular_roi, cfg.edge, backend=backend)
        sil = extract_silhouette(edge_result.edge_map, sam2_mask,
                                 roi_gray=tighten_gray, core_percentile=cfg.wireframe.core_percentile)
        wf = fit_wireframe(sil, edge_result.edge_map, cfg.wireframe)
        candidate = FrameOutput(
            frame=frame, enhanced_bgr=pre.enhanced_bgr, roi=roi, roi_bgr=roi_bgr,
            edge_map=edge_result.edge_map, mask=sil.mask, silhouette_contour=sil.contour,
            wireframe=wf, edge_backend=edge_result.backend_used, sam2_used=sam2_used,
            warnings=list(roi.warnings) + list(edge_result.warnings) + list(wf.warnings),
        )
        if best is None or _score_candidate(wf) > _score_candidate(best.wireframe):
            best = candidate
    if best is None:
        raise RuntimeError(f"[{frame.name}] 没有可用的边缘处理后端")
    log(f"[{frame.name}] roi={roi.scale}({roi.area_ratio*100:.2f}%) sam2={sam2_used} "
        f"backend={best.edge_backend} fit_ready={best.wireframe.fit_ready} "
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


def _consolidate_geometry(pool: List[FrameOutput]) -> Dict[str, float]:
    """把多帧鲁棒共识成单一晶体几何。

    每个参数：先按 MAD 剔除离群帧（如某帧 length 明显偏大），再对剩余帧做质量加权平均。"""
    from .geometry import compute_volume

    weights_all = np.array([_frame_weight(f) for f in pool], dtype=np.float64)

    def robust(key: str) -> float:
        vals = np.array([f.wireframe.geometry_px[key] for f in pool], dtype=np.float64)
        w = weights_all.copy()
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-6
        keep = np.abs(vals - med) <= 3.5 * mad
        if keep.sum() >= 1:
            vals, w = vals[keep], w[keep]
        return float(np.sum(vals * w) / max(np.sum(w), 1e-9))

    length = robust("length_px")
    width = robust("width_px")
    body = robust("body_height_px")
    pyramid = robust("pyramid_height_px")
    return {
        "length_px": length, "width_px": width,
        "body_height_px": body, "pyramid_height_px": pyramid,
        "total_height_px": body + pyramid,
        "volume_px3": compute_volume(length, width, body, pyramid),
    }


def run_stage1(cfg: Stage1Config) -> Dict[str, object]:
    """第一阶段主入口。"""
    section("第一阶段：像素域轮廓与线框重建")
    frames = iter_inputs(
        cfg.input_path, cfg.num_frames, cfg.frame_start_ratio, cfg.frame_end_ratio, cfg.max_input_side,
    )
    log(f"开始流式读取输入（输入: {cfg.input_path}）")

    layout = OutputLayout(Path(cfg.output_dir).expanduser().resolve()).prepare(clean=cfg.clean_output)
    log(f"输出目录: {layout.root}")

    segmenter = build_segmenter(cfg)

    frame_outputs: List[FrameOutput] = []
    failed_frames: list[dict[str, str]] = []
    processed_count = 0
    try:
        for frame in frames:
            try:
                out = _process_frame(frame, cfg, segmenter)
                write_frame_products(layout, out)
                _release_frame_buffers(out)
            except Exception as exc:  # noqa: BLE001
                # 单帧异常不应让整批任务崩溃；保留错误并继续处理后续视图。
                failed_frames.append({"name": frame.name, "error": f"{type(exc).__name__}: {exc}"})
                warn(f"[{frame.name}] 单帧处理失败，跳过并继续：{exc}")
                frame.image_bgr = None
                continue
            frame_outputs.append(out)
            processed_count += 1
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
    geometry_px = _consolidate_geometry(pool)
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
    geometry_payload = {
        "units": "pixel",
        "geometry_params_px": {k: v for k, v in geometry_px.items() if k != "volume_px3"},
        "volume_px3": geometry_px["volume_px3"],
        "depth_estimation_source": frame_outputs[0].wireframe.depth_source if frame_outputs else "none",
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
        "consensus_frames": pool_names,
        "consensus_frame_count": len(pool_names),
        "representative_frame": representative.frame.name,
        "geometry_px": geometry_px,
        "metric": metric_payload,
        "frames": [
            {
                "name": f.frame.name,
                "backend": f.edge_backend,
                "roi_bbox": list(f.roi.bbox),
                "roi_scale": f.roi.scale,
                "area_ratio": f.roi.area_ratio,
                "sam2_used": f.sam2_used,
                "fit_ready": f.wireframe.fit_ready,
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
