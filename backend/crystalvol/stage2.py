# -*- coding: utf-8 -*-
"""第二阶段（公制域）框架。

把第一阶段的像素几何恢复成真实尺度与真实体积。两种模式：

- 模式 A  scale_anchor（当前可运行）：
  已知某条边的真实长度（尺度锚点），线性换算到公制并按体积公式求真实体积。
  只需一条真值边长，无需外参，因此在「只有内参」时即可使用。

- 模式 B  extrinsic_multiview（框架，当前不执行）：
  用参考板外参 + 多视角已知转角做联合几何拟合，直接解出真实尺寸。
  需要：转台/参考板配置(turntable_config) + 每帧角度(angles_file) + 每帧可见参考板。
  当前项目只标定了内参、尚无外参，故本模式仅搭好接口与输入校验；
  待补充转台/参考板的多视角联合优化后，再接入 crystalvol.reference_target
  与多视角优化后端启用（接入点见 _run_extrinsic_multiview）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .calibration import apply_growth_constraints, load_camera_calibration
from .config import Stage2Config
from .logging_utils import log, section, warn
from .metric import convert_pixel_to_metric


def _normalise_geometry_params(params: Dict[str, Any], path: str) -> Dict[str, float]:
    """校验并补齐一组候选像素几何。"""
    if not params:
        raise RuntimeError(f"第一阶段几何 JSON 缺少 geometry_params_px: {path}")
    params = {key: float(value) for key, value in dict(params).items()}
    if "total_height_px" not in params:
        params["total_height_px"] = params.get("body_height_px", 0.0) + params.get("pyramid_height_px", 0.0)
    return params


def _load_stage1_geometry(path: str) -> tuple[Dict[str, float], list[Dict[str, object]]]:
    """读取第一阶段像素几何及候选集合。

    老版本 JSON 没有候选字段时自动包装成一个 ``legacy`` 候选，保持第二阶段
    对已有结果的兼容；新版本会读取第一阶段保存的 Top-K 聚合候选。
    """
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    params = _normalise_geometry_params(dict(payload.get("geometry_params_px", {})), path)
    raw_candidates = payload.get("candidate_geometries", [])
    candidates: list[Dict[str, object]] = []
    if isinstance(raw_candidates, list):
        for index, item in enumerate(raw_candidates):
            if not isinstance(item, dict):
                continue
            raw_params = item.get("geometry_params_px")
            if not isinstance(raw_params, dict):
                continue
            try:
                geometry = _normalise_geometry_params(raw_params, path)
            except (TypeError, ValueError, RuntimeError):
                continue
            candidates.append({
                "candidate": str(item.get("candidate", f"candidate-{index + 1}")),
                "stage1_score": float(item.get("stage1_score", 0.5)),
                "frame_count": int(item.get("frame_count", 0)),
                "geometry_params_px": geometry,
            })
    if not candidates:
        candidates = [{
            "candidate": "legacy",
            "stage1_score": 0.5,
            "frame_count": 0,
            "geometry_params_px": params,
        }]
    return params, candidates


def _run_scale_anchor(cfg: Stage2Config, geometry_px: Dict[str, float]) -> Dict[str, object]:
    """模式 A：尺度锚点换算。"""
    anchor = cfg.metric_anchor
    if not anchor.scale_reference_edge or anchor.scale_reference_value is None:
        raise RuntimeError(
            "scale_anchor 模式需要 --scale-reference-edge 与 --scale-reference-value "
            "（提供一条已知真实边长）。"
        )
    gt = {k: v for k, v in {
        "length": anchor.gt_length, "width": anchor.gt_width,
        "body_height": anchor.gt_body_height, "pyramid_height": anchor.gt_pyramid_height,
    }.items() if v is not None}
    result = convert_pixel_to_metric(
        geometry_px, anchor.scale_reference_edge, anchor.scale_reference_value,
        anchor.metric_length_unit, gt or None,
    )
    result["mode"] = "scale_anchor"
    return result


def _run_extrinsic_multiview(cfg: Stage2Config, geometry_px: Dict[str, float]) -> Dict[str, object]:
    """模式 B：外参 + 多视角联合拟合（框架，当前不执行）。

    接入点：确认下列输入齐备后，在这里调用参考板检测（crystalvol.reference_target）
    与多视角几何优化后端，解出真实尺寸与体积。
    """
    missing = []
    if not cfg.turntable_config:
        missing.append("turntable_config（转台/参考板配置）")
    if not cfg.angles_file:
        missing.append("angles_file（每帧转台角度）")
    raise NotImplementedError(
        "第二阶段 extrinsic_multiview 模式需要相机外参（参考板位姿），当前项目尚未标定外参，"
        "该模式暂未启用。缺少：" + ("；".join(missing) if missing else "参考板外参标定") +
        "。请先准备转台/参考板输入，或改用 scale_anchor 模式（提供一条真实边长）。"
    )


def _run_metric_mode(cfg: Stage2Config, geometry_px: Dict[str, float], mode: str) -> Dict[str, object]:
    """对一组候选几何执行当前第二阶段模式。"""
    if mode == "scale_anchor":
        return _run_scale_anchor(cfg, geometry_px)
    if mode == "extrinsic_multiview":
        return _run_extrinsic_multiview(cfg, geometry_px)
    raise ValueError(f"未知第二阶段模式: {mode}")


def _range_score(value: float, lower: float, upper: float) -> float:
    """把期望体积范围转成连续质量分，范围内为 1。"""
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return float(max(0.0, 1.0 - (lower - value) / max(abs(lower), 1e-12)))
    return float(max(0.0, 1.0 - (value - upper) / max(abs(upper), 1e-12)))


def _evaluate_metric_candidate(
    cfg: Stage2Config,
    candidate: Dict[str, object],
    mode: str,
) -> Dict[str, object]:
    """运行一个候选并计算第二阶段质量分。"""
    name = str(candidate.get("candidate", "unknown"))
    stage1_score = float(np_clip(candidate.get("stage1_score", 0.5)))
    geometry = dict(candidate.get("geometry_params_px", {}))
    result = _run_metric_mode(cfg, geometry, mode)
    result = apply_growth_constraints(result)
    volume_m3 = float(result.get("volume_m3", 0.0))
    in_range = cfg.expected_volume_min_m3 <= volume_m3 <= cfg.expected_volume_max_m3
    result["volume_in_expected_range"] = in_range
    constraints = result.get("physical_constraints", {})
    physical_validity = 1.0 if isinstance(constraints, dict) and constraints.get("valid") else 0.0
    range_quality = _range_score(
        volume_m3, float(cfg.expected_volume_min_m3), float(cfg.expected_volume_max_m3)
    )
    physical_score = 0.65 * physical_validity + 0.35 * range_quality
    total_score = 0.65 * stage1_score + 0.35 * physical_score
    return {
        "candidate": name,
        "stage1_score": stage1_score,
        "physical_score": float(physical_score),
        "total_score": float(total_score),
        "volume_m3": volume_m3,
        "volume_in_expected_range": in_range,
        "physical_constraints_valid": bool(physical_validity),
        "result": result,
    }


def np_clip(value: object) -> float:
    """不引入 NumPy 依赖，仅对候选分数做有限范围保护。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if number != number:
        return 0.5
    return max(0.0, min(1.0, number))


def run_stage2(cfg: Stage2Config) -> Dict[str, object]:
    """第二阶段主入口。"""
    section("第二阶段：公制体积恢复")
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载内参（当前一定有；用于校验与记录，未来 extrinsic 模式会用到）
    calibration = load_camera_calibration(cfg.camera_parameters)
    log(f"已加载相机内参: {calibration.source_path}")

    geometry_px, candidates = _load_stage1_geometry(cfg.stage1_geometry_json)

    # 选择模式
    mode = cfg.mode
    if mode == "auto":
        anchor = cfg.metric_anchor
        if anchor.scale_reference_edge and anchor.scale_reference_value is not None:
            mode = "scale_anchor"
        elif cfg.turntable_config and cfg.angles_file:
            mode = "extrinsic_multiview"
        else:
            raise RuntimeError(
                "无法确定第二阶段模式：请提供尺度锚点（scale_anchor）或外参配置（extrinsic_multiview）。"
                "当前只有内参时，推荐 scale_anchor：给定一条真实边长即可求真实体积。"
            )

    log(f"第二阶段模式: {mode}；待复评候选数: {len(candidates)}")
    evaluated: list[Dict[str, object]] = []
    failures: list[Dict[str, object]] = []
    for candidate in candidates:
        try:
            evaluated.append(_evaluate_metric_candidate(cfg, candidate, mode))
        except NotImplementedError:
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            failures.append({
                "candidate": str(candidate.get("candidate", "unknown")),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
    if not evaluated:
        details = "；".join(f"{item['candidate']}: {item['error']}" for item in failures)
        raise RuntimeError(f"所有第二阶段候选均失败。{details}")
    evaluated.sort(key=lambda item: float(item["total_score"]), reverse=True)
    selected = evaluated[0]
    second_score = float(evaluated[1]["total_score"]) if len(evaluated) > 1 else 0.0
    selection_margin = float(selected["total_score"]) - second_score
    ambiguous = selection_margin < float(cfg.selection_margin_threshold)
    result = dict(selected["result"])
    result["candidate_selection"] = {
        "selected_candidate": selected["candidate"],
        "confidence": float(selected["total_score"]),
        "selection_margin": selection_margin,
        "ambiguous": ambiguous,
        "margin_threshold": float(cfg.selection_margin_threshold),
        "candidates": [
            {
                key: value for key, value in item.items() if key != "result"
            }
            for item in evaluated
        ] + failures,
    }
    if ambiguous:
        warn(
            f"第二阶段候选分数接近（selected={selected['candidate']}, "
            f"margin={selection_margin:.3f}），结果已标记 ambiguous。"
        )

    volume_m3 = float(result.get("volume_m3", 0.0))
    if not bool(result.get("volume_in_expected_range", False)):
        warn(f"估计体积 {volume_m3:.6e} m^3 不在预期范围 "
             f"[{cfg.expected_volume_min_m3}, {cfg.expected_volume_max_m3}] m^3。")

    out_path = output_dir / "stage2_metric.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"第二阶段完成，结果: {out_path}")
    log(f"真实体积: {volume_m3:.6e} m^3")
    return result
