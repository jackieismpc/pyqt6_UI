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
  待用 tools/calibrate_turntable.py 标定外参后，再接入 crystalvol.reference_target
  与多视角优化后端启用（接入点见 _run_extrinsic_multiview）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .calibration import load_camera_calibration
from .config import Stage2Config
from .logging_utils import log, section, warn
from .metric import convert_pixel_to_metric


def _load_stage1_geometry(path: str) -> Dict[str, float]:
    """读取第一阶段像素几何 JSON，返回 geometry_params_px（含 total_height_px）。"""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    params = dict(payload.get("geometry_params_px", {}))
    if "total_height_px" not in params:
        params["total_height_px"] = params.get("body_height_px", 0.0) + params.get("pyramid_height_px", 0.0)
    if not params:
        raise RuntimeError(f"第一阶段几何 JSON 缺少 geometry_params_px: {path}")
    return params


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
        missing.append("turntable_config（转台/参考板配置，用 tools/calibrate_turntable.py 生成）")
    if not cfg.angles_file:
        missing.append("angles_file（每帧转台角度）")
    raise NotImplementedError(
        "第二阶段 extrinsic_multiview 模式需要相机外参（参考板位姿），当前项目尚未标定外参，"
        "该模式暂未启用。缺少：" + ("；".join(missing) if missing else "参考板外参标定") +
        "。请先用 tools/calibrate_turntable.py 标定，或改用 scale_anchor 模式（提供一条真实边长）。"
    )


def run_stage2(cfg: Stage2Config) -> Dict[str, object]:
    """第二阶段主入口。"""
    section("第二阶段：公制体积恢复")
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载内参（当前一定有；用于校验与记录，未来 extrinsic 模式会用到）
    calibration = load_camera_calibration(cfg.camera_calibration)
    log(f"已加载相机内参: {calibration.source_path}")

    geometry_px = _load_stage1_geometry(cfg.stage1_geometry_json)

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

    log(f"第二阶段模式: {mode}")
    if mode == "scale_anchor":
        result = _run_scale_anchor(cfg, geometry_px)
    elif mode == "extrinsic_multiview":
        result = _run_extrinsic_multiview(cfg, geometry_px)
    else:
        raise ValueError(f"未知第二阶段模式: {mode}")

    # 体积范围报警
    volume_m3 = float(result.get("volume_m3", 0.0))
    in_range = cfg.expected_volume_min_m3 <= volume_m3 <= cfg.expected_volume_max_m3
    result["volume_in_expected_range"] = in_range
    if not in_range:
        warn(f"估计体积 {volume_m3:.6e} m^3 不在预期范围 "
             f"[{cfg.expected_volume_min_m3}, {cfg.expected_volume_max_m3}] m^3。")

    out_path = output_dir / "stage2_metric.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"第二阶段完成，结果: {out_path}")
    log(f"真实体积: {volume_m3:.6e} m^3")
    return result
