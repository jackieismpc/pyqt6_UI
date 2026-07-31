# -*- coding: utf-8 -*-
"""命令行入口：stage1 / stage2 / full 三个子命令。

用法示例见 README.md。这里只负责把命令行参数映射到 config.py 的配置对象，
再调用 stage1.run_stage1 / stage2.run_stage2。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    EdgeConfig,
    LocalizeConfig,
    MetricAnchorConfig,
    PreprocessConfig,
    SegmentationConfig,
    Stage1Config,
    Stage2Config,
    WireframeConfig,
)
from .logging_utils import log, warn


# ---------------------------------------------------------------------------
# 参数组：多个子命令共享
# ---------------------------------------------------------------------------
def _add_preprocess_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("预处理（低光/高光）")
    g.add_argument("--lowlight", choices=["auto", "on", "off"], default="auto",
                   help="低光增强模式：auto 按亮度自动，on 强制，off 关闭。默认 auto。")
    g.add_argument("--gamma", type=float, default=0.55, help="gamma 提亮系数（<1 提亮暗部）。默认 0.55。")
    g.add_argument("--clahe-clip", type=float, default=3.0, help="CLAHE 对比度限制。默认 3.0。")
    g.add_argument("--use-msr", action="store_true", help="叠加多尺度 Retinex（更强提亮，更慢）。")
    g.add_argument("--no-denoise", action="store_true", help="关闭增强后的轻度去噪。")
    g.add_argument("--specular-percentile", type=float, default=99.0,
                   help="高光分位数：高于该分位视为镜面高光并抑制其边缘。默认 99.0。")


def _add_localize_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("晶体定位（显著性 ROI，覆盖小→大晶体）")
    g.add_argument("--no-localize", action="store_true",
                   help="关闭显著性定位，直接在整幅上处理（仅适合占画面很大的大晶体）。")
    g.add_argument("--roi-pad", type=float, default=0.6, help="ROI 相对晶体块的外扩比例。默认 0.6。")
    g.add_argument("--localize-center-weight", type=float, default=0.35,
                   help="定位时的中心权重（0 关闭，越大越偏向画面中央的目标）。默认 0.35。")
    g.add_argument("--min-roi-side-ratio", type=float, default=0.12,
                   help="ROI 最短边相对整图短边的下限（给小晶体足够上下文）。默认 0.12。")
    g.add_argument("--fullframe-area-ratio", type=float, default=0.45,
                   help="晶体块超过该占比则直接用整幅（大晶体）。默认 0.45。")


def _add_edge_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("边缘提取")
    g.add_argument("--edge-backend", choices=["auto", "pidinet", "hed", "canny", "lsd"], default="auto",
                   help="边缘后端：auto 在 pidinet 与 canny 间自动择优（推荐）。默认 auto。")
    g.add_argument("--no-fuse-canny", action="store_true", help="深度边缘不与 Canny 融合。")
    g.add_argument("--deep-threshold", type=float, default=0.12, help="深度边缘概率二值化阈值。默认 0.12。")
    g.add_argument("--deep-input-max-side", type=int, default=1024, help="深度模型推理前缩放的最长边。默认 1024。")
    g.add_argument("--deep-repo", default="lllyasviel/Annotators", help="PiDiNet/HED 权重的 HuggingFace 仓库。")
    g.add_argument("--canny-low", type=int, default=24, help="Canny 低阈值。默认 24。")
    g.add_argument("--canny-high", type=int, default=72, help="Canny 高阈值。默认 72。")


def _add_metric_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("尺度锚点（公制换算，可选）")
    g.add_argument("--scale-reference-edge",
                   choices=["length", "width", "body_height", "pyramid_height", "total_height"],
                   default=None, help="用哪条边做尺度锚点。")
    g.add_argument("--scale-reference-value", type=float, default=None, help="该边的真实长度。")
    g.add_argument("--metric-length-unit", default="cm", help="真实长度单位（mm/cm/m）。默认 cm。")
    g.add_argument("--gt-length", type=float, default=None, help="真值 length，用于误差对比。")
    g.add_argument("--gt-width", type=float, default=None, help="真值 width。")
    g.add_argument("--gt-body-height", type=float, default=None, help="真值 body_height。")
    g.add_argument("--gt-pyramid-height", type=float, default=None, help="真值 pyramid_height。")


def _metric_from_args(a: argparse.Namespace) -> MetricAnchorConfig:
    return MetricAnchorConfig(
        scale_reference_edge=a.scale_reference_edge,
        scale_reference_value=a.scale_reference_value,
        metric_length_unit=a.metric_length_unit,
        gt_length=a.gt_length, gt_width=a.gt_width,
        gt_body_height=a.gt_body_height, gt_pyramid_height=a.gt_pyramid_height,
    )


def _stage1_from_args(a: argparse.Namespace) -> Stage1Config:
    return Stage1Config(
        input_path=a.input,
        output_dir=a.output_dir,
        clean_output=a.clean_output,
        device=a.device,
        max_input_side=a.max_input_side,
        num_frames=a.num_frames,
        frame_start_ratio=a.frame_start_ratio,
        frame_end_ratio=a.frame_end_ratio,
        preprocess=PreprocessConfig(
            lowlight_mode=a.lowlight, gamma=a.gamma, clahe_clip=a.clahe_clip,
            use_msr=a.use_msr, denoise=not a.no_denoise, specular_percentile=a.specular_percentile,
        ),
        localize=LocalizeConfig(
            enable=not a.no_localize, center_weight=a.localize_center_weight,
            roi_pad_ratio=a.roi_pad, min_roi_side_ratio=a.min_roi_side_ratio,
            fullframe_area_ratio=a.fullframe_area_ratio,
        ),
        edge=EdgeConfig(
            backend=a.edge_backend, fuse_canny=not a.no_fuse_canny,
            deep_threshold=a.deep_threshold, deep_input_max_side=a.deep_input_max_side,
            deep_repo=a.deep_repo, canny_low=a.canny_low, canny_high=a.canny_high, device=a.device,
        ),
        segmentation=SegmentationConfig(
            enable=not a.no_sam2, world_conf=a.world_conf,
            center_fallback_ratio=a.center_fallback_ratio, device=a.device,
        ),
        wireframe=WireframeConfig(
            min_visible_ratio=a.min_visible_ratio, min_coverage_ratio=a.min_coverage_ratio,
            core_percentile=a.core_percentile,
        ),
        metric_anchor=_metric_from_args(a),
    )


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystalvol",
        description="透明晶体体积估计（长方体 + 四棱锥）。分第一阶段（像素域）与第二阶段（公制域）。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- stage1 ----
    def add_stage1_body(p):
        p.add_argument("input", help="输入：单张图片、图片目录或视频。")
        p.add_argument("--output-dir", default="outputs/stage1", help="输出目录。默认 outputs/stage1。")
        p.add_argument("--clean-output", action="store_true", help="运行前清空输出目录。")
        p.add_argument("--device", default="auto", help="计算设备：auto/cpu/mps/cuda:0。默认 auto。")
        p.add_argument("--max-input-side", type=int, default=2304, help="超大图先缩放到该最长边。默认 2304。")
        p.add_argument("--num-frames", type=int, default=7, help="视频均匀抽帧数。默认 7。")
        p.add_argument("--frame-start-ratio", type=float, default=0.0, help="视频抽帧起点比例。默认 0。")
        p.add_argument("--frame-end-ratio", type=float, default=1.0, help="视频抽帧终点比例。默认 1。")
        p.add_argument("--no-sam2", action="store_true", help="关闭 SAM2 分割前端，纯靠边缘剪影。")
        p.add_argument("--world-conf", type=float, default=0.05, help="YOLO-World 置信度阈值。默认 0.05。")
        p.add_argument("--center-fallback-ratio", type=float, default=0.6,
                       help="YOLO 检不到时的中心先验框比例。默认 0.6。")
        p.add_argument("--min-visible-ratio", type=float, default=0.5,
                       help="判定 fit_ready 的最小关键边可见比例。默认 0.5。")
        p.add_argument("--min-coverage-ratio", type=float, default=0.06,
                       help="剪影相对 ROI 的最小面积占比。默认 0.06。")
        p.add_argument("--core-percentile", type=float, default=55.0,
                       help="亮核收紧：前景内亮度分位，越大越紧（把晶体从背光光晕里分离）。默认 55。")
        _add_preprocess_args(p)
        _add_localize_args(p)
        _add_edge_args(p)
        _add_metric_args(p)

    p_stage1 = sub.add_parser("stage1", help="只运行第一阶段（像素域轮廓与线框重建）。")
    add_stage1_body(p_stage1)

    # ---- stage2 ----
    p_stage2 = sub.add_parser("stage2", help="只运行第二阶段（公制体积恢复）。")
    p_stage2.add_argument("--stage1-geometry", required=True,
                          help="第一阶段产物 geometry/standard_geometry_pixel.json。")
    p_stage2.add_argument("--camera-calibration", required=True, help="相机内参 JSON。")
    p_stage2.add_argument("--output-dir", default="outputs/stage2", help="输出目录。默认 outputs/stage2。")
    p_stage2.add_argument("--mode", choices=["auto", "scale_anchor", "extrinsic_multiview"], default="auto",
                          help="公制恢复模式。默认 auto。")
    p_stage2.add_argument("--turntable-config", default=None, help="转台/参考板配置（extrinsic 模式用）。")
    p_stage2.add_argument("--angles-file", default=None, help="每帧转台角度（extrinsic 模式用）。")
    p_stage2.add_argument("--expected-volume-min-m3", type=float, default=0.1, help="体积下限报警。默认 0.1。")
    p_stage2.add_argument("--expected-volume-max-m3", type=float, default=1.5, help="体积上限报警。默认 1.5。")
    _add_metric_args(p_stage2)

    # ---- full ----
    p_full = sub.add_parser("full", help="依次运行第一阶段 + 第二阶段。")
    add_stage1_body(p_full)
    p_full.add_argument("--camera-calibration", required=True, help="相机内参 JSON（第二阶段用）。")
    p_full.add_argument("--stage2-output-dir", default="outputs/stage2", help="第二阶段输出目录。")
    p_full.add_argument("--mode", choices=["auto", "scale_anchor", "extrinsic_multiview"], default="auto",
                        help="公制恢复模式。默认 auto。")
    p_full.add_argument("--turntable-config", default=None, help="转台/参考板配置（extrinsic 模式用）。")
    p_full.add_argument("--angles-file", default=None, help="每帧转台角度（extrinsic 模式用）。")
    p_full.add_argument("--expected-volume-min-m3", type=float, default=0.1, help="体积下限报警。默认 0.1。")
    p_full.add_argument("--expected-volume-max-m3", type=float, default=1.5, help="体积上限报警。默认 1.5。")
    return parser


def main(argv=None) -> int:
    from .stage1 import run_stage1
    from .stage2 import run_stage2

    args = build_parser().parse_args(argv)

    if args.command == "stage1":
        run_stage1(_stage1_from_args(args))
        return 0

    if args.command == "stage2":
        cfg = Stage2Config(
            stage1_geometry_json=args.stage1_geometry,
            camera_calibration=args.camera_calibration,
            output_dir=args.output_dir,
            mode=args.mode,
            metric_anchor=_metric_from_args(args),
            turntable_config=args.turntable_config,
            angles_file=args.angles_file,
            expected_volume_min_m3=args.expected_volume_min_m3,
            expected_volume_max_m3=args.expected_volume_max_m3,
        )
        try:
            run_stage2(cfg)
        except (NotImplementedError, RuntimeError) as exc:
            warn(f"第二阶段未执行：{exc}")
            return 2
        return 0

    if args.command == "full":
        summary = run_stage1(_stage1_from_args(args))
        geometry_json = str(Path(summary["output_dir"]) / "geometry" / "standard_geometry_pixel.json")
        cfg = Stage2Config(
            stage1_geometry_json=geometry_json,
            camera_calibration=args.camera_calibration,
            output_dir=args.stage2_output_dir,
            mode=args.mode,
            metric_anchor=_metric_from_args(args),
            turntable_config=args.turntable_config,
            angles_file=args.angles_file,
            expected_volume_min_m3=args.expected_volume_min_m3,
            expected_volume_max_m3=args.expected_volume_max_m3,
        )
        try:
            run_stage2(cfg)
        except (NotImplementedError, RuntimeError) as exc:
            warn(f"第二阶段未执行：{exc}")
            log("第一阶段结果已产出；补齐尺度锚点或外参后可单独运行 stage2。")
        return 0

    return 1
