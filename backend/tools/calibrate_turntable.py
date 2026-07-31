#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibrate_turntable.py

第二阶段：生成转台/参考板配置，并可用一张参考图做检测验证。

说明：
1. 这里的“标定转台参数”重点是把参考板类型、物体中心偏移、角度零位定义固化下来；
2. 真正每张图的参考板位姿仍由 VE2.py 在推理时逐帧检测。
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 让脚本能 import crystalvol
from crystalvol.calibration import load_camera_calibration, resolve_camera_matrix_for_image
from crystalvol.reference_target import detect_reference_pose, load_reference_target_config, save_reference_target_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成并验证转台/参考板配置。默认优先使用 ChArUco。")
    parser.add_argument("--target-type", choices=["chessboard", "circles_grid", "asymmetric_circles_grid", "aruco_single", "aruco_board", "charuco_board", "apriltag"], default="charuco_board", help="参考板类型。默认 charuco_board。")
    parser.add_argument("--pattern-size", default="7x5", help="阵列尺寸，如 9x6。ChArUco 默认 7x5。")
    parser.add_argument("--square-size", type=float, default=30.0, help="棋盘格或 ChArUco 方格边长。ChArUco 默认 30 mm。")
    parser.add_argument("--circle-distance", type=float, default=None, help="圆点中心距。")
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="ArUco/AprilTag/ChArUco 字典。默认 DICT_5X5_100。")
    parser.add_argument("--marker-length", type=float, default=15.0, help="单 marker 边长。ChArUco 默认 15 mm。")
    parser.add_argument("--marker-separation", type=float, default=None, help="ArUco board marker 间隔。")
    parser.add_argument("--marker-id", action="append", type=int, default=None, help="允许使用的 marker ID，可重复传入。")
    parser.add_argument("--length-unit", default="mm", help="尺寸单位。默认 mm。")
    parser.add_argument("--mount-mode", choices=["fixed_world", "on_turntable"], default="fixed_world", help="参考板固定方式。推荐 fixed_world。")
    parser.add_argument("--object-offset-x", type=float, default=0.0, help="参考板原点到晶体中心的 X 偏移。")
    parser.add_argument("--object-offset-y", type=float, default=0.0, help="参考板原点到晶体中心的 Y 偏移。")
    parser.add_argument("--object-offset-z", type=float, default=0.0, help="参考板原点到晶体中心的 Z 偏移。")
    parser.add_argument("--angle-offset-deg", type=float, default=0.0, help="转台零位角修正。")
    parser.add_argument("--output", required=True, help="输出 turntable_config.json。")
    parser.add_argument("--validate-image", default=None, help="可选：参考图，用于验证配置可否被检测到。")
    parser.add_argument("--camera-calibration", default=None, help="验证参考图时需要提供相机标定 JSON。")
    return parser.parse_args()


def parse_pattern_size(raw_value: str | None):
    if raw_value is None:
        return None
    raw = raw_value.lower().replace(" ", "")
    if "x" not in raw:
        raise RuntimeError("pattern-size 格式错误，应为 9x6。")
    left, right = raw.split("x", 1)
    return [int(left), int(right)]


def validate_target_arguments(args: argparse.Namespace) -> None:
    """按参考板类型检查必需参数。"""

    if args.target_type == "charuco_board":
        if args.pattern_size is None or args.square_size is None or args.marker_length is None:
            raise RuntimeError("charuco_board 需要同时提供 --pattern-size、--square-size、--marker-length。")
    elif args.target_type == "chessboard":
        if args.pattern_size is None or args.square_size is None:
            raise RuntimeError("chessboard 需要同时提供 --pattern-size 与 --square-size。")
    elif args.target_type in {"circles_grid", "asymmetric_circles_grid"}:
        if args.pattern_size is None or args.circle_distance is None:
            raise RuntimeError(f"{args.target_type} 需要同时提供 --pattern-size 与 --circle-distance。")
    elif args.target_type in {"aruco_single", "apriltag"}:
        if args.marker_length is None:
            raise RuntimeError(f"{args.target_type} 需要提供 --marker-length。")
    elif args.target_type == "aruco_board":
        if args.pattern_size is None or args.marker_length is None or args.marker_separation is None:
            raise RuntimeError("aruco_board 需要同时提供 --pattern-size、--marker-length、--marker-separation。")


def build_payload(args: argparse.Namespace):
    payload = {
        "target_type": args.target_type,
        "length_unit": args.length_unit,
        "mount_mode": args.mount_mode,
        "object_offset_xyz": [args.object_offset_x, args.object_offset_y, args.object_offset_z],
        "angle_offset_deg": args.angle_offset_deg,
    }
    pattern_size = parse_pattern_size(args.pattern_size)
    if pattern_size is not None:
        payload["pattern_size"] = pattern_size
    if args.square_size is not None:
        payload["square_size"] = args.square_size
    if args.circle_distance is not None:
        payload["circle_distance"] = args.circle_distance
    if args.dictionary is not None:
        payload["dictionary"] = args.dictionary
    if args.marker_length is not None:
        payload["marker_length"] = args.marker_length
    if args.marker_separation is not None:
        payload["marker_separation"] = args.marker_separation
    if args.marker_id:
        payload["marker_ids"] = [int(value) for value in args.marker_id]
    return payload


def validate_config(config_path: Path, image_path: Path, camera_path: Path) -> None:
    if cv2 is None:
        raise RuntimeError(f"OpenCV 不可用: {CV2_IMPORT_ERROR}")
    calibration = load_camera_calibration(camera_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取验证图像: {image_path}")
    camera_matrix, distortion = resolve_camera_matrix_for_image(calibration, image.shape[1], image.shape[0])
    target_config = load_reference_target_config(config_path)
    result = detect_reference_pose(image, camera_matrix, distortion, target_config)
    if result is None:
        raise RuntimeError("参考板验证失败：当前配置无法在参考图中检测到目标。")
    debug_path = config_path.with_suffix(".validate.png")
    cv2.imwrite(str(debug_path), result.debug_image)
    print(f"[calibrate_turntable] 验证成功，调试图已保存: {debug_path}")


def main() -> None:
    args = parse_args()
    validate_target_arguments(args)
    payload = build_payload(args)
    output_path = save_reference_target_config(args.output, payload)
    print(f"[calibrate_turntable] 已保存: {output_path}")
    if args.validate_image:
        if not args.camera_calibration:
            raise RuntimeError("使用 --validate-image 时必须同时提供 --camera-calibration。")
        validate_config(
            config_path=output_path,
            image_path=Path(args.validate_image).expanduser().resolve(),
            camera_path=Path(args.camera_calibration).expanduser().resolve(),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[calibrate_turntable][ERROR] {exc}")
        raise SystemExit(1)
