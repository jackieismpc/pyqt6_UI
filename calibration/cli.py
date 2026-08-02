# -*- coding: utf-8 -*-
"""相机标定命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .extrinsics import calibrate_extrinsic
from .intrinsics import calibrate_intrinsics
from .patterns import BoardSpec, PATTERN_TYPES, parse_pattern_size, draw_board, board_metadata
from .schema import load_parameters, save_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETERS = PROJECT_ROOT / "backend" / "crystalvol" / "defaults" / "camera_parameters.json"

# 项目实际使用 ChArUco。棋盘格仍然可通过 --type chessboard 显式选择，
# 但所有不写 --type 的命令必须生成同一套 ChArUco 参数，避免板型和检测器错配。
DEFAULT_PATTERN_TYPE = "charuco"
DEFAULT_PATTERN_SIZE = "5x7"       # 方格列数 x 行数
DEFAULT_SQUARE_SIZE = 30.0          # mm
DEFAULT_MARKER_LENGTH = 22.0        # mm，约为方格边长的 73%
DEFAULT_DICTIONARY = "DICT_5X5_100"


def _add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--type", dest="pattern_type", choices=PATTERN_TYPES, default=DEFAULT_PATTERN_TYPE,
        help="标定板类型。默认 charuco。",
    )
    parser.add_argument(
        "--pattern-size", default=DEFAULT_PATTERN_SIZE,
        help="列x行；棋盘格表示内角点，ChArUco 表示方格。默认 5x7。",
    )
    parser.add_argument(
        "--square-size", type=float, default=DEFAULT_SQUARE_SIZE,
        help="棋盘格/ChArUco 方格边长（默认 30 mm）。",
    )
    parser.add_argument("--circle-distance", type=float, default=30.0, help="圆点板相邻圆心距离。默认 30。")
    parser.add_argument(
        "--marker-length", type=float, default=DEFAULT_MARKER_LENGTH,
        help="ChArUco marker 边长（默认 22 mm）。",
    )
    parser.add_argument(
        "--dictionary", default=DEFAULT_DICTIONARY,
        help="ChArUco 字典。默认 DICT_5X5_100。",
    )
    parser.add_argument("--unit", default="mm", choices=["mm", "cm", "m"], help="尺寸单位。默认 mm。")


def _spec(args: argparse.Namespace) -> BoardSpec:
    return BoardSpec(
        pattern_type=args.pattern_type,
        pattern_size=parse_pattern_size(args.pattern_size),
        square_size=args.square_size,
        circle_distance=args.circle_distance,
        marker_length=args.marker_length,
        dictionary=args.dictionary,
        length_unit=args.unit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calibration",
        description="基于 OpenCV 官方 API 的相机标定工具。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    board = sub.add_parser("board", help="生成可打印的棋盘格、ChArUco 或圆点板图片。")
    _add_board_args(board)
    board.add_argument("--dpi", type=int, default=300, help="输出打印分辨率。默认 300。")
    board.add_argument("--margin-mm", type=float, default=10.0, help="外边距。默认 10 mm。")
    board.add_argument("--output", default="data/calibration/charuco.png", help="输出图片路径。")

    intrinsics = sub.add_parser("intrinsics", help="从图片目录标定内参并输出统一 JSON。")
    intrinsics.add_argument("image_dir", help="标定图片目录。")
    _add_board_args(intrinsics)
    intrinsics.add_argument("--recursive", action="store_true", help="递归读取图片子目录。")
    intrinsics.add_argument("--model", choices=["standard", "rational"], default="standard", help="畸变模型。")
    intrinsics.add_argument("--fix-aspect-ratio", action="store_true", help="固定 fx/fy 比例。")
    intrinsics.add_argument("--zero-tangent-dist", action="store_true", help="固定切向畸变为 0。")
    intrinsics.add_argument("--fix-principal-point", action="store_true", help="固定主点为初始估计值。")
    intrinsics.add_argument("--no-reject-outliers", action="store_true", help="关闭按单视图重投影误差剔除异常图。")
    intrinsics.add_argument("--max-view-error", type=float, default=2.0, help="异常视图最大重投影误差（像素）。默认 2.0。")
    intrinsics.add_argument("--max-rounds", type=int, default=3, help="异常视图剔除最多迭代次数。默认 3。")
    intrinsics.add_argument("--min-views", type=int, default=5, help="最少有效视图数。默认 5。")
    intrinsics.add_argument("--max-iterations", type=int, default=100, help="OpenCV 优化最大迭代次数。默认 100。")
    intrinsics.add_argument("--epsilon", type=float, default=1e-7, help="OpenCV 优化收敛阈值。")
    intrinsics.add_argument("--debug-dir", default=None, help="保存每张图片检测结果的目录。")
    intrinsics.add_argument("--output", default="params/camera_parameters.json", help="输出统一相机参数 JSON。")
    intrinsics.add_argument("--update-default", action="store_true", help="同时更新后端内置默认参数。")

    extrinsics = sub.add_parser("extrinsics", help="使用内参对单张标定板图片求外参。")
    extrinsics.add_argument("--image", required=True, help="单张标定板图片。")
    extrinsics.add_argument("--parameters", required=True, help="统一相机参数 JSON。")
    _add_board_args(extrinsics)
    extrinsics.add_argument("--pose-method", choices=["iterative", "ippe", "ransac"], default="iterative", help="solvePnP 方法。")
    extrinsics.add_argument("--no-refine-pose", action="store_true", help="关闭 LM 位姿细化。")
    extrinsics.add_argument("--object-center", nargs=3, type=float, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.0), help="标定板原点到晶体中心的偏移。")
    extrinsics.add_argument("--expected-distance", type=float, default=None, help="期望的相机到晶体中心距离。")
    extrinsics.add_argument("--distance-tolerance", type=float, default=None, help="距离校验允许误差。")
    extrinsics.add_argument("--append", action="store_true", help="保留输入文件中的外参并追加当前单图结果。")
    extrinsics.add_argument("--debug-output", default=None, help="保存坐标轴调试图。")
    extrinsics.add_argument("--output", default="params/camera_parameters_with_extrinsic.json", help="输出统一相机参数 JSON。")
    extrinsics.add_argument("--update-default", action="store_true", help="同时更新后端内置默认参数。")

    install = sub.add_parser("install-default", help="显式更新后端内置默认相机参数。")
    install.add_argument("--parameters", required=True, help="要安装的统一相机参数 JSON。")

    return parser


def _write_board(output: str | Path, image, metadata: dict) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法保存标定板图片: {output_path}")
    output_path.write_bytes(encoded.tobytes())
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _install_default(payload: dict) -> Path:
    return save_parameters(DEFAULT_PARAMETERS, payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "board":
        spec = _spec(args)
        image = draw_board(spec, dpi=args.dpi, margin_mm=args.margin_mm)
        output = _write_board(args.output, image, board_metadata(spec, args.dpi, args.margin_mm))
        print(f"已生成标定板: {output}")
        print(f"已生成标定板参数: {output.with_suffix('.json')}")
        return 0

    if args.command == "intrinsics":
        spec = _spec(args)
        payload = calibrate_intrinsics(
            args.image_dir,
            spec,
            args.output,
            recursive=args.recursive,
            model=args.model,
            fix_aspect_ratio=args.fix_aspect_ratio,
            zero_tangent_dist=args.zero_tangent_dist,
            fix_principal_point=args.fix_principal_point,
            reject_outliers=not args.no_reject_outliers,
            max_view_error=args.max_view_error,
            max_rounds=args.max_rounds,
            min_views=args.min_views,
            max_iterations=args.max_iterations,
            epsilon=args.epsilon,
            debug_dir=args.debug_dir,
        )
        if args.update_default:
            target = _install_default(payload)
            print(f"已更新后端默认参数: {target}")
        print(f"有效标定图: {len(payload['extrinsics'])}")
        print(f"重投影误差: {payload['calibration']['reprojection_error_px']:.6f} px")
        print(f"已保存: {Path(args.output).expanduser().resolve()}")
        return 0

    if args.command == "extrinsics":
        spec = _spec(args)
        payload = calibrate_extrinsic(
            args.image,
            args.parameters,
            spec,
            args.output,
            pose_method=args.pose_method,
            refine_pose=not args.no_refine_pose,
            object_center=tuple(args.object_center),
            expected_distance=args.expected_distance,
            distance_tolerance=args.distance_tolerance,
            append=args.append,
            debug_output=args.debug_output,
        )
        if args.update_default:
            target = _install_default(payload)
            print(f"已更新后端默认参数: {target}")
        item = payload["extrinsics"][-1]
        print(f"重投影误差: {item['reprojection_error_px']:.6f} px")
        print(f"目标中心距离: {item['distance_to_object_center']:.6f} {spec.length_unit}")
        print(f"已保存: {Path(args.output).expanduser().resolve()}")
        return 0

    if args.command == "install-default":
        payload = load_parameters(args.parameters)
        target = _install_default(payload)
        print(f"已更新后端默认参数: {target}")
        return 0

    raise RuntimeError(f"未知命令: {args.command}")
