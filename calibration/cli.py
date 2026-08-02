# -*- coding: utf-8 -*-
"""相机标定命令行入口。"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
from PIL import Image

from .extrinsics import calibrate_extrinsic
from .intrinsics import calibrate_intrinsics
from .patterns import (
    BoardSpec,
    PATTERN_TYPES,
    board_metadata,
    draw_board,
    parse_pattern_size,
)
from .schema import load_parameters, save_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMETERS = PROJECT_ROOT / "backend" / "crystalvol" / "defaults" / "camera_parameters.json"

# 项目实际使用 ChArUco。棋盘格仍然可通过 --type chessboard 显式选择，
# 但所有不写 --type 的命令必须生成同一套 ChArUco 参数，避免板型和检测器错配。
DEFAULT_PATTERN_TYPE = "charuco"
DEFAULT_PATTERN_SIZE = "5x7"       # 方格列数 x 行数；对应 OpenCV 官方 rows=7, columns=5
DEFAULT_SQUARE_SIZE = 30.0          # mm
DEFAULT_MARKER_LENGTH = 15.0        # mm，OpenCV 官方示例
DEFAULT_DICTIONARY = "DICT_5X5_100"
DEFAULT_PAPER = "a4"
DEFAULT_ORIENTATION = "portrait"
DEFAULT_MARGIN_MM = 0.0


def _add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--type", dest="pattern_type", choices=PATTERN_TYPES, default=DEFAULT_PATTERN_TYPE,
        help="标定板类型。默认 charuco。",
    )
    parser.add_argument(
        "--pattern-size", default=None,
        help="列x行的简写；ChArUco 官方示例为 rows=7, columns=5，即 5x7。",
    )
    parser.add_argument("--columns", type=int, default=None, help="标定板列数（OpenCV x 方向）。")
    parser.add_argument("--rows", type=int, default=None, help="标定板行数（OpenCV y 方向）。")
    parser.add_argument(
        "--square-size", type=float, default=DEFAULT_SQUARE_SIZE,
        help="棋盘格/ChArUco 方格边长（默认 30 mm）。",
    )
    parser.add_argument("--circle-distance", type=float, default=30.0, help="圆点板相邻圆心距离。默认 30。")
    parser.add_argument(
        "--marker-length", type=float, default=DEFAULT_MARKER_LENGTH,
        help="ChArUco marker 边长（默认 15 mm）。",
    )
    parser.add_argument(
        "--dictionary", default=DEFAULT_DICTIONARY,
        help="ChArUco 字典。默认 DICT_5X5_100。",
    )
    parser.add_argument("--unit", default="mm", choices=["mm", "cm", "m"], help="尺寸单位。默认 mm。")


def _spec(args: argparse.Namespace) -> BoardSpec:
    if args.pattern_size is not None and (args.columns is not None or args.rows is not None):
        raise ValueError("--pattern-size 不能与 --columns/--rows 同时使用")
    if (args.columns is None) != (args.rows is None):
        raise ValueError("--columns 和 --rows 必须同时提供")
    if args.columns is not None and args.rows is not None:
        if args.columns < 2 or args.rows < 2:
            raise ValueError("columns 和 rows 都必须至少为 2")
        pattern_size = (args.columns, args.rows)
    else:
        pattern_size = parse_pattern_size(args.pattern_size or DEFAULT_PATTERN_SIZE)
    return BoardSpec(
        pattern_type=args.pattern_type,
        pattern_size=pattern_size,
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
    board.add_argument("--paper", choices=["a4"], default=DEFAULT_PAPER, help="打印纸张。默认 a4。")
    board.add_argument(
        "--orientation",
        choices=["portrait", "landscape"],
        default=DEFAULT_ORIENTATION,
        help="纸张方向。默认 portrait（竖版）。",
    )
    board.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help="标定板图案外附加白边，不改变图案物理尺寸。默认 0 mm。",
    )
    board.add_argument("--output", default="data/calibration/charuco_a4.svg", help="输出 SVG/PNG 路径。")

    intrinsics = sub.add_parser("intrinsics", help="从图片目录标定内参并输出统一 JSON。")
    intrinsics.add_argument("image_dir", help="标定图片目录。")
    _add_board_args(intrinsics)
    intrinsics.add_argument("--recursive", action="store_true", help="递归读取图片子目录。")
    intrinsics.add_argument("--model", choices=["standard", "rational"], default="standard", help="畸变模型。")
    intrinsics.add_argument("--fix-aspect-ratio", action="store_true", help="固定 fx/fy 比例。")
    intrinsics.add_argument("--zero-tangent-dist", action="store_true", help="固定切向畸变为 0。")
    intrinsics.add_argument("--fix-principal-point", action="store_true", help="固定主点为初始估计值。")
    intrinsics.add_argument(
        "--focal-length-mm",
        type=float,
        default=None,
        help="镜头标称物理焦距，作为 OpenCV 初始焦距；例如 8。必须同时提供 --pixel-size-um。",
    )
    intrinsics.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="相机像元尺寸（微米）；例如当前 CH250 相机约为 4.5。",
    )
    intrinsics.add_argument(
        "--fix-focal-length",
        action="store_true",
        help="固定由物理焦距换算出的 fx/fy；只有经过实测确认时使用，默认允许 OpenCV 优化。",
    )
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
    dpi = int(metadata["dpi"])
    if suffix == ".svg":
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"无法编码 SVG 内嵌图像: {output_path}")
        encoded_image = base64.b64encode(encoded.tobytes()).decode("ascii")
        paper = metadata["paper"]
        svg = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{paper["width_mm"]}mm" height="{paper["height_mm"]}mm" '
            f'viewBox="0 0 {paper["width_px"]} {paper["height_px"]}">\n'
            f'  <image width="{paper["width_px"]}" height="{paper["height_px"]}" '
            'preserveAspectRatio="none" '
            f'href="data:image/png;base64,{encoded_image}" />\n'
            '</svg>\n'
        )
        output_path.write_text(svg, encoding="utf-8")
    elif suffix == ".png":
        # Pillow 写入 DPI 元数据；仅依赖像素尺寸会让部分打印程序按屏幕 DPI 缩放。
        Image.fromarray(image).save(output_path, format="PNG", dpi=(dpi, dpi))
    else:
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
        image = draw_board(
            spec,
            dpi=args.dpi,
            margin_mm=args.margin_mm,
            paper=args.paper,
            orientation=args.orientation,
        )
        metadata = board_metadata(
            spec,
            args.dpi,
            args.margin_mm,
            paper=args.paper,
            orientation=args.orientation,
        )
        output = _write_board(args.output, image, metadata)
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
            focal_length_mm=args.focal_length_mm,
            pixel_size_um=args.pixel_size_um,
            fix_focal_length=args.fix_focal_length,
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
