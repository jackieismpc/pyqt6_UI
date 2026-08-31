"""Offline deployment preflight checks.

This module intentionally performs no network operation.  It only checks the
local Python environment and files shipped with the project.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORE_MODULES = {
    "numpy": "numpy",
    "OpenCV": "cv2",
    "scipy": "scipy",
    "scikit-image": "skimage",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "PyTorch": "torch",
    "torchvision": "torchvision",
    "Hydra": "hydra",
    "iopath": "iopath",
    "SAM2": "sam2",
    "Ultralytics": "ultralytics",
    "controlnet-aux": "controlnet_aux",
}

MODE_MODULES = {
    "app": {"PyQt6": "PyQt6"},
    "web": {
        "FastAPI": "fastapi",
        "Uvicorn": "uvicorn",
        "python-multipart": "multipart",
    },
    "backend": {},
}

REQUIRED_ASSETS = {
    "SAM2 checkpoint": ("backend/weights/sam2.1_hiera_tiny.pt", 1_000_000),
    "PiDiNet checkpoint": ("backend/weights/table5_pidinet.pth", 1_000_000),
    "HED checkpoint": ("backend/weights/ControlNetHED.pth", 1_000_000),
    "YOLO checkpoint": ("backend/weights/yolov8s-worldv2.pt", 1_000_000),
    "SAM2 config": (
        "backend/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        1,
    ),
    "camera parameters": (
        "backend/crystalvol/defaults/camera_parameters.json",
        1,
    ),
}


def _check_python(errors: list[str]) -> None:
    if sys.version_info < (3, 13):
        errors.append(
            f"Python 版本过低：当前为 {platform.python_version()}，项目要求 Python 3.13 或更高。"
        )


def _check_modules(mode: str, errors: list[str]) -> None:
    modules = dict(CORE_MODULES)
    modules.update(MODE_MODULES[mode])
    for label, module_name in modules.items():
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            errors.append(f"缺少 Python 依赖：{label}（导入名 {module_name}）。")


def _check_assets(errors: list[str]) -> None:
    for label, (relative_path, minimum_size) in REQUIRED_ASSETS.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            errors.append(f"缺少项目内文件：{label}（{relative_path}）。")
            continue
        size = path.stat().st_size
        if size < minimum_size:
            errors.append(
                f"文件疑似 Git LFS 指针或不完整：{label}（{relative_path}，{size} bytes）。"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="检查晶体体积估计项目是否可离线运行")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_MODULES),
        default="app",
        help="检查目标：app（桌面 GUI）、web（Web 服务）或 backend（后端 CLI）",
    )
    args = parser.parse_args()

    errors: list[str] = []
    _check_python(errors)
    _check_modules(args.mode, errors)
    _check_assets(errors)

    if errors:
        print("离线运行预检失败：", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(
            "请在同一操作系统和 CPU 架构上预置依赖/缓存后，再执行启动脚本。"
            "预检不会联网，也不会自动下载文件。",
            file=sys.stderr,
        )
        return 1

    print(
        f"离线运行预检通过：mode={args.mode}，Python {platform.python_version()}，"
        f"项目目录={PROJECT_ROOT}"
    )
    print("模型、配置和运行依赖均已在本地找到；启动过程不会主动联网。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
