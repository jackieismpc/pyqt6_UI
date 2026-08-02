# -*- coding: utf-8 -*-
"""输入读取与输出目录 / 文件命名规范。

输入支持三类：
- 单张图片
- 图片目录（按文件名排序）
- 视频（均匀抽帧）

输出目录结构（命名统一，便于检索）：
    <output_dir>/
      inputs/     <name>.png                   预处理前留档的输入帧（视频抽帧/缩放后）
      enhanced/   <name>_enhanced.png          低光增强结果
      edges/      <name>_edge.png              融合后的边缘证据图
      masks/      <name>_mask.png              最大晶体剪影掩膜
      contours/   <name>_wireframe.png         轮廓提取图（仅线框，可紧裁放大）
      overlays/   <name>_overlay.png           线框叠加回原图
      geometry/   standard_geometry_pixel.json / .obj / _preview.png
      debug/      <name>_*.png                 其它调试图
      stage1_result.json / stage1_result.txt   汇总
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np

# ── 跨平台安全的图像 I/O（绕过 OpenCV CRT 编码限制）──

def _imread(path: str, flags: int = cv2.IMREAD_COLOR):
    """安全 cv2.imread（支持非 ASCII 路径）。"""
    try:
        with open(path, "rb") as fh:
            data = np.frombuffer(fh.read(), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except (OSError, ValueError, MemoryError, cv2.error):
        return None

def _imwrite(path: str, img: np.ndarray) -> bool:
    """安全 cv2.imwrite（支持非 ASCII 路径）。"""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, data = cv2.imencode(ext, img)
        if not ok:
            return False
        with open(path, "wb") as fh:
            fh.write(data.tobytes())
        return True
    except (OSError, ValueError, MemoryError, cv2.error):
        return False

def _video_capture(path: str) -> cv2.VideoCapture:
    """安全 cv2.VideoCapture（Windows 上自动尝试短路径）。"""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap
    cap.release()
    # Windows 回退：8.3 短路径
    import platform
    if platform.system() == "Windows":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            ret = ctypes.windll.kernel32.GetShortPathNameW(
                os.path.abspath(path), buf, 260,
            )
            if ret > 0 and ret < 260:
                short = buf.value
                cap2 = cv2.VideoCapture(short)
                if cap2.isOpened():
                    return cap2
                cap2.release()
        except Exception:
            pass
    return cv2.VideoCapture(path)

# 静音 FFmpeg/OpenCV 的解码告警（如 AVI 末帧被截断），避免终端出现醒目红字。
# 直接以包方式调用（绕过 run.py）时也生效；有效帧不足时下方仍会抛出明确错误。
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class InputFrame:
    """单个输入帧。"""

    name: str            # 规范化名字（如 frame_01 / 原图 stem）
    image_bgr: np.ndarray | None
    source_path: str
    index: int


def _resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """把最长边限制到 max_side（等比缩放，仅缩不放）。"""
    if max_side <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(
        image, (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _iter_video_frames(
    path: Path,
    num_frames: int,
    start_ratio: float,
    end_ratio: float,
    max_input_side: int,
) -> Iterator[InputFrame]:
    capture = _video_capture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    emitted = 0
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # 部分容器拿不到总帧数，退化为顺序读取；每次只保留一帧。
            index = 0
            while emitted < max(num_frames, 1):
                ok, frame = capture.read()
                if not ok:
                    break
                emitted += 1
                yield InputFrame(
                    f"frame_{emitted:02d}",
                    _resize_max_side(frame, max_input_side),
                    str(path),
                    index,
                )
                index += 1
        else:
            start = int(total * float(np.clip(start_ratio, 0.0, 1.0)))
            end = int(total * float(np.clip(end_ratio, 0.0, 1.0)))
            end = max(end, start + 1)
            positions = np.linspace(
                start, min(end, total) - 1, num=max(num_frames, 1)
            ).astype(int)
            # 短视频可能生成重复位置，跳过重复读取以减少无效推理。
            unique_positions = list(dict.fromkeys(int(pos) for pos in positions))
            for pos in unique_positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = capture.read()
                if not ok:
                    continue
                emitted += 1
                yield InputFrame(
                    f"frame_{emitted:02d}",
                    _resize_max_side(frame, max_input_side),
                    str(path),
                    pos,
                )
    finally:
        capture.release()

    if emitted == 0:
        raise RuntimeError(f"视频没有可读帧或抽帧失败: {path}")


def _safe_frame_name(stem: str, used: dict[str, int]) -> str:
    """将文件名 stem 变成稳定的输出名，并处理同名图片覆盖问题。"""
    safe = "".join(char if (char.isalnum() or char in "._-") else "_" for char in stem)
    safe = safe.strip(" .") or "frame"
    used[safe] = used.get(safe, 0) + 1
    return safe if used[safe] == 1 else f"{safe}_{used[safe]:02d}"


def _iter_image_frames(paths: Iterator[Path], max_input_side: int) -> Iterator[InputFrame]:
    used: dict[str, int] = {}
    for order, item in enumerate(paths):
        image = _imread(str(item))
        if image is None:
            raise RuntimeError(f"无法读取图像: {item}")
        yield InputFrame(
            _safe_frame_name(item.stem, used),
            _resize_max_side(image, max_input_side),
            str(item),
            order,
        )


def iter_inputs(
    input_path: str,
    num_frames: int = 7,
    frame_start_ratio: float = 0.0,
    frame_end_ratio: float = 1.0,
    max_input_side: int = 2304,
) -> Iterator[InputFrame]:
    """流式解析输入，避免图片目录一次性把所有大图放入内存。"""
    path = Path(input_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"输入路径不存在: {path}")

    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        yield from _iter_video_frames(
            path, num_frames, frame_start_ratio, frame_end_ratio, max_input_side
        )
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        yield from _iter_image_frames(iter((path,)), max_input_side)
    elif path.is_dir():
        image_paths = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise RuntimeError(f"目录中没有图像文件: {path}")
        yield from _iter_image_frames(iter(image_paths), max_input_side)
    else:
        raise RuntimeError(f"不支持的输入类型: {path}")


def load_inputs(
    input_path: str,
    num_frames: int = 7,
    frame_start_ratio: float = 0.0,
    frame_end_ratio: float = 1.0,
    max_input_side: int = 2304,
) -> List[InputFrame]:
    """把输入路径解析成一组 InputFrame（兼容调用方的列表接口）。"""
    return list(iter_inputs(
        input_path, num_frames, frame_start_ratio, frame_end_ratio, max_input_side
    ))


@dataclass
class OutputLayout:
    """输出目录布局与统一命名。"""

    root: Path

    def prepare(self, clean: bool = False) -> "OutputLayout":
        if clean and self.root.exists():
            shutil.rmtree(self.root)
        for sub in ("inputs", "enhanced", "edges", "masks", "contours", "overlays", "geometry", "debug"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    # --- 各类文件的规范路径 ---
    def input_frame(self, name: str) -> Path:
        return self.root / "inputs" / f"{name}.png"

    def enhanced(self, name: str) -> Path:
        return self.root / "enhanced" / f"{name}_enhanced.png"

    def edge(self, name: str) -> Path:
        return self.root / "edges" / f"{name}_edge.png"

    def mask(self, name: str) -> Path:
        return self.root / "masks" / f"{name}_mask.png"

    def contour(self, name: str) -> Path:
        return self.root / "contours" / f"{name}_wireframe.png"

    def overlay(self, name: str) -> Path:
        return self.root / "overlays" / f"{name}_overlay.png"

    def debug(self, name: str, tag: str) -> Path:
        return self.root / "debug" / f"{name}_{tag}.png"

    def geometry_json(self) -> Path:
        return self.root / "geometry" / "standard_geometry_pixel.json"

    def geometry_obj(self) -> Path:
        return self.root / "geometry" / "standard_geometry_pixel.obj"

    def geometry_preview(self) -> Path:
        return self.root / "geometry" / "standard_geometry_pixel_preview.png"

    def result_json(self) -> Path:
        return self.root / "stage1_result.json"

    def result_txt(self) -> Path:
        return self.root / "stage1_result.txt"
