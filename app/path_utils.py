"""
跨平台安全的文件路径与图像 I/O 工具。

解决 OpenCV（cv2.imread/imwrite/VideoCapture）在 Windows 上无法处理
非 ASCII 路径的问题（CRT 使用 ANSI API 而非 UTF-8）。
macOS/Linux 通常 UTF-8 是默认 locale，但也统一走安全路径以防万一。
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread_safe(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """安全读取图像文件（支持任意 Unicode 路径）。

    用 Python 的 open() 读二进制 → cv2.imdecode()，避免 OpenCV 的
    CRT fopen 编码问题。

    返回 None 表示读取失败。
    """
    try:
        with open(str(path), "rb") as fh:
            data = np.frombuffer(fh.read(), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img if img is not None else None
    except (OSError, ValueError, MemoryError, cv2.error):
        return None


def imwrite_safe(path: str | Path, image: np.ndarray, params=None) -> bool:
    """安全写入图像文件（支持任意 Unicode 路径）。

    用 cv2.imencode() 编码 → Python open() 写二进制，避免 OpenCV 的 CRT 编码问题。
    """
    try:
        ext = os.path.splitext(str(path))[1].lower()
        if not ext:
            ext = ".png"
        ok, data = cv2.imencode(ext, image, params or [])
        if not ok:
            return False
        with open(str(path), "wb") as fh:
            fh.write(data.tobytes())
        return True
    except (OSError, ValueError, MemoryError, cv2.error):
        return False


_WIN32 = platform.system() == "Windows"


def video_capture_safe(path: str | Path) -> cv2.VideoCapture:
    """安全打开视频文件（支持任意 Unicode 路径）。

    Windows 上 VideoCapture 使用 CRT fopen（ANSI API），非 ASCII 路径会失败。
    尝试使用 GetShortPathNameW 获取 8.3 短路径来规避。
    macOS/Linux 直接使用原路径。
    """
    path_str = str(path)

    if not _WIN32:
        return cv2.VideoCapture(path_str)

    # Windows：先直接尝试（Python 3.8+ 可能会自动转换）
    cap = cv2.VideoCapture(path_str)
    if cap.isOpened():
        return cap
    cap.release()

    # 回退：尝试获取 8.3 短路径名
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        ret = ctypes.windll.kernel32.GetShortPathNameW(
            os.path.abspath(path_str), buf, 260,
        )
        if ret > 0 and ret < 260:
            short = buf.value
            cap2 = cv2.VideoCapture(short)
            if cap2.isOpened():
                return cap2
            cap2.release()
    except Exception:
        pass

    # 最终回退：返回原始 handle（已释放，重新打开一次让调用方检查 isOpened）
    return cv2.VideoCapture(path_str)


def pixmap_from_path(path: str | Path) -> bytes | None:
    """用 Python 安全读取图像文件为 bytes，供 QPixmap.loadFromData 使用。

    返回 None 表示读取失败。
    """
    try:
        with open(str(path), "rb") as fh:
            return fh.read()
    except (OSError, ValueError):
        return None
