"""
跨平台工具模块。

提供 OS 检测、MVS SDK 路径、Native 库加载前准备等功能。
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---- 平台检测 ----

_SYSTEM = platform.system().lower()  # "darwin" | "windows" | "linux"


def is_macos() -> bool:
    return _SYSTEM == "darwin"


def is_windows() -> bool:
    return _SYSTEM == "windows"


def is_linux() -> bool:
    return _SYSTEM == "linux"


def os_label() -> str:
    """用户可读的 OS 标识。"""
    labels = {"darwin": "macOS", "windows": "Windows", "linux": "Linux"}
    return labels.get(_SYSTEM, _SYSTEM)


# ---- MVS SDK 路径（海康机器人工业相机）----

# 常见 MVS SDK 安装根目录
if is_macos():
    _MVS_ROOT_CANDIDATES = [
        "/Library/MVS_SDK",
    ]
    _MVS_DYLIB_NAME = "libMvCameraControl.dylib"
elif is_windows():
    _MVS_ROOT_CANDIDATES = [
        r"D:\MVS",
        r"C:\Program Files (x86)\MVS",
        r"C:\Program Files\MVS",
        r"C:\Program Files (x86)\MvToolKit",
        r"C:\Program Files\MvToolKit",
    ]
    _MVS_DYLIB_NAME = "MvCameraControl.dll"
else:  # linux
    _MVS_ROOT_CANDIDATES = [
        "/opt/Client",       # MVS 5.x 默认安装路径
        "/opt/MVS",
        "/usr/local/MVS",
    ]
    _MVS_DYLIB_NAME = "libMvCameraControl.so"


def _find_mvs_root() -> Optional[Path]:
    """探测当前系统上的 MVS SDK 安装目录。"""
    for candidate in _MVS_ROOT_CANDIDATES:
        p = Path(candidate)
        if p.is_dir():
            return p
    return None


def mvs_sdk_root() -> Optional[Path]:
    """MVS SDK 根目录（跨平台）。"""
    return _find_mvs_root()


def mvs_python_path() -> Optional[Path]:
    """MVS SDK 的 Python 绑定目录（MvImport/）。"""
    root = _find_mvs_root()
    if root is None:
        return None

    # Linux aarch64: <root>/Samples/aarch64/Python/MvImport
    # Linux x86_64:  <root>/Samples/x86_64/Python/MvImport
    # macOS:         <root>/Samples/Python/MvImport
    # Windows:       <root>/Development/Samples/Python/MvImport
    #               <root>/Samples/Python/MvImport
    candidates = []
    if is_linux():
        arch = platform.machine()  # "aarch64", "x86_64", etc.
        candidates.append(f"Samples/{arch}/Python/MvImport")
    candidates.extend([
        "Samples/Python/MvImport",
        "Development/Samples/Python/MvImport",
    ])
    for sub in candidates:
        p = root / sub
        if p.is_dir():
            return p
    return None


def mvs_library_dir() -> Optional[Path]:
    """MVS SDK 的 Native 库目录（含 dylib/dll/so）。"""
    root = _find_mvs_root()
    if root is None:
        return None
    # macOS: <root>/lib
    # Windows: <root>/Development/Libraries/win64
    # Linux: <root>/lib/<arch> (MVS 5.x), <root>/lib, <root>/lib64
    if is_macos():
        candidates = ["lib"]
    elif is_windows():
        candidates = ["Development/Libraries/win64", "Development/Libraries/x64"]
    else:
        arch = platform.machine()  # "aarch64", "x86_64"
        candidates = [
            f"lib/{arch}",
            "lib",
            "lib64",
        ]
    for sub in candidates:
        p = root / sub
        if p.is_dir():
            return p
    return None


def mvs_library_path() -> Optional[Path]:
    """MVS SDK Native 库的完整路径。"""
    lib_dir = mvs_library_dir()
    if lib_dir is None:
        return None
    p = lib_dir / _MVS_DYLIB_NAME
    return p if p.is_file() else None


def setup_mvs_import() -> bool:
    """在 import MvCameraControl_class 前调用，设置 Python 路径和 Native 库路径。

    返回 True 表示 SDK 可用。
    """
    run_python_path = mvs_python_path()
    if run_python_path is None:
        logger.debug("未找到 MVS SDK 安装目录")
        return False

    python_path_str = str(run_python_path)
    if python_path_str not in sys.path:
        sys.path.insert(0, python_path_str)

    # Native 库路径准备
    lib_path = mvs_library_path()
    if lib_path is None:
        logger.debug("MVS SDK Python 绑定存在，但未找到 %s", _MVS_DYLIB_NAME)
        # 仍然尝试导入（系统可能已配好库搜索路径）
    else:
        lib_dir_str = str(lib_path.parent)
        if is_windows():
            try:
                os.add_dll_directory(lib_dir_str)
            except AttributeError:
                # Python < 3.8 不支持 add_dll_directory
                os.environ["PATH"] = lib_dir_str + os.pathsep + os.environ.get("PATH", "")
        elif is_linux():
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir_str not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = (
                    lib_dir_str + (":" + ld_path if ld_path else "")
                )
        # macOS: dyld 已从标准路径搜索，无需额外设置

    return True


def ensure_mvs_importable() -> bool:
    """确保 MVS SDK Python 模块可导入。返回 True 表示可用。

    供 camera_scanner 和 workers 共用。
    """
    if not setup_mvs_import():
        return False
    try:
        import MvCameraControl_class  # noqa: F401
        return True
    except (ImportError, OSError) as exc:
        logger.debug("MVS SDK 导入失败: %s", exc)
        return False
