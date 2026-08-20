"""PyQt6 与 crystalvol 后端之间的唯一适配层。

所有与 UI 无关的逻辑都在 app/backend_core.py 的 BackendCore 中（Web 前端同样复用）；
本模块只是 Qt 侧的别名子类，保持既有 API（workers.py / main_window.py 无需改动）。
"""

from __future__ import annotations

from .backend_core import BackendCore, _imwrite_safe  # noqa: F401


class BackendInterface(BackendCore):
    """Qt 前端使用的后端门面（继承纯 Python 核心层）。"""
