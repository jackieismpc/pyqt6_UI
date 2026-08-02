"""前端运行时相机选项。

相机内参、畸变和外参全部由后端 ``crystalvol.camera_parameters`` 读取；本模块
只保存用户在启动对话框中选择的 UI 选项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    mode: str = "monocular"
    extrinsic_index: int = 0
    scale_anchor_value: Optional[float] = None
    scale_anchor_edge: str = "length"
    parameter_path: Optional[str] = None
