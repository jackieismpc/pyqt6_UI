# -*- coding: utf-8 -*-
"""计算设备自动选择。

选择优先级：
1. 显式指定（--device cuda:0 / mps / cpu）时直接采用；
2. 否则：Linux + CUDA 可用  -> cuda:{gpu_id}
3. 否则：macOS + MPS 可用    -> mps（Apple Silicon，深度边缘/分割可加速）
4. 否则                       -> cpu
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

try:
    import torch
except Exception:  # pragma: no cover - torch 为可选依赖
    torch = None


@dataclass
class DeviceSelection:
    """设备选择结果。

    - torch_device：喂给 torch 的设备字符串（cuda:0 / mps / cpu）
    - label：用于日志展示
    """

    torch_device: str
    label: str


def resolve_device(prefer: str = "auto", gpu_id: int = 0) -> DeviceSelection:
    """按上面的规则解析设备。"""
    prefer = (prefer or "auto").strip().lower()

    if prefer not in ("", "auto"):
        # 显式指定：MPS 场景下打开算子回退，避免个别算子未实现直接报错。
        if prefer.startswith("mps"):
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return DeviceSelection(torch_device=prefer, label=prefer)

    system_name = platform.system().lower()
    if torch is not None and torch.cuda.is_available():
        index = max(int(gpu_id), 0)
        return DeviceSelection(torch_device=f"cuda:{index}", label=f"cuda:{index}")

    if (
        system_name == "darwin"
        and torch is not None
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return DeviceSelection(torch_device="mps", label="mps")

    return DeviceSelection(torch_device="cpu", label="cpu")
