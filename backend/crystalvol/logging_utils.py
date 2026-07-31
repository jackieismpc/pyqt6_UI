# -*- coding: utf-8 -*-
"""统一日志输出。

全流程只用这里的 log / warn / section 打印，方便日后统一改成 logging 或写文件。
"""

from __future__ import annotations

import sys

_TAG = "crystalvol"


def log(message: str) -> None:
    """普通信息日志。"""
    print(f"[{_TAG}] {message}", flush=True)


def warn(message: str) -> None:
    """告警日志（打到 stderr，便于与正常输出区分）。"""
    print(f"[{_TAG}][WARN] {message}", file=sys.stderr, flush=True)


def section(title: str) -> None:
    """阶段分隔标题，方便在长日志里定位。"""
    print(f"\n[{_TAG}] ==== {title} ====", flush=True)
