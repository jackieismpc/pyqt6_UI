"""
摄像头扫描工具。

枚举系统可用的摄像头设备，返回设备列表供 UI 选择。
支持 OpenCV（AVFoundation/DShow/V4L2）及可扩展的 SDK 设备。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .platform_utils import ensure_mvs_importable

logger = logging.getLogger(__name__)


@dataclass
class CameraDevice:
    """一个可用的摄像头设备。"""

    device_id: str           # 唯一标识（传给 RealtimeWorker）
    label: str               # 用户可读名称
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = ""        # OpenCV backend 名称或 SDK 标识
    is_sdk: bool = False     # 是否通过厂商 SDK 访问


def _scan_opencv_cameras(max_index: int = 8) -> list[CameraDevice]:
    """用 OpenCV 按索引扫描摄像头（静默模式，抑制无效索引噪声）。"""
    import os as _os

    import cv2

    # 抑制 OpenCV 扫描不存在摄像头索引时的 stderr 噪声
    _prev_log = _os.environ.get("OPENCV_LOG_LEVEL")
    _os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)  # 0 = silent (OpenCV >=4.8, 5.x 已移除)

    try:
        return _scan_opencv_inner(cv2, max_index)
    finally:
        if _prev_log is not None:
            _os.environ["OPENCV_LOG_LEVEL"] = _prev_log
        else:
            _os.environ.pop("OPENCV_LOG_LEVEL", None)


def _scan_opencv_inner(cv2, max_index: int) -> list[CameraDevice]:
    devices: list[CameraDevice] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend = ""
        try:
            backend = cap.getBackendName()
        except Exception:
            pass

        cap.release()

        label_parts = [f"摄像头 #{i}"]
        if backend:
            label_parts.append(backend)
        if w and h:
            label_parts.append(f"{w}×{h}")

        devices.append(CameraDevice(
            device_id=str(i),
            label="  ".join(label_parts),
            width=w,
            height=h,
            fps=fps,
            backend=backend,
        ))
    return devices


def _scan_hikrobot_cameras() -> list[CameraDevice]:
    """尝试通过 MVS SDK 枚举海康机器人工业相机。

    需要安装 MVS SDK for macOS。返回空列表表示 SDK 不可用。
    """
    devices: list[CameraDevice] = []
    if not ensure_mvs_importable():
        logger.debug("MVS SDK 未安装或不兼容，跳过海康相机扫描")
        return devices

    from MvCameraControl_class import (
        MvCamera,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_GIGE_DEVICE,
        MV_USB_DEVICE,
    )
    from ctypes import POINTER, cast

    try:
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(
            MV_GIGE_DEVICE | MV_USB_DEVICE, device_list
        )
        if ret != 0:
            logger.debug("MVS 枚举设备失败: 0x%x", ret)
            return devices

        for i in range(device_list.nDeviceNum):
            mvcc_dev = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            label = "海康工业相机"

            # 读取设备型号名
            try:
                ch_model_name = mvcc_dev.SpecialInfo.stUsb3VInfo.chModelName
            except Exception:
                ch_model_name = mvcc_dev.SpecialInfo.stGigEInfo.chModelName

            model_name = ""
            for ch in ch_model_name:
                if ch == 0:
                    break
                model_name += chr(ch)
            if model_name:
                label = f"海康 {model_name}"

            devices.append(CameraDevice(
                device_id=f"hikrobot:{i}",
                label=label,
                is_sdk=True,
                backend="MVS(Hikrobot)",
            ))
    except Exception as exc:
        logger.debug("海康相机扫描异常: %s", exc)

    return devices


def scan_cameras(
    max_opencv: int = 8,
    include_hikrobot: bool = True,
) -> list[CameraDevice]:
    """扫描所有可用摄像头，返回统一列表。

    OpenCV 设备排在前，SDK 设备排在后。
    """
    devices = _scan_opencv_cameras(max_index=max_opencv)
    if include_hikrobot:
        devices.extend(_scan_hikrobot_cameras())
    return devices
