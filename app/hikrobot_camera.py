"""海康机器人 MVS 相机访问封装。

该模块只在实际使用海康相机时导入 MVS Python 绑定，因而不会影响没有安装
MVS SDK 的图片/视频模式。封装了设备枚举、稳定设备标识、触发配置、取帧和
资源释放，避免 UI worker 中重复维护 SDK 细节。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .platform_utils import ensure_mvs_importable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MVSDeviceInfo:
    """一个 MVS 设备的稳定描述。"""

    device_id: str
    serial_number: str
    model_name: str
    user_defined_name: str
    transport: str
    index: int
    raw_info: Any


@dataclass
class MVSFrame:
    """从相机取出的 BGR 帧和硬件/主机帧信息。"""

    image_bgr: np.ndarray
    frame_number: int = 0
    host_timestamp: int = 0
    device_timestamp: int = 0


def _mvs_imports() -> dict[str, Any]:
    if not ensure_mvs_importable():
        raise RuntimeError("未找到可用的海康 MVS SDK，请安装 MVS 并重启程序。")
    from MvCameraControl_class import (  # type: ignore[import-untyped]
        MV_ACCESS_Exclusive,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT_INFO_EX,
        MV_GIGE_DEVICE,
        MV_USB_DEVICE,
        MVCC_INTVALUE,
        MvCamera,
    )
    from ctypes import POINTER, byref, cast, c_ubyte

    return {
        "MvCamera": MvCamera,
        "MV_ACCESS_Exclusive": MV_ACCESS_Exclusive,
        "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
        "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
        "MV_FRAME_OUT_INFO_EX": MV_FRAME_OUT_INFO_EX,
        "MV_GIGE_DEVICE": MV_GIGE_DEVICE,
        "MV_USB_DEVICE": MV_USB_DEVICE,
        "MVCC_INTVALUE": MVCC_INTVALUE,
        "POINTER": POINTER,
        "byref": byref,
        "cast": cast,
        "c_ubyte": c_ubyte,
    }


def _decode_chars(value: Any) -> str:
    """解码 MVS 的 c_ubyte/c_char 数组，兼容不同 SDK 版本。"""
    try:
        raw = bytes(value)
    except (TypeError, ValueError):
        raw = bytes(int(item) for item in value)
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


def _special_info(device: Any, gige_value: int) -> tuple[Any, str]:
    """返回设备类型对应的 SpecialInfo 分支和传输层名称。"""
    layer = int(device.nTLayerType)
    if layer == int(gige_value):
        return device.SpecialInfo.stGigEInfo, "GigE"
    return device.SpecialInfo.stUsb3VInfo, "USB3"


def enumerate_mvs_devices() -> list[MVSDeviceInfo]:
    """枚举 USB3/GigE 海康设备，优先使用序列号作为 device_id。"""
    api = _mvs_imports()
    device_list = api["MV_CC_DEVICE_INFO_LIST"]()
    mask = api["MV_GIGE_DEVICE"] | api["MV_USB_DEVICE"]
    ret = api["MvCamera"].MV_CC_EnumDevices(mask, device_list)
    if ret != 0:
        raise RuntimeError(f"MVS 枚举设备失败: 0x{int(ret):x}")

    result: list[MVSDeviceInfo] = []
    for index in range(int(device_list.nDeviceNum)):
        raw = api["cast"](
            device_list.pDeviceInfo[index],
            api["POINTER"](api["MV_CC_DEVICE_INFO"]),
        ).contents
        special, transport = _special_info(raw, api["MV_GIGE_DEVICE"])
        serial = _decode_chars(getattr(special, "chSerialNumber", b""))
        model = _decode_chars(getattr(special, "chModelName", b""))
        user_name = _decode_chars(getattr(special, "chUserDefinedName", b""))
        # 极老固件可能不返回序列号，保留索引作为最后的兼容回退。
        stable = serial or f"index-{index}"
        result.append(
            MVSDeviceInfo(
                device_id=f"hikrobot:{stable}",
                serial_number=serial,
                model_name=model,
                user_defined_name=user_name,
                transport=transport,
                index=index,
                raw_info=raw,
            )
        )
    return result


def _pixel_to_bgr(raw: bytes, info: Any, cv2_module=cv2) -> np.ndarray:
    """把常见 MVS 8-bit 像素格式转换为 OpenCV BGR。"""
    width, height = int(info.nWidth), int(info.nHeight)
    pixel_type = int(info.enPixelType)
    array = np.frombuffer(raw, dtype=np.uint8)

    if pixel_type == 0x01080001:  # Mono8
        return cv2_module.cvtColor(array[: width * height].reshape(height, width), cv2_module.COLOR_GRAY2BGR)
    if pixel_type == 0x01080008:  # BayerGR8
        return cv2_module.cvtColor(array[: width * height].reshape(height, width), cv2_module.COLOR_BayerGR2BGR)
    if pixel_type == 0x01080009:  # BayerRG8
        return cv2_module.cvtColor(array[: width * height].reshape(height, width), cv2_module.COLOR_BayerRG2BGR)
    if pixel_type == 0x0108000A:  # BayerGB8
        return cv2_module.cvtColor(array[: width * height].reshape(height, width), cv2_module.COLOR_BayerGB2BGR)
    if pixel_type == 0x0108000B:  # BayerBG8
        return cv2_module.cvtColor(array[: width * height].reshape(height, width), cv2_module.COLOR_BayerBG2BGR)
    if pixel_type == 0x02180014:  # RGB8 packed
        return cv2_module.cvtColor(array[: width * height * 3].reshape(height, width, 3), cv2_module.COLOR_RGB2BGR)
    if pixel_type == 0x02180015:  # BGR8 packed
        return array[: width * height * 3].reshape(height, width, 3).copy()

    raise RuntimeError(
        f"暂不支持的像素格式 0x{pixel_type:x}（{width}x{height}）。"
        "请在 MVS 中将 Pixel Format 设置为 Mono8/Bayer8/RGB8。"
    )


class HikRobotCamera:
    """一个打开后的海康相机句柄。"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._api: dict[str, Any] | None = None
        self._device: MVSDeviceInfo | None = None
        self._camera: Any = None
        self._payload_size = 0
        self._grabbing = False

    @property
    def serial_number(self) -> str:
        return self._device.serial_number if self._device else self.device_id

    @property
    def model_name(self) -> str:
        return self._device.model_name if self._device else ""

    def open(self) -> None:
        self._api = _mvs_imports()
        devices = enumerate_mvs_devices()
        target = next((item for item in devices if item.device_id == self.device_id), None)
        if target is None and self.device_id.startswith("hikrobot:index-"):
            try:
                index = int(self.device_id.rsplit("-", 1)[1])
                target = next((item for item in devices if item.index == index), None)
            except ValueError:
                target = None
        if target is None:
            raise RuntimeError(f"未找到海康相机：{self.device_id}")

        self._device = target
        camera = self._api["MvCamera"]()
        ret = camera.MV_CC_CreateHandle(target.raw_info)
        if ret != 0:
            raise RuntimeError(f"创建相机句柄失败 {self.device_id}: 0x{int(ret):x}")
        self._camera = camera
        try:
            ret = camera.MV_CC_OpenDevice(self._api["MV_ACCESS_Exclusive"], 0)
            if ret != 0:
                detail = ""
                if int(ret) == 0x80000203:
                    detail = "；设备无访问权限，请关闭 MVS Viewer 或其他占用该相机的程序"
                raise RuntimeError(f"打开相机失败 {self.device_id}: 0x{int(ret):x}{detail}")
        except Exception:
            self.close()
            raise

    def _require_open(self) -> Any:
        if self._camera is None:
            raise RuntimeError(f"相机尚未打开：{self.device_id}")
        return self._camera

    def _set_enum(self, key: str, value: str) -> None:
        ret = self._require_open().MV_CC_SetEnumValueByString(key, value)
        if ret != 0:
            raise RuntimeError(f"设置 {key}={value} 失败: 0x{int(ret):x}")

    def _set_optional_enum(self, key: str, value: str) -> None:
        try:
            self._set_enum(key, value)
        except RuntimeError as exc:
            logger.debug("相机 %s 不支持或拒绝 %s=%s: %s", self.device_id, key, value, exc)

    def _set_optional_float(self, key: str, value: float) -> None:
        try:
            ret = self._require_open().MV_CC_SetFloatValue(key, float(value))
            if ret != 0:
                raise RuntimeError(f"0x{int(ret):x}")
        except Exception as exc:
            logger.debug("相机 %s 不支持或拒绝 %s=%s: %s", self.device_id, key, value, exc)

    def _set_optional_int(self, key: str, value: int) -> None:
        try:
            method = getattr(self._require_open(), "MV_CC_SetIntValueEx", None)
            if method is None:
                method = self._require_open().MV_CC_SetIntValue
            ret = method(key, int(value))
            if ret != 0:
                raise RuntimeError(f"0x{int(ret):x}")
        except Exception as exc:
            logger.debug("相机 %s 不支持或拒绝 %s=%s: %s", self.device_id, key, value, exc)

    def configure(self, trigger_mode: str = "continuous", debounce_us: int = 0) -> None:
        """配置连续、软件触发或 Line0 硬件触发模式。"""
        mode = str(trigger_mode).strip().lower()
        if mode not in {"continuous", "software", "hardware"}:
            raise ValueError("trigger_mode 只支持 continuous、software、hardware")

        # 连续采集模式只表示 SDK 持续提供取流队列，真正是否曝光由下面的
        # TriggerMode/TriggerSource 决定；这样软件/硬件触发可以重复执行。
        self._set_optional_enum("AcquisitionMode", "Continuous")

        if mode == "continuous":
            self._set_enum("TriggerMode", "Off")
        else:
            self._set_enum("TriggerMode", "On")
            self._set_enum("TriggerSource", "Software" if mode == "software" else "Line0")
            if mode == "hardware":
                self._set_optional_enum("TriggerActivation", "RisingEdge")
                self._set_optional_enum("LineSelector", "Line0")
                if debounce_us > 0:
                    self._set_optional_int("LineDebouncerTime", debounce_us)

        self._set_optional_float("TriggerDelay", 0.0)
        # 一个触发对应一张图；高速场景可由调用方改为 True。
        try:
            ret = self._require_open().MV_CC_SetIntValue("AcquisitionBurstFrameCount", 1)
            if ret != 0:
                logger.debug("设置 AcquisitionBurstFrameCount 失败: 0x%x", int(ret))
        except Exception:
            logger.debug("相机不支持 AcquisitionBurstFrameCount", exc_info=True)
        try:
            ret = self._require_open().MV_CC_SetBoolValue("TriggerCacheEnable", False)
            if ret != 0:
                logger.debug("设置 TriggerCacheEnable 失败: 0x%x", int(ret))
        except Exception:
            logger.debug("相机不支持 TriggerCacheEnable", exc_info=True)

    def configure_fixed_imaging(self, exposure_us: float | None = None, gain: float | None = None) -> None:
        """可选地关闭自动曝光/增益，保证左右相机图像亮度稳定。"""
        if exposure_us is not None:
            self._set_optional_enum("ExposureAuto", "Off")
            self._set_optional_float("ExposureTime", exposure_us)
        if gain is not None:
            self._set_optional_enum("GainAuto", "Off")
            self._set_optional_float("Gain", gain)

    def start_grabbing(self) -> None:
        camera = self._require_open()
        info = self._api["MVCC_INTVALUE"]()
        ret = camera.MV_CC_GetIntValue("PayloadSize", info)
        if ret != 0:
            raise RuntimeError(f"获取 PayloadSize 失败: 0x{int(ret):x}")
        self._payload_size = int(info.nCurValue)
        ret = camera.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开始采集失败: 0x{int(ret):x}")
        self._grabbing = True

    def software_trigger(self) -> None:
        ret = self._require_open().MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            raise RuntimeError(f"软件触发失败 {self.device_id}: 0x{int(ret):x}")

    def read(self, timeout_ms: int = 1000) -> MVSFrame | None:
        if not self._grabbing:
            raise RuntimeError(f"相机尚未开始采集：{self.device_id}")
        api = self._api
        info = api["MV_FRAME_OUT_INFO_EX"]()
        buffer = (api["c_ubyte"] * self._payload_size)()
        ret = self._require_open().MV_CC_GetOneFrameTimeout(
            api["byref"](buffer), self._payload_size, info, int(timeout_ms)
        )
        if ret != 0 or int(info.nFrameLen) <= 0:
            return None
        image = _pixel_to_bgr(bytes(buffer[: int(info.nFrameLen)]), info)
        device_ts = (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow)
        return MVSFrame(
            image_bgr=image,
            frame_number=int(info.nFrameNum),
            host_timestamp=int(info.nHostTimeStamp),
            device_timestamp=device_ts,
        )

    def close(self) -> None:
        camera = self._camera
        was_grabbing = self._grabbing
        self._camera = None
        self._grabbing = False
        if camera is None:
            return
        try:
            if was_grabbing:
                camera.MV_CC_StopGrabbing()
        except Exception:
            pass
        if not was_grabbing:
            try:
                camera.MV_CC_StopGrabbing()
            except Exception:
                pass
        try:
            camera.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            camera.MV_CC_DestroyHandle()
        except Exception:
            pass


__all__ = ["HikRobotCamera", "MVSDeviceInfo", "MVSFrame", "enumerate_mvs_devices"]
