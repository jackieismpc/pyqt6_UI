"""
后台线程封装（QThread）。

真实推理会加载 torch/SAM2 并耗时数秒到数十秒，必须放到后台线程，避免阻塞 UI。
本模块提供两个 worker：

- RunWorker：      对视频/图片目录做一次性 stage1 推理，完成后发信号回主线程。
- RealtimeWorker： 打开摄像头持续出预览帧；收到「拍摄」请求时抓当前帧并入增量会话，
                   每张处理完发信号回主线程增量刷新界面。
"""

from __future__ import annotations

import time
import threading
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from .backend_interface import BackendInterface
from .models import Stage1Result


class RunWorker(QThread):
    """一次性推理（视频 / 图片目录）。"""

    resultReady = pyqtSignal(object)   # Stage1Result
    failed = pyqtSignal(str)

    def __init__(self, backend: BackendInterface, input_path: str,
                 input_type: str, options: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._input_path = input_path
        self._input_type = input_type
        self._options = options or {}

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            result: Stage1Result = self._backend.run(
                self._input_path, self._input_type, self._options,
            )
            if not self.isInterruptionRequested():
                self.resultReady.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        """请求取消；后端算法在当前不可中断阶段结束后不会再更新 UI。"""
        self.requestInterruption()


from .platform_utils import ensure_mvs_importable


class RealtimeWorker(QThread):
    """摄像头实时预览 + 按需拍摄增量估计。

    camera_id 支持：
      - 纯数字字符串（如 "0", "1"）→ OpenCV index
      - "hikrobot:N" → 海康 MVS SDK 设备
    """

    previewFrame = pyqtSignal(object, float)  # np.ndarray（BGR）+ 时间戳，主线程可丢弃过期帧
    shotProcessed = pyqtSignal(object, int)  # Stage1Result, 已拍张数
    cameraOpened = pyqtSignal(bool)
    processingStarted = pyqtSignal()       # 一张照片开始推理
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, backend: BackendInterface, camera_id: str = "0",
                 save: bool = False, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._camera_id = camera_id
        self._save = save
        self._running = True
        self._capture_requested = False
        self._state_lock = threading.Lock()
        self._latest_frame = None
        self._sdk_cam = None

    # ---- 主线程调用 ----
    def request_capture(self) -> None:
        with self._state_lock:
            if self._running:
                self._capture_requested = True

    def stop(self) -> None:
        with self._state_lock:
            self._running = False

    # ---- 线程体 ----
    def run(self) -> None:
        import cv2

        cap = None
        sdk_reader = None
        session_started = False
        try:
            self._backend.start_realtime_session(save=self._save)
            session_started = True
            try:
                cap, sdk_reader = self._open_camera(cv2)
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"打开摄像头时发生异常：{exc}")
                return
            if cap is None and sdk_reader is None:
                self.cameraOpened.emit(False)
                return
            self.cameraOpened.emit(True)

            last_preview = 0.0
            preview_interval = 1.0 / 15.0
            while self._running:
                if sdk_reader:
                    ok, frame = sdk_reader()
                else:
                    ok, frame = cap.read()
                if ok and frame is not None:
                    self._latest_frame = frame
                    now = time.monotonic()
                    if now - last_preview >= preview_interval:
                        last_preview = now
                        # 缩放到预览尺寸（最长边 ≤ 640px），减少主线程 QImage 转换开销
                        h, w = frame.shape[:2]
                        max_side = max(h, w)
                        if max_side > 640:
                            scale = 640.0 / float(max_side)
                            preview = cv2.resize(
                                frame, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)
                        else:
                            preview = frame
                        self.previewFrame.emit(preview, now)

                with self._state_lock:
                    capture_requested = self._capture_requested
                    self._capture_requested = False
                if capture_requested:
                    if self._latest_frame is None:
                        self.error.emit("尚未取到画面，请稍候再拍。")
                    else:
                        self.processingStarted.emit()
                        try:
                            result = self._backend.add_realtime_photo(
                                self._latest_frame.copy())
                            self.shotProcessed.emit(
                                result, self._backend.realtime_count())
                        except Exception as exc:  # noqa: BLE001
                            self.error.emit(f"这张照片处理失败：{exc}")

                # 控制预览刷新率，同时给主线程留出事件处理时间
                self.msleep(10)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"实时任务异常：{exc}")
        finally:
            if cap is not None:
                cap.release()
            if sdk_reader is not None:
                self._close_sdk_camera()
            if session_started:
                try:
                    self._backend.end_realtime_session()
                except Exception as exc:  # noqa: BLE001
                    self.error.emit(f"实时会话清理失败：{exc}")
            self.stopped.emit()

    # ---- 摄像头打开逻辑 ----
    def _open_camera(self, cv2) -> tuple:
        cam_id = self._camera_id
        if cam_id.startswith("hikrobot:"):
            return self._open_hikrobot(cv2)

        try:
            idx = int(cam_id)
        except ValueError:
            idx = cam_id

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            self.error.emit(
                f"无法打开摄像头（id={cam_id}）。请检查连接与权限。")
            return None, None
        return cap, None

    def _open_hikrobot(self, cv2) -> tuple:
        if not ensure_mvs_importable():
            self.error.emit("未安装海康 MVS SDK，无法打开工业相机。")
            return None, None

        from ctypes import POINTER, cast, byref, c_ubyte
        # NOTE: MV_CC_GetIntValue / MV_CC_SetEnumValue are instance methods
        # on MvCamera (not module-level exports in MVS SDK v5.x).
        from MvCameraControl_class import (  # type: ignore[import-untyped]
            MvCamera,
            MV_ACCESS_Exclusive,
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_FRAME_OUT_INFO_EX,
            MV_GIGE_DEVICE,
            MV_TRIGGER_MODE_OFF,
            MV_USB_DEVICE,
            MVCC_INTVALUE,
        )

        try:
            idx = int(self._camera_id.split(":", 1)[1])
        except (IndexError, ValueError):
            idx = 0

        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(
            MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            self.error.emit("未检测到海康相机。请检查连接与供电。")
            return None, None

        if idx >= device_list.nDeviceNum:
            self.error.emit(
                f"海康相机索引 {idx} 越界（共 {device_list.nDeviceNum} 台）。")
            return None, None

        st_device = cast(
            device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents

        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(st_device)
        if ret != 0:
            self.error.emit(f"创建海康相机句柄失败: 0x{ret:x}")
            return None, None

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.error.emit(f"打开海康相机失败: 0x{ret:x}")
            cam.MV_CC_DestroyHandle()
            return None, None

        # 关闭触发模式（连续采集）
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            self.error.emit(f"设置 TriggerMode 失败: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return None, None

        # 自动曝光 & 自动增益（非致命，失败了继续）
        # ExposureAuto: Off=0, Once=1, Continuous=2
        # GainAuto:     Off=0, Once=1, Continuous=2
        for _name, _val in (("ExposureAuto", 2), ("GainAuto", 2)):
            _ret = cam.MV_CC_SetEnumValue(_name, _val)
            if _ret != 0:
                print(f"[HikRobot] 设置 {_name}=Continuous 失败: 0x{_ret:x}")

        # 获取 PayloadSize 用于分配帧缓冲
        st_param = MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            self.error.emit(f"获取 PayloadSize 失败: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return None, None
        n_payload = st_param.nCurValue

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.error.emit(f"海康相机开始采集失败: 0x{ret:x}")
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            return None, None

        self._sdk_cam = cam

        def _sdk_reader():
            import numpy as np

            st_info = MV_FRAME_OUT_INFO_EX()
            data_buf = (c_ubyte * n_payload)()
            ret = cam.MV_CC_GetOneFrameTimeout(
                byref(data_buf), n_payload, st_info, 1000)
            if ret != 0 or st_info.nFrameLen == 0:
                return False, None

            raw = bytes(data_buf[:st_info.nFrameLen])
            arr = np.frombuffer(raw, dtype=np.uint8)
            h, w = st_info.nHeight, st_info.nWidth

            pixel_type = st_info.enPixelType
            # PixelType_Gvsp_Mono8      = 0x01080001
            # PixelType_Gvsp_BayerRG8   = 0x01080009
            # PixelType_Gvsp_RGB8_Packed = 0x02180014
            # PixelType_Gvsp_BGR8_Packed = 0x02180015
            if pixel_type == 0x01080001:  # Mono8
                arr = arr.reshape((h, w))
                frame = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif pixel_type == 0x01080009:  # BayerRG8
                arr = arr.reshape((h, w))
                frame = cv2.cvtColor(arr, cv2.COLOR_BayerRG2BGR)
            elif pixel_type in (0x02180014, 0x02180015):
                arr = arr.reshape((h, w, 3))
                frame = arr
                if pixel_type == 0x02180014:  # RGB8 -> BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                arr = arr.reshape((h, w))
                frame = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            return True, frame

        return None, _sdk_reader

    def _close_sdk_camera(self) -> None:
        if self._sdk_cam is None:
            return
        try:
            self._sdk_cam.MV_CC_StopGrabbing()
            self._sdk_cam.MV_CC_CloseDevice()
            self._sdk_cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        self._sdk_cam = None
