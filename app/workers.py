"""后台线程封装（QThread）。

支持单相机连续预览、海康 MVS 软件触发、双相机软件触发配对，以及硬件
触发后的双帧配对。所有耗时的图像处理仍然在本线程执行。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from .backend_interface import BackendInterface
from .hikrobot_camera import HikRobotCamera, MVSFrame
from .models import Stage1Result


class RunWorker(QThread):
    """一次性推理（视频 / 图片目录）。"""

    resultReady = pyqtSignal(object)
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
        self.requestInterruption()


class _OpenCVReader:
    """免驱摄像头回退读取器；软件触发表示抓取下一帧。"""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._cap = None
        self._frame_number = 0

    def open(self) -> None:
        try:
            source = int(self.camera_id)
        except ValueError:
            source = self.camera_id
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self.close()
            raise RuntimeError(f"无法打开 OpenCV 摄像头：{self.camera_id}")

    def configure(self, _trigger_mode: str) -> None:
        return

    def start_grabbing(self) -> None:
        if self._cap is None:
            raise RuntimeError("OpenCV 摄像头尚未打开")

    def software_trigger(self) -> None:
        return

    def read(self, _timeout_ms: int = 1000) -> MVSFrame | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._frame_number += 1
        return MVSFrame(frame, frame_number=self._frame_number, host_timestamp=time.time_ns())

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class RealtimeWorker(QThread):
    """单相机或双相机实时采集线程。"""

    previewFrame = pyqtSignal(object, float)  # 单相机 ndarray，双相机 dict
    shotProcessed = pyqtSignal(object, int)  # 结果，照片/双视图组计数
    cameraOpened = pyqtSignal(bool)
    processingStarted = pyqtSignal()
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(
        self,
        backend: BackendInterface,
        camera_id: str = "0",
        camera_ids: Optional[list[str]] = None,
        trigger_mode: str = "continuous",
        save: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._backend = backend
        self._camera_ids = list(camera_ids or [camera_id])
        if not self._camera_ids:
            self._camera_ids = [camera_id]
        if len(self._camera_ids) > 2:
            raise ValueError("当前实时模式最多支持两台相机")
        self._trigger_mode = str(trigger_mode).lower()
        if self._trigger_mode not in {"continuous", "software", "hardware"}:
            raise ValueError("trigger_mode 只支持 continuous、software、hardware")
        self._save = save
        self._running = True
        self._capture_requested = False
        self._state_lock = threading.Lock()
        self._latest_frames: list[MVSFrame | None] = [None] * len(self._camera_ids)
        self._readers: list[object] = []

    def _is_running(self) -> bool:
        with self._state_lock:
            return self._running

    def request_capture(self) -> None:
        with self._state_lock:
            if self._running:
                self._capture_requested = True

    def _pop_capture_request(self) -> bool:
        with self._state_lock:
            requested = self._capture_requested
            self._capture_requested = False
            return requested

    def stop(self) -> None:
        with self._state_lock:
            self._running = False

    def _open_reader(self, camera_id: str):
        if self._trigger_mode == "hardware" and not camera_id.startswith("hikrobot:"):
            raise RuntimeError("硬件触发模式只支持海康 MVS 相机")
        reader = HikRobotCamera(camera_id) if camera_id.startswith("hikrobot:") else _OpenCVReader(camera_id)
        reader.open()
        reader.configure(self._trigger_mode)
        reader.start_grabbing()
        return reader

    def _emit_preview(self, frames: list[MVSFrame | None]) -> None:
        if any(frame is None for frame in frames):
            return
        now = time.monotonic()
        if len(frames) == 1:
            self.previewFrame.emit(frames[0].image_bgr, now)
            return
        self.previewFrame.emit(
            {
                "left": frames[0].image_bgr,
                "right": frames[1].image_bgr,
                "left_frame_number": frames[0].frame_number,
                "right_frame_number": frames[1].frame_number,
            },
            now,
        )

    def _read_pair(self, timeout_ms: int) -> list[MVSFrame | None]:
        return [reader.read(timeout_ms) for reader in self._readers]

    def _trigger_pair(self) -> None:
        # 先对所有句柄发送命令，再读取图像，避免第一台读取等待阻塞第二台触发。
        for reader in self._readers:
            reader.software_trigger()

    def _process_capture(self, frames: list[MVSFrame | None]) -> None:
        if any(frame is None for frame in frames):
            self.error.emit("本次触发未同时收到所有相机图像，已丢弃不完整视图组。")
            return
        self.processingStarted.emit()
        try:
            if len(frames) == 1:
                result = self._backend.add_realtime_photo(
                    frames[0].image_bgr,
                    camera_id=self._camera_ids[0],
                )
            else:
                result = self._backend.add_realtime_pair(
                    frames[0].image_bgr,
                    frames[1].image_bgr,
                    left_camera_id=self._camera_ids[0],
                    right_camera_id=self._camera_ids[1],
                )
            self.shotProcessed.emit(result, self._backend.realtime_count())
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"图像处理失败：{exc}")

    def run(self) -> None:
        session_started = False
        try:
            self._backend.start_realtime_session(
                save=self._save,
                camera_ids=self._camera_ids,
            )
            session_started = True
            self._readers = []
            for camera_id in self._camera_ids:
                # 逐台加入列表，第二台打开失败时 finally 仍能释放第一台句柄。
                self._readers.append(self._open_reader(camera_id))
            self.cameraOpened.emit(True)

            last_preview = 0.0
            preview_interval = 1.0 / 15.0
            while self._is_running():
                if self._trigger_mode == "software":
                    if self._pop_capture_request():
                        self._trigger_pair()
                        frames = self._read_pair(2000)
                        self._latest_frames = frames
                        self._emit_preview(frames)
                        self._process_capture(frames)
                    self.msleep(10)
                    continue

                # 连续模式或硬件触发模式：持续取流。硬件触发时，SDK 只会在外部
                # 脉冲到达后返回一帧；软件线程不会发送 TriggerSoftware。
                frames = self._read_pair(100)
                if all(frame is not None for frame in frames):
                    self._latest_frames = frames
                    now = time.monotonic()
                    if now - last_preview >= preview_interval:
                        last_preview = now
                        self._emit_preview(frames)

                if self._pop_capture_request():
                    if all(frame is not None for frame in self._latest_frames):
                        self._process_capture(self._latest_frames)
                    else:
                        self.error.emit("尚未收到完整图像，请稍候再拍。")
                self.msleep(5)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"实时相机任务异常：{exc}")
            self.cameraOpened.emit(False)
        finally:
            for reader in self._readers:
                try:
                    reader.close()
                except Exception:
                    pass
            self._readers.clear()
            if session_started:
                try:
                    self._backend.end_realtime_session()
                except Exception as exc:  # noqa: BLE001
                    self.error.emit(f"实时会话清理失败：{exc}")
            self._latest_frames = [None] * len(self._camera_ids)
            self.stopped.emit()
