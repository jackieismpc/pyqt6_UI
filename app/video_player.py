"""
视频循环播放控件 VideoPlayer。

用于在图像面板中替代静态图片，循环播放视频文件。
基于 QLabel + QTimer，用 OpenCV 逐帧读取。
"""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from .path_utils import video_capture_safe


class VideoPlayer(QWidget):
    """循环播放一段视频文件的控件。

    用法：set_video(path) 开始播放，set_message(text) 或 set_image(path)
    切换回静态模式时会自动停止。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._fps = 25.0
        self._last_frame_time = 0.0

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setMinimumSize(120, 120)

        # 让 label 填满整个 widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def set_video(self, path: str) -> None:
        """开始循环播放视频文件。"""
        self.stop()
        if not _HAS_CV2:
            self.set_message("（无 OpenCV）")
            return

        cap = video_capture_safe(path)
        if not cap.isOpened():
            self.set_message("（视频无法打开）")
            return

        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = max(fps, 1.0) if fps > 0 else 25.0
        # 动态帧间隔：按实际帧率设置 timer，而非固定间隔
        interval = int(1000.0 / self._fps)
        interval = max(int(1000.0 / self._fps), 16)  # 不低于 ~60fps，避免太高占满 CPU
        self._last_frame_time = time.monotonic()
        self._timer.start(interval)
        self._next_frame()

    def stop(self) -> None:
        """停止播放并释放视频资源。"""
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_message(self, text: str) -> None:
        """显示一段文字（停止视频播放）。"""
        self.stop()
        self._label.setText(text)

    def set_image(self, path: str | None) -> None:
        """显示一张静态图片（停止视频播放）。"""
        self.stop()
        if not path:
            self._label.setText("（无图像）")
            return
        # 跨平台安全加载
        pixmap = QPixmap()
        try:
            with open(path, "rb") as fh:
                pixmap.loadFromData(fh.read())
        except (OSError, ValueError):
            pixmap = QPixmap()
        if pixmap.isNull():
            self._label.setText("（图像加载失败）")
            return
        self._label.setPixmap(pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def set_np_bgr(self, array) -> None:
        """显示 BGR numpy 数组（停止视频播放）。"""
        self.stop()
        if array is None or getattr(array, "size", 0) == 0:
            self._label.setText("（无图像）")
            return
        import numpy as np
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        h, w = array.shape[:2]
        # Format_BGR888 零拷贝
        qimg = QImage(array.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self._label.setPixmap(pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _next_frame(self) -> None:
        if self._cap is None:
            return

        ok, frame = self._cap.read()
        if not ok or frame is None:
            # 循环：回到开头
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._timer.stop()
                self._label.setText("（视频读帧失败）")
                return

        # BGR 零拷贝 → QImage（Format_BGR888）
        if not frame.flags["C_CONTIGUOUS"]:
            frame = frame.copy()
        h, w = frame.shape[:2]
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)

        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # 窗口尺寸变化时重新缩放当前 pixmap
        current = self._label.pixmap()
        if current and not current.isNull():
            scaled = current.scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._label.setPixmap(scaled)
