"""首帧晶体预选对话框。

只负责将用户在首帧上拖出的矩形框转换成归一化图像坐标；晶体边界仍由
后端的 SAM2/边缘剪影流程自动细化，避免把粗略人工框误当成精确分割掩膜。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


NormalizedRoi = Tuple[float, float, float, float]


class RoiSelectionCanvas(QWidget):
    """按比例显示首帧并支持鼠标拖动框选。"""

    roiChanged = pyqtSignal(object)  # NormalizedRoi | None

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            raise ValueError("预选图像为空。")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("预选图像必须是 BGR 三通道图像。")

        self._image = np.ascontiguousarray(image_bgr)
        self._height, self._width = self._image.shape[:2]
        qimage = QImage(
            self._image.data,
            self._width,
            self._height,
            self._width * 3,
            QImage.Format.Format_BGR888,
        )
        self._pixmap = QPixmap.fromImage(qimage)
        self._roi: Optional[NormalizedRoi] = None
        self._drag_start: Optional[QPointF] = None
        self._drag_current: Optional[QPointF] = None
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("在晶体外接区域上按住鼠标左键拖动框选")

    def clear_selection(self) -> None:
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self.roiChanged.emit(None)
        self.update()

    def selected_roi(self) -> Optional[NormalizedRoi]:
        return self._roi

    def _display_rect(self) -> QRectF:
        margin = 12.0
        available_w = max(float(self.width()) - margin * 2, 1.0)
        available_h = max(float(self.height()) - margin * 2, 1.0)
        scale = min(available_w / self._width, available_h / self._height)
        draw_w = self._width * scale
        draw_h = self._height * scale
        return QRectF(
            (self.width() - draw_w) * 0.5,
            (self.height() - draw_h) * 0.5,
            draw_w,
            draw_h,
        )

    def _widget_to_image(self, point: QPointF) -> Optional[QPointF]:
        rect = self._display_rect()
        if not rect.contains(point):
            return None
        x = (point.x() - rect.left()) / rect.width() * self._width
        y = (point.y() - rect.top()) / rect.height() * self._height
        return QPointF(
            min(max(x, 0.0), float(self._width)),
            min(max(y, 0.0), float(self._height)),
        )

    def _image_to_widget(self, point: QPointF) -> QPointF:
        rect = self._display_rect()
        return QPointF(
            rect.left() + point.x() / self._width * rect.width(),
            rect.top() + point.y() / self._height * rect.height(),
        )

    def _update_roi(self, first: QPointF, second: QPointF) -> None:
        x1, x2 = sorted((first.x(), second.x()))
        y1, y2 = sorted((first.y(), second.y()))
        if x2 - x1 < 4.0 or y2 - y1 < 4.0:
            self._roi = None
        else:
            self._roi = (
                float(x1 / self._width),
                float(y1 / self._height),
                float(x2 / self._width),
                float(y2 / self._height),
            )
        self.roiChanged.emit(self._roi)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = self._widget_to_image(event.position())
        if point is None:
            return
        self._drag_start = point
        self._drag_current = point
        self._roi = None
        self.roiChanged.emit(None)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        point = self._widget_to_image(event.position())
        if point is None:
            return
        self._drag_current = point
        self._update_roi(self._drag_start, point)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            point = self._widget_to_image(event.position()) or self._drag_current
            if point is not None:
                self._update_roi(self._drag_start, point)
            self._drag_start = None
            self._drag_current = None
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        display = self._display_rect()
        painter.drawPixmap(display.toRect(), self._pixmap)

        roi = self._roi
        if self._drag_start is not None and self._drag_current is not None:
            x1, x2 = sorted((self._drag_start.x(), self._drag_current.x()))
            y1, y2 = sorted((self._drag_start.y(), self._drag_current.y()))
            roi = (
                float(x1 / self._width), float(y1 / self._height),
                float(x2 / self._width), float(y2 / self._height),
            )
        if roi is not None:
            x1, y1, x2, y2 = roi
            top_left = self._image_to_widget(QPointF(x1 * self._width, y1 * self._height))
            bottom_right = self._image_to_widget(QPointF(x2 * self._width, y2 * self._height))
            selection = QRectF(top_left, bottom_right).normalized()
            painter.setOpacity(0.22)
            painter.fillRect(selection, Qt.GlobalColor.blue)
            painter.setOpacity(1.0)
            painter.setPen(QPen(Qt.GlobalColor.green, 2.0))
            painter.drawRect(selection)
        painter.end()


class RoiSelectionDialog(QDialog):
    """首帧框选对话框，返回 (x1, y1, x2, y2) 归一化坐标。"""

    def __init__(self, image_bgr: np.ndarray, frame_name: str = "首帧", parent=None):
        super().__init__(parent)
        self.setWindowTitle("首帧预选晶体区域")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel(f"{frame_name}：请框选完整晶体区域")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        hint = QLabel("建议保留约 10%～20%边缘背景；只框选晶体，不要把大段台面或反光区域包含进来。")
        hint.setObjectName("panelSecondary")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.canvas = RoiSelectionCanvas(image_bgr, self)
        layout.addWidget(self.canvas, 1)

        self.selection_label = QLabel("尚未框选")
        self.selection_label.setObjectName("panelSecondary")
        layout.addWidget(self.selection_label)
        self.canvas.roiChanged.connect(self._on_roi_changed)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认预选")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_roi_changed(self, roi) -> None:
        if roi is None:
            self.selection_label.setText("尚未框选")
            return
        x1, y1, x2, y2 = roi
        self.selection_label.setText(
            f"已框选：左{x1:.3f} 上{y1:.3f} 右{x2:.3f} 下{y2:.3f}（归一化坐标）"
        )

    def _accept_selection(self) -> None:
        if self.canvas.selected_roi() is None:
            QMessageBox.warning(self, "尚未框选", "请先在首帧上拖动鼠标框选晶体区域。")
            return
        self.accept()

    def selected_roi(self) -> Optional[NormalizedRoi]:
        return self.canvas.selected_roi()

    @classmethod
    def get_roi(
        cls,
        image_bgr: np.ndarray,
        frame_name: str = "首帧",
        parent=None,
    ) -> Optional[NormalizedRoi]:
        dialog = cls(image_bgr, frame_name=frame_name, parent=parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_roi()
