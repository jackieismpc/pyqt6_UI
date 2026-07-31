"""
启动配置对话框。

在主窗口加载前弹出，让用户选择：
- 相机模式：单目 / 双目
- 外参组（1-12 组标定图对应的相机位姿）
- 可选的尺度锚点校正
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .camera_config import CameraConfig, CameraParams, parse_camera_params


class StartupDialog(QDialog):
    """启动配置对话框（模态）。"""

    def __init__(self, camera_params: CameraParams | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相机配置")
        self.setMinimumWidth(420)
        self.setStyleSheet("background-color: #f5f5f7;")

        self._params = camera_params or parse_camera_params()

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 22)
        root.setSpacing(18)

        # 标题
        title = QLabel("晶体体积估计 · 相机配置")
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #1d1d1f;")
        root.addWidget(title)

        # 描述
        desc = QLabel(
            "选择相机模式与外参组（标定板位姿）。\n"
            "若晶体真实尺寸已知，启用「尺度锚点」输入一条真实边长即可自动校正。"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #86868b;")
        root.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(14)
        form.setContentsMargins(0, 4, 0, 0)

        # 相机模式
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("单目相机", "monocular")
        self._mode_combo.addItem("双目相机（实验性）", "binocular")
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.setToolTip(
            "单目：单相机拍摄；双目：双相机立体匹配增强深度估计（实验性）"
        )
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.addWidget(self._mode_combo)
        form.addRow("相机模式", mode_widget)

        # 外参组选择
        self._ext_combo = QComboBox()
        for i, ext in enumerate(self._params.extrinsics):
            import math
            d = math.sqrt(sum(v * v for v in ext.t))
            self._ext_combo.addItem(
                f"外参组 {i + 1}　·　距离 ≈ {d:.0f} mm", i
            )
        self._ext_combo.setCurrentIndex(0)
        form.addRow("外参（相机位姿）", self._ext_combo)

        # 尺度锚点（可选）
        self._anchor_check = QCheckBox(
            "启用尺度锚点校正 — 输入一条已知真实边长，自动校正所有尺寸"
        )
        self._anchor_check.setStyleSheet("font-size: 12px; font-weight: 600; color: #1d1d1f;")
        form.addRow("", self._anchor_check)

        anchor_row = QHBoxLayout()
        anchor_row.setSpacing(8)

        self._anchor_edge = QComboBox()
        self._anchor_edge.addItem("长度 L", "length")
        self._anchor_edge.addItem("宽度 W", "width")
        self._anchor_edge.addItem("体高 Hb", "body_height")
        self._anchor_edge.addItem("锥高 Hp", "pyramid_height")
        self._anchor_edge.addItem("总高", "total_height")
        anchor_row.addWidget(self._anchor_edge)

        self._anchor_value = QDoubleSpinBox()
        self._anchor_value.setRange(0.01, 9999.0)
        self._anchor_value.setValue(1.0)
        self._anchor_value.setSuffix(" cm")
        self._anchor_value.setDecimals(2)
        anchor_row.addWidget(self._anchor_value)

        self._anchor_widgets = [self._anchor_edge, self._anchor_value]
        for w in self._anchor_widgets:
            w.setVisible(False)

        anchor_container = QWidget()
        anchor_container.setLayout(anchor_row)
        form.addRow("", anchor_container)

        self._anchor_check.toggled.connect(
            lambda checked: [w.setVisible(checked) for w in self._anchor_widgets]
        )

        root.addLayout(form)
        root.addStretch(1)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def get_config(self) -> CameraConfig:
        """返回用户选择的 CameraConfig。"""
        anchor_val = (
            self._anchor_value.value()
            if self._anchor_check.isChecked()
            else None
        )
        return CameraConfig(
            mode=self._mode_combo.currentData(),
            extrinsic_index=self._ext_combo.currentData(),
            scale_anchor_value=anchor_val,
            scale_anchor_edge=(
                self._anchor_edge.currentData()
                if self._anchor_check.isChecked()
                else "length"
            ),
        )
