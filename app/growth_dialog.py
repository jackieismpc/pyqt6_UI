"""
晶体生长预测对话框。

基于指数生长模型模拟透明晶体在未来 1-6 个月的尺寸与体积变化。
初始晶体假设长度约 1cm，6 个月后生长到约 60cm（60 倍）。
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---- 生长模型 ----
GROWTH_TARGET = 60.0               # 6 个月后总生长倍数
TOTAL_MONTHS = 6
INITIAL_LENGTH_CM = 1.0            # 初始晶体长度约 1cm


def simulate_growth(
    geometry_px: dict, months: int
) -> dict:
    """指数生长模拟：长度在 6 个月内从~1cm 增长到~60cm。

    Args:
        geometry_px: 当前像素域几何（含 length_px 等键）。
        months: 1–6 个月的生长周期。

    Returns:
        包含 length_cm / width_cm / body_height_cm /
        pyramid_height_cm / total_height_cm / volume_cm3 的字典。
    """
    # 归一化：当前像素域 length 对应初始 ~1 cm
    length_px = float(geometry_px.get("length_px", 100))
    px_to_cm = INITIAL_LENGTH_CM / max(length_px, 1e-3)

    base = {
        "length_cm": length_px * px_to_cm,
        "width_cm": float(geometry_px.get("width_px", length_px)) * px_to_cm,
        "body_height_cm": float(geometry_px.get("body_height_px", 100)) * px_to_cm,
        "pyramid_height_cm": float(
            geometry_px.get("pyramid_height_px", 50)
        )
        * px_to_cm,
    }

    # 指数生长
    scale = GROWTH_TARGET ** (months / TOTAL_MONTHS)

    L = base["length_cm"] * scale
    W = base["width_cm"] * scale
    Hb = base["body_height_cm"] * scale
    Hp = base["pyramid_height_cm"] * scale
    Ht = Hb + Hp
    vol = L * W * (Hb + Hp / 3.0)

    return {
        "length_cm": L,
        "width_cm": W,
        "body_height_cm": Hb,
        "pyramid_height_cm": Hp,
        "total_height_cm": Ht,
        "volume_cm3": vol,
    }


# ---- 3D 线框可视化组件 ----
class _CrystalWireframe(QWidget):
    """使用 QPainter 绘制等轴测晶体线框图（长方体在下 + 四棱锥在上）。

    使用固定参考尺度（第 6 个月）确保不同月份的晶体呈现真实相对大小。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 280)
        self._geometry: dict | None = None
        # 以第 6 个月最大尺寸为参考，保证不同月份能看到大小变化
        self._ref_max_dim: float = 1.0

    def set_ref_max_dim(self, dim: float) -> None:
        """设置参考最大尺寸（用于固定缩放基准）。"""
        self._ref_max_dim = max(dim, 1e-3)

    def set_geometry(self, geometry: dict) -> None:
        self._geometry = geometry
        self.repaint()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if self._geometry is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # 先填充白色背景
        painter.fillRect(0, 0, w, h, QColor("#ffffff"))

        L = self._geometry["length_cm"]
        Wc = self._geometry["width_cm"]
        Hb = self._geometry["body_height_cm"]
        Hp = self._geometry["pyramid_height_cm"]

        # 构造 9 个顶点（与 backend/crystalvol/geometry.py 一致）
        hl, hw = L * 0.5, Wc * 0.5
        top = Hb
        apex_z = Hb + Hp
        pts = [
            [-hl, -hw, 0.0],
            [hl, -hw, 0.0],
            [hl, hw, 0.0],
            [-hl, hw, 0.0],
            [-hl, -hw, top],
            [hl, -hw, top],
            [hl, hw, top],
            [-hl, hw, top],
            [0.0, 0.0, apex_z],
        ]

        # 等轴测投影：绕 Y 30°，再绕 X 35°（让高度清晰可见）
        ay = math.radians(30)
        ax = math.radians(35)
        cos_y, sin_y = math.cos(ay), math.sin(ay)
        cos_x, sin_x = math.cos(ax), math.sin(ax)

        def project(x, y, z):
            z = -z  # 翻转 z：让 z 轴正值映射到屏幕上方
            # rot Y
            rx = x * cos_y + z * sin_y
            rz = -x * sin_y + z * cos_y
            # rot X
            ry = y * cos_x - rz * sin_x
            return rx, ry

        proj = [project(*p) for p in pts]

        # 固定参考缩放：以第 6 个月最大尺寸为基准，不做逐月归一化
        margin = min(w, h) * 0.12
        scale = (min(w, h) - margin * 2) / self._ref_max_dim
        cx, cy = w * 0.5, h * 0.6
        pp = [(int(px * scale + cx), int(cy - py * scale)) for px, py in proj]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),   # 肩部
            (0, 4), (1, 5), (2, 6), (3, 7),   # 竖直棱
            (4, 8), (5, 8), (6, 8), (7, 8),   # 屋顶棱
        ]

        # 绘制面填充（浅蓝色半透明，区分远近）
        faces = [
            (0, 1, 2, 3),  # 底面
            (4, 5, 6, 7),  # 肩部顶面
            (0, 1, 5, 4),  # 前面
            (1, 2, 6, 5),  # 右面
            (2, 3, 7, 6),  # 后面
            (3, 0, 4, 7),  # 左面
            (4, 5, 8),     # 屋顶前三角
            (5, 6, 8),     # 屋顶右三角
            (6, 7, 8),     # 屋顶后三角
            (7, 4, 8),     # 屋顶左三角
        ]
        # 根据面法线方向近似区分明暗
        light_fill = QColor(0, 122, 255, 15)    # 亮面
        dark_fill = QColor(0, 122, 255, 30)     # 暗面
        face_light = [False, True, True, False, True, False, True, False, True, False]
        for fi, face in enumerate(faces):
            poly_points = [pp[i] for i in face]
            qpoly = QPolygonF([QPointF(p[0], p[1]) for p in poly_points])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(light_fill if face_light[fi] else dark_fill)
            painter.drawPolygon(qpoly)

        # 绘制棱线
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor("#007aff"), 2)
        painter.setPen(pen)
        for i, j in edges:
            p1, p2 = pp[i], pp[j]
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        # 顶点
        pen = QPen(QColor("#007aff"), 5)
        painter.setPen(pen)
        for p in pp:
            painter.drawPoint(p[0], p[1])

        painter.end()


# ---- 主对话框 ----
class CrystalGrowthDialog(QDialog):
    """晶体生长预测对话框。"""

    def __init__(self, aggregate_geometry_px: dict | None = None, parent=None):
        super().__init__(parent)
        self._initial_geometry = aggregate_geometry_px or {
            "length_px": 100,
            "width_px": 100,
            "body_height_px": 120,
            "pyramid_height_px": 60,
        }

        self.setWindowTitle("晶体生长预测")
        self.resize(560, 620)
        self.setMinimumSize(480, 540)
        self.setStyleSheet("background-color: #f5f5f7;")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # 标题行
        title = QLabel("晶体生长预测")
        title.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #1d1d1f;"
        )
        root.addWidget(title)

        # 生长周期选择
        slider_row = QHBoxLayout()
        slider_row.setSpacing(12)

        month_hint = QLabel("生长周期")
        month_hint.setStyleSheet("font-size: 13px; color: #86868b;")
        slider_row.addWidget(month_hint)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 6)
        self._slider.setValue(0)  # 默认显示当前晶体（第 0 个月）
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #e3e3e8; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #007aff; width: 16px; height: 16px; "
            "margin: -6px 0; border-radius: 8px; }"
            "QSlider::sub-page:horizontal { background: #007aff; border-radius: 2px; }"
        )
        slider_row.addWidget(self._slider, 1)

        self._month_label = QLabel("3 个月")
        self._month_label.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #007aff; min-width: 64px;"
        )
        slider_row.addWidget(self._month_label)

        # 月份刻度（0 = 当前晶体）
        ticks_row = QHBoxLayout()
        ticks_row.setContentsMargins(16, 0, 48, 0)
        tick_labels = ["当前"] + [f"{m}" for m in range(1, 7)]
        for label in tick_labels:
            t = QLabel(label)
            t.setStyleSheet("font-size: 11px; color: #86868b;")
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ticks_row.addWidget(t)
        root.addLayout(slider_row)
        root.addLayout(ticks_row)

        # 预计算第 6 个月参考尺寸（固定缩放基准）
        ref = simulate_growth(self._initial_geometry, TOTAL_MONTHS)
        ref_max = max(ref["length_cm"], ref["width_cm"],
                      ref["body_height_cm"] + ref["pyramid_height_cm"])

        # 3D 线框
        self._wireframe = _CrystalWireframe(self)
        self._wireframe.set_ref_max_dim(ref_max)
        self._wireframe.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #e3e3e8; border-radius: 12px;"
        )
        root.addWidget(self._wireframe, 1)

        # 尺寸指标
        dims_card = QWidget()
        dims_card.setObjectName("panelCard")
        dims_card.setStyleSheet(
            "QWidget#panelCard {"
            "background-color: #ffffff;"
            "border: 1px solid #e3e3e8;"
            "border-radius: 12px;"
            "}"
        )
        dims_root = QVBoxLayout(dims_card)
        dims_root.setContentsMargins(18, 14, 18, 14)
        dims_root.setSpacing(2)

        dims_title = QLabel("晶体尺寸")
        dims_title.setStyleSheet(
            "font-weight: 600; font-size: 13px; color: #86868b;"
        )
        dims_root.addWidget(dims_title)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(0, 8, 0, 0)

        params = [
            ("length_cm",        "长度 L"),
            ("width_cm",         "宽度 W"),
            ("body_height_cm",   "体高 Hb"),
            ("pyramid_height_cm", "锥高 Hp"),
            ("total_height_cm",  "总高"),
            ("volume_cm3",       "体积"),
        ]

        self._value_labels: dict[str, QLabel] = {}
        for i, (key, name) in enumerate(params):
            r, c = i // 3, i % 3
            lbl = QLabel(name)
            lbl.setStyleSheet("font-size: 11px; color: #86868b;")
            grid.addWidget(lbl, r * 2, c)
            val = QLabel("--")
            val.setStyleSheet(
                "font-size: 15px; font-weight: 600; color: #1d1d1f;"
            )
            grid.addWidget(val, r * 2 + 1, c)
            self._value_labels[key] = val

        dims_root.addLayout(grid)
        root.addWidget(dims_card)

        # 关闭按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("关闭")
        close_btn.setObjectName("primaryButton")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # 连接信号
        self._slider.valueChanged.connect(self._on_slider_changed)

        # 初始刷新（第 0 个月 = 当前晶体）
        self._on_slider_changed(0)

    def _on_slider_changed(self, months: int) -> None:
        """滑块变化时重新计算并刷新所有展示。"""
        if months == 0:
            self._month_label.setText("当前晶体")
        else:
            self._month_label.setText(f"{months} 个月")

        geo = simulate_growth(self._initial_geometry, months)
        self._wireframe.set_geometry(geo)

        units: dict[str, str] = {}
        for k in geo:
            units[k] = "cm³" if "cm3" in k else "cm"

        for key, val in geo.items():
            unit = units.get(key, "")
            self._value_labels[key].setText(f"{val:,.1f} {unit}")
