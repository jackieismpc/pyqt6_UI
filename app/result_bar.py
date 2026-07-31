"""
底部结果栏控件 ResultBar。

设计原则：左下角只突出两个最重要的指标——「估计体积」与「置信度」，
二者以等权重的"标题 + 数值"块并排呈现；帧的技术细节（fit_ready / 可见比 /
覆盖比）与汇总(共识)信息不再占用版面，改为鼠标悬停在置信度上时的 tooltip 显示。
右侧提供帧切换器（上一帧 / 下拉 / 下一帧）。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .models import Stage1Result, aggregate_confidence, confidence_color, frame_confidence
from .widgets import StyledComboBox


class ResultBar(QWidget):
    """底部横向结果栏。"""

    # 帧切换信号，值为帧索引
    frameChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultBar")

        self._result: Stage1Result = None
        self._updating_combo = False  # 防止程序化设置下拉框时触发信号递归

        row = QHBoxLayout(self)
        row.setContentsMargins(18, 12, 18, 12)
        row.setSpacing(20)

        # ---- 指标一：估计体积（紧凑）----
        row.addLayout(self._make_stat_block("估计体积", "volume_label"))

        # ---- 细分隔线 ----
        divider1 = QWidget()
        divider1.setObjectName("statDivider")
        divider1.setFixedSize(1, 34)
        row.addWidget(divider1)

        # ---- 指标二：置信度 ----
        row.addLayout(self._make_stat_block("置信度", "conf_label"))

        # ---- 细分隔线 ----
        divider2 = QWidget()
        divider2.setObjectName("statDivider")
        divider2.setFixedSize(1, 34)
        row.addWidget(divider2)

        # ---- 指标三：尺寸（置信度右侧空白区域）----
        dims_block = QVBoxLayout()
        dims_block.setSpacing(2)
        dims_cap = QLabel("尺寸")
        dims_cap.setObjectName("panelSecondary")
        dims_block.addWidget(dims_cap)
        self.dims_label = QLabel("")
        self.dims_label.setObjectName("statValue")
        self.dims_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #1d1d1f;")
        dims_block.addWidget(self.dims_label)
        row.addLayout(dims_block)

        row.addStretch(1)

        # ---- 帧选择器 ----
        self.btn_prev = QPushButton("‹ 上一帧")
        self.frame_combo = StyledComboBox()
        self.frame_combo.setMinimumWidth(150)
        self.btn_next = QPushButton("下一帧 ›")
        row.addWidget(self.btn_prev)
        row.addWidget(self.frame_combo)
        row.addWidget(self.btn_next)

        # 初始占位
        self.volume_label.setText("--")
        self.conf_label.setText("--")

        # 信号连接
        self.btn_prev.clicked.connect(self._on_prev_clicked)
        self.btn_next.clicked.connect(self._on_next_clicked)
        self.frame_combo.currentIndexChanged.connect(self._on_combo_changed)

    def _make_stat_block(self, caption: str, value_attr: str) -> QVBoxLayout:
        """构造一个"小标题 + 大数值"的指标块，并把数值 QLabel 存到 self.<value_attr>。"""
        box = QVBoxLayout()
        box.setSpacing(1)
        cap = QLabel(caption)
        cap.setObjectName("panelSecondary")
        value = QLabel("--")
        value.setObjectName("statValue")
        box.addWidget(cap)
        box.addWidget(value)
        setattr(self, value_attr, value)
        return box

    def clear(self):
        """清空所有显示，回退到初始占位。"""
        self._result = None
        self.volume_label.setText("--")
        self.conf_label.setText("--")
        self.dims_label.setText("")
        self.conf_label.setToolTip("")
        self.volume_label.setToolTip("")
        self.conf_label.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #1d1d1f;"
        )
        self._updating_combo = True
        self.frame_combo.clear()
        self._updating_combo = False

    def set_result(self, res: Stage1Result):
        """根据新的 Stage1Result 填充帧下拉框，默认选中代表帧。"""
        self._result = res
        self._updating_combo = True
        self.frame_combo.clear()
        default_index = 0
        for i, fr in enumerate(res.frames):
            label = fr.name
            if fr.name == res.representative_frame:
                label = f"{fr.name}（代表帧）"
                default_index = i
            self.frame_combo.addItem(label, i)
        self._updating_combo = False

        if res.frames:
            self.frame_combo.setCurrentIndex(default_index)
        self.update_for_frame(res, default_index if res.frames else 0)

    def update_for_frame(self, res: Stage1Result, frame_index: int):
        """刷新体积与置信度两个指标；技术细节写入 tooltip。"""
        self._result = res
        if not res.frames:
            return
        frame_index = max(0, min(frame_index, len(res.frames) - 1))
        fr = res.frames[frame_index]

        # ---- 体积与尺寸 ----
        metric = res.metric if isinstance(res.metric, dict) else None
        if metric and metric.get("volume") is not None:
            unit = metric.get("unit", "cm³")
            self.volume_label.setText(f"{metric['volume']:,.1f} {unit}")

            dims = metric.get("dimensions_cm", {})
            if dims:
                L = dims.get("length", 0)
                W = dims.get("width", 0)
                Hb = dims.get("body_height", 0)
                Hp = dims.get("pyramid_height", 0)
                self.dims_label.setText(
                    f"长{L:.1f}  宽{W:.1f}  体高{Hb:.1f}  锥高{Hp:.1f} cm"
                )
            else:
                self.dims_label.setText("--")
        else:
            self.volume_label.setText(f"{res.aggregate_volume_px3:,.0f} px³")
            self.dims_label.setText(
                f"长{res.aggregate_geometry.get('length_px',0):.0f} × "
                f"宽{res.aggregate_geometry.get('width_px',0):.0f} px"
            )

        # 置信度：前置状态圆点 + 彩色数值（不再用大块填充徽章）
        label, pct = frame_confidence(fr)
        color = confidence_color(label)
        self.conf_label.setText(f"● {label} · {pct}%")
        self.conf_label.setStyleSheet(
            f"font-size: 16px; font-weight: 600; color: {color};"
        )

        # 技术细节 + 汇总信息：折叠进 tooltip，保持版面清爽
        agg_label, agg_pct = aggregate_confidence(res)
        px_vol = f"像素体积：{res.aggregate_volume_px3:,.0f} px³"
        metric_info = ""
        if isinstance(res.metric, dict) and res.metric.get("scale_info"):
            si = res.metric["scale_info"]
            method = si.get("method", "?")
            dist = si.get("distance_mm", 0)
            metric_info = f"\n换算方式：{method} · 距离 {dist:.0f}mm"
            if si.get("corrected_by"):
                metric_info += f"（已用尺度锚点校正 ×{si.get('correction_factor',1):.2f}）"
        tip = (
            f"{px_vol}{metric_info}\n"
            f"当前帧 {fr.name} 单帧体积：{fr.volume_px3:,.0f} px³（仅诊断）\n"
            f"fit_ready：{'是' if fr.fit_ready else '否'}　可见比：{fr.visible_ratio:.2f}　"
            f"覆盖比：{fr.coverage_ratio:.2f}\n"
            f"fit_ready {res.fit_ready_count}/{res.frame_count}　总体 {agg_label} {agg_pct}%"
        )
        self.conf_label.setToolTip(tip)
        self.volume_label.setToolTip(tip)

        # 同步下拉框
        if self.frame_combo.currentData() != frame_index:
            self._updating_combo = True
            idx = self.frame_combo.findData(frame_index)
            if idx >= 0:
                self.frame_combo.setCurrentIndex(idx)
            self._updating_combo = False

    def _on_prev_clicked(self):
        if self._result is None or not self._result.frames:
            return
        current = self.frame_combo.currentData() or 0
        self.frameChanged.emit(max(0, current - 1))

    def _on_next_clicked(self):
        if self._result is None or not self._result.frames:
            return
        current = self.frame_combo.currentData() or 0
        self.frameChanged.emit(min(len(self._result.frames) - 1, current + 1))

    def _on_combo_changed(self, index: int):
        if self._updating_combo or index < 0:
            return
        frame_index = self.frame_combo.currentData()
        if frame_index is not None:
            self.frameChanged.emit(frame_index)
