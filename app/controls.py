"""
顶部控制栏控件 ControlBar。

提供输入类型选择（视频/图片/实时）、运行按钮、状态标签，以及：
- 视频专用「帧数」选择器（从视频均匀抽多少帧参与联合建模）；
- 实时专用「拍摄」控件（目标张数 + 拍摄按钮 + 计数 + 结束），实时会话激活时才显示。

输入类型采用 macOS 风格的分段控件(segmented control)呈现。
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .camera_scanner import CameraDevice


class ControlBar(QWidget):
    """顶部横向控制栏。"""

    # 输入类型变化信号，值为 "video" / "image" / "realtime"
    inputTypeChanged = pyqtSignal(str)
    # 运行请求信号，发出当前输入类型
    runRequested = pyqtSignal(str)
    # 实时：请求拍摄一张
    captureRequested = pyqtSignal()
    # 实时：结束会话
    stopRealtimeRequested = pyqtSignal()
    # 生长预测
    growthRequested = pyqtSignal()
    # 摄像头选择（实时模式）
    cameraChanged = pyqtSignal(str)  # device_id

    def __init__(self, cameras: list[CameraDevice] | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("controlBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        # 左侧：一句轻描述
        self.section_label = QLabel("输入")
        self.section_label.setObjectName("panelSecondary")
        layout.addWidget(self.section_label)

        # 输入类型分段控件（互斥 checkable 按钮，装在一个圆角底槽里）
        self.segmented = QWidget()
        self.segmented.setObjectName("segmented")
        seg_layout = QHBoxLayout(self.segmented)
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(0)

        self.btn_video = QPushButton("视频")
        self.btn_image = QPushButton("图片")
        self.btn_realtime = QPushButton("实时")

        for btn in (self.btn_video, self.btn_image, self.btn_realtime):
            btn.setCheckable(True)
            seg_layout.addWidget(btn)

        self.input_type_group = QButtonGroup(self)
        self.input_type_group.setExclusive(True)
        self.input_type_group.addButton(self.btn_video)
        self.input_type_group.addButton(self.btn_image)
        self.input_type_group.addButton(self.btn_realtime)

        # 默认选中"图片"
        self.btn_image.setChecked(True)
        self.btn_realtime.setToolTip("实时：打开摄像头，多角度拍摄同一晶体做增量联合估计")

        layout.addWidget(self.segmented)

        # 视频专用：抽帧数量
        self.frame_label = QLabel("帧数")
        self.frame_label.setObjectName("panelSecondary")
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 60)
        self.frame_spin.setValue(7)
        self.frame_spin.setToolTip("视频均匀抽取多少帧参与联合建模（默认 7）")
        layout.addWidget(self.frame_label)
        layout.addWidget(self.frame_spin)

        # 保存结果开关（默认不保存；勾选后本次产物写入 assets/<日期-时间>/）
        self.save_check = QCheckBox("保存结果")
        self.save_check.setChecked(False)
        self.save_check.setToolTip("勾选后，本次推理产物保存到 assets/<日期-时间>/ 留档；默认不保存")
        layout.addWidget(self.save_check)

        # 主运行按钮
        self.btn_run = QPushButton("选择并运行")
        self.btn_run.setObjectName("primaryButton")
        layout.addWidget(self.btn_run)

        # 生长预测按钮（默认隐藏，有结果时显示）
        self.btn_growth = QPushButton("生长预测")
        self.btn_growth.setToolTip("基于当前晶体模型预测未来 1-6 个月的长势")
        self.btn_growth.setVisible(False)
        layout.addWidget(self.btn_growth)

        # ---- 实时专用控件（默认隐藏，实时会话激活时显示）----
        self.cam_label = QLabel("摄像头")
        self.cam_label.setObjectName("panelSecondary")
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(180)
        self.cam_combo.setToolTip("选择要使用的摄像头设备")
        self.cam_combo.currentIndexChanged.connect(
            lambda: self.cameraChanged.emit(self.current_camera_id())
        )

        self.shots_label = QLabel("目标张数")
        self.shots_label.setObjectName("panelSecondary")
        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(1, 30)
        self.shots_spin.setValue(5)
        self.shots_spin.setToolTip("计划拍摄多少张（可随时结束）")
        self.btn_capture = QPushButton("拍摄")
        self.btn_capture.setObjectName("primaryButton")
        self.shot_counter = QLabel("已拍 0/5")
        self.shot_counter.setObjectName("panelSecondary")
        self.btn_stop_realtime = QPushButton("结束实时")
        for w in (self.cam_label, self.cam_combo, self.shots_label, self.shots_spin,
                  self.btn_capture, self.shot_counter, self.btn_stop_realtime):
            layout.addWidget(w)

        layout.addStretch(1)

        # 右侧：状态标签
        self.status_label = QLabel("当前使用示例数据")
        self.status_label.setObjectName("panelSecondary")
        layout.addWidget(self.status_label)

        # 信号连接
        self.btn_video.toggled.connect(self._on_type_toggled)
        self.btn_image.toggled.connect(self._on_type_toggled)
        self.btn_realtime.toggled.connect(self._on_type_toggled)
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_capture.clicked.connect(self.captureRequested.emit)
        self.btn_stop_realtime.clicked.connect(self.stopRealtimeRequested.emit)
        self.btn_growth.clicked.connect(self.growthRequested.emit)

        # 初始可见性
        self._populate_cameras(cameras)
        self.set_realtime_active(False)
        self._sync_frame_visibility()

    # ---- 交互回调 ----
    def _on_type_toggled(self, checked: bool):
        """任一互斥按钮状态变化时，仅在被选中时发出信号并同步控件可见性。"""
        if checked:
            self._sync_frame_visibility()
            self.inputTypeChanged.emit(self.current_input_type())

    def _on_run_clicked(self):
        self.runRequested.emit(self.current_input_type())

    def _sync_frame_visibility(self):
        """帧数选择器仅在「视频」输入时显示。"""
        is_video = self.btn_video.isChecked()
        self.frame_label.setVisible(is_video)
        self.frame_spin.setVisible(is_video)

    # ---- 对外查询 ----
    def current_input_type(self) -> str:
        """返回当前选中的输入类型：'video' / 'image' / 'realtime'。"""
        if self.btn_video.isChecked():
            return "video"
        if self.btn_realtime.isChecked():
            return "realtime"
        return "image"

    def num_frames(self) -> int:
        """视频抽帧数量。"""
        return int(self.frame_spin.value())

    def target_shots(self) -> int:
        """实时目标拍摄张数。"""
        return int(self.shots_spin.value())

    def save_results(self) -> bool:
        """是否保存本次推理结果到 assets/<日期-时间>/。"""
        return self.save_check.isChecked()

    # ---- 状态设置 ----
    def _populate_cameras(self, cameras: list[CameraDevice] | None):
        """填充摄像头下拉列表。"""
        self.cam_combo.clear()
        if not cameras:
            self.cam_combo.addItem("未检测到摄像头", "")
            self.cam_combo.setEnabled(False)
            return
        for cam in cameras:
            self.cam_combo.addItem(cam.label, cam.device_id)
        self.cam_combo.setEnabled(True)

    def current_camera_id(self) -> str:
        """返回当前选中的摄像头 device_id。"""
        return self.cam_combo.currentData() or "0"

    def set_status(self, text: str):
        """更新右侧状态标签文本。"""
        self.status_label.setText(text)

    def set_realtime_active(self, active: bool):
        """切换到实时拍摄模式布局：隐藏类型/运行/帧数，显示拍摄控件（反之亦然）。"""
        # 常规控件
        self.section_label.setVisible(not active)
        self.segmented.setVisible(not active)
        self.btn_run.setVisible(not active)
        self.save_check.setVisible(not active)
        if active:
            self.frame_label.setVisible(False)
            self.frame_spin.setVisible(False)
        else:
            self._sync_frame_visibility()
        # 实时控件
        for w in (self.cam_label, self.cam_combo, self.shots_label, self.shots_spin,
                  self.btn_capture, self.shot_counter, self.btn_stop_realtime):
            w.setVisible(active)

    def set_shot_counter(self, done: int, total: int):
        self.shot_counter.setText(f"已拍 {done}/{total}")

    def set_capture_enabled(self, enabled: bool):
        self.btn_capture.setEnabled(enabled)

    def set_run_enabled(self, enabled: bool):
        self.btn_run.setEnabled(enabled)
        for btn in (self.btn_video, self.btn_image, self.btn_realtime):
            btn.setEnabled(enabled)

    def set_growth_enabled(self, enabled: bool):
        """控制「生长预测」按钮的可见性（有推理结果时启用）。"""
        self.btn_growth.setVisible(enabled)

    def set_cameras(self, cameras: list[CameraDevice] | None):
        """更新摄像头下拉列表（用于延迟扫描后填充）。"""
        self._populate_cameras(cameras)
