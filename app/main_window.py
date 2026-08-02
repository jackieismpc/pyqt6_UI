"""
主窗口 MainWindow。

组织顶部控制栏、三个并排图像面板（原始输入 / 预处理后的输入 / 晶体几何模型）
以及底部结果栏，串联各控件信号，驱动整体交互流程。

三种输入：
- 视频：选一个视频文件 + 帧数，均匀抽帧联合建模；
- 图片：选一个目录，目录内所有图片视为同一晶体，联合建模；
- 实时：打开摄像头，多角度拍摄同一晶体，每拍一张在已有模型上增量优化。

真实推理耗时，全部放到后台线程（app/workers.py），推理期间面板显示「推理中…」优雅等待。
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .backend_interface import BackendInterface
from .camera_config import CameraConfig
from .camera_scanner import CameraDevice, scan_cameras
from .controls import ControlBar
from .growth_dialog import CrystalGrowthDialog
from .image_panel import ImagePanel
from .result_bar import ResultBar
from .widgets import StyledComboBox, apply_card_shadow
from .workers import RealtimeWorker, RunWorker

# 文件对话框的视频扩展名过滤
_VIDEO_FILTER = "视频文件 (*.mp4 *.mov *.avi *.mkv *.m4v);;所有文件 (*)"


class MainWindow(QMainWindow):
    """应用主窗口。"""

    def __init__(self, camera_config: CameraConfig | None = None):
        super().__init__()
        self.setWindowTitle("透明晶体体积估计 · 可视化")
        self.resize(1280, 820)

        self.backend = BackendInterface(camera_config=camera_config)
        self.result = None  # 当前 Stage1Result
        self.frame_index = 0  # 当前展示的帧索引

        self._run_worker = None      # 一次性推理线程（防 GC）
        self._rt_worker = None       # 实时线程（防 GC）
        self._realtime = False       # 是否处于实时模式（实时时左面板为摄像头实时预览）
        self._selected_camera_id: str = "0"  # 当前选中的摄像头
        self._last_input_path = ""  # 最近一次的输入路径（用于视频回放）
        self._last_input_type = ""  # 最近一次的输入类型

        # 摄像头延迟扫描（仅在用户切到实时模式时才枚举，避免启动时噪声警告）
        self._available_cameras: list[CameraDevice] = []
        self._cameras_scanned = False

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)

        # 顶部控制栏（摄像头延迟扫描，初始不传列表）
        self.control_bar = ControlBar(cameras=None)
        apply_card_shadow(self.control_bar)
        main_layout.addWidget(self.control_bar)

        # 中部：预处理产物类型选择下拉框（放入中间面板标题栏右侧）
        self.preprocess_selector = StyledComboBox()
        self.preprocess_selector.addItem("低光增强", "enhanced")
        self.preprocess_selector.addItem("边缘证据", "edges")
        self.preprocess_selector.addItem("剪影掩膜", "mask")
        self.preprocess_selector.addItem("线框叠加", "overlay")
        self.preprocess_selector.setMinimumWidth(120)
        self.preprocess_selector.setCurrentIndex(0)

        # 三个并排图像面板
        self.panel_raw = ImagePanel("原始输入")
        self.panel_preprocess = ImagePanel("预处理后的输入", header_widget=self.preprocess_selector)
        self.panel_geometry = ImagePanel("晶体几何模型（估计）")

        for panel in (self.panel_raw, self.panel_preprocess, self.panel_geometry):
            apply_card_shadow(panel, blur=22, y_offset=5, alpha=26)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.panel_raw)
        splitter.addWidget(self.panel_preprocess)
        splitter.addWidget(self.panel_geometry)
        splitter.setSizes([1, 1, 1])
        splitter.setHandleWidth(16)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter, 1)

        # 底部结果栏
        self.result_bar = ResultBar()
        apply_card_shadow(self.result_bar)
        main_layout.addWidget(self.result_bar)

        # 信号连接
        self.control_bar.runRequested.connect(self.on_run)
        self.control_bar.captureRequested.connect(self._on_capture_requested)
        self.control_bar.stopRealtimeRequested.connect(self._stop_realtime)
        self.control_bar.growthRequested.connect(self._on_growth_requested)
        self.control_bar.cameraChanged.connect(self._on_camera_changed)
        # 同步初始摄像头 ID（控制栏可能在信号连接前就填充了设备列表）
        self._on_camera_changed(self.control_bar.current_camera_id())
        self.result_bar.frameChanged.connect(self.on_frame_changed)
        self.preprocess_selector.currentIndexChanged.connect(self._on_preprocess_changed)

        # 启动时显示空白占位，等待用户选择输入
        self._show_blank_panels()

    # ==================================================================
    # 展示刷新
    # ==================================================================
    def load_and_show(self, res):
        """加载一个新的 Stage1Result 并刷新整个界面。"""
        self.result = res
        self.result_bar.set_result(res)

        # 有结果后启用生长预测按钮
        self.control_bar.set_growth_enabled(
            bool(res.aggregate_geometry and res.frames)
        )

        # 默认展示代表帧
        self.frame_index = 0
        for i, fr in enumerate(res.frames):
            if fr.name == res.representative_frame:
                self.frame_index = i
                break

        self.refresh_panels()

    def refresh_panels(self):
        """根据当前 self.result 和 self.frame_index 刷新图像面板与结果栏。

        实时模式下左面板保持摄像头实时预览，不被产物图覆盖。
        """
        if self.result is None or not self.result.frames:
            return

        frame_index = max(0, min(self.frame_index, len(self.result.frames) - 1))
        fr = self.result.frames[frame_index]

        # 左侧：原始输入（实时模式保留摄像头预览）
        if not self._realtime:
            self.panel_raw.set_image(fr.images.get("raw"))

        # 右侧：晶体几何模型（整段输入的汇总单图，所有帧共用）
        self.panel_geometry.set_image(self.result.geometry_preview)

        # 中间：预处理后的输入，取决于下拉框当前选中的产物类型
        preprocess_key = self.preprocess_selector.currentData()
        self.panel_preprocess.set_image(fr.images.get(preprocess_key))

        # 结果栏刷新
        self.result_bar.update_for_frame(self.result, frame_index)

    def _show_blank_panels(self):
        """清空三个面板与结果栏，显示占位文字。"""
        self.result = None
        self.frame_index = 0
        self.panel_raw.set_message("（无图像）")
        self.panel_preprocess.set_message("（无图像）")
        self.panel_geometry.set_message("（无图像）")
        self.result_bar.clear()
        self.control_bar.set_growth_enabled(False)

    def on_frame_changed(self, frame_index: int):
        self.frame_index = frame_index
        self.refresh_panels()

    def _on_preprocess_changed(self, _index: int):
        self.refresh_panels()

    def _on_growth_requested(self):
        """打开晶体生长预测对话框。"""
        geo = self.result.aggregate_geometry if self.result is not None else None
        dlg = CrystalGrowthDialog(aggregate_geometry_px=geo, parent=self)
        dlg.exec()

    def _on_camera_changed(self, device_id: str):
        """用户切换了摄像头选择。"""
        self._selected_camera_id = device_id

    def _ensure_cameras_scanned(self):
        """延迟扫描摄像头（仅首次调用时执行），避免启动时无用的设备枚举与 OpenCV 噪声。"""
        if self._cameras_scanned:
            return
        self._cameras_scanned = True
        try:
            self._available_cameras = scan_cameras()
        except Exception as exc:  # noqa: BLE001
            self._available_cameras = []
            self.control_bar.set_status(f"摄像头扫描失败：{exc}")
        self.control_bar.set_cameras(self._available_cameras)

    # ==================================================================
    # 一次性推理：视频 / 图片目录
    # ==================================================================
    def on_run(self, input_type: str):
        """响应运行请求：按输入类型选择输入并启动后台推理。"""
        if self._run_worker is not None and self._run_worker.isRunning():
            return  # 已有任务在跑

        if input_type == "realtime":
            self._start_realtime()
            return

        save = self.control_bar.save_results()
        camera_mode = self.backend._camera_config.mode

        if input_type == "video":
            path, _ = QFileDialog.getOpenFileName(self, "选择一个视频文件", "", _VIDEO_FILTER)
            if not path:
                return
            options = {"num_frames": self.control_bar.num_frames(), "save": save}
        elif camera_mode == "binocular":
            # 双目：依次选择左目、右目图片目录
            path_left = QFileDialog.getExistingDirectory(
                self, "选择左目相机图片目录（目录内所有图片视为同一晶体）")
            if not path_left:
                return
            path_right = QFileDialog.getExistingDirectory(
                self, "选择右目相机图片目录（目录内所有图片视为同一晶体）")
            if not path_right:
                return
            path = path_left
            options = {"save": save, "input_path2": path_right, "mode": "binocular"}
        else:  # image, monocular
            path = QFileDialog.getExistingDirectory(
                self, "选择一个图片目录（目录内所有图片视为同一晶体）")
            if not path:
                return
            options = {"save": save}

        self._last_input_path = path
        self._last_input_type = input_type
        self._begin_busy_state(input_type)
        self._run_worker = RunWorker(self.backend, path, input_type, options)
        self._run_worker.resultReady.connect(self._on_run_finished)
        self._run_worker.failed.connect(self._on_run_failed)
        self._run_worker.start()

    def _begin_busy_state(self, input_type: str):
        """进入等待态：禁用控件，三面板显示「推理中…」。"""
        self.control_bar.set_run_enabled(False)
        self.control_bar.set_status(f"推理中… · 输入类型：{input_type}")
        for panel in (self.panel_raw, self.panel_preprocess, self.panel_geometry):
            panel.set_message("推理中…\n算法运行中，请稍候")

    def _on_run_finished(self, result):
        self.control_bar.set_run_enabled(True)
        self.load_and_show(result)
        self.control_bar.set_status(
            f"完成 · 帧数 {result.frame_count} · fit_ready {result.fit_ready_count}/{result.frame_count}")
        # 视频输入：在原始面板回放视频
        if self._last_input_type == "video" and self._last_input_path:
            self.panel_raw.set_video(self._last_input_path)

    def _on_run_failed(self, message: str):
        self.control_bar.set_run_enabled(True)
        self.control_bar.set_status("推理失败")
        self._show_blank_panels()
        QMessageBox.critical(self, "推理失败", f"后端运行出错：\n{message}")

    # ==================================================================
    # 实时：摄像头多视角增量估计
    # ==================================================================
    def _start_realtime(self):
        """打开摄像头，进入实时增量拍摄模式。"""
        if self._rt_worker is not None and self._rt_worker.isRunning():
            return

        # 延迟扫描摄像头（仅在真正需要时才枚举设备）
        self._ensure_cameras_scanned()

        # 检查是否有可用摄像头（使用控制栏最新选中的 ID）
        cam_id = self.control_bar.current_camera_id()
        self._selected_camera_id = cam_id
        if not cam_id or not self._available_cameras:
            QMessageBox.warning(
                self, "实时模式",
                "未检测到任何摄像头设备。\n"
                "请连接 USB 摄像头（免驱）或海康工业相机后再试。\n\n"
                "Windows 用户请检查：设置 → 隐私与安全性 → 摄像头 → 允许应用访问。",
            )
            return

        self._realtime = True
        self.control_bar.set_realtime_active(True)
        self.control_bar.set_shot_counter(0, self.control_bar.target_shots())
        self.control_bar.set_capture_enabled(False)  # 摄像头就绪前不可拍
        self.control_bar.set_status("正在打开摄像头…")
        self.panel_raw.set_message("正在打开摄像头…")
        self.panel_preprocess.set_message("等待拍摄…\n多角度拍摄同一晶体做增量联合估计")
        self.panel_geometry.set_message("等待拍摄…")

        self._rt_worker = RealtimeWorker(
            self.backend,
            camera_id=self._selected_camera_id,
            save=self.control_bar.save_results(),
        )
        self._rt_worker.previewFrame.connect(self._on_preview_frame)
        self._rt_worker.cameraOpened.connect(self._on_camera_opened)
        self._rt_worker.processingStarted.connect(self._on_rt_processing_started)
        self._rt_worker.shotProcessed.connect(self._on_shot_processed)
        self._rt_worker.error.connect(self._on_rt_error)
        self._rt_worker.stopped.connect(self._on_rt_stopped)
        self._rt_worker.start()

    def _on_preview_frame(self, frame, _timestamp: float = 0.0):
        """摄像头预览帧 -> 左面板实时显示。

        携带时间戳，跳过过期帧避免主线程信号队列积压导致卡顿。
        """
        if not self._realtime:
            return
        # 帧丢弃：如果上一帧还在处理中，跳过中间帧
        if not hasattr(self, "_preview_busy"):
            self._preview_busy = False
        if self._preview_busy:
            return
        self._preview_busy = True
        try:
            self.panel_raw.set_np_bgr(frame)
        finally:
            self._preview_busy = False

    def _on_camera_opened(self, ok: bool):
        if ok:
            self.control_bar.set_capture_enabled(True)
            self.control_bar.set_status("摄像头就绪 · 点击「拍摄」采集多视角照片")
        else:
            # 摄像头打开失败 — worker 随后会 emit stopped
            self.control_bar.set_status("摄像头打开失败，即将退出实时模式")

    def _on_capture_requested(self):
        """用户点「拍摄」：转发给实时线程抓帧处理。"""
        if self._rt_worker is not None and self._rt_worker.isRunning():
            self.control_bar.set_capture_enabled(False)
            self._rt_worker.request_capture()

    def _on_rt_processing_started(self):
        """一张照片开始推理：中右面板进入等待态。"""
        self.control_bar.set_capture_enabled(False)
        self.control_bar.set_status("正在处理这张照片…")
        self.panel_preprocess.set_message("推理中…")
        self.panel_geometry.set_message("推理中…")

    def _on_shot_processed(self, result, count: int):
        """一张照片处理完：增量刷新中右面板与结果栏（左面板保持实时预览）。"""
        self.result = result
        self.result_bar.set_result(result)
        # 展示最新并入的这一帧
        self.frame_index = max(0, len(result.frames) - 1)
        self.refresh_panels()

        target = self.control_bar.target_shots()
        self.control_bar.set_shot_counter(count, target)
        self.control_bar.set_capture_enabled(True)
        if count >= target:
            self.control_bar.set_status(f"已达目标 {count}/{target} 张 · 可继续拍摄或点「结束实时」")
        else:
            self.control_bar.set_status(f"已并入第 {count} 张 · 联合模型已更新")

    def _on_rt_error(self, message: str):
        self.control_bar.set_capture_enabled(True)
        self.control_bar.set_status("实时出错")
        QMessageBox.warning(self, "实时模式", message)

    def _stop_realtime(self):
        """请求结束实时会话。"""
        if self._rt_worker is not None and self._rt_worker.isRunning():
            self.control_bar.set_status("正在结束实时…")
            self._rt_worker.stop()

    def _on_rt_stopped(self):
        """实时线程已停止：恢复常规布局。"""
        self._realtime = False
        self.control_bar.set_realtime_active(False)
        self.control_bar.set_run_enabled(True)
        count = self.result.frame_count if self.result is not None else 0
        self.control_bar.set_status(f"实时已结束 · 本次采集 {count} 张")
        # 清理预览状态
        self._preview_busy = False
        # 恢复面板：如果有之前的结果就展示，否则显示空白
        if self.result is not None and self.result.frames:
            self.refresh_panels()
        else:
            self._show_blank_panels()
        self._rt_worker = None

    def closeEvent(self, event):
        """关闭窗口时确保摄像头线程退出。"""
        if self._run_worker is not None and self._run_worker.isRunning():
            self._run_worker.cancel()
            if not self._run_worker.wait(5000):
                QMessageBox.warning(
                    self,
                    "任务仍在运行",
                    "后端任务尚未安全结束，请等待当前任务完成后再关闭窗口。",
                )
                event.ignore()
                return
        if self._rt_worker is not None and self._rt_worker.isRunning():
            self._rt_worker.stop()
            if not self._rt_worker.wait(3000):
                QMessageBox.warning(
                    self,
                    "摄像头仍在使用",
                    "摄像头任务尚未安全结束，请稍候再关闭窗口。",
                )
                event.ignore()
                return
        super().closeEvent(event)
