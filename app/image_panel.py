"""
可复用的图像展示面板控件。

包含：
    ScaledImageLabel: 按比例缩放并居中显示图片的 QLabel，无图时显示占位文字。
    ImagePanel: 带标题栏的图像面板卡片，标题栏右侧可插入自定义控件（如下拉框）。
"""

import numpy as np

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QImageReader, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QStackedLayout, QVBoxLayout, QWidget

from .video_player import VideoPlayer


MAX_STATIC_PREVIEW_SIDE = 1280


def _scaled_pixmap_from_path(path: str, max_side: int = MAX_STATIC_PREVIEW_SIDE) -> QPixmap:
    """读取图片时直接限制解码尺寸，避免把超大原图解码成完整 QPixmap。

    先让 Qt 通过 Unicode 路径读取；某些旧平台插件若不支持该路径，再回退到
    Python 二进制 + QImageReader。两条路径都设置 scaled size，前端只保留预览图。
    """
    if not path:
        return QPixmap()

    def read(reader: QImageReader) -> QPixmap:
        reader.setAutoTransform(True)
        original = reader.size()
        if original.isValid() and max(original.width(), original.height()) > max_side:
            scale = float(max_side) / max(original.width(), original.height())
            reader.setScaledSize(QSize(
                max(1, round(original.width() * scale)),
                max(1, round(original.height() * scale)),
            ))
        image = reader.read()
        return QPixmap.fromImage(image) if not image.isNull() else QPixmap()

    pixmap = read(QImageReader(path))
    if not pixmap.isNull():
        return pixmap

    # 兼容 Windows 上少数 Qt 图像插件的路径编码问题；这里只保留压缩字节，
    # 不再用 QPixmap.loadFromData 解码完整原图。
    try:
        from PyQt6.QtCore import QBuffer, QByteArray, QIODevice

        with open(path, "rb") as handle:
            buffer = QBuffer()
            buffer.setData(QByteArray(handle.read()))
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            pixmap = read(QImageReader(buffer))
            buffer.close()
            return pixmap
    except (OSError, ValueError):
        return QPixmap()


class ScaledImageLabel(QLabel):
    """
    始终按比例缩放填充显示图片的 QLabel。

    保存原始 QPixmap，在 resizeEvent 中根据当前控件大小重新生成缩放后的
    pixmap 并显示，避免图片被拉伸变形。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_pixmap = None
        self.setObjectName("scaledImageLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._show_placeholder()

    def _show_placeholder(self, text: str = "（无图像）"):
        """显示占位文字（无图 / 等待 / 推理中等状态）。"""
        self._original_pixmap = None
        self.setObjectName("imagePlaceholder")
        # 触发样式重新计算
        self.style().unpolish(self)
        self.style().polish(self)
        super().setPixmap(QPixmap())
        self.setText(text)

    def set_message(self, text: str):
        """清空图片并显示一段状态文字（如“推理中…”“等待拍摄…”）。"""
        self._show_placeholder(text)

    def set_np_bgr(self, array):
        """直接显示一张 BGR 的 numpy 图（用于摄像头实时预览，避免落盘）。

        优化：使用 Format_BGR888 避免 BGR→RGB 的内存拷贝，QImage 零拷贝引用 ndarray。
        ndarray 需是连续内存（C-contiguous）；如果不是则拷贝一份。
        """
        if array is None or getattr(array, "size", 0) == 0:
            self._show_placeholder()
            return
        if max(array.shape[:2]) > MAX_STATIC_PREVIEW_SIDE:
            # 即使调用方忘记预缩放，也不要把工业相机原始大帧转成 QPixmap。
            import cv2

            scale = MAX_STATIC_PREVIEW_SIDE / float(max(array.shape[:2]))
            array = cv2.resize(
                array,
                (max(1, round(array.shape[1] * scale)), max(1, round(array.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        h, w = array.shape[:2]
        # Format_BGR888：零拷贝，无需把 BGR 转 RGB
        qimg = QImage(array.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        self.setObjectName("scaledImageLabel")
        self.style().unpolish(self)
        self.style().polish(self)
        self.setText("")
        self._original_pixmap = QPixmap.fromImage(qimg)
        self._update_scaled_pixmap()

    def set_image(self, path):
        """
        设置要显示的图片。

        参数：
            path: 图片文件路径；为 None 或文件不存在时清空显示并显示占位文字。

        使用 Python open() 读 bytes 再构造 QPixmap，避免 Qt 内部路径编码问题
        （尤其是 Windows 上非 ASCII 路径）。
        """
        if not path:
            self._show_placeholder()
            return

        pixmap = _scaled_pixmap_from_path(path)
        if pixmap.isNull():
            self._show_placeholder()
            return

        self.setObjectName("scaledImageLabel")
        self.style().unpolish(self)
        self.style().polish(self)
        self.setText("")
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()

    def setPixmap(self, pixmap):
        """重写 setPixmap，保存原始图并按当前尺寸缩放显示。"""
        self._original_pixmap = pixmap
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scaled = self._original_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        """控件尺寸变化时重新缩放当前图片。"""
        super().resizeEvent(event)
        self._update_scaled_pixmap()


class ImagePanel(QWidget):
    """
    带标题栏的图像面板卡片。

    布局：
        顶部标题栏：左侧标题文字 + 右侧可选自定义控件（如下拉框）
        下方：ScaledImageLabel，占满剩余空间
    """

    def __init__(self, title: str, header_widget: QWidget = None, parent=None):
        super().__init__(parent)
        self.setObjectName("panelCard")

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(14, 12, 14, 14)
        outer_layout.setSpacing(10)

        # 标题栏
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        if header_widget is not None:
            header_layout.addWidget(header_widget)
        outer_layout.addLayout(header_layout)

        # 图像展示区（StackedLayout 切换静态/视频两种模式）
        self._display_stack = QStackedLayout()
        self.image_label = ScaledImageLabel(self)
        self._video_player = VideoPlayer(self)
        self._display_stack.addWidget(self.image_label)
        self._display_stack.addWidget(self._video_player)
        self._display_stack.setCurrentWidget(self.image_label)
        outer_layout.addLayout(self._display_stack, 1)

    def set_image(self, path):
        """转发给内部 ScaledImageLabel，切换到静态图模式。"""
        self._video_player.stop()
        self._display_stack.setCurrentWidget(self.image_label)
        self.image_label.set_image(path)

    def set_video(self, path: str):
        """切换到视频循环播放模式。"""
        self._display_stack.setCurrentWidget(self._video_player)
        self._video_player.set_video(path)

    def set_message(self, text: str):
        """在图像区显示状态文字（等待 / 推理中等）。"""
        self._video_player.stop()
        self._display_stack.setCurrentWidget(self.image_label)
        self.image_label.set_message(text)

    def set_np_bgr(self, array):
        """在图像区显示一张 BGR numpy 图（摄像头实时预览），切换到静态模式。"""
        self._video_player.stop()
        self._display_stack.setCurrentWidget(self.image_label)
        self.image_label.set_np_bgr(array)

    def stop(self) -> None:
        """停止视频定时器并释放视频句柄。"""
        self._video_player.stop()
