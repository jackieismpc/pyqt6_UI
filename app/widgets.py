"""
通用自定义控件与视觉辅助函数。

包含：
    StyledComboBox : 修复"下拉弹出时选项文字被截断"问题的下拉框。
    apply_card_shadow : 给卡片控件添加柔和投影，营造 Apple 风格的悬浮质感。
"""

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QGraphicsDropShadowEffect, QListView


class StyledComboBox(QComboBox):
    """
    带弹出层宽度自适应的下拉框。

    原生 QComboBox 的下拉弹出层(popup)宽度默认等于下拉框本身宽度，
    当选项文字比框更宽时，展开选择的瞬间文字会被截断，只有选完收起后才完整显示。
    这里重写 showPopup()，在弹出前把弹出视图的最小宽度撑到"最宽选项文字 + 余量"，
    从而保证展开时每一项文字都完整可见。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 用 QListView 作为弹出视图，圆角/内边距等样式更可控。
        self.setView(QListView())
        # 首次展示时按内容自适应下拉框自身宽度。
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

    def showPopup(self):
        """弹出前把弹出视图加宽到能容纳最宽的一项文字。"""
        fm = self.fontMetrics()
        widest = 0
        for i in range(self.count()):
            widest = max(widest, fm.horizontalAdvance(self.itemText(i)))
        # 余量：给箭头、选中标记、内边距留出空间。
        self.view().setMinimumWidth(widest + 56)
        super().showPopup()


def apply_card_shadow(widget, blur: int = 24, y_offset: int = 6, alpha: int = 30):
    """
    给控件添加一层柔和投影，让卡片有 Apple 式的轻微悬浮感。

    参数：
        widget   : 目标控件（每个控件只能有一个图形效果）。
        blur     : 阴影模糊半径，越大越柔和。
        y_offset : 阴影竖直偏移（向下为正）。
        alpha    : 阴影颜色透明度(0~255)，越小越淡。
    """
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setXOffset(0)
    effect.setYOffset(y_offset)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)
