"""
浅色主题定义模块（Apple 风格）。

提供统一的 QSS 样式字符串 LIGHT_QSS 以及 apply_theme(app) 函数，
供 main.py 在创建 QApplication 后调用，统一全局观感。

设计语言参考 macOS：
    - 背景使用 Apple 浅灰 #f5f5f7，卡片纯白 + 细发丝边框 + 大圆角(12px)；
    - 强调色使用系统蓝 #007aff；
    - 文字主色 #1d1d1f、次要色 #86868b；
    - 分段控件(segmented control)用于输入类型切换；
    - macOS/Windows 使用系统原生 CJK 回退字体；
    - Linux 需安装 Noto Sans CJK 或文泉驿字体（详见 README），代码自动检测可用字体。
配色统一在文件顶部集中定义，便于整体调色。
"""

# ---- 统一调色板（改这里即可全局换色） ----
BG = "#f5f5f7"            # 窗口背景（Apple 浅灰）
CARD = "#ffffff"          # 卡片背景
HAIRLINE = "#e3e3e8"      # 发丝分隔线/边框
TEXT = "#1d1d1f"          # 主文字
TEXT_2 = "#86868b"        # 次要文字
ACCENT = "#007aff"        # 系统蓝
ACCENT_HOVER = "#0a6cff"  # 蓝色 hover
ACCENT_PRESS = "#0060df"  # 蓝色按下
TINT = "#eaf2ff"          # 蓝色浅底(hover 提示)
SEG_BG = "#e9e9ec"        # 分段控件底槽
FIELD = "#f2f2f4"         # 输入类控件底色

LIGHT_QSS = f"""
* {{
    /* 不在此处硬编码字体族：应用字体由 apply_theme() 用系统原生 UI 字体设置，
       避免命名不存在的字体族（如 "SF Pro Text"）触发启动告警与字体别名查找开销。 */
    font-size: 13px;
    color: {TEXT};
}}

QMainWindow, QWidget#centralWidget {{
    background-color: {BG};
}}

QWidget {{
    background-color: transparent;
}}

/* ---- 卡片面板 ---- */
QWidget#panelCard {{
    background-color: {CARD};
    border: 1px solid {HAIRLINE};
    border-radius: 12px;
}}

/* 图像展示区：浅灰内衬 + 圆角，让画面像被托住 */
QLabel#scaledImageLabel, QLabel#imagePlaceholder {{
    background-color: {FIELD};
    border-radius: 9px;
}}

QLabel#panelTitle {{
    color: {TEXT};
    font-weight: 600;
    font-size: 15px;
    padding: 2px 2px;
}}

QLabel#panelSecondary {{
    color: {TEXT_2};
    font-size: 12px;
}}

QLabel#statValue {{
    color: {TEXT};
    font-size: 16px;
    font-weight: 600;
}}

QWidget#statDivider {{
    background-color: {HAIRLINE};
}}

QLabel#imagePlaceholder {{
    color: #b0b0b6;
    font-size: 13px;
}}

/* ---- 顶部控制栏 / 底部结果栏（卡片） ---- */
QWidget#controlBar, QWidget#resultBar {{
    background-color: {CARD};
    border: 1px solid {HAIRLINE};
    border-radius: 12px;
}}

/* ---- 分段控件（输入类型三选一） ---- */
QWidget#segmented {{
    background-color: {SEG_BG};
    border-radius: 9px;
}}

QWidget#segmented QPushButton {{
    background-color: transparent;
    color: {TEXT};
    border: none;
    border-radius: 7px;
    padding: 5px 18px;
    margin: 2px;
    font-weight: 500;
}}

QWidget#segmented QPushButton:hover {{
    color: #000000;
}}

QWidget#segmented QPushButton:checked {{
    background-color: {CARD};
    color: {TEXT};
    font-weight: 600;
}}

/* ---- 普通描边按钮（帧切换等） ---- */
QPushButton {{
    background-color: {CARD};
    color: {ACCENT};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {TINT};
    border-color: {ACCENT};
}}

QPushButton:pressed {{
    background-color: #dfebff;
}}

QPushButton:disabled {{
    color: #b7b7bd;
    border-color: {HAIRLINE};
    background-color: {FIELD};
}}

/* ---- 主按钮（加载/运行）：实心系统蓝胶囊 ---- */
QPushButton#primaryButton {{
    background-color: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 9px;
    padding: 7px 20px;
    font-weight: 600;
}}

QPushButton#primaryButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton#primaryButton:pressed {{
    background-color: {ACCENT_PRESS};
}}

/* ---- 下拉框 ---- */
QComboBox {{
    background-color: {FIELD};
    color: {TEXT};
    border: 1px solid {HAIRLINE};
    border-radius: 8px;
    padding: 5px 12px;
    min-height: 22px;
}}

QComboBox:hover {{
    border-color: {ACCENT};
}}

QComboBox:focus {{
    border-color: {ACCENT};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

/* 弹出选项列表 */
QComboBox QAbstractItemView {{
    background-color: {CARD};
    color: {TEXT};
    border: 1px solid {HAIRLINE};
    border-radius: 10px;
    padding: 4px;
    outline: none;
    selection-background-color: {TINT};
    selection-color: {ACCENT};
}}

QComboBox QAbstractItemView::item {{
    min-height: 26px;
    padding: 4px 12px;
    border-radius: 6px;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {FIELD};
}}

/* ---- 普通标签 ---- */
QLabel {{
    color: {TEXT};
    background-color: transparent;
}}

/* ---- 分隔条：透明留白，形成卡片之间的间隙 ---- */
QSplitter::handle {{
    background-color: transparent;
}}

QToolTip {{
    background-color: {TEXT};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 4px 8px;
}}
"""


def apply_theme(app):
    """
    应用浅色（Apple 风格）主题。

    macOS / Windows：使用系统原生 UI 字体，CJK 回退由系统内置字体链处理。
    Linux：系统字体通常不含中文字形，需显式指定 CJK 字体族。
    按优先级检测：Noto Sans CJK SC → 文泉驿微米黑 → 文泉驿正黑 →
    Noto Sans SC → 思源黑体 → AR PL 系列。
    若都不可用（未安装任何中文字体），回退到系统字体并提示用户安装。
    """
    import platform as _platform
    from PyQt6.QtGui import QFont, QFontDatabase

    if _platform.system() == "Linux":
        # Linux: 显式指定 CJK 字体族
        available = QFontDatabase.families()
        cjk_candidates = [
            "Noto Sans CJK SC",      # fonts-noto-cjk (推荐)
            "WenQuanYi Micro Hei",   # fonts-wqy-microhei
            "WenQuanYi Zen Hei",     # fonts-wqy-zenhei
            "Noto Sans SC",          # fonts-noto-cjk 变体
            "Source Han Sans SC",    # 思源黑体
            "AR PL UMing CN",        # 旧版中文宋体
            "AR PL UKai CN",         # 旧版中文楷体
        ]
        font = None
        for family in cjk_candidates:
            if family in available:
                font = QFont(family, 10)
                break
        if font is None:
            # 未安装任何中文字体 — 使用系统字体并打印警告
            import logging
            logging.getLogger(__name__).warning(
                "Linux 系统未检测到中文字体（Noto Sans CJK SC / 文泉驿等），"
                "界面中文将显示为乱码。请安装："
                "sudo apt install fonts-noto-cjk"
            )
            font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    else:
        # macOS / Windows：系统字体自带 CJK 回退
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)

    app.setFont(font)
    app.setStyleSheet(LIGHT_QSS)
