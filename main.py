"""
应用入口。

创建 QApplication，应用浅色主题，展示主窗口。
"""

import os as _os

# ── 完全离线运行：禁止所有 HuggingFace / transformers 联网 ──
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import sys

from PyQt6.QtWidgets import QApplication

from app.camera_config import parse_camera_params
from app.main_window import MainWindow
from app.startup_dialog import StartupDialog
from app.theme import apply_theme


def main():
    """应用入口函数，供直接运行 main.py 或通过 uv run / 脚本方式调用。"""
    app = QApplication(sys.argv)
    apply_theme(app)

    # 启动配置对话框
    camera_params = parse_camera_params()
    dlg = StartupDialog(camera_params=camera_params)
    if dlg.exec() != StartupDialog.DialogCode.Accepted:
        sys.exit(0)
    camera_config = dlg.get_config()

    window = MainWindow(camera_config=camera_config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
