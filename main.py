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

from app.backend_interface import BackendInterface
from app.main_window import MainWindow
from app.startup_dialog import StartupDialog
from app.theme import apply_theme


def main():
    """应用入口函数，供直接运行 main.py 或通过 uv run / 脚本方式调用。"""
    app = QApplication(sys.argv)
    apply_theme(app)

    # 由后端加载相机参数；前端只接收轻量摘要和用户选择。
    parameter_service = BackendInterface()
    parameter_summary = parameter_service.camera_parameter_summary()
    dlg = StartupDialog(parameter_summary=parameter_summary)
    if dlg.exec() != StartupDialog.DialogCode.Accepted:
        sys.exit(0)
    camera_config = dlg.get_config()

    window = MainWindow(camera_config=camera_config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
