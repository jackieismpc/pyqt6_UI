"""
应用入口（平台分派）。

- macOS / Windows：默认启动 PyQt6 GUI
- Linux 有 DISPLAY：默认启动 PyQt6 GUI
- Linux 无 DISPLAY（服务器）：自动启动 Web 服务并打印访问地址
- 任意平台加 --web 参数：强制启动 Web 服务

Web 服务也可以直接 `uv run python -m web --host 0.0.0.0 --port 8000` 启动。
"""

import os as _os
import sys

# ── 完全离线运行：禁止所有 HuggingFace / transformers 联网 ──
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"
_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
_os.environ["WANDB_DISABLED"] = "true"
_os.environ["WANDB_MODE"] = "disabled"
_os.environ["ULTRALYTICS_OFFLINE"] = "1"


def _want_web(argv: list[str]) -> bool:
    """判断应走 Web 还是 Qt GUI。"""
    if "--web" in argv:
        return True
    # Linux 服务器通常没有 DISPLAY，无法初始化 Qt xcb 平台插件
    if sys.platform.startswith("linux") and not _os.environ.get("DISPLAY"):
        return True
    return False


def _run_web(argv: list[str]) -> int:
    """启动 FastAPI Web 服务（无 Qt 依赖）。"""
    from web.__main__ import main as web_main

    # 去掉应用自己的 --web 标记，其余参数（--host/--port 等）透传给 web 服务
    rest = [arg for arg in argv if arg != "--web"]
    return web_main(rest)


def _run_gui(argv: list[str]) -> int:
    """启动 PyQt6 桌面 GUI。"""
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from app.backend_interface import BackendInterface
    from app.main_window import MainWindow
    from app.startup_dialog import StartupDialog
    from app.theme import apply_theme

    app = QApplication(argv)
    apply_theme(app)

    # 由后端加载相机参数；前端只接收轻量摘要和用户选择。
    parameter_service = BackendInterface()
    try:
        parameter_summary = parameter_service.camera_parameter_summary()
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(
            None,
            "相机参数加载失败",
            "后端默认参数也无法读取，程序无法安全启动。\n\n"
            f"{type(exc).__name__}: {exc}",
        )
        return 2
    dlg = StartupDialog(parameter_summary=parameter_summary)
    if dlg.exec() != StartupDialog.DialogCode.Accepted:
        return 0
    camera_config = dlg.get_config()

    window = MainWindow(camera_config=camera_config)
    window.show()

    return app.exec()


def main(argv: list[str] | None = None) -> int:
    """应用入口函数，供直接运行 main.py 或通过 uv run / 脚本方式调用。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if _want_web(argv):
        return _run_web(argv)
    return _run_gui(argv)


if __name__ == "__main__":
    raise SystemExit(main())
