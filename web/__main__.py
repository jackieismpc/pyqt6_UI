"""Web 服务入口：uv run python -m web --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 完全离线运行（与 main.py 一致）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="透明晶体体积估计 Web 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载")
    args = parser.parse_args(argv)

    import uvicorn

    from web.server import app

    print(f"\n晶体体积估计 Web 服务已启动： http://{args.host}:{args.port}")
    lan = _lan_ip()
    if args.host in ("0.0.0.0", "::") and lan:
        print(f"本机浏览器打开：     http://127.0.0.1:{args.port}")
        print(f"局域网其他电脑打开： http://{lan}:{args.port}")
    print("按 Ctrl+C 停止。\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


def _lan_ip() -> str | None:
    """探测本机局域网 IP（用于提示其他电脑的访问地址）。"""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
