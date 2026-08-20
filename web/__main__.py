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
    parser.add_argument("--no-reload", action="store_true", help="关闭自动重载")
    args = parser.parse_args(argv)

    import uvicorn

    from web.server import app

    print(f"\n晶体体积估计 Web 服务已启动： http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止。\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, reload=not args.no_reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
