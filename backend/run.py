#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""透明晶体体积估计统一入口。

用法：
    python run.py stage1 <输入图/目录/视频> [选项]      # 只跑第一阶段
    python run.py stage2 --stage1-geometry <json> --camera-parameters <json> [选项]
    python run.py full   <输入> --camera-parameters <json> [选项]   # 一二阶段串跑

详细参数见 README.md 或 `python run.py stage1 -h`。
"""

import os
import sys

# 在导入 torch 之前打开 MPS 算子回退，避免 Apple 芯片上个别未实现算子直接报错。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# 静音 FFmpeg 解码器的良性告警（如 AVI 末帧被截断打成红色 stderr）；
# 真正无法读帧时管线会明确报错，不会被这条掩盖。
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

# 后端 CLI 也必须和 GUI/Web 一样完全离线。所有模型都必须通过项目内的
# backend/weights 路径加载；缺失时直接给出错误/回退，不允许运行时从 Hub 下载。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["ULTRALYTICS_OFFLINE"] = "1"

from crystalvol.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
