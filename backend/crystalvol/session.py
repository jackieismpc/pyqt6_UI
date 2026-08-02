# -*- coding: utf-8 -*-
"""增量多视角联合估计会话（实时拍摄用）。

用于「实时」模式：用户对同一个晶体从不同视角连续拍多张照片，每拍一张就在原来
已建好的几何模型上继续优化。语义是「累积 + 重新联合拟合」：

- 分割器（SAM2）与配置只初始化一次，常驻整个会话，避免每张照片重复加载模型；
- 每调用一次 add_frame()，就把这张照片处理成一帧、并入已累积的帧集合，
  然后对「当前累积的全部帧」重新做跨帧稳健联合拟合（复用 stage1 的
  _select_consensus_pool / _consolidate_geometry），得到在旧模型基础上被这张
  新照片进一步约束、优化后的单一晶体几何；
- 产物目录布局与命名与批处理 stage1 完全一致，因此前端展示逻辑无需区分两条路径。

与「单目单视角深度为启发式」的现实一致：视角越多、约束越强，联合拟合越稳。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Stage1Config
from .io import InputFrame, OutputLayout, _resize_max_side
from .logging_utils import log, section
from .segmentation import CrystalSegmenter
from .stage1 import (
    FrameOutput,
    _process_frame,
    build_segmenter,
    finalize_stage1,
    write_frame_products,
)


class Stage1Session:
    """一个晶体的增量多视角会话。"""

    def __init__(self, output_dir: str, cfg: Optional[Stage1Config] = None,
                 clean: bool = True) -> None:
        self.cfg: Stage1Config = cfg or Stage1Config(
            input_path="realtime://camera", output_dir=output_dir, device="auto",
        )
        # 以传入的 output_dir 为准
        self.cfg.output_dir = output_dir
        self.layout = OutputLayout(Path(output_dir).expanduser().resolve()).prepare(clean=clean)
        section("实时增量会话：初始化分割前端（仅一次）")
        self.segmenter: Optional[CrystalSegmenter] = build_segmenter(self.cfg)
        self.frame_outputs: List[FrameOutput] = []
        self._total_count = 0

    @property
    def count(self) -> int:
        """已累积并成功处理的照片数。"""
        return self._total_count

    def add_frame(self, image_bgr: np.ndarray, name: Optional[str] = None) -> Dict[str, object]:
        """并入一张新照片，重新联合拟合，返回与 stage1 一致的 summary。

        参数：
            image_bgr: 摄像头/文件读入的 BGR 图（np.ndarray）。
            name:      帧名，缺省自动编号 frame_01/frame_02/...
        返回：
            summary dict —— 结构与 stage1_result.json 完全一致，可直接喂给前端。
        """
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            raise ValueError("传入的图像为空，无法加入会话。")

        index = self._total_count
        frame_name = name or f"frame_{index + 1:02d}"
        image = _resize_max_side(image_bgr, self.cfg.max_input_side)
        frame = InputFrame(name=frame_name, image_bgr=image,
                           source_path=f"realtime://{frame_name}", index=index)

        out = _process_frame(frame, self.cfg, self.segmenter)
        write_frame_products(self.layout, out)
        from .stage1 import _release_frame_buffers
        _release_frame_buffers(out)
        self.frame_outputs.append(out)
        self._total_count += 1
        max_frames = max(int(self.cfg.max_session_frames), 1)
        if len(self.frame_outputs) > max_frames:
            # 历史产物文件保留在 data/results 供追溯，但只保留最近窗口参与
            # 联合拟合，防止实时会话的 CPU/元数据成本无限增长。
            del self.frame_outputs[:-max_frames]

        summary = finalize_stage1(self.cfg, self.layout, self.frame_outputs)
        log(f"实时增量：已并入第 {self.count} 张，联合体积 "
            f"volume_px3={summary['geometry_px']['volume_px3']:.3e}")
        return summary

    def reset(self, clean: bool = True) -> None:
        """清空累积帧（开始对一个新晶体建模）；分割器常驻不重建。"""
        self.frame_outputs.clear()
        self._total_count = 0
        self.layout.prepare(clean=clean)

    def close(self) -> None:
        """结束会话并释放分割模型和历史元数据。"""
        self.frame_outputs.clear()
        if self.segmenter is not None:
            self.segmenter.close()
            self.segmenter = None
