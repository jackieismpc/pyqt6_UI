# -*- coding: utf-8 -*-
"""预定义几何体：长方体 + 四棱锥（屋顶形）。

四个尺寸参数：
- length  (L)  ：前向长度（宽度方向可见的水平跨度）
- width   (W)  ：侧向深度
- body_height     (Hb)：长方体高度
- pyramid_height  (Hp)：四棱锥（屋顶）高度

体积公式（长方体 + 四棱锥）：
    V = L * W * (Hb + Hp / 3)

顶点编号（物体坐标系，底面中心为原点，z 向上）：
    0-3：底面矩形     z = 0
    4-7：肩部矩形     z = Hb
    8  ：顶点 apex    z = Hb + Hp
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def build_vertices(length: float, width: float, body_height: float, pyramid_height: float) -> np.ndarray:
    """构造 9 个标准顶点，返回 (9,3) 数组。"""
    hl = float(length) * 0.5
    hw = float(width) * 0.5
    top = float(body_height)
    apex = float(body_height) + float(pyramid_height)
    return np.array(
        [
            [-hl, -hw, 0.0],
            [hl, -hw, 0.0],
            [hl, hw, 0.0],
            [-hl, hw, 0.0],
            [-hl, -hw, top],
            [hl, -hw, top],
            [hl, hw, top],
            [-hl, hw, top],
            [0.0, 0.0, apex],
        ],
        dtype=np.float64,
    )


def edge_index_pairs() -> List[Tuple[int, int]]:
    """16 条棱线的顶点索引对。"""
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),   # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),   # 肩部
        (0, 4), (1, 5), (2, 6), (3, 7),   # 竖直棱
        (4, 8), (5, 8), (6, 8), (7, 8),   # 屋顶棱
    ]


def compute_volume(length: float, width: float, body_height: float, pyramid_height: float) -> float:
    """标准晶体体积：长方体 + 四棱锥。"""
    return float(length * width * (body_height + pyramid_height / 3.0))


def volume_breakdown(length: float, width: float, body_height: float, pyramid_height: float) -> dict:
    """分别给出长方体和四棱锥体积，便于核对。"""
    cuboid = float(length * width * body_height)
    pyramid = float(length * width * pyramid_height / 3.0)
    return {"cuboid": cuboid, "pyramid": pyramid, "total": cuboid + pyramid}
