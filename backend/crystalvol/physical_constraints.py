# -*- coding: utf-8 -*-
"""晶体尺寸与生长过程的物理约束。

这些约束是后端质量控制的一部分，不依赖前端是否打开或使用某个控件：

* 晶体有效长边范围为 1--70 cm；
* 实时序列中晶体应持续生长，单帧突然缩小超过 25% 会被标记为异常；
* 异常结果仍会返回，便于诊断，但不会被当作下一帧的生长基准。
"""

from __future__ import annotations

from typing import Optional


MIN_CRYSTAL_LENGTH_CM = 1.0
MAX_CRYSTAL_LENGTH_CM = 70.0
MAX_REALTIME_SHRINK_RATIO = 0.25


def apply_growth_constraints(
    metric: dict[str, object],
    previous_length_cm: Optional[float] = None,
    *,
    min_length_cm: float = MIN_CRYSTAL_LENGTH_CM,
    max_length_cm: float = MAX_CRYSTAL_LENGTH_CM,
    max_shrink_ratio: float = MAX_REALTIME_SHRINK_RATIO,
) -> dict[str, object]:
    """给公制结果附加可审计的物理约束判定。

    函数不会丢弃或篡改测量值，只在 ``physical_constraints`` 中写入状态、
    原因和建议。这样前端可以继续展示问题帧，工程流程也不会因为一个坏帧
    崩溃；调用方应只把 ``accepted_for_growth`` 为真的结果写入历史基准。
    """
    dimensions = metric.get("dimensions_cm")
    if not isinstance(dimensions, dict):
        return {**metric, "physical_constraints": {
            "valid": False,
            "accepted_for_growth": False,
            "warnings": ["缺少 dimensions_cm，无法执行晶体长度物理约束。"],
        }}

    try:
        length_cm = float(dimensions.get("length", 0.0))
    except (TypeError, ValueError):
        length_cm = 0.0

    warnings: list[str] = []
    valid = True
    if length_cm < float(min_length_cm):
        valid = False
        warnings.append(f"晶体长边 {length_cm:.3f} cm 小于物理下限 {float(min_length_cm):.1f} cm。")
    if length_cm > float(max_length_cm):
        valid = False
        warnings.append(f"晶体长边 {length_cm:.3f} cm 超过物理上限 {float(max_length_cm):.1f} cm。")

    previous = None if previous_length_cm is None else float(previous_length_cm)
    growth_ratio = None
    if previous is not None and previous > 0:
        growth_ratio = length_cm / previous
        if growth_ratio < 1.0 - float(max_shrink_ratio):
            valid = False
            warnings.append(
                f"相邻帧长边从 {previous:.3f} cm 降至 {length_cm:.3f} cm，"
                f"超过允许收缩 {float(max_shrink_ratio) * 100:.0f}%，疑似分割/标定异常。"
            )

    constraints = {
        "valid": valid,
        "accepted_for_growth": valid,
        "length_range_cm": [float(min_length_cm), float(max_length_cm)],
        "previous_length_cm": previous,
        "growth_ratio": growth_ratio,
        "max_realtime_shrink_ratio": float(max_shrink_ratio),
        "warnings": warnings,
    }
    return {**metric, "physical_constraints": constraints}

