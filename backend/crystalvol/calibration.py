# -*- coding: utf-8 -*-
"""相机参数公共入口。

实际实现位于 :mod:`crystalvol.camera_parameters`；保留这个模块名是为了让
后端内部的标定相关代码使用同一个参数服务，而不是各自解析文件。
"""

from .camera_parameters import (
    CameraCalibration,
    CameraParameters,
    DEFAULT_PARAMETERS_PATH,
    ExtrinsicParameters,
    apply_scale_anchor_correction,
    camera_parameters_summary,
    length_unit_to_meter,
    load_camera_calibration,
    load_camera_parameters,
    pinhole_pixel_to_cm,
    resolve_camera_matrix_for_image,
    resolve_camera_parameters_path,
    save_camera_calibration,
    save_camera_parameters,
)
from .physical_constraints import (
    MAX_CRYSTAL_LENGTH_CM,
    MAX_REALTIME_SHRINK_RATIO,
    MIN_CRYSTAL_LENGTH_CM,
    apply_growth_constraints,
)

__all__ = [
    "CameraCalibration",
    "CameraParameters",
    "DEFAULT_PARAMETERS_PATH",
    "ExtrinsicParameters",
    "apply_scale_anchor_correction",
    "camera_parameters_summary",
    "length_unit_to_meter",
    "load_camera_calibration",
    "load_camera_parameters",
    "pinhole_pixel_to_cm",
    "resolve_camera_matrix_for_image",
    "resolve_camera_parameters_path",
    "save_camera_calibration",
    "save_camera_parameters",
    "MIN_CRYSTAL_LENGTH_CM",
    "MAX_CRYSTAL_LENGTH_CM",
    "MAX_REALTIME_SHRINK_RATIO",
    "apply_growth_constraints",
]
