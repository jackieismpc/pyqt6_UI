# -*- coding: utf-8 -*-
"""图像预处理：针对透明晶体拍摄的三大难点。

难点与对策：
1. 整体偏暗（晶体透明、背景黑）——> LAB-CLAHE 局部对比度增强 + gamma 提亮，
   可选多尺度 Retinex（MSR）进一步压暗提亮。
2. 强高光 / 镜面反射（灯光在晶体表面形成刺眼亮斑）——> 检测高光掩膜，
   在后续边缘提取时抑制高光边缘，避免把「高光轮廓」误当成晶体棱线。
3. 暗部遮挡（部分晶体面与黑背景亮度接近、被暗区吞没）——> 增强放大暗部细节，
   让深度边缘检测器（PiDiNet）能捕捉到弱棱线。

主入口 `run_preprocess` 返回增强图、高光掩膜与元信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from .config import PreprocessConfig


@dataclass
class PreprocessResult:
    """预处理结果。"""

    enhanced_bgr: np.ndarray      # 增强后的 BGR 图，供边缘/分割使用
    specular_mask: np.ndarray     # 高光/镜面掩膜（uint8 0/255）
    applied_lowlight: bool        # 是否实际执行了低光增强
    mean_luma: float              # 原图平均亮度（用于 auto 判定与记录）
    meta: Dict[str, float]


def _gamma_lut(gamma: float) -> np.ndarray:
    """构造 gamma 查找表。gamma<1 提亮暗部。"""
    gamma = max(float(gamma), 1e-3)
    xs = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    return np.clip((xs ** gamma) * 255.0, 0, 255).astype(np.uint8)


def _multi_scale_retinex(gray: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    """多尺度 Retinex（在对数域压暗提亮），返回归一化到 0~255 的单通道。"""
    img = gray.astype(np.float32) + 1.0
    log_img = np.log(img)
    retinex = np.zeros_like(log_img)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))
        retinex += log_img - np.log(blur + 1.0)
    retinex /= float(len(sigmas))
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def enhance_lowlight(image_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """低光增强：LAB-CLAHE + gamma（+ 可选 MSR + 轻度去噪）。"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(cfg.clahe_clip),
        tileGridSize=(int(cfg.clahe_grid), int(cfg.clahe_grid)),
    )
    l_channel = clahe.apply(l_channel)

    if cfg.use_msr:
        # 用 MSR 结果与 CLAHE 结果做加权，兼顾局部对比度与暗部提亮
        msr = _multi_scale_retinex(l_channel)
        l_channel = cv2.addWeighted(l_channel, 0.6, msr, 0.4, 0)

    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    enhanced = cv2.LUT(enhanced, _gamma_lut(cfg.gamma))

    if cfg.denoise:
        # 轻度双边滤波：抑制水珠/散景噪点，同时尽量保住棱线
        enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=40, sigmaSpace=5)
    return enhanced


def compute_specular_mask(image_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """检测镜面高光区域，返回膨胀后的掩膜（uint8 0/255）。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    percentile_value = float(np.percentile(gray, float(cfg.specular_percentile)))
    threshold = max(percentile_value, float(cfg.specular_min_value))
    mask = (gray >= threshold).astype(np.uint8) * 255
    if cfg.specular_dilate > 0:
        ksize = int(cfg.specular_dilate) | 1  # 保证奇数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def run_preprocess(image_bgr: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    """预处理主入口。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))

    if cfg.lowlight_mode == "on":
        do_enhance = True
    elif cfg.lowlight_mode == "off":
        do_enhance = False
    else:  # auto
        do_enhance = mean_luma < float(cfg.lowlight_luma_threshold)

    enhanced = enhance_lowlight(image_bgr, cfg) if do_enhance else image_bgr.copy()
    specular = compute_specular_mask(image_bgr, cfg)

    return PreprocessResult(
        enhanced_bgr=enhanced,
        specular_mask=specular,
        applied_lowlight=do_enhance,
        mean_luma=mean_luma,
        meta={
            "mean_luma": mean_luma,
            "specular_ratio": float(np.count_nonzero(specular)) / float(specular.size),
        },
    )
