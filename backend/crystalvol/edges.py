# -*- coding: utf-8 -*-
"""边缘提取：传统 Canny/LSD + 深度 PiDiNet/HED，融合与择优。

设计要点（对应透明晶体难点）：
- 深度边缘（PiDiNet）：对反光、暗部弱棱线远比 Canny 鲁棒，是难图的主力。
  权重从本地 backend/weights/ 加载（零网络），缺失时优雅回退到 Canny。
- 融合：深度边缘（保主结构）按位或 Canny（保锐利细边）。
- 高光抑制：把镜面高光区域内的边缘清零，避免高光边被误当成棱线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import List, Optional

import cv2
import numpy as np

from .config import EdgeConfig
from .logging_utils import warn

# 深度检测器缓存：避免重复加载权重
_DEEP_CACHE: "OrderedDict[str, DeepEdgeDetector]" = OrderedDict()
_MAX_DEEP_CACHE = 4


def _resolve_deep_device(prefer: str) -> str:
    """深度边缘设备：显式指定优先；否则 CUDA 可用则 CUDA，否则 CPU。

    深度边缘模型很小，CPU 足够快；Mac 上默认走 CPU 以规避个别 MPS 算子问题。
    """
    prefer = (prefer or "auto").strip().lower()
    if prefer not in ("", "auto"):
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class DeepEdgeDetector:
    """controlnet_aux 深度边缘检测器封装（PiDiNet / HED）。

    repo 参数：可以是 HuggingFace 仓库名（如 "lllyasviel/Annotators"），
    也可以是本地目录路径（如 project/weights/）。传入本地目录时完全不连网。
    """

    def __init__(self, backend: str, repo: str, device: str) -> None:
        self.backend = backend
        self.repo = repo
        self.device = device
        self._model = None
        self._load_error: Optional[str] = None

    def _load(self):
        if self._model is not None or self._load_error is not None:
            return self._model
        try:
            import warnings
            warnings.filterwarnings("ignore")

            # 本地路径：直接读文件，始终传 local_files_only=True 确保零网络
            # controlnet_aux 的 from_pretrained 对 isdir() 路径走本地读，
            # 但显式传参可以 100% 杜绝任何 HF Hub 校验回退。
            repo_or_path = self.repo
            load_kwargs: dict = {"local_files_only": True}

            if self.backend == "pidinet":
                from controlnet_aux import PidiNetDetector
                model = PidiNetDetector.from_pretrained(repo_or_path, **load_kwargs)
            elif self.backend == "hed":
                from controlnet_aux import HEDdetector
                model = HEDdetector.from_pretrained(repo_or_path, **load_kwargs)
            else:
                raise ValueError(f"未知深度边缘 backend: {self.backend}")
            try:
                model = model.to(self.device)
            except Exception:
                pass  # 某些版本不支持 .to()，保持默认设备
            self._model = model
        except Exception as exc:  # 权重缺失 / 未安装 controlnet_aux
            self._load_error = f"{type(exc).__name__}: {exc}"
        return self._model

    def is_available(self) -> bool:
        return self._load() is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def edge_prob(self, image_bgr: np.ndarray, input_max_side: int) -> Optional[np.ndarray]:
        """返回与输入同分辨率的边缘概率图（float32, 0~1）。失败返回 None。"""
        model = self._load()
        if model is None:
            return None
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detect_resolution = min(int(input_max_side), max(height, width))
        try:
            out = model(
                rgb,
                detect_resolution=detect_resolution,
                image_resolution=max(height, width),
                safe=True,
            )
        except Exception as exc:
            warn(f"深度边缘推理失败，将回退传统边缘: {exc}")
            return None
        arr = np.asarray(out)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if arr.shape[:2] != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        return arr.astype(np.float32) / 255.0

    def close(self) -> None:
        """释放深度边缘模型引用。"""
        self._model = None


def get_deep_detector(cfg: EdgeConfig, backend: str) -> DeepEdgeDetector:
    """按 (backend, repo, device) 取/建缓存的深度检测器。"""
    device = _resolve_deep_device(cfg.device)
    key = f"{backend}|{cfg.deep_repo}|{device}"
    detector = _DEEP_CACHE.get(key)
    if detector is None:
        if len(_DEEP_CACHE) >= _MAX_DEEP_CACHE:
            _, old_detector = _DEEP_CACHE.popitem(last=False)
            old_detector.close()
        detector = DeepEdgeDetector(backend, cfg.deep_repo, device)
        _DEEP_CACHE[key] = detector
    else:
        _DEEP_CACHE.move_to_end(key)
    return detector


def clear_deep_detector_cache() -> None:
    """释放所有深度边缘模型，供应用退出或设备切换时调用。"""
    while _DEEP_CACHE:
        _, detector = _DEEP_CACHE.popitem(last=False)
        detector.close()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def canny_edge_map(image_bgr: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    """传统 Canny 边缘（uint8 0/255）。输入应为已增强图。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, int(cfg.canny_low), int(cfg.canny_high))


def lsd_segments(image_bgr: np.ndarray) -> np.ndarray:
    """LSD 直线段检测，返回 (N,4) 端点数组。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except Exception:
        return np.empty((0, 4), dtype=np.float32)
    lines = detector.detect(gray)[0]
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def suppress_specular(edge_map: np.ndarray, specular_mask: Optional[np.ndarray]) -> np.ndarray:
    """把高光区域内的边缘清零，抑制高光引起的伪棱线。"""
    if specular_mask is None or specular_mask.size == 0:
        return edge_map
    result = edge_map.copy()
    result[specular_mask > 0] = 0
    return result


@dataclass
class EdgeResult:
    """边缘提取结果。"""

    edge_map: np.ndarray            # 融合并抑制高光后的二值边缘（uint8 0/255）
    backend_used: str               # 实际使用的 backend
    deep_available: bool            # 深度边缘是否成功启用
    deep_prob: Optional[np.ndarray] = None   # 深度概率图（如有）
    warnings: List[str] = field(default_factory=list)


def compute_edge_map(
    image_bgr: np.ndarray,
    specular_mask: Optional[np.ndarray],
    cfg: EdgeConfig,
    backend: Optional[str] = None,
) -> EdgeResult:
    """计算一张融合边缘图。

    backend 为 None 时用 cfg.backend；auto 视为 pidinet(+canny)，
    多后端择优交给上层（wireframe 阶段）。
    """
    backend = (backend or cfg.backend).strip().lower()
    warnings_list: List[str] = []

    if backend == "canny":
        edge = canny_edge_map(image_bgr, cfg)
        return EdgeResult(suppress_specular(edge, specular_mask), "canny", False, None, warnings_list)

    if backend == "lsd":
        segments = lsd_segments(image_bgr)
        edge = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in segments.astype(int):
            cv2.line(edge, (x1, y1), (x2, y2), 255, 2)
        return EdgeResult(suppress_specular(edge, specular_mask), "lsd", False, None, warnings_list)

    # 深度边缘（pidinet / hed / auto）
    deep_backend = "pidinet" if backend == "auto" else backend
    detector = get_deep_detector(cfg, deep_backend)
    prob = detector.edge_prob(image_bgr, cfg.deep_input_max_side)

    if prob is None:
        # 深度不可用 -> 回退 Canny
        if detector.load_error:
            warnings_list.append(f"深度边缘不可用（{detector.load_error}），回退 Canny。")
        edge = canny_edge_map(image_bgr, cfg)
        return EdgeResult(suppress_specular(edge, specular_mask), "canny(fallback)", False, None, warnings_list)

    deep_binary = (prob >= float(cfg.deep_threshold)).astype(np.uint8) * 255
    if cfg.fuse_canny:
        canny = canny_edge_map(image_bgr, cfg)
        fused = cv2.bitwise_or(deep_binary, canny)
        used = f"{deep_backend}+canny"
    else:
        fused = deep_binary
        used = deep_backend
    fused = suppress_specular(fused, specular_mask)
    return EdgeResult(fused, used, True, prob, warnings_list)
