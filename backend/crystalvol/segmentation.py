# -*- coding: utf-8 -*-
"""YOLO-World + SAM2 分割前端（只保留最大晶体）。

流程：
1. YOLO-World 用文本提示词（crystal/quartz...）检测主体框；
   透明物体常检不到，检不到时回退「中心先验框」。
2. 用该框提示 SAM2，得到像素级掩膜。
3. 后处理只保留「面积最大的连通域」——对应「每次只拍一个晶体、只取最大」。

对透明晶体，SAM2 本身并不稳，因此本模块是可选增强（Stage1Config.segmentation.enable），
最终剪影会与边缘证据一起在 silhouette.py 里融合。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import SegmentationConfig
from .device import resolve_device

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class SegmentationResult:
    """分割结果。"""

    mask: np.ndarray                 # uint8 0/255，最大连通域
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float                # YOLO 置信度 * SAM2 掩膜分数
    prompt_label: str
    mode: str                        # world+sam2 / sam2-center-fallback
    warnings: List[str] = field(default_factory=list)


class CrystalSegmenter:
    """封装 YOLO-World 选框 + SAM2 精分割。"""

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device).torch_device
        # SAM2 在 Apple MPS 上有多个算子未实现（如 upsample_bicubic2d），
        # 即便开启 PYTORCH_ENABLE_MPS_FALLBACK 也不稳；ROI 裁剪后目标很小，
        # 直接用 CPU 跑 SAM2-tiny 足够快，故 mps 一律降级到 cpu。
        if self.device.startswith("mps"):
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            self.device = "cpu"
        self._world_model = None
        self._sam2_predictor = None

    # ---- 模型懒加载 ----
    def _load_world(self):
        if self._world_model is not None:
            return self._world_model
        from ultralytics import YOLO, YOLOWorld
        model_path = str(Path(self.config.world_model_path).expanduser().resolve())
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO-World 权重不存在: {model_path}")
        try:
            model = YOLOWorld(model_path)
        except Exception:
            model = YOLO(model_path)
        if hasattr(model, "set_classes") and self.config.world_classes:
            model.set_classes(list(self.config.world_classes))
        self._world_model = model
        return model

    def _load_sam2(self):
        if self._sam2_predictor is not None:
            return self._sam2_predictor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        checkpoint = Path(self.config.sam2_checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM2 权重不存在: {checkpoint}")
        model = build_sam2(
            config_file=self._normalize_sam2_config(self.config.sam2_config_path),
            ckpt_path=str(checkpoint),
            device=self.device,
        )
        self._sam2_predictor = SAM2ImagePredictor(model)
        return self._sam2_predictor

    @staticmethod
    def _normalize_sam2_config(raw_value: str) -> str:
        config_path = Path(raw_value).expanduser()
        if not config_path.exists():
            return raw_value
        parts = list(config_path.resolve().parts)
        if "configs" in parts:
            return "/".join(parts[parts.index("configs"):])
        return str(config_path.resolve())

    # ---- 选框 ----
    def _score_box(self, bbox, confidence, image_shape) -> float:
        x1, y1, x2, y2 = map(float, bbox)
        area = max(x2 - x1, 1.0) * max(y2 - y1, 1.0)
        image_h, image_w = image_shape
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        image_center = np.array([image_w * 0.5, image_h * 0.5])
        distance = float(np.linalg.norm(center - image_center))
        diagonal = float(np.hypot(image_w, image_h))
        center_term = max(0.0, 1.0 - distance / max(diagonal, 1.0))
        area_term = area / max(float(image_w * image_h), 1.0)
        # 中心 + 面积 + 置信度综合，倾向画面中央的大目标
        return float(0.5 * confidence + 0.3 * center_term + 0.2 * area_term)

    def _pick_box(self, image_bgr, warnings):
        # 默认不走 YOLO-World（需下载 CLIP 文本模型、且透明晶体常检不到）；直接用中心先验框。
        if not self.config.use_yolo:
            return self._center_box(image_bgr.shape[:2], warnings)
        try:
            model = self._load_world()
            results = model.predict(
                source=image_bgr, conf=float(self.config.world_conf),
                iou=float(self.config.world_iou), device=self.device, verbose=False,
            )
            boxes = getattr(results[0], "boxes", None) if results else None
            if boxes is None or len(boxes) == 0:
                return self._center_box(image_bgr.shape[:2], warnings)
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
            confs = boxes.conf.detach().cpu().numpy().astype(np.float64)
            names = getattr(results[0], "names", {}) or {}
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            best = int(np.argmax([self._score_box(xyxy[i], confs[i], image_bgr.shape[:2]) for i in range(len(xyxy))]))
            return xyxy[best], float(confs[best]), str(names.get(int(cls_ids[best]), cls_ids[best])), "world+sam2"
        except Exception as exc:
            warnings.append(f"YOLO-World 不可用（{exc}），回退中心先验框。")
            return self._center_box(image_bgr.shape[:2], warnings)

    def _center_box(self, image_shape, warnings):
        ratio = float(np.clip(self.config.center_fallback_ratio, 0.0, 1.0))
        if ratio <= 0.0:
            raise RuntimeError("YOLO-World 未检测到目标，且未启用中心框回退。")
        image_h, image_w = image_shape
        box_w, box_h = int(image_w * ratio), int(image_h * ratio)
        x1, y1 = (image_w - box_w) // 2, (image_h - box_h) // 2
        warnings.append("YOLO-World 未检测到主体，回退中心先验框继续用 SAM2 分割。")
        return np.array([x1, y1, x1 + box_w, y1 + box_h], dtype=np.float64), 0.0, "center-fallback", "sam2-center-fallback"

    def _expand_box(self, bbox, image_shape):
        x1, y1, x2, y2 = map(float, bbox)
        image_h, image_w = image_shape
        ex = (x2 - x1) * float(self.config.box_expand_ratio)
        ey = (y2 - y1) * float(self.config.box_expand_ratio)
        return np.array([
            max(int(x1 - ex), 0), max(int(y1 - ey), 0),
            min(int(x2 + ex), image_w - 1), min(int(y2 + ey), image_h - 1),
        ], dtype=np.int32)

    @staticmethod
    def _box_center_points(bbox) -> tuple:
        """在框内取中心 + 四个内偏点作为 SAM2 正点提示，帮助锁定透明晶体主体。"""
        x1, y1, x2, y2 = map(float, bbox)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        dx, dy = (x2 - x1) * 0.22, (y2 - y1) * 0.22
        pts = np.array([
            [cx, cy], [cx - dx, cy], [cx + dx, cy], [cx, cy - dy], [cx, cy + dy],
        ], dtype=np.float32)
        labels = np.ones(len(pts), dtype=np.int32)
        return pts, labels

    def _run_sam2(self, image_bgr, bbox, positive_point=None):
        predictor = self._load_sam2()
        predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        points, labels = self._box_center_points(bbox)
        if positive_point is not None:
            # 把定位阶段给的晶体质心作为额外正点，帮助锁定主体（而非框内背景）
            points = np.vstack([np.asarray(positive_point, np.float32).reshape(1, 2), points]).astype(np.float32)
            labels = np.concatenate([np.ones(1, np.int32), labels]).astype(np.int32)
        ctx = torch.inference_mode() if torch is not None else _NullCtx()
        with ctx:
            masks, scores, _ = predictor.predict(
                box=np.asarray(bbox, np.float32).reshape(4),
                point_coords=points, point_labels=labels,
                multimask_output=True,
            )
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 未返回掩膜。")
        best = int(np.argmax(np.asarray(scores, np.float32)))
        return (np.asarray(masks[best], np.uint8) * 255), float(np.asarray(scores, np.float32).reshape(-1)[best])

    def _largest_component(self, mask):
        """只保留最大连通域（对应「只取最大晶体」）。"""
        binary = (mask > 0).astype(np.uint8) * 255
        ksize = int(self.config.morphology_kernel) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if num <= 1:
            return binary
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + int(np.argmax(areas))
        return np.where(labels == best_label, 255, 0).astype(np.uint8)

    def segment(
        self,
        image_bgr: np.ndarray,
        prompt_box: Optional[Sequence[float]] = None,
        positive_point: Optional[Sequence[float]] = None,
    ) -> SegmentationResult:
        """对一张图分割出最大晶体掩膜。

        prompt_box:      定位阶段给的晶体框（图像坐标）。给了就直接用它提示 SAM2，
                         不再走「中心先验框」——这对小晶体至关重要。
        positive_point:  晶体质心正点，作为额外提示。
        """
        warnings: List[str] = []
        if prompt_box is not None:
            box = np.asarray(prompt_box, dtype=np.float64).reshape(4)
            conf, label, mode = 0.5, "roi-prompt", "roi+sam2"
        else:
            box, conf, label, mode = self._pick_box(image_bgr, warnings)
        expanded = self._expand_box(box, image_bgr.shape[:2])
        mask, score = self._run_sam2(image_bgr, expanded, positive_point=positive_point)
        mask = self._largest_component(mask)
        return SegmentationResult(
            mask=mask,
            bbox_xyxy=tuple(int(v) for v in expanded.tolist()),
            confidence=float(max(conf, 0.0) * max(score, 0.0)),
            prompt_label=label,
            mode=mode,
            warnings=warnings,
        )

    def close(self) -> None:
        """释放模型引用和设备缓存，供批处理/实时会话结束时调用。"""
        self._sam2_predictor = None
        self._world_model = None
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                # 清理失败不应覆盖主流程结果；Python/torch 后续仍会回收对象。
                pass


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
