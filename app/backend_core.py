"""纯 Python 后端核心层（不依赖 PyQt）。

Qt 前端（app/backend_interface.py）与 Web 前端（web/）共用这一层：结果目录加载、
公制换算、Stage1 编排、实时增量会话。Qt 适配层只负责线程/信号等 UI 侧差异。
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from .camera_config import CameraConfig
from .models import FrameResult, Stage1Result

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"


def _imwrite_safe(path: str, image) -> bool:
    """跨平台安全保存图片，支持非 ASCII 路径。"""
    try:
        extension = Path(path).suffix or ".png"
        ok, encoded = cv2.imencode(extension, image)
        if not ok:
            return False
        Path(path).write_bytes(encoded.tobytes())
        return True
    except (OSError, cv2.error):
        logger.exception("保存图片失败: %s", path)
        return False


def _ensure_backend_importable() -> None:
    """把 backend 源码目录加入导入路径，并保持推理离线。"""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    backend_string = str(_BACKEND_DIR)
    if backend_string not in sys.path:
        sys.path.insert(0, backend_string)


class BackendCore:
    """前端（Qt/Web 共用）使用的后端门面。"""

    RESULTS_DIR = _PROJECT_ROOT / "data" / "results"

    def __init__(self, camera_config: CameraConfig | None = None) -> None:
        self._session = None
        self._camera_config = camera_config or CameraConfig()
        self._camera_params = None
        self._realtime_previous_length_cm: float | None = None
        self._temporary_output_dirs: set[Path] = set()

    @staticmethod
    def _load_backend_camera_parameters(parameter_path: str | None = None):
        _ensure_backend_importable()
        from crystalvol.camera_parameters import load_camera_parameters  # noqa: WPS433

        try:
            return load_camera_parameters(parameter_path)
        except Exception:
            if parameter_path:
                raise
            logger.exception("项目相机参数加载失败，回退到后端内置默认参数")
            from crystalvol.camera_parameters import DEFAULT_PARAMETERS_PATH  # noqa: WPS433
            return load_camera_parameters(DEFAULT_PARAMETERS_PATH)

    def camera_parameter_summary(self) -> dict:
        """返回启动配置所需摘要，不加载 Torch、SAM2 或其他重模型。"""
        _ensure_backend_importable()
        from crystalvol.camera_parameters import camera_parameters_summary  # noqa: WPS433

        params = self._load_backend_camera_parameters(self._camera_config.parameter_path)
        self._camera_params = params
        return camera_parameters_summary(params)

    def _compute_metric(
        self,
        aggregate_geometry: dict,
        previous_length_cm: float | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> dict | None:
        if not aggregate_geometry or not aggregate_geometry.get("length_px"):
            return None
        if self._camera_params is None:
            self._camera_params = self._load_backend_camera_parameters(
                self._camera_config.parameter_path
            )

        _ensure_backend_importable()
        from crystalvol.calibration import (  # noqa: WPS433
            apply_scale_anchor_correction,
            apply_growth_constraints,
            pinhole_pixel_to_cm,
        )

        try:
            metric = pinhole_pixel_to_cm(
                aggregate_geometry,
                self._camera_params,
                self._camera_config.extrinsic_index,
                image_size=image_size,
            )
        except (IndexError, RuntimeError, ValueError) as exc:
            logger.warning("无法完成公制换算: %s", exc)
            return None
        if not metric:
            return None
        if self._camera_config.scale_anchor_value is not None:
            metric = apply_scale_anchor_correction(
                metric,
                self._camera_config.scale_anchor_edge,
                self._camera_config.scale_anchor_value,
            )
        return apply_growth_constraints(metric, previous_length_cm)

    def _make_output_dir(self, save: bool, prefix: str) -> str:
        if save:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = self.RESULTS_DIR / stamp
            if output.exists():
                output = self.RESULTS_DIR / f"{stamp}-{datetime.now().strftime('%f')[:3]}"
            output.mkdir(parents=True, exist_ok=True)
            return str(output)
        path = Path(tempfile.mkdtemp(prefix=prefix))
        self._temporary_output_dirs.add(path)
        return str(path)

    def _cleanup_temporary_outputs(self) -> None:
        """清理本适配层创建的临时产物，避免长时间运行占满系统磁盘。"""
        paths = list(self._temporary_output_dirs)
        self._temporary_output_dirs.clear()
        for path in paths:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning("临时结果目录清理失败: %s", path, exc_info=True)

    def _build_image_paths(self, base_dir: Path, name: str) -> dict:
        candidates = {
            "raw": base_dir / "inputs" / f"{name}.png",
            "enhanced": base_dir / "enhanced" / f"{name}_enhanced.png",
            "edges": base_dir / "edges" / f"{name}_edge.png",
            "mask": base_dir / "masks" / f"{name}_mask.png",
            "overlay": base_dir / "overlays" / f"{name}_overlay.png",
            "contour": base_dir / "contours" / f"{name}_wireframe.png",
        }
        return {
            key: (str(path) if path.exists() else None)
            for key, path in candidates.items()
        }

    def _load_result_dir(self, base_dir: Path, data: Optional[dict] = None) -> Stage1Result:
        base_dir = Path(base_dir)
        if data is None:
            with (base_dir / "stage1_result.json").open("r", encoding="utf-8") as handle:
                data = json.load(handle)

        frames = []
        for frame_data in data.get("frames", []):
            name = frame_data["name"]
            geometry_px = frame_data.get("geometry_px", {})
            frames.append(
                FrameResult(
                    name=name,
                    backend=frame_data.get("backend", ""),
                    fit_ready=frame_data.get("fit_ready", False),
                    visible_ratio=frame_data.get("visible_ratio", 0.0),
                    coverage_ratio=frame_data.get("coverage_ratio", 0.0),
                    volume_px3=geometry_px.get("volume_px3", 0.0),
                    geometry=geometry_px,
                    warnings=frame_data.get("warnings", []),
                    images=self._build_image_paths(base_dir, name),
                )
            )

        aggregate_geometry = data.get("geometry_px", {})
        raw_size = data.get("processing_image_size")
        processing_image_size = None
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            try:
                width, height = int(raw_size[0]), int(raw_size[1])
                if width > 0 and height > 0:
                    processing_image_size = (width, height)
            except (TypeError, ValueError):
                processing_image_size = None
        preview_path = base_dir / "geometry" / "standard_geometry_pixel_preview.png"
        return Stage1Result(
            input=data.get("input", ""),
            frame_count=data.get("frame_count", 0),
            fit_ready_count=data.get("fit_ready_count", 0),
            edge_backend=data.get("edge_backend", ""),
            consensus_frames=data.get("consensus_frames", []),
            consensus_frame_count=data.get("consensus_frame_count", 0),
            representative_frame=data.get("representative_frame", ""),
            aggregate_volume_px3=aggregate_geometry.get("volume_px3", 0.0),
            aggregate_geometry=aggregate_geometry,
            metric=data.get("metric"),
            geometry_preview=str(preview_path) if preview_path.exists() else None,
            frames=frames,
            processing_image_size=processing_image_size,
        )

    def run(
        self,
        input_path: Optional[str],
        input_type: str,
        options: Optional[dict] = None,
        progress_callback=None,
        should_cancel=None,
    ) -> Stage1Result:
        """执行一次视频或图片目录推理；调用方必须在后台线程中运行。

        progress_callback: 每处理一帧调用（参数为已处理帧数），供 Web 轮询进度。
        should_cancel:     返回 True 时任务安全中断（抛 Stage1Cancelled）。
        """
        if not input_path:
            raise ValueError("input_path 为空：请先选择视频文件或图片目录。")
        _ensure_backend_importable()
        from crystalvol.config import Stage1Config  # noqa: WPS433
        from crystalvol.stage1 import run_stage1  # noqa: WPS433

        options = options or {}
        self._cleanup_temporary_outputs()
        output_dir = self._make_output_dir(bool(options.get("save")), "crystalvol_ui_")
        config = Stage1Config(
            input_path=str(input_path),
            output_dir=output_dir,
            clean_output=True,
            camera_parameters=self._camera_config.parameter_path,
            undistort=True,
        )
        if input_type == "video" and "num_frames" in options:
            config.num_frames = int(options["num_frames"])
        if options.get("device"):
            config.device = str(options["device"])

        # UI 的首帧预选以归一化坐标传入，避免输入缩放后像素坐标失效。
        # 仅在明确启用且框有效时打开硬锚定模式；未启用时保持原有自动定位行为。
        preselection = options.get("preselection")
        if isinstance(preselection, dict) and preselection.get("enabled"):
            roi = preselection.get("roi_norm")
            if isinstance(roi, (list, tuple)) and len(roi) == 4:
                config.localize.preselection_roi = tuple(float(value) for value in roi)
                point = preselection.get("point_norm")
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    config.localize.preselection_point = tuple(float(value) for value in point)
                config.localize.preselection_enabled = True
                # 图片目录默认逐图独立定位；有首帧锚点时必须启用跨图跟踪。
                config.localize.tracking_force = True
                if "search_margin" in preselection:
                    config.localize.preselection_search_margin = max(
                        float(preselection["search_margin"]), 0.0
                    )
                if "max_jump_ratio" in preselection:
                    config.localize.preselection_max_jump_ratio = max(
                        float(preselection["max_jump_ratio"]), 1e-3
                    )

        summary = run_stage1(
            config,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
        )
        result = self._load_result_dir(Path(summary["output_dir"]), data=summary)
        metric = self._compute_metric(
            result.aggregate_geometry, image_size=result.processing_image_size
        )
        if metric:
            result.metric = metric
        return result

    def load_first_input_frame(
        self,
        input_path: str,
        input_type: str,
        max_input_side: int = 2304,
    ) -> dict:
        """读取批处理输入的第一帧，供 UI 首帧预选使用。

        该方法只做解码、统一缩放和可用时的去畸变，不加载 SAM2/边缘模型，
        因此可以在主线程中快速完成；返回的图像坐标与 stage1 的处理坐标一致。
        """
        if not input_path:
            raise ValueError("input_path 为空：无法读取首帧。")
        _ensure_backend_importable()
        from crystalvol.io import iter_inputs  # noqa: WPS433
        from crystalvol.camera_parameters import (  # noqa: WPS433
            load_camera_calibration,
            undistort_image_for_calibration,
        )

        frame = next(iter_inputs(str(input_path), num_frames=1, max_input_side=max_input_side))
        image = frame.image_bgr
        if image is None:
            raise RuntimeError("首帧图像为空，无法进行预选。")

        # stage1 的去畸变失败时会保留原图继续处理；这里采用相同的容错策略，
        # 不能因为预览阶段缺少标定文件而阻断用户框选。
        if self._camera_config.parameter_path:
            try:
                calibration = load_camera_calibration(self._camera_config.parameter_path)
                image = undistort_image_for_calibration(image, calibration)
            except Exception:
                logger.debug("首帧预选去畸变跳过", exc_info=True)

        height, width = image.shape[:2]
        return {
            "name": frame.name,
            "image_bgr": image,
            "image_size": (int(width), int(height)),
        }

    def start_realtime_session(self, save: bool = False) -> None:
        _ensure_backend_importable()
        from crystalvol.config import Stage1Config  # noqa: WPS433
        from crystalvol.session import Stage1Session  # noqa: WPS433

        self._cleanup_temporary_outputs()
        output_dir = self._make_output_dir(save, "crystalvol_rt_")
        session_config = Stage1Config(
            input_path="realtime://camera",
            output_dir=output_dir,
            clean_output=True,
            camera_parameters=self._camera_config.parameter_path,
            undistort=True,
        )
        self._session = Stage1Session(
            output_dir=output_dir, cfg=session_config, clean=True
        )
        self._realtime_previous_length_cm = None

    def add_realtime_photo(self, image_bgr) -> Stage1Result:
        if self._session is None:
            self.start_realtime_session()
        summary = self._session.add_frame(image_bgr)
        result = self._load_result_dir(Path(summary["output_dir"]), data=summary)
        metric = self._compute_metric(
            result.aggregate_geometry,
            previous_length_cm=self._realtime_previous_length_cm,
            image_size=result.processing_image_size,
        )
        if metric:
            result.metric = metric
            constraints = metric.get("physical_constraints", {})
            if constraints.get("accepted_for_growth"):
                self._realtime_previous_length_cm = float(
                    metric["dimensions_cm"]["length"]
                )
        return result

    def realtime_count(self) -> int:
        return self._session.count if self._session is not None else 0

    def end_realtime_session(self) -> None:
        if self._session is not None:
            close = getattr(self._session, "close", None)
            if close is not None:
                close()
        self._session = None
        self._realtime_previous_length_cm = None

    def close(self) -> None:
        """应用退出时释放实时会话和未保存的临时产物。"""
        self.end_realtime_session()
        self._cleanup_temporary_outputs()
        try:
            _ensure_backend_importable()
            from crystalvol.edges import clear_deep_detector_cache  # noqa: WPS433
            clear_deep_detector_cache()
        except Exception:
            logger.debug("深度边缘缓存清理失败", exc_info=True)
