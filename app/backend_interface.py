"""PyQt6 与 crystalvol 后端之间的唯一适配层。"""

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

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    backend_string = str(_BACKEND_DIR)
    if backend_string not in sys.path:
        sys.path.insert(0, backend_string)


class BackendInterface:
    """前端使用的后端门面。"""

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
        )

    def run(
        self,
        input_path: Optional[str],
        input_type: str,
        options: Optional[dict] = None,
    ) -> Stage1Result:
        """执行一次视频或图片目录推理；调用方必须在后台线程中运行。"""
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
        )
        if input_type == "video" and "num_frames" in options:
            config.num_frames = int(options["num_frames"])
        if options.get("device"):
            config.device = str(options["device"])

        summary = run_stage1(config)
        result = self._load_result_dir(Path(summary["output_dir"]), data=summary)
        metric = self._compute_metric(result.aggregate_geometry)
        if metric:
            result.metric = metric
        return result

    def start_realtime_session(self, save: bool = False) -> None:
        _ensure_backend_importable()
        from crystalvol.session import Stage1Session  # noqa: WPS433

        self._cleanup_temporary_outputs()
        output_dir = self._make_output_dir(save, "crystalvol_rt_")
        self._session = Stage1Session(output_dir=output_dir, clean=True)
        self._realtime_previous_length_cm = None

    def add_realtime_photo(self, image_bgr) -> Stage1Result:
        if self._session is None:
            self.start_realtime_session()
        summary = self._session.add_frame(image_bgr)
        result = self._load_result_dir(Path(summary["output_dir"]), data=summary)
        metric = self._compute_metric(
            result.aggregate_geometry,
            previous_length_cm=self._realtime_previous_length_cm,
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
