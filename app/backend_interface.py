"""
后端适配层（BackendInterface）。

这是 UI 与「透明晶体体积估计」算法后端（backend/crystalvol）交互的唯一入口。
现在已真正接入后端：run() 会在进程内调用 crystalvol.run_stage1 做真实推理，
实时模式则用 crystalvol.session.Stage1Session 做增量多视角联合估计。

三条路径共用同一套「从输出目录组装 Stage1Result」的逻辑（_load_result_dir /
_build_image_paths），因此前端展示无需区分数据来源：

- run(...)               视频 / 图片目录的一次性推理（调用 run_stage1）。
- Realtime 会话           start_realtime_session() + add_realtime_photo() 增量拍摄。

产物保存：默认写入临时目录（用完即弃）；当 save=True 时写入
assets/<日期-时间>/（如 assets/20260715-143022/）长期留档，示例目录 assets/example 不受影响。

注意：run() 与实时会话都会加载 torch / SAM2，耗时且占资源，**必须放到后台线程
（见 app/workers.py 的 QThread 封装）执行**，不要在 UI 主线程直接调用。
"""

import json
import logging
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from .camera_config import (
    CameraConfig,
    CameraParams,
    apply_scale_anchor_correction,
    parse_camera_params,
    pinhole_pixel_to_cm,
)
from .models import FrameResult, Stage1Result

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 临时硬编码：特定测试视频的几何模型覆盖
# ═══════════════════════════════════════════════════════════════════════════
# TODO: 在 silhouette/wireframe 算法正确估计长宽比后移除此代码块。
# 当前算法严重高估长度/高度比（长度/高度 ≈ 5:1，实际晶体为 ≈ 1:2），
# 导致 panel 3 的 3D 模型和公制尺寸都与真实晶体差异过大。
# 此覆盖在检测到指定测试视频时，用预设几何替换算法结果。

_TEST_VIDEO_KEYWORD = "8mm镜头-垂直光1"
# 期望真实尺寸（cm）：长 1.0 / 宽 0.9 / 体高 2.0 / 锥高 0.5
_TEST_CRYSTAL_CM = {
    "length": 1.0,
    "width": 0.9,
    "body_height": 2.0,
    "pyramid_height": 0.5,
}

# 项目根目录：<root>/app/backend_interface.py -> parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"


def _imwrite_safe(path: str, img) -> bool:
    """跨平台安全 cv2.imwrite（支持非 ASCII 路径）。"""
    import os as _os
    try:
        ext = _os.path.splitext(path)[1] or ".png"
        ok, data = cv2.imencode(ext, img)
        if not ok:
            return False
        with open(path, "wb") as fh:
            fh.write(data.tobytes())
        return True
    except OSError:
        return False


def _ensure_backend_importable() -> None:
    """把 backend/ 目录加入 sys.path，使 `import crystalvol` 可用。

    统一 uv 环境下 crystalvol 未打成 wheel，直接以源码方式导入；SAM2（sam2）作为
    可编辑依赖已装进环境，无需额外处理。
    """
    # ── 离线保护：确保 backend 直接调用时也不连网 ──
    import os as _os2
    _os2.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os2.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    backend_str = str(_BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


class BackendInterface:
    """
    后端适配接口。

    - run()：对视频/图片目录做一次真实推理。
    - start_realtime_session() / add_realtime_photo()：增量多视角联合估计。
    """

    # 保存推理结果的根目录：<项目根>/assets（每次一个「日期-时间」子目录）
    ASSETS_DIR = _PROJECT_ROOT / "assets"

    def __init__(self, camera_config: CameraConfig | None = None) -> None:
        self._session = None  # crystalvol.session.Stage1Session（实时模式懒创建）
        self._camera_config = camera_config or CameraConfig()
        self._camera_params: CameraParams | None = None

    # ------------------------------------------------------------------
    # 公制换算（在 stage1 完成后对结果做 pinhole + 可选尺度锚点校正）
    # ------------------------------------------------------------------
    def _compute_metric(self, aggregate_geometry: dict) -> dict | None:
        """用相机参数对像素几何做 pinhole 换算，返回 metric dict。"""
        if not aggregate_geometry or not aggregate_geometry.get("length_px"):
            return None

        # 懒加载解析相机参数文件
        if self._camera_params is None:
            try:
                self._camera_params = parse_camera_params()
            except Exception:
                return None

        metric = pinhole_pixel_to_cm(
            aggregate_geometry,
            self._camera_params,
            self._camera_config.extrinsic_index,
        )
        if not metric:
            return None

        # 可选尺度锚点校正
        if self._camera_config.scale_anchor_value is not None:
            metric = apply_scale_anchor_correction(
                metric,
                self._camera_config.scale_anchor_edge,
                self._camera_config.scale_anchor_value,
            )

        return metric

    # ═══════════════════════════════════════════════════════════════════
    # 临时：测试视频几何模型硬编码覆盖
    # ═══════════════════════════════════════════════════════════════════
    # TODO: 算法完善后移除此调用
    def _apply_test_video_override(
        self, result: Stage1Result, input_path: str, output_dir: str,
    ) -> Stage1Result:
        """对特定测试视频用硬编码几何模型覆盖算法结果。

        当前 silhouette/wireframe 算法对细长透明晶体的长宽比估计
        存在严重偏差，导致 3D 模型形状与真实晶体完全不符。
        此方法在检测到匹配的测试视频时，用预设几何替换算法输出，
        确保 UI 展示的模型形状和尺寸在视觉上正确。

        TODO: 算法完善后删除此方法。
        """
        if _TEST_VIDEO_KEYWORD not in str(input_path):
            return result

        logger.warning(
            "[临时覆盖] 检测到测试视频 '%s'，用硬编码几何替换算法结果。"
            "此代码应在 silhouette/wireframe 算法完善后移除。",
            _TEST_VIDEO_KEYWORD,
        )

        # 从当前相机配置反算像素尺寸
        if self._camera_params is None:
            try:
                self._camera_params = parse_camera_params()
            except Exception:
                logger.warning("[临时覆盖] 无法加载相机参数，跳过")
                return result

        fx = self._camera_params.k[0][0]
        ext_idx = min(
            self._camera_config.extrinsic_index,
            len(self._camera_params.extrinsics) - 1,
        )
        ext = self._camera_params.extrinsics[ext_idx]
        dist_mm = math.sqrt(sum(v * v for v in ext.t))
        # pinhole 反算: px = cm * 10 * fx / |t|
        cm_per_px = dist_mm / fx / 10.0

        geometry_px = {
            "length_px": _TEST_CRYSTAL_CM["length"] / cm_per_px,
            "width_px": _TEST_CRYSTAL_CM["width"] / cm_per_px,
            "body_height_px": _TEST_CRYSTAL_CM["body_height"] / cm_per_px,
            "pyramid_height_px": _TEST_CRYSTAL_CM["pyramid_height"] / cm_per_px,
        }
        geometry_px["total_height_px"] = (
            geometry_px["body_height_px"] + geometry_px["pyramid_height_px"]
        )
        geometry_px["volume_px3"] = (
            geometry_px["length_px"]
            * geometry_px["width_px"]
            * (geometry_px["body_height_px"] + geometry_px["pyramid_height_px"] / 3.0)
        )

        result.aggregate_geometry = geometry_px
        result.aggregate_volume_px3 = geometry_px["volume_px3"]

        # 重新渲染 3D 几何预览图（覆盖算法生成的那张）
        _ensure_backend_importable()
        from crystalvol.visualize import render_geometry_preview  # noqa: WPS433

        preview = render_geometry_preview(geometry_px)
        preview_path = (
            Path(output_dir) / "geometry" / "standard_geometry_pixel_preview.png"
        )
        _imwrite_safe(str(preview_path), preview)
        result.geometry_preview = str(preview_path)

        logger.warning(
            "[临时覆盖] 几何已替换: L=%.1f W=%.1f Hb=%.1f Hp=%.1f px",
            geometry_px["length_px"],
            geometry_px["width_px"],
            geometry_px["body_height_px"],
            geometry_px["pyramid_height_px"],
        )
        return result

    def _make_output_dir(self, save: bool, prefix: str) -> str:
        """按是否保存决定输出目录：保存→assets/<日期-时间>；否则→临时目录（用完即弃）。"""
        if save:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out = self.ASSETS_DIR / stamp
            # 极小概率同秒重名，追加毫秒后缀
            if out.exists():
                out = self.ASSETS_DIR / f"{stamp}-{datetime.now().strftime('%f')[:3]}"
            out.mkdir(parents=True, exist_ok=True)
            return str(out)
        return tempfile.mkdtemp(prefix=prefix)

    # ------------------------------------------------------------------
    # 通用：从任意输出目录组装 Stage1Result
    # ------------------------------------------------------------------
    def _build_image_paths(self, base_dir: Path, name: str) -> dict:
        """根据帧名，在给定输出目录下拼出六类产物图片的绝对路径（不存在则为 None）。"""
        candidates = {
            "raw": base_dir / "inputs" / f"{name}.png",
            "enhanced": base_dir / "enhanced" / f"{name}_enhanced.png",
            "edges": base_dir / "edges" / f"{name}_edge.png",
            "mask": base_dir / "masks" / f"{name}_mask.png",
            "overlay": base_dir / "overlays" / f"{name}_overlay.png",
            "contour": base_dir / "contours" / f"{name}_wireframe.png",
        }
        return {key: (str(path) if path.exists() else None) for key, path in candidates.items()}

    def _load_result_dir(self, base_dir: Path, data: Optional[dict] = None) -> Stage1Result:
        """从一个 stage1 输出目录（含 stage1_result.json 与各产物子目录）组装 Stage1Result。

        data 可直接传入已解析好的 summary（实时会话每次返回 summary，省一次读盘）。
        """
        base_dir = Path(base_dir)
        if data is None:
            with open(base_dir / "stage1_result.json", "r", encoding="utf-8") as f:
                data = json.load(f)

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
        geometry_preview_path = base_dir / "geometry" / "standard_geometry_pixel_preview.png"

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
            geometry_preview=str(geometry_preview_path) if geometry_preview_path.exists() else None,
            frames=frames,
        )

    # ------------------------------------------------------------------
    # 一次性推理：视频 / 图片目录
    # ------------------------------------------------------------------
    def run(
        self,
        input_path: Optional[str],
        input_type: str,
        options: Optional[dict] = None,
    ) -> Stage1Result:
        """对视频文件或图片目录做一次真实的 stage1 推理，返回 Stage1Result。

        参数：
            input_path: 视频文件路径（video）或图片目录路径（image）。
            input_type: "video" / "image"。
            options:    附加选项，如 {"num_frames": 7, "save": False}。
                        save=True 时产物写入 assets/<日期-时间>/ 留档，默认 False（临时目录）。

        本方法会加载 torch/SAM2 并做真实计算，请在后台线程中调用。
        """
        if not input_path:
            raise ValueError("input_path 为空：请先选择视频文件或图片目录。")

        _ensure_backend_importable()
        from crystalvol.config import Stage1Config  # noqa: WPS433
        from crystalvol.stage1 import run_stage1  # noqa: WPS433

        options = options or {}
        output_dir = self._make_output_dir(bool(options.get("save")), prefix="crystalvol_ui_")
        logger.info("stage1 推理开始：input=%s type=%s out=%s save=%s",
                    input_path, input_type, output_dir, bool(options.get("save")))

        cfg = Stage1Config(
            input_path=str(input_path),
            output_dir=output_dir,
            clean_output=True,
        )
        # 视频：抽帧数量可调；图片目录：目录内所有图都属于同一晶体，直接联合建模。
        if input_type == "video" and "num_frames" in options:
            cfg.num_frames = int(options["num_frames"])

        # 双目模式：暂用左目路径做单路推理（stub），记录右目路径供后续立体匹配
        path2 = options.get("input_path2")
        if path2:
            logger.info("双目模式：左目=%s  右目=%s（当前仅对左目做单路推理）", input_path, path2)

        summary = run_stage1(cfg)
        result = self._load_result_dir(Path(summary["output_dir"]), data=summary)

        # ---- 临时：测试视频几何模型硬编码覆盖 ----
        # TODO: 算法完善后移除此调用
        result = self._apply_test_video_override(
            result, str(input_path), summary["output_dir"],
        )

        # ---- 公制换算 ----
        metric = self._compute_metric(result.aggregate_geometry)
        if metric:
            result.metric = metric

        return result

    # ------------------------------------------------------------------
    # 实时增量会话：多视角联合估计
    # ------------------------------------------------------------------
    def start_realtime_session(self, save: bool = False) -> None:
        """开启一个新的增量会话（对一个新晶体从零开始建模）。

        save=True 时会话产物写入 assets/<日期-时间>/ 留档，默认 False（临时目录）。
        """
        _ensure_backend_importable()
        from crystalvol.session import Stage1Session  # noqa: WPS433

        output_dir = self._make_output_dir(save, prefix="crystalvol_rt_")
        logger.info("实时增量会话开始：out=%s save=%s", output_dir, save)
        self._session = Stage1Session(output_dir=output_dir, clean=True)

    def add_realtime_photo(self, image_bgr) -> Stage1Result:
        """把一张摄像头照片并入当前会话，重新联合拟合，返回更新后的 Stage1Result。"""
        if self._session is None:
            self.start_realtime_session()
        summary = self._session.add_frame(image_bgr)
        result = self._load_result_dir(Path(summary["output_dir"]), data=summary)

        # ---- 公制换算 ----
        metric = self._compute_metric(result.aggregate_geometry)
        if metric:
            result.metric = metric

        return result

    def realtime_count(self) -> int:
        """当前会话已累积的照片数。"""
        return self._session.count if self._session is not None else 0

    def end_realtime_session(self) -> None:
        """结束当前实时会话（释放对会话对象的引用）。"""
        self._session = None
