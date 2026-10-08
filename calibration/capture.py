"""使用海康 MVS API 采集单相机或双相机标定图像。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from app.hikrobot_camera import HikRobotCamera


def _save_png(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"无法编码标定图像: {path}")
    path.write_bytes(encoded.tobytes())


def capture_calibration_images(
    camera_ids: list[str],
    output_dirs: list[str | Path],
    count: int,
    trigger_mode: str = "software",
    interval_ms: int = 300,
    timeout_ms: int = 3000,
) -> int:
    """采集标定图像；两台相机时使用完全相同的序号保存左右图像。"""
    if len(camera_ids) not in {1, 2}:
        raise ValueError("标定采集只支持 1 或 2 台相机")
    if len(output_dirs) != len(camera_ids):
        raise ValueError("输出目录数量必须与相机数量一致")
    if count < 1:
        raise ValueError("count 必须大于 0")
    mode = str(trigger_mode).lower()
    if mode not in {"software", "hardware"}:
        raise ValueError("标定采集 trigger_mode 只支持 software 或 hardware")

    cameras = [HikRobotCamera(camera_id) for camera_id in camera_ids]
    try:
        for camera in cameras:
            camera.open()
            camera.configure(mode)
            camera.start_grabbing()

        for index in range(1, count + 1):
            if mode == "software":
                # 先触发所有相机，再读取所有相机，保持 pair 序号一致。
                for camera in cameras:
                    camera.software_trigger()
            frames = [camera.read(timeout_ms) for camera in cameras]
            if any(frame is None for frame in frames):
                raise RuntimeError(f"第 {index} 组标定图像未收到完整帧")
            for output_dir, frame in zip(output_dirs, frames):
                _save_png(Path(output_dir).expanduser().resolve() / f"{index:04d}.png", frame.image_bgr)
            if index < count and interval_ms > 0:
                time.sleep(interval_ms / 1000.0)
        return count
    finally:
        for camera in cameras:
            camera.close()


__all__ = ["capture_calibration_images"]
