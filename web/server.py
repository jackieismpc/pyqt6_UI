"""FastAPI Web 服务：任务队列 + 产物文件 + 实时摄像头 + 静态前端。

启动方式见 web/__main__.py（uv run python -m web --host 0.0.0.0 --port 8000）。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.backend_core import BackendCore
from app.models import Stage1Result

from . import tasks as task_queue

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="透明晶体体积估计 · Web", version="0.2.0")

# ── 任务 API ──────────────────────────────────────────────────────────────


def _result_payload(res: Stage1Result, task_id: str, output_dir: str) -> dict:
    """把 Stage1Result 转成前端可用的 JSON（图像路径换成 /api/tasks/{id}/files/...）。"""

    def file_url(abs_path: Optional[str]) -> Optional[str]:
        if not abs_path:
            return None
        try:
            rel = Path(abs_path).resolve().relative_to(Path(output_dir).resolve())
        except ValueError:
            return None
        return f"/api/tasks/{task_id}/files/{rel.as_posix()}"

    return {
        "input": res.input,
        "frame_count": res.frame_count,
        "fit_ready_count": res.fit_ready_count,
        "edge_backend": res.edge_backend,
        "consensus_frames": res.consensus_frames,
        "consensus_frame_count": res.consensus_frame_count,
        "representative_frame": res.representative_frame,
        "aggregate_volume_px3": res.aggregate_volume_px3,
        "aggregate_geometry": res.aggregate_geometry,
        "metric": res.metric,
        "geometry_preview": file_url(res.geometry_preview),
        "frames": [
            {
                "name": fr.name,
                "backend": fr.backend,
                "fit_ready": fr.fit_ready,
                "visible_ratio": fr.visible_ratio,
                "coverage_ratio": fr.coverage_ratio,
                "volume_px3": fr.volume_px3,
                "geometry": fr.geometry,
                "warnings": fr.warnings,
                "images": {key: file_url(value) for key, value in fr.images.items()},
            }
            for fr in res.frames
        ],
    }


def _task_detail(task: task_queue.TaskRecord) -> dict:
    base = task.as_dict()
    if task.result is not None:
        base["result"] = _result_payload(task.result, task.id, task.output_dir)
    return base


@app.post("/api/tasks")
async def create_task(
    input_type: str = Form(...),          # video | zip | path
    file: Optional[UploadFile] = File(None),
    path: Optional[str] = Form(None),     # 服务器本地路径（input_type=path）
    num_frames: int = Form(7),
):
    """创建一次 stage1 推理任务（视频上传 / zip 图片目录 / 服务器本地路径）。"""
    options = {"num_frames": int(num_frames), "save": True}
    if input_type == "video":
        if file is None:
            raise HTTPException(400, "video 类型需要上传视频文件")
        data = await file.read()
        saved = task_queue.save_upload(file.filename or "video.mp4", data)
        task = task_queue.get_manager().submit(str(saved), "video", options)
        return {"task": task.as_dict()}
    if input_type == "zip":
        if file is None:
            raise HTTPException(400, "zip 类型需要上传压缩包")
        data = await file.read()
        saved = task_queue.save_upload(file.filename or "images.zip", data)
        extracted = task_queue.extract_zip(saved)
        task = task_queue.get_manager().submit(str(extracted), "dir", options)
        return {"task": task.as_dict()}
    if input_type == "path":
        if not path:
            raise HTTPException(400, "path 类型需要提供服务器本地路径")
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise HTTPException(400, f"路径不存在: {p}")
        if p.is_file():
            from backend.crystalvol.io import VIDEO_EXTENSIONS

            ext = p.suffix.lower()
            actual_type = "video" if ext in VIDEO_EXTENSIONS else "dir"
        else:
            actual_type = "dir"
        task = task_queue.get_manager().submit(str(p), actual_type, options)
        return {"task": task.as_dict()}
    raise HTTPException(400, f"未知 input_type: {input_type}")


@app.get("/api/tasks")
async def list_tasks():
    return {"tasks": [t.as_dict() for t in task_queue.get_manager().list_tasks()]}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = task_queue.get_manager().get(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    return _task_detail(task)


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = task_queue.get_manager().cancel(task_id)
    if not ok:
        raise HTTPException(404, "任务不存在或已完成")
    return {"ok": True}


@app.get("/api/tasks/{task_id}/files/{file_path:path}")
async def task_file(task_id: str, file_path: str):
    """提供任务输出目录下的产物文件（图像 / JSON / OBJ 等）。"""
    task = task_queue.get_manager().get(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    if not task.output_dir:
        raise HTTPException(404, "任务尚未产生产物")
    target = task_queue._safe_resolve(Path(task.output_dir), file_path)  # noqa: SLF001
    if target is None or not target.is_file():
        raise HTTPException(404, f"文件不存在: {file_path}")
    return FileResponse(target)


# ── 摄像头实时 API ────────────────────────────────────────────────────────


class CameraManager:
    """管理共享摄像头：MJPEG 预览流 + 抓帧拍摄（增量会话）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = None  # 最近一帧 BGR（用于 capture）
        self._backend: Optional[BackendCore] = None
        self._session_started = False

    def list(self) -> list[dict]:
        import cv2

        found = []
        for index in range(4):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                found.append({"id": str(index), "label": f"摄像头 {index}"})
            cap.release()
        return found

    def set_frame(self, frame) -> None:
        with self._lock:
            self._latest = frame

    def latest_frame(self):
        with self._lock:
            return self._latest

    def _ensure_session(self) -> BackendCore:
        if self._backend is None:
            self._backend = BackendCore()
            self._backend.start_realtime_session(save=True)
            self._session_started = True
        return self._backend

    def capture(self) -> dict:
        frame = self.latest_frame()
        if frame is None:
            # 无人观看预览流时直接开摄像头抓一帧
            import cv2

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise HTTPException(400, "无法打开摄像头，请先启动预览流")
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                raise HTTPException(400, "摄像头没有画面")
        backend = self._ensure_session()
        result = backend.add_realtime_photo(frame.copy())
        output_dir = (
            str(Path(result.geometry_preview).parents[1])
            if result.geometry_preview else ""
        )
        return {
            "count": backend.realtime_count(),
            "result": _result_payload(result, "realtime", output_dir),
        }

    def end_session(self) -> None:
        if self._backend is not None:
            self._backend.end_realtime_session()
            self._backend.close()
            self._backend = None
        self._session_started = False
        with self._lock:
            self._latest = None


_cameras = CameraManager()


@app.get("/api/cameras/list")
async def camera_list():
    return {"cameras": _cameras.list()}


@app.get("/api/cameras/stream.mjpg")
async def camera_stream():
    """MJPEG 实时预览（multipart/x-mixed-replace）。"""
    import cv2

    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield b"--frame\r\nContent-Type: text/plain\r\n\r\ncamera error\r\n--frame--\r\n"
            return
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                _cameras.set_frame(frame)
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
        finally:
            cap.release()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/cameras/capture")
async def camera_capture():
    return JSONResponse(_cameras.capture())


@app.post("/api/cameras/end")
async def camera_end():
    _cameras.end_session()
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"ok": True}


# ── 静态前端（最后挂载，API 路由优先）────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
