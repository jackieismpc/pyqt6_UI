"""Web 任务队列：单 worker 串行执行 stage1 推理。

设计约束：同一时刻只跑一个任务，避免并发加载 torch/SAM2 等重模型导致显存/内存
竞争；任务在后台线程执行，FastAPI 只做轻量轮询。输出目录在 data/results 下持久化，
产物文件由 server.py 的 /api/tasks/{id}/files/* 直接映射提供。
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.backend_core import BackendCore, Stage1Result

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = _PROJECT_ROOT / "data" / "results"
UPLOADS_DIR = _PROJECT_ROOT / "data" / "web_uploads"


def _safe_resolve(root: Path, relative: str) -> Path | None:
    """把相对路径安全地解析到 root 之内，防目录穿越。"""
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


@dataclass
class TaskRecord:
    id: str
    input_type: str
    input_label: str
    status: str = "queued"          # queued | running | done | failed | cancelled
    progress: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Stage1Result] = None
    cancel_requested: bool = False
    output_dir: str = ""
    _options: dict = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "input_type": self.input_type,
                "input_label": self.input_label,
                "status": self.status,
                "progress": self.progress,
                "error": self.error,
                "output_dir": self.output_dir,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


class TaskManager:
    """串行任务队列。"""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="web-task-worker", daemon=True)
        self._worker.start()

    # ---- 对外 API ----
    def submit(self, input_path: str, input_type: str, options: Optional[dict] = None) -> TaskRecord:
        task = TaskRecord(
            id=uuid.uuid4().hex[:12],
            input_type=input_type,
            input_label=str(input_path),
            _options=options or {},
        )
        with self._lock:
            self._tasks[task.id] = task
            self._order.append(task.id)
        self._wakeup.set()
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskRecord]:
        with self._lock:
            return [self._tasks[tid] for tid in self._order]

    def cancel(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task is None:
            return False
        with task._lock:
            if task.status in {"queued", "running"}:
                task.cancel_requested = True
                return True
        return False

    # ---- 内部 ----
    def _loop(self) -> None:
        while True:
            self._wakeup.wait()
            self._wakeup.clear()
            task_id = None
            with self._lock:
                for tid in self._order:
                    rec = self._tasks[tid]
                    if rec.status == "queued":
                        task_id = tid
                        break
            if task_id is None:
                continue
            self._execute(self._tasks[task_id])

    def _execute(self, task: TaskRecord) -> None:
        options: dict = task._options
        with task._lock:
            task.status = "running"
            task.started_at = time.time()
            task.progress = 0
        backend = BackendCore()
        try:
            result = backend.run(
                task.input_label,
                task.input_type,
                options,
                progress_callback=lambda n: self._set_progress(task, n),
                should_cancel=lambda: self._is_cancel_requested(task),
            )
            with task._lock:
                task.status = "done"
                task.result = result
                # Stage1Result 不保留 output_dir；由几何预览图路径反推输出目录。
                preview = Path(result.geometry_preview) if result.geometry_preview else None
                task.output_dir = str(preview.parents[1]) if preview else ""
                task.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with task._lock:
                task.status = "cancelled" if task.cancel_requested else "failed"
                task.error = f"{type(exc).__name__}: {exc}"
                task.finished_at = time.time()
        finally:
            try:
                backend.close()
            except Exception:
                pass

    @staticmethod
    def _set_progress(task: TaskRecord, count: int) -> None:
        with task._lock:
            if task.status == "running":
                task.progress = count

    @staticmethod
    def _is_cancel_requested(task: TaskRecord) -> bool:
        with task._lock:
            return task.cancel_requested


_manager: TaskManager | None = None


def get_manager() -> TaskManager:
    global _manager
    if _manager is None:
        _manager = TaskManager()
    return _manager


def save_upload(filename: str, data: bytes) -> Path:
    """保存上传文件到 data/web_uploads，返回绝对路径。"""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in filename)
    safe = safe.strip(" .") or "upload"
    path = UPLOADS_DIR / f"{int(time.time()*1000)}_{safe}"
    path.write_bytes(data)
    return path


def extract_zip(zip_path: Path) -> Path:
    """解压 zip 到临时目录，返回包含图片的目录。"""
    import zipfile

    dest = Path(tempfile.mkdtemp(prefix="web_zip_"))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    # 若解压后只有一层目录，就提升那一层为输入目录
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1 and not any(p.is_file() for p in dest.iterdir()):
        return children[0]
    return dest


def cleanup_upload(path: Path) -> None:
    """清理上传/解压的临时文件。"""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass
