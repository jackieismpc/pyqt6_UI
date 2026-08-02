# app：PyQt6 前端

`app/` 只负责桌面交互和展示，不解析相机参数文件，也不在主线程执行推理。后端唯一适配点是 `backend_interface.py`。

## 启动流程

根目录 `main.py` 创建 `BackendInterface`，先请求后端的轻量相机参数摘要，再打开启动配置对话框。对话框只保存以下 UI 选项：

- `mode`：当前单目模式；
- `extrinsic_index`：选择后端参数中的外参组；
- `scale_anchor_edge` / `scale_anchor_value`：可选的真实边长锚点，单位 cm；
- `parameter_path`：可选显式 JSON 路径。

内参矩阵、畸变系数、外参单位和相机距离始终由 `backend/crystalvol/camera_parameters.py` 读取。这样前后端不会各自维护一份格式或产生不同的优先级。

## 输入模式

顶部控制栏支持：

1. **视频**：均匀抽帧，对同一晶体做跨帧共识；
2. **图片目录**：目录内图片视为同一晶体的不同视角；
3. **实时摄像头**：预览线程持续取帧，用户点击拍摄时把当前帧送入增量会话。

“保存结果”关闭时使用系统临时目录；打开时写入 `data/results/<时间戳>/`。所有中间图都通过后端返回的结果路径加载，不把大图缓存到 Qt 对象之外。

## 线程与异常恢复

`workers.py` 中有两个线程：

- `RunWorker`：一次性视频/图片目录推理。支持中断请求，异常通过 `failed` 信号回传；
- `RealtimeWorker`：摄像头预览和拍摄任务。预览限制为约 15 FPS，最长边缩放到 640 px；拍摄请求使用锁保护，避免 UI 点击和采集循环竞争。

主线程不得直接调用 `BackendInterface.run()` 或 `add_realtime_photo()`。耗时操作要通过 worker 信号更新 UI。实时 worker 在 `finally` 中释放 OpenCV/工业相机资源、结束后端会话并发出 `stopped`，所以摄像头打开失败、单张处理失败或窗口关闭都不会遗留句柄。

窗口关闭时会先请求停止实时线程或取消一次性任务，并等待有限时间；没有完成的任务不会继续向已销毁的窗口发信号。摄像头枚举是延迟执行的，启动时不会因为没有权限或没有设备而阻塞。

## UI 与结果模型

- `models.py`：`Stage1Result`、`FrameResult` 和置信度计算；
- `image_panel.py`：三面板等比显示，图片缺失时显示可读的占位提示；
- `result_bar.py`：总体体积、当前帧质量和警告；
- `main_window.py`：只编排信号与展示，不实现图像算法；
- `camera_config.py`：仅保存 UI 选择，不保存 K、畸变或外参；
- `platform_utils.py` / `camera_scanner.py`：处理平台相关的工业相机发现和 DLL 加载。

## 调试方式

```bash
uv run python main.py
```

运行结果和后端警告在 `data/results/` 对应任务目录中查看。遇到问题时优先检查：

1. `data/results/.../stage1_result.json` 是否有 `failed_frames` 或逐帧 `warnings`；
2. 右侧线框预览和中间的边缘/剪影产物是否贴合晶体；
3. `params/camera_parameters.json` 是否被后端成功加载；
4. 摄像头权限、MVS SDK 和 Git LFS 权重是否准备好。
