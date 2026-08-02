# 运行数据目录

这里保存原始采集数据、标定图片、临时中间结果和推理结果。除本说明文件外，目录内容默认不纳入 Git；不要用 `git add -f` 把大视频、图片或运行结果强行加入仓库。

推荐子目录：

- `inputs/`：用户输入的视频和图片目录。
- `raw/`：原始采集文件和未处理数据。
- `calibration/`：标定板图片、生成的棋盘格和调试图。
- `results/`：前端或后端运行生成的结果。
- `tmp/`：可删除的临时中间文件。

推荐校准数据布局：

```text
data/calibration/
├── board.png/.json       生成的标定板和元数据
├── intrinsics/           内参标定图片
├── intrinsics_debug/     角点/圆点检测调试图
└── pose/                 用于单图外参的图片和坐标轴调试图
```

模型权重不放在这里，统一放在 `backend/weights/` 并由 Git LFS 跟踪。
