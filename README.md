# 透明晶体体积估计 · 前端可视化 + 算法后端（统一工程）

一个完整的桌面应用：**PyQt6 可视化前端**（`app/`）+ **crystalvol 算法后端**（`backend/`），
用同一个 **uv** 环境统一管理。对**透明规则晶体**（长方体 + 四棱锥屋顶形）做体积估计，
针对透明、强高光反射、暗部遮挡三大拍摄难点做了专门处理。

前端三个并排图像面板（原始输入 / 预处理后 / 几何模型）+ 顶部输入控制栏 + 底部体积与置信度结果栏，
浅色主题。**后端算法已真正接入**：前端在后台线程内直接调用 `crystalvol` 做真实推理，
默认启动为空白面板，选择输入后开始推理。

---

## 架构总览

```
pyqt6_UI/
├── main.py                     应用入口
├── pyproject.toml              统一依赖声明（前端 + 后端，uv 管理）
├── .python-version             3.13
├── app/                        —— 前端 UI ——
│   ├── main_window.py          主窗口：三面板 + 控制栏 + 结果栏，编排三种输入
│   ├── controls.py             顶部控制栏：视频/图片/实时、帧数、拍摄控件
│   ├── backend_interface.py    后端适配层：进程内调用 crystalvol（唯一对接点）
│   ├── workers.py              后台线程：RunWorker（一次性推理）/ RealtimeWorker（摄像头增量）
│   ├── image_panel.py          图像面板：等比缩放 + 等待态文字 + 实时预览
│   ├── result_bar.py           结果栏：体积、置信度、帧切换
│   ├── models.py               数据模型 Stage1Result / FrameResult + 置信度公式
│   ├── widgets.py / theme.py   通用控件与浅色主题
├── assets/<日期-时间>/         勾选「保存结果」后每次推理的产物目录（gitignore）
└── backend/                    —— crystalvol 算法后端 ——
    ├── run.py                  命令行入口（也可独立使用）
    ├── crystalvol/             核心包（预处理/边缘/分割/线框/几何/编排/会话…）
    │   ├── stage1.py           第一阶段编排（已抽出可复用的 process/finalize helper）
    │   ├── session.py          增量多视角会话（实时拍摄用，累积 + 重新联合拟合）
    │   └── …
    ├── third_party/sam2/       SAM2 源码（以可编辑依赖接入）
    ├── weights/                SAM2 / YOLO-World 权重（已随工程附带）
    ├── data/inputs/            示例输入
    └── docs/                   方案设计文档
```

前端与后端的**唯一对接点**是 `app/backend_interface.py`：它把 `backend/` 加入 `sys.path`，
在后台线程里 `import crystalvol` 并调用 `run_stage1`（一次性）或 `Stage1Session`（实时增量），
再把输出目录组装成 `Stage1Result` 交给界面展示。

---

## 统一环境安装（跨平台）

全项目只有一个 uv 环境，前端（PyQt6）与后端（torch / SAM2 / ultralytics / controlnet_aux …）
都装在里面。Python 版本为 **3.13**（见 `.python-version`）。

### 0. 获取代码（所有平台通用）

本仓库使用 **Git LFS** 管理模型权重文件（约 214 MB），请先安装 Git LFS 再克隆：

```bash
# 安装 Git LFS（仅需一次）
# macOS:
brew install git-lfs
# Ubuntu/Debian:
sudo apt install git-lfs
# Windows: 从 https://git-lfs.com 下载安装包

# 初始化 LFS
git lfs install

# 克隆仓库（LFS 文件会自动下载）
git clone <仓库地址>
cd pyqt6_UI
```

> **如果 LFS 下载失败或耗时过长**（网络受限等），克隆后 `backend/weights/` 下的 `.pt`/`.pth` 文件
> 可能是 LFS 指针（仅几百字节），需要手动下载权重文件。见下方 [LFS 备用方案](#lfs-备用方案手动权重拷贝)。

### macOS

```bash
# 1) 安装 uv（若未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
#   或： brew install uv

# 2) 进入工程目录，一键创建环境并安装全部依赖
cd /path/to/pyqt6_UI
SAM2_BUILD_CUDA=0 uv sync          # macOS 必须带 SAM2_BUILD_CUDA=0（不编译 CUDA 扩展）

# 3) 运行前端
uv run python main.py
```

### Windows（全新电脑完整指南）

以下是从零开始在全新 Windows 电脑上安装并运行本工程的详细步骤。

#### 前置准备

本工程运行需要 **Git**（用于克隆仓库）和 **uv**（包管理器，会自动下载 Python 3.13）。
你不需要单独安装 Python，uv 会自行管理 Python 版本。

**1. 安装 Git**

从 https://git-scm.com/download/win 下载并安装。安装过程中全部默认选项即可。
安装完成后，按 `Win + R`，输入 `powershell`，回车打开 PowerShell，验证：

```powershell
git --version
# 应输出类似: git version 2.47.0.windows.1
```

**2. 安装 uv（Python 包管理器）**

打开 PowerShell（不是 CMD），执行：

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后，**关闭当前 PowerShell 窗口，重新打开一个新的 PowerShell 窗口**，然后验证：

```powershell
uv --version
# 应输出类似: uv 0.6.x
```

> 如果提示 `uv` 命令找不到，按 `Win + X` →「系统」→「高级系统设置」→「环境变量」，
> 检查用户变量 `Path` 里是否有 `%USERPROFILE%\.local\bin`，没有则手动添加，然后重启 PowerShell。

#### 获取工程代码

**方式 A：从 GitHub 克隆（推荐）**

```powershell
cd %USERPROFILE%\Downloads
git clone <你的仓库地址>
cd pyqt6_UI
```

**方式 B：从 U 盘/移动硬盘拷贝**

将整个 `pyqt6_UI` 文件夹复制到电脑上的某个位置（如 `C:\Users\你的用户名\Downloads\pyqt6_UI`）。
**确保 `backend\weights\` 文件夹一并复制**（里面是 SAM2/YOLO-World 模型权重文件，约 150 MB）。
然后在 PowerShell 中进入该目录：

```powershell
cd C:\Users\你的用户名\Downloads\pyqt6_UI
```

#### 安装依赖并运行

在 `pyqt6_UI` 目录下，执行以下命令：

```powershell
# 设置环境变量（告诉 SAM2 不要编译 CUDA 扩展，Windows 没有 CUDA）
$env:SAM2_BUILD_CUDA = "0"

# 一键安装全部依赖（首次约需 5–15 分钟，取决于网速）
uv sync

# 运行应用
uv run python main.py
```

> **注意**：`uv sync` 首次运行会下载 PyTorch（约 2 GB）等大量依赖，
> 请确保网络畅通且有足够磁盘空间（约 5 GB）。之后再次运行 `uv sync` 秒级完成。

#### 后续每次运行

以后每次使用，只需打开 PowerShell，进入工程目录，执行一行命令：

```powershell
cd C:\Users\你的用户名\Downloads\pyqt6_UI
uv run python main.py
```

> 不需要重复 `uv sync`，也不需要再次设置 `SAM2_BUILD_CUDA`（环境变量只在首次安装时需要）。

#### 常见问题

**Q: 运行 `uv sync` 时提示 `error: failed to build` 之类的错误？**

通常是因为忘了设置 `SAM2_BUILD_CUDA=0`。关闭 PowerShell 重新打开，先执行
`$env:SAM2_BUILD_CUDA = "0"`，再执行 `uv sync`。

**Q: 运行 `main.py` 后窗口闪退？**

用命令行启动（`uv run python main.py`），不要双击 `main.py` 文件。
如果命令行有报错信息，根据错误提示排查。

**Q: 提示找不到 `python` 或 `uv`？**

确认已安装 uv（见前置准备第 2 步），且使用 **PowerShell** 而非 CMD。

**Q: 运行后提示 `ImportError: No module named 'PyQt6'`？**

说明依赖未正确安装。重新运行：

```powershell
$env:SAM2_BUILD_CUDA = "0"
uv sync
```

**Q: 海康工业相机（MVS SDK）怎么办？**

如果你需要使用海康工业相机，需额外安装 MVS SDK for Windows，
从海康官网下载安装（默认路径 `C:\Program Files (x86)\MVS`），应用会自动检测。
不使用工业相机则无需安装。

**Q: 虚拟环境装在哪里？占多大空间？**

uv 将虚拟环境创建在 `pyqt6_UI\.venv\` 目录下，约 3–5 GB（含 PyTorch）。
如需清理，删除 `.venv` 文件夹后重新 `uv sync` 即可。

### Linux

```bash
# 1) 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 安装依赖
cd /path/to/pyqt6_UI
SAM2_BUILD_CUDA=0 uv sync

# 3) 运行
uv run python main.py
```

说明：

- `SAM2_BUILD_CUDA=0` 让 SAM2 以纯 Python 方式可编辑安装（源码在 `backend/third_party/sam2`）；
  非 CUDA 环境必须设置，否则 `uv sync` 构建 SAM2 时会报错。
- **模型权重**（SAM2 / YOLO-World / PiDiNet / ControlNet-HED）放在 `backend/weights/`，通过 Git LFS 管理。
  正常 `git clone` 即自动下载（约 214 MB）。若 LFS 下载失败，见下方 [LFS 备用方案](#lfs-备用方案手动权重拷贝)。
- **深度边缘权重（PiDiNet / HED）** 已随工程放在 `backend/weights/`，直接从本地加载，零网络依赖。
  权重缺失时，边缘提取自动回退到 Canny，流程不中断。
- 设备：macOS/Linux 默认 CPU；SAM2 在 Apple MPS 上有个别算子未实现，代码里已自动把 SAM2 降级到 CPU
  跑（ROI 裁剪后目标很小，CPU 上 SAM2-tiny 足够快）。
- **Linux 中文显示**：Ubuntu 需要安装中文字体，否则界面中文会乱码：
  `sudo apt install fonts-noto-cjk`（推荐）或 `sudo apt install fonts-wqy-microhei`。
  应用启动时会自动检测可用中文字体，无需额外配置。
- **Linux glibc 要求**：PyQt6-Qt6 的 aarch64 轮子需要 glibc ≥ 2.39（`manylinux_2_39`），
  因此 Ubuntu aarch64 必须升级到 24.04 或更高版本。Ubuntu 22.04 无法直接安装 PyQt6。
  x86_64 不受此限制（有 `manylinux_2_34` 轮子）。

### 迁移到另一台电脑（快速安装）

整个 `pyqt6_UI/` 目录是自包含的（含 `backend/weights/` 权重与 `third_party/sam2` 源码），
迁移只需三步：

```bash
# 1) 拷贝/克隆整个 pyqt6_UI 目录到新机器（确保 backend/weights/ 一并带上）
# 2) 安装 uv（见上），然后：
cd /path/to/pyqt6_UI
# macOS / Linux:
SAM2_BUILD_CUDA=0 uv sync
# Windows (PowerShell):
$env:SAM2_BUILD_CUDA = "0"; uv sync
# 3) 运行
uv run python main.py
```

`uv sync` 会依据 `pyproject.toml` 自动解析、下载并锁定（生成 `uv.lock`）全部依赖，
包含 torch/torchvision 的平台对应版本。首次同步会下载较多依赖（torch 等），请耐心等待；
之后再次 `uv sync` 会命中缓存，非常快。

> 提示：若新机器联网受限，PiDiNet 权重下载会失败，但不影响运行——边缘会自动回退 Canny。

### LFS 备用方案：手动权重拷贝

如果 `git clone` 时 LFS 下载失败（`.pt`/`.pth` 文件仅 1–2 KB 即为失败），
或网络条件不允许 LFS 传输，可以从已有安装的机器上手动复制权重：

**需要复制的文件**（位于 `backend/weights/`，共 4 个，约 214 MB）：

| 文件 | 大小 | 用途 |
|------|------|------|
| `sam2.1_hiera_tiny.pt` | 156 MB | SAM2 视频分割模型 |
| `yolov8s-worldv2.pt` | 26 MB | YOLO-World 目标检测 |
| `ControlNetHED.pth` | 29 MB | HED 边缘检测 |
| `table5_pidinet.pth` | 2.9 MB | PiDiNet 深度边缘检测 |

**操作步骤**：

```bash
# 从源机器（有权重的）复制到目标机器：
scp backend/weights/*.pt backend/weights/*.pth user@target:/path/to/pyqt6_UI/backend/weights/

# 或用 U 盘/移动硬盘拷到目标机器的 pyqt6_UI/backend/weights/ 目录下
```

复制完成后，`uv sync && uv run python main.py` 即可正常使用。
缺少这些权重文件时，应用启动不会报错，但在运行推理时会在对应步骤失败。

---

## 前端使用方法

启动后顶部控制栏有三种输入类型，选好后点「选择并运行」：

**视频**
选择一个视频文件；用「帧数」选择器设定从视频中均匀抽取多少帧（默认 7）参与联合建模。
一段视频视为同一个晶体的多帧采样，跨帧稳健共识出一套几何。

**图片**
选择一个**目录**；该目录下的所有图片都被视为**同一个晶体**的不同照片，用于联合建模
（跨图共识出一个几何）。

**实时（摄像头多视角增量估计）**
点「选择并运行」后打开摄像头，左面板显示实时预览。设定「目标张数」，从不同角度对准同一晶体，
每点一次「拍摄」就采一张照片并入模型——**后端在已建好的几何上继续优化（增量估计）**，
中间/右侧面板与体积、置信度随每张照片实时更新。拍够或随时点「结束实时」即可停止。

**保存结果**：控制栏的「保存结果」勾选框，**默认不保存**（产物写入临时目录，用完即弃）。
勾选后本次推理产物保存到 `assets/<日期-时间>/`（如 `assets/20260715-143022/`）长期留档；
时间戳目录默认被 gitignore。

**等待态**：真实推理耗时（数秒到数十秒），推理在后台线程进行，期间对应面板显示「推理中…」，
界面不会卡死；完成后自动填充结果。

底部结果栏的**估计体积**展示的是**联合模型的汇总(共识)体积**——它由本次全部帧共同拟合得到，
因此对一次视频/图片推理是**固定值，切换帧不会变**；只有实时模式每并入一张新照片重新联合拟合时，
体积才会**增量变化**。逐帧的单帧体积仅作诊断，悬停在体积/置信度上即可在 tooltip 中查看
（连同 fit_ready / 可见比 / 覆盖比 / 汇总信息）。右侧可切换帧；中间面板标题右侧下拉框
可切换查看四类中间产物：低光增强 / 边缘证据 / 剪影掩膜 / 线框叠加。

---

## 后端独立使用（命令行）

后端也可脱离前端单独用（同一个 uv 环境）：

```bash
# 单张图片 / 图片目录 / 视频（输入换成对应路径即可）
uv run python backend/run.py stage1 backend/data/inputs/xxx.jpg \
    --device cpu --edge-backend auto --clean-output --output-dir backend/outputs/run1

uv run python backend/run.py stage1 /path/to/frames_dir --output-dir backend/outputs/run1 --clean-output
uv run python backend/run.py stage1 /path/to/video.mp4 --num-frames 9 --output-dir backend/outputs/run1 --clean-output
```

输出目录结构（前端也按此解析）：`inputs/ enhanced/ edges/ masks/ contours/ overlays/`
每帧一图，`geometry/standard_geometry_pixel_preview.png(.json/.obj)` 为汇总几何，
`stage1_result.json` 为汇总结果。完整参数见 `uv run python backend/run.py stage1 -h`。

几何与体积公式：四个尺寸 `length`(L) / `width`(W) / `body_height`(Hb) / `pyramid_height`(Hp)，
体积 `V = L * W * (Hb + Hp/3)`。单目单视角的侧向深度 `width` 为启发式估计，视角越多约束越强。

### 线框拟合可调参数（WireframeConfig）

以下参数可通过修改 `backend/crystalvol/config.py` 中的 `WireframeConfig` 调整，
适配不同形状的晶体：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `depth_ratio` | `0.9` | 宽度/长度比（单目深度启发式）；典型晶体截面接近正方形（0.8–1.0） |
| `pyramid_top_width_ratio` | `0.7` | 顶部宽度 < 该比例×满宽才认为有屋顶（越小越严格） |
| `pyramid_min_taper_ratio` | `0.06` | 渐缩段高度 ≥ 该比例×总高才认为有屋顶 |
| `min_pyramid_fraction` | `0.15` | 未检测到屋顶时的最小棱锥高度比例（保底值，避免平顶） |
| `core_percentile` | `40.0` | 前景内亮度分位，高于它视为晶体亮核（越小保留越多边缘像素；透明晶体推荐 35–45） |

形状比例校准参考（测试视频期望值）：长 1cm / 宽 0.9cm / 体高 2cm / 锥高 0.5cm →
`depth_ratio≈0.9`，`min_pyramid_fraction≈0.2`。

### 尺度锚点校准（推荐）

相机外参估算的绝对尺寸可能因标定条件与拍摄条件不一致而产生数倍偏差。
**推荐做法**：在启动对话框的「尺度锚点」区域设置一条已知真实边长：

1. 选择锚点边：`body_height`（体高，最容易测量的维度）
2. 输入真实值：如 `2.0`（cm）

这会将所有像素维度按 `真实值 / 外参估算值` 的比例统一缩放，
得到正确的绝对尺寸。相比纯依赖相机外参，这是更稳健的校准方式。

如果启动时未设置锚点，运行后可在结果栏看到外参估算值，
记下偏差倍数后重新运行并填入正确锚点即可。

---

## 置信度说明

置信度为经验公式，仅用于 UI 展示（定义见 `app/models.py`）：

- 单帧：`score = 0.6 * visible_ratio + (0.4 if fit_ready else 0)`
- 整体：`score = 0.4 * fit_ready占比 + 0.4 * 代表帧visible_ratio + 0.2 * 共识帧占比`

百分比 = `round(score*100)`；`≥66% 高`、`40%~65% 中`、`<40% 低`（绿/橙/红）。

---

## 已知限制

- 单张/单视角下 `width`（侧向深度）为比例启发式；多视角（视频多帧 / 图片目录 / 实时多拍）
  能显著增强约束、提升稳健性。
- 极端反光或遮挡的单帧可能 `fit_ready=false`，此时仍 best-effort 输出，请以边缘证据图判断可信度。
- 当前体积默认为像素域 `px³`；公制换算需提供尺度锚点（一条已知真实边长）或相机外参
  （第二阶段 `stage2`，见后端 `crystalvol/stage2.py`）。
- 实时模式需系统授予摄像头权限；SAM2 在 macOS 上以 CPU 运行，多拍时每张有数秒处理时间。
- Windows 兼容：工程已全面支持 Windows，使用 `pathlib` 处理路径分隔符、`platform` 模块做 OS 检测、
  `os.add_dll_directory()` 加载 MVS SDK DLL。OpenCV 通过 DirectShow/MSMF 后端访问摄像头，
  所有平台相关代码均有 `try/except` 保护。
