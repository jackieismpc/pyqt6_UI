/* 透明晶体体积估计 · Web —— 前端逻辑（无构建，原生 JS） */

"use strict";

// ── 状态 ──────────────────────────────────────────────────────────────────
let currentMode = "video";
let currentTaskId = null;
let pollTimer = null;
let result = null;          // 最近一次任务/拍摄的结果 payload
let frameIndex = 0;
let realtimeMode = false;
let realtimeSession = false;

// ── DOM ───────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const modeTabs = document.querySelectorAll(".mode-tab");
const modePanes = {
  video: $("pane-video"),
  zip: $("pane-zip"),
  path: $("pane-path"),
  realtime: $("pane-realtime"),
};

function showPlaceholder(panel, text) {
  const body = panel.querySelector(".panel-body");
  const img = body.querySelector("img");
  const ph = body.querySelector(".placeholder");
  img.classList.remove("visible", "rt-live");
  img.removeAttribute("src");
  ph.textContent = text;
  ph.style.display = "";
}

function showImage(panel, url) {
  const body = panel.querySelector(".panel-body");
  const img = body.querySelector("img");
  const ph = body.querySelector(".placeholder");
  ph.style.display = "none";
  img.src = url;
  img.classList.add("visible");
}

// ── 模式切换 ──────────────────────────────────────────────────────────────
function switchMode(mode) {
  currentMode = mode;
  modeTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  Object.entries(modePanes).forEach(([key, pane]) =>
    pane.classList.toggle("active", key === mode)
  );
  if (mode === "realtime") {
    loadCameras();
    startRealtimePreview();
    realtimeMode = true;
  } else {
    stopRealtimePreview();
    realtimeMode = false;
  }
}

modeTabs.forEach((tab) =>
  tab.addEventListener("click", () => switchMode(tab.dataset.mode))
);

// ── 任务创建与轮询 ────────────────────────────────────────────────────────
async function startTask() {
  $("btn-run").disabled = true;
  $("btn-cancel").classList.remove("hidden");
  showProgress("创建任务…", 0);

  const numFrames = parseInt($("num-frames").value, 10) || 7;
  const form = new FormData();
  form.append("num_frames", numFrames);

  if (currentMode === "video") {
    const file = $("video-file").files[0];
    if (!file) { alert("请先选择视频文件"); $("btn-run").disabled = false; return; }
    form.append("input_type", "video");
    form.append("file", file);
  } else if (currentMode === "zip") {
    const file = $("zip-file").files[0];
    if (!file) { alert("请先选择 zip 压缩包"); $("btn-run").disabled = false; return; }
    form.append("input_type", "zip");
    form.append("file", file);
  } else if (currentMode === "path") {
    const p = $("server-path").value.trim();
    if (!p) { alert("请填写服务器本地路径"); $("btn-run").disabled = false; return; }
    form.append("input_type", "path");
    form.append("path", p);
  } else {
    return; // 实时模式用拍摄按钮
  }

  try {
    const resp = await fetch("/api/tasks", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) { throw new Error(data.detail || "创建任务失败"); }
    currentTaskId = data.task.id;
    pollTimer = setInterval(pollTask, 1000);
    pollTask();
  } catch (err) {
    showProgress(`失败：${err.message}`, 0);
    $("btn-run").disabled = false;
    $("btn-cancel").classList.add("hidden");
  }
}

async function pollTask() {
  if (!currentTaskId) return;
  try {
    const resp = await fetch(`/api/tasks/${currentTaskId}`);
    if (!resp.ok) throw new Error("获取任务状态失败");
    const data = await resp.json();
    const status = data.status;

    if (status === "running" || status === "queued") {
      const label = status === "queued" ? "排队中…" : "处理中…";
      showProgress(`${label} ${data.progress} 帧`, data.progress > 0 ? 50 : 5);
      return;
    }

    clearInterval(pollTimer);
    pollTimer = null;
    $("btn-run").disabled = false;
    $("btn-cancel").classList.add("hidden");

    if (status === "done") {
      showProgress("完成", 100);
      renderResult(data.result);
    } else if (status === "cancelled") {
      showProgress("已取消", 0);
    } else {
      showProgress(`失败：${data.error || "未知错误"}`, 0);
    }
  } catch (err) {
    showProgress(`轮询错误：${err.message}`, 0);
  }
}

async function cancelTask() {
  if (!currentTaskId) return;
  await fetch(`/api/tasks/${currentTaskId}/cancel`, { method: "POST" });
}

// ── 渲染结果 ──────────────────────────────────────────────────────────────
function renderResult(res) {
  result = res;
  frameIndex = 0;
  // 默认展示代表帧
  res.frames.forEach((fr, i) => {
    if (fr.name === res.representative_frame) frameIndex = i;
  });
  populateFrameSelect();
  refreshPanels();
  updateResultBar(res);
}

function populateFrameSelect() {
  const sel = $("frame-select");
  sel.innerHTML = "";
  result.frames.forEach((fr, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = fr.name + (fr.name === result.representative_frame ? "（代表帧）" : "");
    sel.appendChild(opt);
  });
  sel.value = frameIndex;
}

function refreshPanels() {
  if (!result || !result.frames.length) return;
  const fr = result.frames[frameIndex];

  // 左侧：原始输入（实时模式保留摄像头预览）
  if (!realtimeMode) {
    if (fr.images.raw) showImage($("panel-raw"), fr.images.raw);
    else showPlaceholder($("panel-raw"), "（无图像）");
  }

  // 右侧：几何模型（汇总单图）
  if (result.geometry_preview) showImage($("panel-geometry"), result.geometry_preview);
  else showPlaceholder($("panel-geometry"), "（无几何预览）");

  // 中间：预处理产物，跟随下拉框
  const key = $("preprocess-select").value;
  const url = fr.images[key];
  if (url) showImage($("panel-preprocess"), url);
  else showPlaceholder($("panel-preprocess"), `（无 ${key} 产物）`);
}

function updateResultBar(res) {
  const geo = res.aggregate_geometry || {};

  // 体积
  if (res.metric && res.metric.volume != null) {
    const unit = res.metric.unit || "cm³";
    $("volume-label").textContent = `${Number(res.metric.volume).toFixed(1)} ${unit}`;
    const dims = res.metric.dimensions_cm || {};
    const fmt = (v) => (v == null ? 0 : Number(v).toFixed(1));
    $("dims-label").textContent =
      `长${fmt(dims.length)} 宽${fmt(dims.width)} 体高${fmt(dims.body_height)} 锥高${fmt(dims.pyramid_height)} cm`;
  } else {
    $("volume-label").textContent = `${Number(res.aggregate_volume_px3 || 0).toFixed(0)} px³`;
    $("dims-label").textContent =
      `长${Number(geo.length_px || 0).toFixed(0)} × 宽${Number(geo.width_px || 0).toFixed(0)} px`;
  }

  // 不确定性 / 深度来源
  const unc = geo.volume_uncertainty_pct;
  const depthSrc = geo.depth_source || "";
  if (unc != null) {
    let note = `±${Number(unc)}%`;
    if (depthSrc === "cross_frame_projection_range") note += " · 多视角约束";
    else if (depthSrc === "adaptive_single_view_shape_prior") note += " · 先验深度";
    $("volume-sub").textContent = note;
  } else {
    $("volume-sub").textContent = "";
  }

  // 置信度（用代表帧的单帧置信度 + fit_ready 汇总）
  const fr = res.frames[frameIndex] || res.frames[0];
  const conf = Math.round(100 * (0.6 * (fr.visible_ratio || 0) + (fr.fit_ready ? 0.4 : 0)));
  const label = conf >= 66 ? "高" : conf >= 40 ? "中" : "低";
  const color = label === "高" ? getComputedStyle(document.documentElement).getPropertyValue("--green").trim()
    : label === "中" ? getComputedStyle(document.documentElement).getPropertyValue("--orange").trim()
    : getComputedStyle(document.documentElement).getPropertyValue("--red").trim();
  const confEl = $("conf-label");
  confEl.textContent = `● ${label} · ${conf}%`;
  confEl.style.color = color;
  confEl.title = `fit_ready ${res.fit_ready_count}/${res.frame_count} · 代表帧 ${res.representative_frame}`;
}

// ── 帧切换 ────────────────────────────────────────────────────────────────
$("frame-select").addEventListener("change", (e) => {
  frameIndex = Number(e.target.value);
  refreshPanels();
  updateResultBar(result);
});
$("btn-prev").addEventListener("click", () => {
  if (!result) return;
  frameIndex = Math.max(0, frameIndex - 1);
  $("frame-select").value = frameIndex;
  refreshPanels();
  updateResultBar(result);
});
$("btn-next").addEventListener("click", () => {
  if (!result) return;
  frameIndex = Math.min(result.frames.length - 1, frameIndex + 1);
  $("frame-select").value = frameIndex;
  refreshPanels();
  updateResultBar(result);
});

// ── 实时摄像头 ────────────────────────────────────────────────────────────
let rtStreamActive = false;

async function loadCameras() {
  try {
    const resp = await fetch("/api/cameras/list");
    const data = await resp.json();
    const sel = $("camera-select");
    sel.innerHTML = "";
    data.cameras.forEach((cam) => {
      const opt = document.createElement("option");
      opt.value = cam.id;
      opt.textContent = cam.label;
      sel.appendChild(opt);
    });
    if (!data.cameras.length) {
      const opt = document.createElement("option");
      opt.value = "0";
      opt.textContent = "未检测到摄像头";
      sel.appendChild(opt);
    }
  } catch (err) {
    console.warn("摄像头列表加载失败", err);
  }
}

function startRealtimePreview() {
  if (rtStreamActive) return;
  rtStreamActive = true;
  const img = $("img-raw");
  img.classList.add("rt-live");
  img.src = "/api/cameras/stream.mjpg";
  $("panel-raw").querySelector(".placeholder").style.display = "none";
}

function stopRealtimePreview() {
  rtStreamActive = false;
  const img = $("img-raw");
  img.classList.remove("rt-live");
  img.removeAttribute("src");
  if (!realtimeMode) showPlaceholder($("panel-raw"), "（无图像）");
}

$("btn-capture").addEventListener("click", async () => {
  $("btn-capture").disabled = true;
  try {
    const resp = await fetch("/api/cameras/capture", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "拍摄失败");
    realtimeSession = true;
    $("rt-count").textContent = `已拍 ${data.count} 张`;
    renderResult(data.result);
  } catch (err) {
    alert(`拍摄失败：${err.message}`);
  } finally {
    $("btn-capture").disabled = false;
  }
});

$("btn-rt-end").addEventListener("click", async () => {
  await fetch("/api/cameras/end", { method: "POST" });
  realtimeSession = false;
  $("rt-count").textContent = "已拍 0 张";
  $("frame-select").innerHTML = "";
  showPlaceholder($("panel-geometry"), "（无几何预览）");
});

// ── 进度条 ────────────────────────────────────────────────────────────────
function showProgress(text, pct) {
  const row = $("progress-row");
  row.classList.remove("hidden");
  $("progress-fill").style.width = `${pct}%`;
  $("progress-text").textContent = text;
}

// ── 事件绑定 ──────────────────────────────────────────────────────────────
$("btn-run").addEventListener("click", startTask);
$("btn-cancel").addEventListener("click", cancelTask);
$("preprocess-select").addEventListener("change", refreshPanels);

// 初始化
showPlaceholder($("panel-raw"), "（无图像）");
showPlaceholder($("panel-preprocess"), "（无图像）");
showPlaceholder($("panel-geometry"), "（无图像）");
