const state = {
  runId: null,
  tool: null,
  activeTool: null,
  imageWidth: 0,
  imageHeight: 0,
  inputUrl: null,
  opencvOutputs: [],
  taskRois: {},
  manualMode: false,
  manualRois: [],
  selectedRoiIndex: null,
  dragAction: null,
  dragStart: null,
  dragOriginal: null,
  apiKey: "",
  agentPlan: null,
  failedTools: new Set(),
};

const $ = (id) => document.getElementById(id);
const message = $("message");
const imageInput = $("imageInput");
const agentButton = $("agentButton");
const rerunButton = $("rerunButton");
const manualButton = $("manualButton");
const manualGuideButton = $("manualGuideButton");
const deleteManualButton = $("deleteManualButton");
const clearManualButton = $("clearManualButton");
const previewImage = $("previewImage");
const roiCanvas = $("roiCanvas");
const ctx = roiCanvas.getContext("2d");
const validateKeyButton = $("validateKeyButton");
const closeManualGuideButton = $("closeManualGuideButton");
const startManualFromGuideButton = $("startManualFromGuideButton");

validateKeyButton.addEventListener("click", async () => {
  const apiKey = $("gateApiKey").value.trim();
  if (!apiKey) {
    $("keyStatus").textContent = "请输入 DeepSeek API Key。";
    $("keyStatus").className = "key-status error";
    return;
  }
  validateKeyButton.disabled = true;
  validateKeyButton.textContent = "正在验证...";
  try {
    const data = await readJson(await fetch("/api/validate-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey }),
    }));
    state.apiKey = apiKey;
    $("keyStatus").textContent = `验证成功。可用模型：${(data.models || []).join("、") || "DeepSeek"}`;
    $("keyStatus").className = "key-status success";
    $("serverStatus").textContent = "DeepSeek Agent 已连接";
    agentButton.disabled = false;
    setTimeout(() => $("keyGate").classList.add("hidden"), 450);
  } catch (error) {
    $("keyStatus").textContent = error.message;
    $("keyStatus").className = "key-status error";
  } finally {
    validateKeyButton.disabled = false;
    validateKeyButton.textContent = "验证 API Key 并进入";
  }
});

document.querySelectorAll(".prompt-example").forEach((button) => {
  button.addEventListener("click", () => {
    message.value = button.dataset.message;
    resetSession(false);
    message.focus();
  });
});

imageInput.addEventListener("change", () => {
  $("fileName").textContent = imageInput.files[0]?.name || "点击选择 JPG / PNG / BMP / TIF";
  resetSession(false);
});

message.addEventListener("input", () => resetSession(false));

function resetSession(clearImage = true) {
  state.runId = null;
  state.tool = null;
  state.activeTool = null;
  state.inputUrl = null;
  state.opencvOutputs = [];
  state.taskRois = {};
  state.manualRois = [];
  state.selectedRoiIndex = null;
  state.dragAction = null;
  state.dragStart = null;
  state.dragOriginal = null;
  state.manualMode = false;
  state.agentPlan = null;
  state.failedTools = new Set();
  rerunButton.disabled = true;
  manualButton.disabled = true;
  manualGuideButton.disabled = true;
  deleteManualButton.disabled = true;
  clearManualButton.disabled = true;
  $("workflowTitle").textContent = "等待任务";
  $("workflowTimeline").innerHTML = '<div class="workflow-empty">运行后会在这里显示真实的意图识别、规划和工具调用记录。</div>';
  $("intentBadge").classList.add("hidden");
  $("resultTitle").textContent = "尚未评价";
  $("agentAnswer").classList.add("hidden");
  $("agentAnswer").textContent = "";
  $("resultContent").className = "result-placeholder";
  $("resultContent").textContent = "Agent 会在这里输出自然语言总结和光学指标。";
  $("qualityMessages").innerHTML = "";
  $("traceBox").classList.add("hidden");
  $("downloadResult").classList.add("hidden");
  $("opencvTabs").classList.add("hidden");
  $("chartGallery").classList.add("hidden");
  hideManualGuide();
  if (clearImage) {
    $("imageWrap").classList.add("hidden");
    $("emptyState").classList.remove("hidden");
    $("imageStage").classList.add("empty");
  }
  drawManualRois();
}

function setBusy(isBusy, text = "Agent 正在理解请求...") {
  $("progressBox").classList.toggle("hidden", !isBusy);
  $("progressText").textContent = text;
  agentButton.disabled = isBusy;
  rerunButton.disabled = isBusy || !manualRoisValid();
}

function showRunningWorkflow() {
  $("workflowTitle").textContent = "Agent 正在执行";
  $("workflowTimeline").innerHTML = [
    ["理解自然语言", "识别用户真正想评价的像质指标"],
    ["制定计划", "选择视觉定位工具与光学计算工具"],
    ["调用 OpenCV", "自动寻找并框选测量区域"],
    ["执行测量", "调用确定性光学算法计算指标"],
    ["整理回答", "校验结果并生成自然语言总结"],
  ].map(([title, text], index) => `
    <div class="workflow-step ${index === 0 ? "running" : "waiting"}">
      <span class="workflow-index">${index + 1}</span>
      <div><strong>${title}</strong><p>${text}</p></div>
    </div>
  `).join("");
}

async function readJson(response) {
  const data = await response.json();
  if (!response.ok || !data.success) {
    throw new Error(data.error || "操作失败");
  }
  return data;
}

agentButton.addEventListener("click", async () => {
  if (!message.value.trim()) {
    alert("请用一句自然语言告诉 Agent 想评价什么。");
    return;
  }
  if (!imageInput.files[0]) {
    alert("请先选择一张待评价图片。");
    return;
  }
  const body = new FormData();
  body.append("message", message.value.trim());
  body.append("image", imageInput.files[0]);
  body.append("api_key", state.apiKey);
  showRunningWorkflow();
  setBusy(true, "Agent 正在识别意图并规划工具...");
  try {
    const data = await readJson(await fetch("/api/agent", { method: "POST", body }));
    await acceptAgentResult(data);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});

rerunButton.addEventListener("click", async () => {
  if (!state.runId || !manualRoisValid()) return;
  setBusy(true, "Agent 正在使用人工修正 ROI 重新计算...");
  try {
    const data = await readJson(await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: state.runId,
        message: message.value.trim(),
        api_key: state.apiKey,
        tool: state.activeTool,
        manual_rois: state.manualRois.filter(Boolean),
      }),
    }));
    await acceptAgentResult(data);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});

async function acceptAgentResult(data) {
  state.runId = data.run_id;
  state.agentPlan = data.agent_plan || null;
  state.failedTools = new Set(
    (data.task_results || [])
      .filter((task) => !task.success)
      .map((task) => task.tool),
  );
  const requestedTools = data.intent.tools || [data.intent.tool];
  state.tool = requestedTools.length === 1 ? requestedTools[0] : null;
  state.imageHeight = data.image_shape[0];
  state.imageWidth = data.image_shape[1];
  state.inputUrl = data.input_url || state.inputUrl;
  state.taskRois = {};
  (data.task_results || []).forEach((task) => {
    if (task.success && task.roi_coordinates) {
      state.taskRois[task.tool] = task.roi_coordinates.map((roi) => [...roi]);
    }
  });
  if (!data.task_results && data.roi_coordinates) {
    state.taskRois[data.intent.tool] = data.roi_coordinates.map((roi) => [...roi]);
  }
  state.manualRois = [];
  state.selectedRoiIndex = null;
  state.manualMode = false;
  const opencvOutputs = data.opencv_outputs || [
    { tool: data.intent.tool, url: data.annotated_url, title: "OpenCV框选" },
  ];
  state.opencvOutputs = opencvOutputs;
  state.activeTool = opencvOutputs[0].tool;
  await showImage(`${opencvOutputs[0].url}?t=${Date.now()}`);
  $("viewerTitle").textContent = `${requestedTools.map(toolName).join(" + ")} · Agent OpenCV输出`;
  $("previewInfo").innerHTML = [
    `<span class="info-chip">Agent识别：${escapeHtml(requestedTools.map(toolName).join("、"))}</span>`,
    `<span class="info-chip">意图方法：${escapeHtml(data.intent.method)}</span>`,
    `<span class="info-chip">模型：${escapeHtml(data.intent.model || "DeepSeek")}</span>`,
    `<span class="info-chip">子任务：${requestedTools.length}</span>`,
    `<span class="info-chip">ROI：${data.roi_count}</span>`,
  ].join("");
  $("previewInfo").classList.remove("hidden");
  renderOpenCvTabs(opencvOutputs);
  updateManualControls();
  deleteManualButton.disabled = true;
  clearManualButton.disabled = true;
  rerunButton.disabled = true;
  renderWorkflow(data);
  renderResult(data);
}

function renderOpenCvTabs(outputs) {
  if (!outputs.length) {
    $("opencvTabs").classList.add("hidden");
    return;
  }
  $("opencvTabs").innerHTML = outputs.map((item, index) =>
    `<button class="output-tab ${index === 0 ? "active" : ""}" data-tool="${escapeHtml(item.tool)}" data-url="${escapeHtml(item.url)}">${escapeHtml(item.title || `${toolName(item.tool)} 框选图`)}</button>`
  ).join("");
  $("opencvTabs").classList.toggle("hidden", outputs.length <= 1);
  $("opencvTabs").querySelectorAll(".output-tab").forEach((button) => {
    button.addEventListener("click", async () => {
      $("opencvTabs").querySelectorAll(".output-tab").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      exitManualMode();
      state.activeTool = button.dataset.tool;
      updateManualControls();
      await showImage(`${button.dataset.url}?t=${Date.now()}`);
    });
  });
}

function renderWorkflow(data) {
  $("workflowTitle").textContent = "完整流程已执行";
  $("intentBadge").textContent = `${toolName(data.intent.tool)} · ${(data.intent.confidence * 100).toFixed(0)}%`;
  $("intentBadge").classList.remove("hidden");
  $("workflowTimeline").innerHTML = data.agent_trace.map((item, index) => {
    const detail = workflowDetail(item);
    return `<div class="workflow-step done">
      <span class="workflow-index">${index + 1}</span>
      <div>
        <strong>${escapeHtml(item.title || item.stage)}</strong>
        <p>${escapeHtml(detail)}</p>
      </div>
      <span class="workflow-check">完成</span>
    </div>`;
  }).join("");
}

function workflowDetail(item) {
  const detail = item.detail || {};
  if (item.stage === "intent_recognition") {
    return `${detail.message}；方式=${detail.method}，置信度=${Math.round(detail.confidence * 100)}%`;
  }
  if (item.stage === "ai_planning") {
    return `模型生成 ${detail.tasks?.length || 0} 个子任务：${(detail.tasks || []).map((task) => toolName(task.tool)).join("、")}`;
  }
  if (item.stage === "task_planning") {
    return `计划调用 ${detail.vision_tool}，再调用 ${detail.measurement_tool}`;
  }
  if (item.stage === "image_loading") {
    return `图像尺寸 ${detail.shape?.join(" × ")}`;
  }
  if (item.stage === "automatic_roi_detection") {
    return `调用 ${detail.tool_call}，自动输出 ${detail.roi_count} 个 ROI`;
  }
  if (item.stage === "manual_roi_validation") {
    return `采用人工修正区域，共 ${detail.roi_count} 个 ROI`;
  }
  if (item.stage === "tool_execution") {
    return `调用 ${detail.tool_call} 完成指标计算`;
  }
  if (item.stage === "result_validation") {
    return detail.assessment || "测量结果结构完整";
  }
  if (item.stage === "result_synthesis") {
    return "将工具输出整理为自然语言回答和结构化 JSON";
  }
  if (item.stage === "ai_synthesis") {
    return "模型根据真实工具数据生成最终回答";
  }
  if (item.stage === "task_failed") {
    return detail.error || "该子任务未完成";
  }
  return item.stage;
}

const editableTools = new Set(["sfr", "color_check", "step_chart"]);

function expectedManualCount(tool) {
  return { color_check: 24, step_chart: 20 }[tool] || null;
}

function activeManualRoiCount() {
  return state.manualRois.filter(Boolean).length;
}

function manualRoisValid() {
  if (!editableTools.has(state.activeTool)) return false;
  const count = activeManualRoiCount();
  const expected = expectedManualCount(state.activeTool);
  return expected === null ? count >= 1 : count === expected;
}

function updateManualControls() {
  const editable = editableTools.has(state.activeTool);
  manualButton.disabled = !editable;
  manualGuideButton.disabled = !editable;
  manualButton.textContent = state.manualMode
    ? "退出人工修正"
    : `人工修正 ${editable ? toolName(state.activeTool) : ""} ROI`;
  updateManualCount();
}

function updateManualCount() {
  const count = activeManualRoiCount();
  const expected = expectedManualCount(state.activeTool);
  $("manualCount").textContent = expected
    ? `当前 ${count}/${expected} 个框。`
    : `当前 ${count} 个框，可继续添加。`;
  deleteManualButton.disabled = state.selectedRoiIndex === null;
  clearManualButton.disabled = count === 0;
  rerunButton.disabled = !manualRoisValid();
}

function exitManualMode() {
  state.manualMode = false;
  state.manualRois = [];
  state.selectedRoiIndex = null;
  state.dragAction = null;
  state.dragStart = null;
  state.dragOriginal = null;
  roiCanvas.classList.remove("drawing");
  $("manualHint").classList.add("hidden");
  drawManualRois();
  updateManualControls();
}

manualButton.addEventListener("click", async () => {
  if (!editableTools.has(state.activeTool)) return;
  if (state.manualMode) {
    const output = state.opencvOutputs.find((item) => item.tool === state.activeTool);
    exitManualMode();
    if (output) await showImage(`${output.url}?t=${Date.now()}`);
    return;
  }
  state.manualMode = true;
  state.manualRois = (state.taskRois[state.activeTool] || []).map((roi) => [...roi]);
  state.selectedRoiIndex = null;
  roiCanvas.classList.add("drawing");
  $("manualHint").classList.remove("hidden");
  if (state.inputUrl) await showImage(`${state.inputUrl}?t=${Date.now()}`);
  updateManualControls();
  drawManualRois();
  showManualGuide(state.activeTool);
});

manualGuideButton.addEventListener("click", () => {
  if (!editableTools.has(state.activeTool)) return;
  showManualGuide(state.activeTool);
});

closeManualGuideButton.addEventListener("click", hideManualGuide);
startManualFromGuideButton.addEventListener("click", hideManualGuide);
$("manualGuideModal").addEventListener("click", (event) => {
  if (event.target.id === "manualGuideModal") hideManualGuide();
});

deleteManualButton.addEventListener("click", () => {
  if (state.selectedRoiIndex === null) return;
  state.manualRois[state.selectedRoiIndex] = null;
  state.selectedRoiIndex = null;
  updateManualCount();
  drawManualRois();
});

clearManualButton.addEventListener("click", () => {
  state.manualRois = [];
  state.selectedRoiIndex = null;
  updateManualCount();
  drawManualRois();
});

roiCanvas.addEventListener("pointerdown", (event) => {
  if (!state.manualMode) return;
  roiCanvas.setPointerCapture(event.pointerId);
  const point = canvasToImagePoint(event);
  const hitIndex = hitTestRoi(point);
  state.dragStart = point;
  if (hitIndex !== null) {
    state.selectedRoiIndex = hitIndex;
    state.dragAction = "move";
    state.dragOriginal = [...state.manualRois[hitIndex]];
  } else {
    const emptyIndex = state.manualRois.findIndex((roi) => roi === null);
    state.selectedRoiIndex = emptyIndex >= 0 ? emptyIndex : state.manualRois.length;
    state.dragAction = "draw";
    state.manualRois[state.selectedRoiIndex] = [point.x, point.y, 0, 0];
  }
  updateManualCount();
  drawManualRois();
});

roiCanvas.addEventListener("pointermove", (event) => {
  if (!state.manualMode || !state.dragStart || state.selectedRoiIndex === null) return;
  const current = canvasToImagePoint(event);
  if (state.dragAction === "draw") {
    state.manualRois[state.selectedRoiIndex] = [
      Math.round(Math.min(state.dragStart.x, current.x)),
      Math.round(Math.min(state.dragStart.y, current.y)),
      Math.round(Math.abs(current.x - state.dragStart.x)),
      Math.round(Math.abs(current.y - state.dragStart.y)),
    ];
  } else if (state.dragAction === "move") {
    const [x, y, width, height] = state.dragOriginal;
    const nextX = Math.max(
      0,
      Math.min(state.imageWidth - width, x + current.x - state.dragStart.x),
    );
    const nextY = Math.max(
      0,
      Math.min(state.imageHeight - height, y + current.y - state.dragStart.y),
    );
    state.manualRois[state.selectedRoiIndex] = [
      Math.round(nextX),
      Math.round(nextY),
      width,
      height,
    ];
  }
  drawManualRois();
});

roiCanvas.addEventListener("pointerup", () => {
  if (!state.manualMode || !state.dragStart || state.selectedRoiIndex === null) return;
  const roi = state.manualRois[state.selectedRoiIndex];
  if (state.dragAction === "draw" && (!roi || roi[2] < 12 || roi[3] < 12)) {
    state.manualRois[state.selectedRoiIndex] = null;
    state.selectedRoiIndex = null;
    alert("框选区域太小，请重新绘制。");
  }
  state.dragAction = null;
  state.dragStart = null;
  state.dragOriginal = null;
  updateManualCount();
  drawManualRois();
});

function canvasPoint(event) {
  const rect = roiCanvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(rect.width, event.clientX - rect.left)),
    y: Math.max(0, Math.min(rect.height, event.clientY - rect.top)),
  };
}

function canvasToImagePoint(event) {
  const point = canvasPoint(event);
  const rect = roiCanvas.getBoundingClientRect();
  return {
    x: point.x / rect.width * state.imageWidth,
    y: point.y / rect.height * state.imageHeight,
  };
}

function hitTestRoi(point) {
  for (let index = state.manualRois.length - 1; index >= 0; index -= 1) {
    const roi = state.manualRois[index];
    if (!roi) continue;
    const [x, y, width, height] = roi;
    if (point.x >= x && point.x <= x + width && point.y >= y && point.y <= y + height) {
      return index;
    }
  }
  return null;
}

function imageToDisplayRoi(roi) {
  const rect = roiCanvas.getBoundingClientRect();
  return [
    roi[0] / state.imageWidth * rect.width,
    roi[1] / state.imageHeight * rect.height,
    roi[2] / state.imageWidth * rect.width,
    roi[3] / state.imageHeight * rect.height,
  ];
}

function resizeCanvas() {
  const rect = previewImage.getBoundingClientRect();
  roiCanvas.width = Math.max(1, Math.round(rect.width));
  roiCanvas.height = Math.max(1, Math.round(rect.height));
  drawManualRois();
}

function drawManualRois() {
  ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  state.manualRois.forEach((imageRoi, index) => {
    if (!imageRoi) return;
    const roi = imageToDisplayRoi(imageRoi);
    const selected = index === state.selectedRoiIndex;
    ctx.save();
    ctx.strokeStyle = selected ? "#ffe100" : "#ff8a00";
    ctx.fillStyle = selected
      ? "rgba(255, 225, 0, 0.18)"
      : "rgba(255, 138, 0, 0.13)";
    ctx.lineWidth = selected ? 4 : 3;
    ctx.setLineDash([9, 6]);
    ctx.fillRect(...roi);
    ctx.strokeRect(...roi);
    ctx.setLineDash([]);
    ctx.fillStyle = selected ? "#806d00" : "#9b4f00";
    ctx.font = "bold 16px Microsoft YaHei";
    ctx.fillText(String(index + 1), roi[0] + 5, Math.max(18, roi[1] - 5));
    ctx.restore();
  });
}

function activeTask() {
  return (state.agentPlan?.tasks || []).find((task) => task.tool === state.activeTool) || null;
}

function activeTaskParameters() {
  return activeTask()?.parameters || {};
}

function showManualGuide(tool = state.activeTool) {
  const guide = manualGuideHtml(tool);
  $("manualGuideTitle").textContent = guide.title;
  $("manualGuideContent").innerHTML = guide.html;
  $("manualGuideModal").classList.remove("hidden");
  $("manualGuideModal").setAttribute("aria-hidden", "false");
}

function hideManualGuide() {
  $("manualGuideModal").classList.add("hidden");
  $("manualGuideModal").setAttribute("aria-hidden", "true");
}

function manualGuideHtml(tool) {
  if (tool === "sfr") return sfrManualGuide();
  if (tool === "color_check") return colorManualGuide();
  if (tool === "step_chart") return grayManualGuide();
  return {
    title: "人工 ROI 框选说明",
    html: "<p>当前工具暂时没有专门的人工框选说明。</p>",
  };
}

function sfrManualGuide() {
  const params = activeTaskParameters();
  const mtfDirections = params.mtf_directions || [];
  const fieldPosition = params.field_position === "edge" ? "画面边缘/四周" : "画面中心";
  const directionText = mtfDirections.length
    ? mtfDirections.map((item) => item === "horizontal" ? "水平 MTF" : "竖直 MTF").join("、")
    : "水平 MTF 和竖直 MTF";
  return {
    title: "MTF / SFR 应该怎么手动画框",
    html: `
      <div class="guide-grid">
        <div>
          <p class="guide-lead">当前任务：测 ${escapeHtml(fieldPosition)} 的 ${escapeHtml(directionText)}。</p>
          <ol>
            <li>先找测试卡上的斜正方形，不要框文字、线对、圆形图案或整张测试卡。</li>
            <li>真正送入 SFR 算法的是跨过一条斜边的窄矩形，不是整个斜正方形。</li>
            <li>如果要测 <strong>水平 MTF</strong>，请框 <strong>近竖直的斜边</strong>；如果要测 <strong>竖直 MTF</strong>，请框 <strong>近水平的斜边</strong>。</li>
            <li>框要同时包含边两侧的亮区和暗区，边缘尽量穿过框的中间；不要把框画到图案内部纹理或边角外面。</li>
            <li>边缘 MTF 可以画多个斜正方形的边，程序会分别计算每个 ROI 的 ESF 和 MTF 曲线。</li>
          </ol>
        </div>
        <div class="guide-figure">
          ${sfrGuideSvg()}
          <p>绿色窄框才是 ROI；青色外框只是帮助你确认找到了斜正方形。</p>
        </div>
      </div>
      <div class="guide-warning">
        自动识别失败常见原因：斜正方形太小、曝光反光、边缘过模糊、对比度太低、画面中有太多类似方块，或用户要求边缘 MTF 但边缘斜方块被裁掉。
      </div>
    `,
  };
}

function colorManualGuide() {
  return {
    title: "色彩还原度应该怎么手动画框",
    html: `
      <div class="guide-grid">
        <div>
          <p class="guide-lead">ColorChecker 需要 24 个 ROI，按 6 列 × 4 行的色块中心依次框选。</p>
          <ol>
            <li>从左到右、从上到下依次画 1 到 24 号框。</li>
            <li>每个框放在色块中心，尽量避开色块边界、阴影、反光和脏点。</li>
            <li>框的大小可以略小于色块，不要跨到相邻色块。</li>
            <li>如果色卡有明显透视，仍按画面里看到的色块中心逐个框。</li>
          </ol>
        </div>
        <div class="guide-figure">
          ${colorGuideSvg()}
          <p>绿色小框放在每个色块中心，编号顺序会影响 ΔE 计算。</p>
        </div>
      </div>
      <div class="guide-warning">
        自动识别失败常见原因：色卡占画面太小、局部过曝或阴影、色块被遮挡、色卡不是标准 6×4 排列，或背景中存在大量相似彩色方块。
      </div>
    `,
  };
}

function grayManualGuide() {
  return {
    title: "20级灰阶应该怎么手动画框",
    html: `
      <div class="guide-grid">
        <div>
          <p class="guide-lead">20 级灰阶需要 20 个 ROI，框在每个灰度方块中心。</p>
          <ol>
            <li>按测试卡印刷的编号 1 到 20 依次框选，尤其不要把左右对称位置的顺序画反。</li>
            <li>每个框只覆盖灰阶方块中心，避开数字、边界线、彩色色块和背景纸面。</li>
            <li>很黑的 1、2、3 级容易连在一起，请按方块质心分别画三个框。</li>
            <li>很亮的 18、19、20 级容易接近背景，也要框在实际灰块中心而不是白纸背景上。</li>
          </ol>
        </div>
        <div class="guide-figure">
          ${grayGuideSvg()}
          <p>绿色框沿环形灰阶分布，红点表示采样质心。</p>
        </div>
      </div>
      <div class="guide-warning">
        自动识别失败常见原因：亮灰阶和背景太接近、暗灰阶粘连、拍摄透视明显、局部模糊、灰阶环不完整，或画面中其它灰色方块干扰。
      </div>
    `,
  };
}

function sfrGuideSvg() {
  return `
    <svg class="guide-svg" viewBox="0 0 220 160" aria-label="MTF框选示意图">
      <rect width="220" height="160" rx="12" fill="#f6f4ee"/>
      <g transform="translate(110 80) rotate(-7)">
        <rect x="-48" y="-48" width="96" height="96" fill="#222"/>
        <rect x="-48" y="-48" width="96" height="96" fill="none" stroke="#18bfc0" stroke-width="3"/>
        <rect x="31" y="-34" width="22" height="68" fill="none" stroke="#13d44b" stroke-width="4"/>
        <rect x="-34" y="-53" width="68" height="22" fill="none" stroke="#13d44b" stroke-width="4"/>
      </g>
      <text x="145" y="78" fill="#0b6b4f" font-size="12" font-weight="700">水平MTF</text>
      <text x="130" y="94" fill="#0b6b4f" font-size="11">框竖直边</text>
      <text x="58" y="24" fill="#0b6b4f" font-size="12" font-weight="700">竖直MTF：框水平边</text>
    </svg>
  `;
}

function colorGuideSvg() {
  const colors = ["#7d5a46", "#c28d75", "#6d7fa3", "#6d623b", "#8aa64d", "#5c6fb0", "#c15048", "#4aa0b5", "#c34e82", "#dfc447", "#5b8f54", "#755ba0", "#ffffff", "#dedede", "#bcbcbc", "#999999", "#777777", "#555555", "#333333", "#222222", "#111111", "#3aa0d8", "#f08b2b", "#d83b3b"];
  return `
    <svg class="guide-svg" viewBox="0 0 260 170" aria-label="色卡框选示意图">
      <rect width="260" height="170" rx="12" fill="#f6f4ee"/>
      ${colors.map((color, index) => {
        const col = index % 6;
        const row = Math.floor(index / 6);
        const x = 22 + col * 36;
        const y = 20 + row * 34;
        return `<g>
          <rect x="${x}" y="${y}" width="30" height="28" fill="${color}" stroke="#c9c3b6"/>
          <rect x="${x + 8}" y="${y + 7}" width="14" height="14" fill="none" stroke="#13d44b" stroke-width="2"/>
          <text x="${x + 2}" y="${y + 10}" fill="#0b6b4f" font-size="8" font-weight="700">${index + 1}</text>
        </g>`;
      }).join("")}
    </svg>
  `;
}

function grayGuideSvg() {
  const centers = [
    [130, 133], [116, 133], [158, 127], [88, 127], [176, 113],
    [70, 113], [188, 94], [58, 94], [194, 74], [52, 74],
    [194, 54], [52, 54], [188, 35], [58, 35], [176, 22],
    [70, 22], [158, 15], [88, 15], [137, 11], [112, 11],
  ];
  return `
    <svg class="guide-svg" viewBox="0 0 250 155" aria-label="灰阶框选示意图">
      <rect width="250" height="155" rx="12" fill="#f6f4ee"/>
      <path d="M42,132 C50,36 82,8 125,8 C168,8 200,36 208,132" fill="none" stroke="#b8b0a2" stroke-width="2" stroke-dasharray="5 5"/>
      ${centers.map(([x, y], index) => {
        const shade = Math.round(35 + index * 9);
        return `<g>
          <rect x="${x - 8}" y="${y - 8}" width="16" height="16" fill="rgb(${shade},${shade},${shade})" stroke="#888"/>
          <rect x="${x - 11}" y="${y - 11}" width="22" height="22" fill="none" stroke="#13d44b" stroke-width="2"/>
          <circle cx="${x}" cy="${y}" r="2" fill="#e53935"/>
          <text x="${x + 8}" y="${y - 8}" fill="#0b6b4f" font-size="8" font-weight="700">${index + 1}</text>
        </g>`;
      }).join("")}
    </svg>
  `;
}

async function showImage(url) {
  await new Promise((resolve, reject) => {
    previewImage.onload = resolve;
    previewImage.onerror = reject;
    previewImage.src = url;
  });
  $("imageWrap").classList.remove("hidden");
  $("emptyState").classList.add("hidden");
  $("imageStage").classList.remove("empty");
  resizeCanvas();
}

function renderResult(data) {
  const tools = data.intent.tools || [data.intent.tool];
  $("resultTitle").textContent = `${tools.map(toolName).join(" + ")} · Agent 已回答`;
  $("agentAnswer").textContent = data.agent_response;
  $("agentAnswer").classList.remove("hidden");
  $("qualityMessages").innerHTML = (data.quality_messages || []).map((item) =>
    `<div class="message ${escapeHtml(item.level)}">${escapeHtml(item.text)}</div>`
  ).join("");
  $("resultContent").className = "";
  $("resultContent").innerHTML = resultHtml(data);
  renderCharts(data.chart_outputs || []);
  $("downloadResult").href = data.result_url;
  $("downloadResult").classList.remove("hidden");
  $("traceContent").textContent = [
    "Agent Plan:",
    JSON.stringify(data.agent_plan, null, 2),
    "",
    "Tool Trace:",
    JSON.stringify(data.agent_trace, null, 2),
    "",
    "Algorithm log:",
    data.algorithm_log || "(无)",
  ].join("\n");
  $("traceBox").classList.remove("hidden");
}

function resultHtml(data) {
  if (data.task_results) {
    return data.task_results.map((task) => {
      if (!task.success) {
        return `<section class="task-result"><h3>${escapeHtml(toolName(task.tool))}</h3><div class="message warning">${escapeHtml(task.error)}</div></section>`;
      }
      return `<section class="task-result"><h3>${escapeHtml(toolName(task.tool))}</h3>${singleResultHtml(task.tool, task.result)}</section>`;
    }).join("");
  }
  return singleResultHtml(data.intent.tool, data.result);
}

function singleResultHtml(tool, result) {
  if (tool === "sfr") {
    const rows = [];
    result.measurements.forEach((measurement) => {
      Object.entries(measurement.channels).forEach(([channel, values]) => {
        rows.push(`<tr>
          <td>${measurement.roi}</td>
          <td>${escapeHtml(channel)}</td>
          <td>${formatNumber(values.mtf50_cy_per_pixel, 4)}</td>
          <td>${formatNumber(values.mtf30_cy_per_pixel, 4)}</td>
        </tr>`);
      });
    });
    return `<table class="result-table">
      <thead><tr><th>ROI</th><th>通道</th><th>MTF50 (cy/px)</th><th>MTF30 (cy/px)</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table><div class="assessment">${escapeHtml(result.assessment)}</div>`;
  }
  if (tool === "color_check") {
    return metricGrid([
      ["平均 ΔE", result.mean_delta_e],
      ["最大 ΔE", result.max_delta_e],
      ["平均 ΔC", result.mean_delta_c],
      ["最大 ΔC", result.max_delta_c],
      ["平均饱和度", `${formatNumber(result.mean_saturation_percent, 1)}%`],
    ]) + `<div class="assessment">${escapeHtml(result.assessment)}</div>`;
  }
  if (tool === "step_chart") {
    return metricGrid([
      ["可分辨相邻过渡", `${result.detected_transitions} / 19`],
      ["归一化对比度", formatNumber(result.contrast_normalized, 4)],
    ]) + `<div class="assessment">${escapeHtml(result.assessment)}</div>`;
  }
  return `<pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
}

function renderCharts(charts) {
  if (!charts.length) {
    $("chartGallery").classList.add("hidden");
    $("chartGallery").innerHTML = "";
    return;
  }
  $("chartGallery").innerHTML = charts.map((chart) =>
    `<div class="chart-card"><strong>${escapeHtml(chart.title || chart.type)}</strong><img src="${escapeHtml(chart.url)}" alt="${escapeHtml(chart.title || "评价曲线")}"></div>`
  ).join("");
  $("chartGallery").classList.remove("hidden");
}

function metricGrid(items) {
  return `<div class="metric-grid">${items.map(([label, value]) =>
    `<div class="metric"><span>${escapeHtml(label)}</span><strong>${typeof value === "number" ? formatNumber(value, 3) : escapeHtml(String(value))}</strong></div>`
  ).join("")}</div>`;
}

function showError(text) {
  $("qualityMessages").innerHTML = `<div class="message warning">${escapeHtml(text)}</div>`;
  $("resultTitle").textContent = "Agent 未完成任务";
  $("workflowTitle").textContent = "执行中断";
}

function toolName(tool) {
  return {
    sfr: "MTF / SFR",
    color_check: "色彩还原度",
    step_chart: "20级灰阶",
    white_balance: "白平衡",
    distortion: "畸变",
    dark_field: "暗场",
  }[tool] || tool;
}

function formatNumber(value, digits) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "未得到";
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.addEventListener("resize", resizeCanvas);

window.addEventListener("DOMContentLoaded", async () => {
  const runId = new URLSearchParams(window.location.search).get("run");
  if (!runId) return;
  setBusy(true, "正在恢复 Agent 会话...");
  try {
    const data = await readJson(await fetch(`/api/session/${encodeURIComponent(runId)}`));
    message.value = data.request || "";
    await acceptAgentResult(data);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});
