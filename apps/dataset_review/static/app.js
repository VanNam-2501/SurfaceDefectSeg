const $ = (id) => document.getElementById(id);

const decisionLabels = {
  approved: "Duyệt — giữ nguyên",
  acceptable_mark: "Dấu chấp nhận được (Good)",
  relabel_good: "Đổi nhãn thành Good",
  relabel_defect: "Đổi nhãn thành Defect",
  fix_mask: "Cần sửa / đã sửa mask",
  uncertain: "Chưa chắc — cần hội chẩn",
  exclude: "Loại khỏi dataset sạch",
};

const tagLabels = {
  hidden_defect: "Lỗi ẩn trong Good",
  false_alarm: "False alarm",
  mask_misaligned: "Mask lệch",
  mask_incomplete: "Mask thiếu",
  border_issue: "Lỗi sát biên",
  annotation_ambiguous: "Nhãn mơ hồ",
  duplicate: "Trùng lặp",
  hard_negative: "Hard negative",
  lighting: "Ánh sáng / phản xạ",
  texture: "Texture bình thường",
  model_disagreement: "Model bất đồng",
  model_mask_accepted: "Dùng mask model",
};

const candidateLabels = {
  false_positive: "FP",
  high_score_good: "Good score cao",
  false_negative: "FN",
  zero_overlap: "Không chồng GT",
  model_disagreement: "Model bất đồng",
};

const state = {
  bootstrap: null,
  items: [],
  total: 0,
  offset: 0,
  limit: 100,
  currentIndex: -1,
  details: null,
  selected: new Set(),
  loading: false,
  saving: false,
  filterTimer: null,
};

const editor = {
  open: false,
  original: new Image(),
  view: $("maskCanvas"),
  mask: document.createElement("canvas"),
  mode: "draw",
  brush: 24,
  zoom: .75,
  drawing: false,
  last: null,
  history: [],
  historyIndex: -1,
  sourceModel: null,
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(data.detail || data || `HTTP ${response.status}`);
  return data;
}

function toast(message, type = "ok") {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show ${type === "error" ? "error" : ""}`;
  clearTimeout(node.timer);
  node.timer = setTimeout(() => node.className = "toast", 3200);
}

function setSaveState(message) { $("saveState").textContent = message; }
function formatNumber(value) { return Number(value || 0).toLocaleString("vi-VN"); }
function formatPercent(value, digits = 1) { return `${(Number(value || 0) * 100).toFixed(digits)}%`; }
function cacheBust(url) { return `${url}${url.includes("?") ? "&" : "?"}v=${Date.now()}`; }
function activePredictionModel() {
  if (!state.details) return null;
  const models = state.details.prediction_models || [];
  const selected = $("modelFilter").value;
  if (selected !== "all" && models.includes(selected)) return selected;
  return models[0] || null;
}
function clearRemoteImage(imageId) {
  const image = $(imageId);
  image.onload = null;
  image.onerror = null;
  image.removeAttribute("src");
  image.classList.add("hidden");
}

function loadRemoteImage(imageId, url, emptyId, errorMessage) {
  const image = $(imageId);
  const empty = $(emptyId);
  const token = String(Date.now()) + Math.random();
  image.dataset.loadToken = token;
  image.classList.remove("hidden");
  empty.classList.remove("hidden");
  empty.textContent = "Đang tải ảnh dự báo…";
  image.onload = () => {
    if (image.dataset.loadToken !== token) return;
    empty.classList.add("hidden");
  };
  image.onerror = () => {
    if (image.dataset.loadToken !== token) return;
    image.removeAttribute("src");
    image.classList.add("hidden");
    empty.classList.remove("hidden");
    empty.textContent = errorMessage;
  };
  image.src = url;
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

function updateStats(summary) {
  if (!summary) return;
  $("statTotal").textContent = formatNumber(summary.total);
  $("statReviewed").textContent = formatNumber(summary.reviewed);
  $("statRemaining").textContent = formatNumber(summary.remaining);
  $("statCandidates").textContent = formatNumber(summary.candidates);
  $("statHardNegatives").textContent = formatNumber(summary.hard_negatives);
  const progress = summary.total ? summary.reviewed / summary.total * 100 : 0;
  $("progressBar").style.width = `${progress}%`;
}

function populateOptions() {
  const data = state.bootstrap;
  $("groupFilter").insertAdjacentHTML("beforeend", data.groups.map(group => `<option value="${escapeHtml(group)}">${escapeHtml(group)}</option>`).join(""));
  $("groupList").innerHTML = data.groups.map(group => `<option value="${escapeHtml(group)}"></option>`).join("");
  $("modelFilter").insertAdjacentHTML("beforeend", data.models.map(model => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join(""));
  const decisions = data.decisions.map(value => `<option value="${value}">${decisionLabels[value] || value}</option>`).join("");
  $("decisionInput").innerHTML = decisions;
  $("decisionFilter").insertAdjacentHTML("beforeend", decisions);
  $("tagPicker").innerHTML = data.issue_tags.map(tag => `
    <label class="tag-option"><input type="checkbox" name="issueTag" value="${tag}"><span>${tagLabels[tag] || tag}</span></label>
  `).join("");
}

function filterParams() {
  const params = new URLSearchParams({
    split: $("splitFilter").value,
    label: $("labelFilter").value,
    group: $("groupFilter").value,
    review_status: $("reviewFilter").value,
    decision: $("decisionFilter").value,
    candidate: $("candidateFilter").value,
    model: $("modelFilter").value,
    search: $("searchFilter").value.trim(),
    sort: $("sortFilter").value,
    descending: "true",
    offset: String(state.offset),
    limit: String(state.limit),
  });
  const minScore = $("scoreFilter").value.trim();
  if (minScore !== "") params.set("min_score", minScore);
  return params;
}

async function loadItems(reset = false, preferredId = null) {
  if (state.loading) return;
  state.loading = true;
  if (reset) state.offset = 0;
  setSaveState("Đang lọc dữ liệu…");
  try {
    const data = await api(`/api/items?${filterParams()}`);
    state.items = data.items;
    state.total = data.total;
    state.offset = data.offset;
    state.currentIndex = preferredId ? state.items.findIndex(item => item.image_id === preferredId) : 0;
    if (state.currentIndex < 0 && state.items.length) state.currentIndex = 0;
    renderQueue();
    if (state.items.length) await openItem(state.currentIndex);
    else clearViewer();
    setSaveState("Sẵn sàng");
  } catch (error) {
    toast(error.message, "error");
    setSaveState("Có lỗi");
  } finally {
    state.loading = false;
  }
}

function renderQueue() {
  const list = $("queueList");
  list.innerHTML = state.items.map((item, index) => {
    const score = item.max_score == null ? "" : `<span class="score-pill">${Number(item.max_score).toFixed(3)}</span>`;
    const reviewed = item.review ? "reviewed" : "";
    const active = index === state.currentIndex ? "active" : "";
    const checked = state.selected.has(item.image_id) ? "checked" : "";
    const subtitle = `${item.split} · ${item.label_name}${item.candidate_reasons.length ? " · " + item.candidate_reasons.map(v => candidateLabels[v] || v).join(", ") : ""}`;
    return `<div class="queue-item ${reviewed} ${active}" data-index="${index}">
      <input class="select-item" type="checkbox" data-id="${escapeHtml(item.image_id)}" ${checked} aria-label="Chọn ${escapeHtml(item.image_id)}">
      <button class="queue-open" data-index="${index}"><strong>${escapeHtml(item.image_id)}</strong><small>${escapeHtml(subtitle)}</small></button>${score}
    </div>`;
  }).join("");
  $("queueCount").textContent = `${formatNumber(state.total)} mẫu`;
  const start = state.total ? state.offset + 1 : 0;
  const end = Math.min(state.offset + state.items.length, state.total);
  $("pageInfo").textContent = `${formatNumber(start)}–${formatNumber(end)} / ${formatNumber(state.total)}`;
  $("pagePrev").disabled = state.offset === 0;
  $("pageNext").disabled = state.offset + state.items.length >= state.total;
  list.querySelectorAll(".queue-open").forEach(button => button.onclick = () => openItem(Number(button.dataset.index)));
  list.querySelectorAll(".select-item").forEach(input => input.onchange = () => {
    if (input.checked) state.selected.add(input.dataset.id); else state.selected.delete(input.dataset.id);
    updateSelection();
  });
  updateSelection();
}

function updateSelection() {
  $("selectedCount").textContent = `${state.selected.size} đã chọn`;
  $("bulkApply").disabled = state.selected.size === 0;
}

function clearViewer() {
  state.currentIndex = -1;
  state.details = null;
  $("sampleId").textContent = "Không có mẫu phù hợp";
  $("samplePosition").textContent = "Hãy thay đổi bộ lọc";
  $("sampleChips").innerHTML = "";
  ["originalImage", "overlayImage", "maskImage"].forEach(id => $(id).removeAttribute("src"));
  ["predictionOverlayImage", "probabilityImage", "binaryImage", "qualitativeImage"].forEach(clearRemoteImage);
  $("qualitativeCard").classList.add("hidden");
  $("predictionOverlayEmpty").classList.remove("hidden");
  $("probabilityEmpty").classList.remove("hidden");
  $("binaryEmpty").classList.remove("hidden");
  $("quickModelMetrics").innerHTML = '<span class="quick-empty">Không có mẫu phù hợp</span>';
  $("applyPredictionMaskButton").disabled = true;
  $("applyPredictionMaskButton").classList.add("hidden");
  ["quickApproveButton", "quickFalseAlarmButton", "quickDefectButton", "quickEditMaskButton", "quickModelEditButton", "quickUncertainButton"]
    .forEach(id => $(id).disabled = true);
  $("sampleFacts").innerHTML = "";
  $("diagnostics").innerHTML = "";
  $("modelMetrics").innerHTML = '<div class="empty-note">Không có dữ liệu.</div>';
  renderQueue();
}

async function openItem(index) {
  if (index < 0 || index >= state.items.length) return;
  state.currentIndex = index;
  renderQueue();
  const summary = state.items[index];
  $("samplePosition").textContent = `${formatNumber(state.offset + index + 1)} / ${formatNumber(state.total)}`;
  $("sampleId").textContent = summary.image_id;
  try {
    state.details = await api(`/api/items/${encodeURIComponent(summary.image_id)}`);
    renderDetails(state.details);
    const active = $("queueList").querySelector(".queue-item.active");
    active?.scrollIntoView({block: "nearest"});
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderDetails(item) {
  const id = encodeURIComponent(item.image_id);
  $("predictionOverview").style.setProperty(
    "--sample-aspect",
    `${Math.max(1, Number(item.width) || 1)} / ${Math.max(1, Number(item.height) || 1)}`,
  );
  ["quickApproveButton", "quickFalseAlarmButton", "quickDefectButton", "quickEditMaskButton", "quickUncertainButton"]
    .forEach(buttonId => $(buttonId).disabled = false);
  $("sampleId").textContent = item.image_id;
  $("originalImage").src = `/api/items/${id}/image`;
  $("overlayImage").src = cacheBust(`/api/items/${id}/overlay`);
  $("maskImage").src = cacheBust(`/api/items/${id}/mask`);
  const selectedModel = $("modelFilter").value;
  const availablePreviews = item.qualitative_models || [];
  const predictionModels = item.prediction_models || [];
  const previewModel = selectedModel !== "all" ? selectedModel : (predictionModels[0] || availablePreviews[0]);
  $("qualitativeTitle").textContent = previewModel ? `Dự báo · ${previewModel}` : "Dự báo model";
  renderQuickModelMetrics(previewModel, previewModel ? item.predictions[previewModel] : null);
  if (previewModel && predictionModels.includes(previewModel)) {
    const encodedModel = encodeURIComponent(previewModel);
    document.querySelectorAll(".prediction-zoom").forEach(button => button.classList.remove("hidden"));
    loadRemoteImage("predictionOverlayImage", cacheBust(`/api/items/${id}/prediction?model=${encodedModel}&view=overlay`), "predictionOverlayEmpty", "Không tải được model overlay. Hãy restart Data Review Studio rồi F5.");
    loadRemoteImage("probabilityImage", cacheBust(`/api/items/${id}/prediction?model=${encodedModel}&view=probability`), "probabilityEmpty", "Không tải được probability map. Hãy restart Data Review Studio rồi F5.");
    loadRemoteImage("binaryImage", cacheBust(`/api/items/${id}/prediction?model=${encodedModel}&view=binary`), "binaryEmpty", "Không tải được binary mask. Hãy restart Data Review Studio rồi F5.");
    const threshold = Number(item.predictions[previewModel]?.threshold ?? 0.5);
    $("binaryTitle").textContent = `3 · Binary @ ${threshold.toFixed(2)}`;
    const canApplyPredictionMask = Boolean(item.predictions[previewModel]?.image_pred) && Number(item.predictions[previewModel]?.predicted_positive_pixels || 0) > 0;
    $("applyPredictionMaskButton").classList.toggle("hidden", !canApplyPredictionMask);
    $("applyPredictionMaskButton").disabled = !canApplyPredictionMask;
    $("applyPredictionMaskButton").dataset.model = previewModel;
    $("qualitativeCard").classList.add("hidden");
  } else if (previewModel && availablePreviews.includes(previewModel)) {
    ["predictionOverlayImage", "probabilityImage", "binaryImage"].forEach(clearRemoteImage);
    $("predictionOverlayEmpty").classList.remove("hidden");
    $("probabilityEmpty").classList.remove("hidden");
    $("binaryEmpty").classList.remove("hidden");
    $("predictionOverlayEmpty").textContent = "Chưa xuất model overlay cho ảnh này.";
    $("probabilityEmpty").textContent = "Chưa xuất probability cho ảnh này.";
    $("binaryEmpty").textContent = "Chưa xuất binary cho ảnh này.";
    document.querySelectorAll(".prediction-zoom").forEach(button => button.classList.add("hidden"));
    $("applyPredictionMaskButton").disabled = true;
    $("applyPredictionMaskButton").classList.add("hidden");
    $("qualitativeCard").classList.remove("hidden");
    $("qualitativeZoom").classList.remove("hidden");
    loadRemoteImage("qualitativeImage", `/api/items/${id}/qualitative?model=${encodeURIComponent(previewModel)}`, "qualitativeEmpty", "Không tải được ảnh qualitative của model này.");
  } else {
    ["predictionOverlayImage", "probabilityImage", "binaryImage", "qualitativeImage"].forEach(clearRemoteImage);
    $("predictionOverlayEmpty").classList.remove("hidden");
    $("probabilityEmpty").classList.remove("hidden");
    $("binaryEmpty").classList.remove("hidden");
    const missingText = previewModel ? "Đang chờ xuất dự báo cho ảnh này." : "Hãy chọn model để xem dự báo.";
    $("predictionOverlayEmpty").textContent = missingText;
    $("probabilityEmpty").textContent = missingText;
    $("binaryEmpty").textContent = missingText;
    $("binaryTitle").textContent = "3 · Binary";
    document.querySelectorAll(".prediction-zoom").forEach(button => button.classList.add("hidden"));
    $("applyPredictionMaskButton").disabled = true;
    $("applyPredictionMaskButton").classList.add("hidden");
    $("qualitativeCard").classList.add("hidden");
    $("qualitativeImage").removeAttribute("src");
    $("qualitativeImage").classList.add("hidden");
    $("qualitativeZoom").classList.add("hidden");
  }

  const editablePrediction = previewModel && predictionModels.includes(previewModel)
    && Number(item.predictions[previewModel]?.predicted_positive_pixels || 0) > 0;
  $("quickModelEditButton").disabled = !editablePrediction;
  $("quickModelEditButton").title = editablePrediction
    ? `Nạp Binary của ${previewModel} vào editor để chỉnh`
    : "Ảnh này chưa có mask dự báo dương";
  $("loadPredictionMask").disabled = !editablePrediction;
  $("quickDefectButton").querySelector("span").textContent = item.label ? "Xác nhận Defect" : "Có lỗi → vẽ mask";

  const chips = [
    `<span class="chip ${item.label ? "defect" : "good"}">${item.label_name}</span>`,
    `<span class="chip">${escapeHtml(item.split)}</span>`,
    `<span class="chip">${escapeHtml(item.defect_group)}</span>`,
    ...item.candidate_reasons.map(value => `<span class="chip ${value === "false_positive" || value === "zero_overlap" ? "alert" : "warn"}">${candidateLabels[value] || value}</span>`),
  ];
  if (item.review) chips.push(`<span class="chip good">${decisionLabels[item.review.decision] || item.review.decision}</span>`);
  $("sampleChips").innerHTML = chips.join("");

  const mask = item.diagnostics.mask;
  const factRows = [
    ["Nhãn gốc", item.label_name], ["Nhóm", item.defect_group], ["Split", item.split],
    ["Kích thước", `${item.width} × ${item.height}`], ["GT pixels", formatNumber(mask.pixels)],
    ["Components", formatNumber(mask.component_count)], ["Nhỏ nhất", formatNumber(mask.smallest_component)],
    ["Lớn nhất", formatNumber(mask.largest_component)], ["Tỷ lệ mask", formatPercent(mask.ratio, 3)],
  ];
  $("sampleFacts").innerHTML = factRows.map(([key, value]) => `<div><dt>${key}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  const issues = [...item.diagnostics.issues];
  if (!issues.length) $("diagnostics").innerHTML = '<span class="diagnostic ok">Kích thước ảnh/mask hợp lệ</span>';
  else $("diagnostics").innerHTML = issues.map(issue => `<span class="diagnostic">${escapeHtml(issue)}</span>`).join("");
  $("diagnosticStatus").textContent = issues.length ? `${issues.length} cảnh báo` : "Không có lỗi cấu trúc";
  $("sourcePaths").textContent = `Image: ${item.source_image_path}\nMask: ${item.source_mask_path || "(mask rỗng)"}`;

  renderModelMetrics(item.predictions, selectedModel);
  populateReviewForm(item);
  $("maskEditorTitle").textContent = `Sửa mask · ${item.image_id}`;
}

function renderModelMetrics(predictions, selectedModel = "all") {
  const entries = Object.entries(predictions).filter(([model]) => selectedModel === "all" || model === selectedModel);
  $("modelCount").textContent = entries.length ? `${entries.length} model` : "Chưa nạp model";
  if (!entries.length) {
    $("modelMetrics").innerHTML = '<div class="empty-note">Khởi động công cụ với ResultsRoot để nạp per_image_metrics.csv.</div>';
    return;
  }
  $("modelMetrics").innerHTML = entries.map(([model, value]) => `<div class="model-row">
    <strong title="${escapeHtml(model)}">${escapeHtml(model)}</strong>
    <div class="metric"><small>Score</small><b>${value.image_score.toFixed(4)}</b></div>
    <div class="metric"><small>Pred</small><b>${value.image_pred ? "Defect" : "Good"}</b></div>
    <div class="metric"><small>Pred pixels</small><b>${formatNumber(value.predicted_positive_pixels)}</b></div>
    <div class="metric"><small>Overlap GT</small><b>${formatNumber(value.pixel_tp)}</b></div>
  </div>`).join("");
}

function renderQuickModelMetrics(model, value) {
  if (!model || !value) {
    $("quickModelMetrics").innerHTML = '<span class="quick-empty">Chưa có chỉ số model cho ảnh này</span>';
    return;
  }
  const overlap = value.pixel_tp == null ? "—" : formatNumber(value.pixel_tp);
  $("quickModelMetrics").innerHTML = `
    <div class="quick-metric"><small>Score</small><strong>${Number(value.image_score || 0).toFixed(4)}</strong></div>
    <div class="quick-metric ${value.image_pred ? "is-defect" : "is-good"}"><small>Kết luận</small><strong>${value.image_pred ? "Defect" : "Good"}</strong></div>
    <div class="quick-metric"><small>Pred pixels</small><strong>${formatNumber(value.predicted_positive_pixels)}</strong></div>
    <div class="quick-metric"><small>Overlap GT</small><strong>${overlap}</strong></div>`;
}

function populateReviewForm(item) {
  const review = item.review;
  $("decisionInput").value = review?.decision || "approved";
  $("correctedLabelInput").value = review?.corrected_label == null ? "" : String(review.corrected_label);
  $("correctedGroupInput").value = review?.corrected_group || "";
  $("reviewerInput").value = review?.reviewer || localStorage.getItem("reviewer") || "";
  $("noteInput").value = review?.note || "";
  $("hardNegativeInput").checked = Boolean(review?.hard_negative);
  $("excludedInput").checked = Boolean(review?.excluded);
  document.querySelectorAll('input[name="issueTag"]').forEach(input => input.checked = Boolean(review?.tags?.includes(input.value)));
}

function reviewPayload(imageId = state.details?.image_id) {
  const corrected = $("correctedLabelInput").value;
  return {
    image_id: imageId,
    decision: $("decisionInput").value,
    corrected_label: corrected === "" ? null : Number(corrected),
    corrected_group: $("correctedGroupInput").value.trim(),
    tags: [...document.querySelectorAll('input[name="issueTag"]:checked')].map(input => input.value),
    note: $("noteInput").value.trim(),
    hard_negative: $("hardNegativeInput").checked,
    excluded: $("excludedInput").checked,
    reviewer: $("reviewerInput").value.trim(),
  };
}

async function saveReview(advance = false) {
  if (!state.details || state.saving) return false;
  state.saving = true;
  const payload = reviewPayload();
  if (payload.decision === "relabel_defect" && payload.corrected_label !== 1) payload.corrected_label = 1;
  if (payload.decision === "relabel_good" || payload.decision === "acceptable_mark") payload.corrected_label = 0;
  localStorage.setItem("reviewer", payload.reviewer);
  setSaveState("Đang lưu…");
  try {
    const result = await api("/api/reviews", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload)});
    state.details.review = result.review;
    state.items[state.currentIndex].review = result.review;
    updateStats(result.summary);
    renderQueue();
    toast("Đã lưu kết luận và audit log");
    setSaveState("Đã lưu");
    if (advance) await nextItem();
    return true;
  } catch (error) {
    toast(error.message, "error");
    setSaveState("Lưu thất bại");
    return false;
  } finally {
    state.saving = false;
  }
}

function checkIssueTag(value, checked = true) {
  const input = document.querySelector(`input[name="issueTag"][value="${value}"]`);
  if (input) input.checked = checked;
}

function prepareQuickDecision(decision) {
  setDecision(decision);
  $("excludedInput").checked = decision === "exclude";
  if (decision === "approved") {
    $("correctedLabelInput").value = "";
    $("correctedGroupInput").value = "";
    $("hardNegativeInput").checked = false;
    checkIssueTag("false_alarm", false);
    checkIssueTag("hard_negative", false);
  }
  if (decision === "acceptable_mark") {
    $("hardNegativeInput").checked = true;
    checkIssueTag("false_alarm", true);
    checkIssueTag("hard_negative", true);
  }
  if (decision === "relabel_defect" && !$("correctedGroupInput").value.trim()) {
    $("correctedGroupInput").value = state.details?.defect_group !== "good"
      ? state.details.defect_group
      : "unclassified";
  }
}

async function quickReview(decision) {
  if (!state.details || state.saving) return;
  prepareQuickDecision(decision);
  await saveReview(true);
}

async function quickDefectReview() {
  if (!state.details || state.saving) return;
  if (Number(state.details.label) === 1) {
    await quickReview("approved");
    return;
  }
  prepareQuickDecision("relabel_defect");
  await openMaskEditor();
}

async function applyBulk() {
  if (!state.selected.size) return;
  const payload = reviewPayload();
  payload.image_ids = [...state.selected];
  delete payload.image_id;
  if (!confirm(`Áp dụng “${decisionLabels[payload.decision]}” cho ${payload.image_ids.length} mẫu đã chọn?`)) return;
  try {
    const result = await api("/api/reviews/bulk", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload)});
    state.selected.clear();
    updateStats(result.summary);
    toast(`Đã cập nhật ${result.saved} mẫu`);
    await loadItems(false, state.details?.image_id);
  } catch (error) { toast(error.message, "error"); }
}

async function nextItem() {
  if (state.currentIndex < state.items.length - 1) return openItem(state.currentIndex + 1);
  if (state.offset + state.items.length < state.total) {
    state.offset += state.limit;
    await loadItems(false);
  }
}

async function prevItem() {
  if (state.currentIndex > 0) return openItem(state.currentIndex - 1);
  if (state.offset > 0) {
    state.offset = Math.max(0, state.offset - state.limit);
    await loadItems(false);
    if (state.items.length) await openItem(state.items.length - 1);
  }
}

function setDecision(value) {
  $("decisionInput").value = value;
  if (value === "relabel_good" || value === "acceptable_mark") $("correctedLabelInput").value = "0";
  if (value === "relabel_defect") $("correctedLabelInput").value = "1";
  if (value === "acceptable_mark") {
    $("hardNegativeInput").checked = true;
    const hardTag = document.querySelector('input[name="issueTag"][value="hard_negative"]');
    if (hardTag) hardTag.checked = true;
  }
  if (value === "exclude") $("excludedInput").checked = true;
  if (value === "fix_mask") openMaskEditor();
}

function scheduleFilter() {
  clearTimeout(state.filterTimer);
  state.filterTimer = setTimeout(() => loadItems(true), 260);
}

function pushEditorHistory() {
  const context = editor.mask.getContext("2d");
  const snapshot = context.getImageData(0, 0, editor.mask.width, editor.mask.height);
  editor.history = editor.history.slice(0, editor.historyIndex + 1);
  editor.history.push(snapshot);
  if (editor.history.length > 20) editor.history.shift();
  editor.historyIndex = editor.history.length - 1;
  updateEditorHistoryButtons();
}

function updateEditorHistoryButtons() {
  $("undoMask").disabled = editor.historyIndex <= 0;
  $("redoMask").disabled = editor.historyIndex >= editor.history.length - 1;
}

function restoreEditorHistory(index) {
  if (index < 0 || index >= editor.history.length) return;
  editor.historyIndex = index;
  editor.mask.getContext("2d").putImageData(editor.history[index], 0, 0);
  renderEditor();
  updateEditorHistoryButtons();
}

function setEditorMode(mode) {
  editor.mode = mode;
  $("drawTool").classList.toggle("active", mode === "draw");
  $("eraseTool").classList.toggle("active", mode === "erase");
  $("eraseComponentTool").classList.toggle("active", mode === "component_erase");
  editor.view.style.cursor = mode === "component_erase" ? "cell" : "crosshair";
}

function maskFromImage(image) {
  const temp = document.createElement("canvas");
  temp.width = editor.mask.width;
  temp.height = editor.mask.height;
  const context = temp.getContext("2d", {willReadFrequently: true});
  context.drawImage(image, 0, 0, temp.width, temp.height);
  const source = context.getImageData(0, 0, temp.width, temp.height).data;
  const targetContext = editor.mask.getContext("2d");
  const target = targetContext.createImageData(temp.width, temp.height);
  for (let index = 0; index < source.length; index += 4) {
    const active = Math.max(source[index], source[index + 1], source[index + 2]) > 20;
    target.data[index] = 255;
    target.data[index + 1] = 55;
    target.data[index + 2] = 70;
    target.data[index + 3] = active ? 205 : 0;
  }
  targetContext.clearRect(0, 0, temp.width, temp.height);
  targetContext.putImageData(target, 0, 0);
}

function renderEditor() {
  const view = editor.view;
  view.width = editor.original.naturalWidth;
  view.height = editor.original.naturalHeight;
  const context = view.getContext("2d");
  context.clearRect(0, 0, view.width, view.height);
  context.drawImage(editor.original, 0, 0);
  context.drawImage(editor.mask, 0, 0);
  view.style.width = `${Math.round(view.width * editor.zoom)}px`;
  view.style.height = `${Math.round(view.height * editor.zoom)}px`;
  $("zoomValue").textContent = `${Math.round(editor.zoom * 100)}%`;
}

async function openMaskEditor(source = "current") {
  if (!state.details || editor.open) return;
  const id = encodeURIComponent(state.details.image_id);
  const model = source === "prediction" ? activePredictionModel() : null;
  if (source === "prediction" && !model) {
    toast("Ảnh này chưa có mask dự báo để chỉnh", "error");
    return;
  }
  editor.open = true;
  editor.sourceModel = model;
  setSaveState("Đang mở mask editor…");
  try {
    const maskUrl = model
      ? cacheBust(`/api/items/${id}/prediction?model=${encodeURIComponent(model)}&view=binary`)
      : cacheBust(`/api/items/${id}/mask`);
    const [original, mask] = await Promise.all([
      loadImage(`/api/items/${id}/image`),
      loadImage(maskUrl),
    ]);
    editor.original = original;
    editor.mask.width = original.naturalWidth;
    editor.mask.height = original.naturalHeight;
    maskFromImage(mask);
    const automaticCleanup = removeSmallMaskComponents({minimum: 2, minComponentCount: 32, silent: true});
    const fit = Math.min(1, (window.innerWidth - 100) / original.naturalWidth, (window.innerHeight - 260) / original.naturalHeight);
    editor.zoom = Math.max(.25, Math.round(fit * 20) / 20);
    $("editorZoom").value = String(editor.zoom);
    editor.history = [];
    editor.historyIndex = -1;
    pushEditorHistory();
    renderEditor();
    $("decisionInput").value = "fix_mask";
    if (model || Number(state.details.label) === 0) {
      $("correctedLabelInput").value = "1";
      if (!$("correctedGroupInput").value.trim()) $("correctedGroupInput").value = "unclassified";
    }
    checkIssueTag("model_mask_accepted", Boolean(model));
    $("maskDialog").showModal();
    setSaveState("Sẵn sàng");
    if (automaticCleanup.removedPixels) {
      toast(`Đã tự loại ${formatNumber(automaticCleanup.removedPixels)} chấm nhiễu 1 pixel; hãy kiểm tra rồi lưu`);
    }
    if (model) toast(`Đã nạp Binary của ${model}; hãy xóa/vẽ lại phần cần chỉnh`);
  } catch (error) {
    editor.open = false;
    toast(error.message, "error");
  }
}

async function loadPredictionIntoEditor() {
  if (!editor.open || !state.details) return;
  const model = activePredictionModel();
  if (!model) {
    toast("Ảnh này chưa có mask dự báo để nạp", "error");
    return;
  }
  try {
    const id = encodeURIComponent(state.details.image_id);
    const mask = await loadImage(cacheBust(`/api/items/${id}/prediction?model=${encodeURIComponent(model)}&view=binary`));
    maskFromImage(mask);
    const automaticCleanup = removeSmallMaskComponents({minimum: 2, minComponentCount: 32, silent: true});
    editor.sourceModel = model;
    $("correctedLabelInput").value = "1";
    if (!$("correctedGroupInput").value.trim()) $("correctedGroupInput").value = "unclassified";
    checkIssueTag("model_mask_accepted", true);
    pushEditorHistory();
    renderEditor();
    toast(`Đã nạp Binary của ${model}${automaticCleanup.removedPixels ? ` và loại ${formatNumber(automaticCleanup.removedPixels)} chấm nhiễu` : ""}`);
  } catch (error) {
    toast(error.message, "error");
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Không tải được ảnh: ${url}`));
    image.src = url;
  });
}

function editorPoint(event) {
  const rect = editor.view.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * editor.view.width / rect.width,
    y: (event.clientY - rect.top) * editor.view.height / rect.height,
  };
}

function drawEditorSegment(from, to) {
  const context = editor.mask.getContext("2d");
  context.save();
  context.globalCompositeOperation = editor.mode === "erase" ? "destination-out" : "source-over";
  context.strokeStyle = "rgba(255,55,70,.8)";
  context.lineWidth = editor.brush;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
  context.restore();
  renderEditor();
}

function eraseMaskComponent(point) {
  const context = editor.mask.getContext("2d", {willReadFrequently: true});
  const width = editor.mask.width;
  const height = editor.mask.height;
  const x = Math.max(0, Math.min(width - 1, Math.floor(point.x)));
  const y = Math.max(0, Math.min(height - 1, Math.floor(point.y)));
  const image = context.getImageData(0, 0, width, height);
  const alpha = index => image.data[index * 4 + 3];
  const start = y * width + x;
  if (alpha(start) <= 20) {
    toast("Hãy click đúng vào vùng mask màu đỏ cần xóa", "error");
    return 0;
  }
  const queue = new Int32Array(width * height);
  let head = 0;
  let tail = 0;
  let removed = 0;
  queue[tail++] = start;
  image.data[start * 4 + 3] = 0;
  while (head < tail) {
    const index = queue[head++];
    removed += 1;
    const px = index % width;
    const py = Math.floor(index / width);
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (!dx && !dy) continue;
        const nx = px + dx;
        const ny = py + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const next = ny * width + nx;
        if (alpha(next) <= 20) continue;
        image.data[next * 4 + 3] = 0;
        queue[tail++] = next;
      }
    }
  }
  context.putImageData(image, 0, 0);
  pushEditorHistory();
  renderEditor();
  toast(`Đã xóa một vùng gồm ${formatNumber(removed)} pixels`);
  return removed;
}

function removeSmallMaskComponents(options = {}) {
  const minimum = Math.max(1, Number(options.minimum ?? $("smallComponentSize").value) || 30);
  const minComponentCount = Math.max(1, Number(options.minComponentCount || 1));
  const silent = Boolean(options.silent);
  const context = editor.mask.getContext("2d", {willReadFrequently: true});
  const width = editor.mask.width;
  const height = editor.mask.height;
  const image = context.getImageData(0, 0, width, height);
  const visited = new Uint8Array(width * height);
  const queue = new Int32Array(width * height);
  let removedComponents = 0;
  let removedPixels = 0;
  for (let start = 0; start < width * height; start += 1) {
    if (visited[start] || image.data[start * 4 + 3] <= 20) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;
    while (head < tail) {
      const index = queue[head++];
      const px = index % width;
      const py = Math.floor(index / width);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (!dx && !dy) continue;
          const nx = px + dx;
          const ny = py + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const next = ny * width + nx;
          if (visited[next] || image.data[next * 4 + 3] <= 20) continue;
          visited[next] = 1;
          queue[tail++] = next;
        }
      }
    }
    if (tail >= minimum) continue;
    removedComponents += 1;
    removedPixels += tail;
    for (let index = 0; index < tail; index += 1) image.data[queue[index] * 4 + 3] = 0;
  }
  if (removedComponents < minComponentCount) {
    if (!silent) toast(`Không có đốm nào nhỏ hơn ${formatNumber(minimum)} pixels`);
    return {removedComponents: 0, removedPixels: 0};
  }
  context.putImageData(image, 0, 0);
  if (!silent) {
    pushEditorHistory();
    renderEditor();
    toast(`Đã xóa ${formatNumber(removedComponents)} đốm (${formatNumber(removedPixels)} pixels)`);
  }
  return {removedComponents, removedPixels};
}

function editorMaskPixelCount() {
  const data = editor.mask.getContext("2d", {willReadFrequently: true})
    .getImageData(0, 0, editor.mask.width, editor.mask.height).data;
  let count = 0;
  for (let index = 3; index < data.length; index += 4) if (data[index] > 20) count += 1;
  return count;
}

function binaryMaskDataUrl() {
  const source = editor.mask.getContext("2d").getImageData(0, 0, editor.mask.width, editor.mask.height).data;
  const output = document.createElement("canvas");
  output.width = editor.mask.width;
  output.height = editor.mask.height;
  const context = output.getContext("2d");
  const pixels = context.createImageData(output.width, output.height);
  for (let index = 0; index < source.length; index += 4) {
    const value = source[index + 3] > 20 ? 255 : 0;
    pixels.data[index] = value;
    pixels.data[index + 1] = value;
    pixels.data[index + 2] = value;
    pixels.data[index + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  return output.toDataURL("image/png");
}

async function saveMask(advance = false) {
  if (!state.details || state.saving) return;
  state.saving = true;
  $("saveMaskButton").disabled = true;
  $("saveMaskNextButton").disabled = true;
  setSaveState("Đang lưu mask…");
  try {
    const imageId = state.details.image_id;
    const automaticCleanup = removeSmallMaskComponents({minimum: 2, minComponentCount: 32, silent: true});
    if (automaticCleanup.removedPixels) renderEditor();
    const maskPixels = editorMaskPixelCount();
    const result = await api(`/api/items/${encodeURIComponent(imageId)}/mask`, {
      method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({data_url: binaryMaskDataUrl()}),
    });
    const payload = reviewPayload(imageId);
    payload.decision = "fix_mask";
    payload.tags = [...new Set([...(payload.tags || []), ...(result.review?.tags || [])])];
    if (editor.sourceModel) payload.tags = [...new Set([...payload.tags, "model_mask_accepted"])];
    if (maskPixels > 0 && Number(state.details.label) === 0) {
      payload.corrected_label = 1;
      if (!payload.corrected_group) payload.corrected_group = "unclassified";
    }
    const saved = await api("/api/reviews", {
      method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload),
    });
    state.details.review = saved.review;
    state.details.diagnostics = result.diagnostics;
    state.items[state.currentIndex].review = saved.review;
    updateStats(saved.summary);
    renderQueue();
    $("maskDialog").close();
    editor.open = false;
    editor.sourceModel = null;
    if (advance) {
      await nextItem();
    } else {
      $("maskImage").src = cacheBust(`/api/items/${encodeURIComponent(imageId)}/mask`);
      $("overlayImage").src = cacheBust(`/api/items/${encodeURIComponent(imageId)}/overlay`);
      renderDetails(state.details);
    }
    const removedNoise = Number(result.removed_noise_pixels || 0) + automaticCleanup.removedPixels;
    toast(`Đã lưu mask ${formatNumber(maskPixels)} pixels${removedNoise ? ` · loại ${formatNumber(removedNoise)} chấm nhiễu` : ""}${advance ? " và sang ảnh tiếp" : ""}`);
    setSaveState("Đã lưu mask");
  } catch (error) {
    toast(error.message, "error");
    setSaveState("Lưu mask thất bại");
  } finally {
    state.saving = false;
    $("saveMaskButton").disabled = false;
    $("saveMaskNextButton").disabled = false;
  }
}

async function applyPredictionMask() {
  if (!state.details) return;
  const model = $("applyPredictionMaskButton").dataset.model;
  if (!model) return;
  const confirmed = confirm(
    `Dùng Binary của ${model} làm GT mask?\n\nTool sẽ lưu một mask chỉnh sửa riêng, đặt mẫu là Defect và đánh dấu “Fix mask”. Mask nguồn không bị ghi đè.`
  );
  if (!confirmed) return;
  $("applyPredictionMaskButton").disabled = true;
  setSaveState("Đang áp dụng mask từ model…");
  try {
    const result = await api(`/api/items/${encodeURIComponent(state.details.image_id)}/prediction-mask`, {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({model}),
    });
    state.details.review = result.review;
    state.details.diagnostics = result.diagnostics;
    const id = encodeURIComponent(state.details.image_id);
    $("maskImage").src = cacheBust(`/api/items/${id}/mask`);
    $("overlayImage").src = cacheBust(`/api/items/${id}/overlay`);
    renderDetails(state.details);
    toast(`Đã dùng Binary của ${model} làm GT mask (${formatNumber(result.mask_pixels)} pixels)`);
    setSaveState("Đã áp dụng mask model");
  } catch (error) {
    toast(error.message, "error");
    setSaveState("Không thể áp dụng mask model");
  }
}

async function exportDataset() {
  const name = prompt("Tên bản export (để trống để dùng timestamp):", "clean_reviewed");
  if (name === null) return;
  setSaveState("Đang xuất dữ liệu…");
  try {
    const result = await api("/api/export", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({name})});
    const summary = result.summary;
    const trainingPath = summary.training_dataset_path || result.path;
    toast(`Đã xuất ${formatNumber(summary.exported_records)} mẫu train-ready → ${trainingPath}`);
    setSaveState("Xuất hoàn tất");
    alert(`Xuất dataset hoàn tất.\n\nDataset để train: ${trainingPath}\nTrain: ${summary.training_dataset_split_counts?.train ?? "—"}\nVal: ${summary.training_dataset_split_counts?.val ?? "—"}\nTest: ${summary.training_dataset_split_counts?.test ?? "—"}\n\nMỗi ảnh đã có image_path và mask_path tương đối. Good dùng mask đen; Defect dùng mask cuối cùng sau review.\n\nUnresolved: ${summary.unresolved_records}\nHard negatives: ${summary.hard_negatives}\n\nDataset nguồn không bị thay đổi.`);
  } catch (error) {
    toast(error.message, "error");
    setSaveState("Xuất thất bại");
  }
}

function bindEvents() {
  ["splitFilter", "labelFilter", "groupFilter", "reviewFilter", "decisionFilter", "candidateFilter", "modelFilter", "sortFilter"]
    .forEach(id => $(id).addEventListener("change", () => loadItems(true)));
  ["searchFilter", "scoreFilter"].forEach(id => $(id).addEventListener("input", scheduleFilter));
  $("resetFilters").onclick = () => {
    ["splitFilter", "labelFilter", "groupFilter", "reviewFilter", "decisionFilter", "modelFilter"].forEach(id => $(id).value = "all");
    $("candidateFilter").value = "any";
    $("searchFilter").value = ""; $("scoreFilter").value = ""; $("sortFilter").value = "priority"; loadItems(true);
  };
  $("refreshButton").onclick = () => loadItems(false, state.details?.image_id);
  $("exportButton").onclick = exportDataset;
  $("prevButton").onclick = prevItem; $("nextButton").onclick = nextItem;
  $("pagePrev").onclick = async () => { state.offset = Math.max(0, state.offset - state.limit); await loadItems(false); };
  $("pageNext").onclick = async () => { state.offset += state.limit; await loadItems(false); };
  $("saveButton").onclick = () => saveReview(false); $("saveNextButton").onclick = () => saveReview(true);
  $("quickApproveButton").onclick = () => quickReview("approved");
  $("quickFalseAlarmButton").onclick = () => quickReview("acceptable_mark");
  $("quickDefectButton").onclick = quickDefectReview;
  $("quickEditMaskButton").onclick = () => openMaskEditor();
  $("quickModelEditButton").onclick = () => openMaskEditor("prediction");
  $("quickUncertainButton").onclick = () => quickReview("uncertain");
  $("bulkApply").onclick = applyBulk;
  $("editMaskButton").onclick = openMaskEditor;
  $("applyPredictionMaskButton").onclick = applyPredictionMask;
  $("decisionInput").onchange = () => {
    const value = $("decisionInput").value;
    if (value === "relabel_good" || value === "acceptable_mark") $("correctedLabelInput").value = "0";
    if (value === "relabel_defect") $("correctedLabelInput").value = "1";
    if (value === "exclude") $("excludedInput").checked = true;
  };
  document.querySelectorAll(".zoom-image").forEach(button => button.onclick = () => {
    const source = $(button.dataset.target).src;
    if (!source) return;
    $("largeImage").src = source;
    $("imageDialog").showModal();
  });

  $("drawTool").onclick = () => setEditorMode("draw");
  $("eraseTool").onclick = () => setEditorMode("erase");
  $("eraseComponentTool").onclick = () => setEditorMode("component_erase");
  $("brushSize").oninput = event => { editor.brush = Number(event.target.value); $("brushValue").textContent = `${editor.brush} px`; };
  $("editorZoom").oninput = event => { editor.zoom = Number(event.target.value); renderEditor(); };
  $("undoMask").onclick = () => restoreEditorHistory(editor.historyIndex - 1);
  $("redoMask").onclick = () => restoreEditorHistory(editor.historyIndex + 1);
  $("clearMask").onclick = () => {
    if (!confirm("Xóa sạch toàn bộ mask hiện tại? Sau thao tác, mask sẽ còn đúng 0 pixel.")) return;
    editor.mask.getContext("2d").clearRect(0, 0, editor.mask.width, editor.mask.height);
    pushEditorHistory();
    renderEditor();
    toast("Đã xóa sạch mask — còn 0 pixel");
  };
  $("removeSmallComponents").onclick = () => removeSmallMaskComponents();
  $("loadPredictionMask").onclick = loadPredictionIntoEditor;
  $("importMask").onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    const url = URL.createObjectURL(file);
    try {
      const image = await loadImage(url);
      if (image.naturalWidth !== editor.mask.width || image.naturalHeight !== editor.mask.height) throw new Error(`PNG phải có kích thước ${editor.mask.width} × ${editor.mask.height}`);
      maskFromImage(image); pushEditorHistory(); renderEditor();
    } catch (error) { toast(error.message, "error"); }
    finally { URL.revokeObjectURL(url); event.target.value = ""; }
  };
  $("saveMaskButton").onclick = () => saveMask(false);
  $("saveMaskNextButton").onclick = () => saveMask(true);
  $("maskDialog").addEventListener("close", () => { editor.open = false; editor.drawing = false; editor.sourceModel = null; });

  editor.view.addEventListener("pointerdown", event => {
    if (!editor.open) return;
    if (editor.mode === "component_erase") {
      event.preventDefault();
      eraseMaskComponent(editorPoint(event));
      return;
    }
    editor.drawing = true;
    editor.view.setPointerCapture(event.pointerId);
    editor.last = editorPoint(event);
    drawEditorSegment(editor.last, editor.last);
  });
  editor.view.addEventListener("pointermove", event => {
    if (!editor.drawing) return;
    const point = editorPoint(event); drawEditorSegment(editor.last, point); editor.last = point;
  });
  const finishStroke = event => {
    if (!editor.drawing) return;
    editor.drawing = false;
    try { editor.view.releasePointerCapture(event.pointerId); } catch (_) {}
    pushEditorHistory();
  };
  editor.view.addEventListener("pointerup", finishStroke);
  editor.view.addEventListener("pointercancel", finishStroke);

  document.addEventListener("keydown", event => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (editor.open) {
      if (!typing && event.key.toLowerCase() === "b") setEditorMode("draw");
      if (!typing && event.key.toLowerCase() === "e") setEditorMode("erase");
      if (!typing && event.key.toLowerCase() === "r") setEditorMode("component_erase");
      if (!typing && event.key.toLowerCase() === "p") loadPredictionIntoEditor();
      if (event.ctrlKey && event.key.toLowerCase() === "z") { event.preventDefault(); restoreEditorHistory(editor.historyIndex - 1); }
      if (event.ctrlKey && event.key === "Enter") { event.preventDefault(); saveMask(true); }
      return;
    }
    if (event.ctrlKey && event.key === "Enter") { event.preventDefault(); saveReview(true); return; }
    if (typing) return;
    if (event.key === "ArrowLeft") prevItem();
    if (event.key === "ArrowRight") nextItem();
    if (event.key === "1") { event.preventDefault(); quickReview("approved"); return; }
    if (event.key === "2") { event.preventDefault(); quickReview("acceptable_mark"); return; }
    if (event.key === "3") { event.preventDefault(); quickDefectReview(); return; }
    if (event.key.toLowerCase() === "p") { event.preventDefault(); openMaskEditor("prediction"); return; }
    if (event.key.toLowerCase() === "u") { event.preventDefault(); quickReview("uncertain"); return; }
    const keys = {a: "approved", h: "acceptable_mark", g: "relabel_good", d: "relabel_defect", m: "fix_mask", u: "uncertain", x: "exclude"};
    const decision = keys[event.key.toLowerCase()];
    if (decision) { event.preventDefault(); setDecision(decision); }
  });
}

async function init() {
  try {
    state.bootstrap = await api("/api/bootstrap");
    updateStats(state.bootstrap.summary);
    populateOptions();
    bindEvents();
    await loadItems(true);
  } catch (error) {
    toast(error.message, "error");
    $("sampleId").textContent = "Không khởi tạo được công cụ";
    $("samplePosition").textContent = error.message;
  }
}

init();
