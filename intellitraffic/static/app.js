const els = {
  html: document.documentElement,
  themeToggle: document.querySelector("#themeToggle"),
  statusStrip: document.querySelector("#statusStrip"),
  statusText: document.querySelector("#statusText"),
  statusList: document.querySelector("#statusList"),
  imageInput: document.querySelector("#imageInput"),
  dropZone: document.querySelector("#dropZone"),
  previewFrame: document.querySelector("#previewFrame"),
  previewImage: document.querySelector("#previewImage"),
  predictButton: document.querySelector("#predictButton"),
  resetButton: document.querySelector("#resetButton"),
  sampleGrid: document.querySelector("#sampleGrid"),
  resultEmpty: document.querySelector("#resultEmpty"),
  resultContent: document.querySelector("#resultContent"),
  latencyPill: document.querySelector("#latencyPill"),
  topLabel: document.querySelector("#topLabel"),
  imageMeta: document.querySelector("#imageMeta"),
  confidenceRing: document.querySelector("#confidenceRing"),
  confidenceText: document.querySelector("#confidenceText"),
  topList: document.querySelector("#topList"),
  probabilityBars: document.querySelector("#probabilityBars"),
  classCount: document.querySelector("#classCount"),
  historySearch: document.querySelector("#historySearch"),
  historyBody: document.querySelector("#historyBody"),
  toast: document.querySelector("#toast")
};

let selectedFile = null;
let toastTimer = null;

function formatPercent(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "-";
  const units = ["B", "KB", "MB"];
  let size = Number(bytes);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.add("hidden"), 4200);
}

function setTheme(theme) {
  els.html.dataset.theme = theme;
  localStorage.setItem("intellitraffic-theme", theme);
  els.themeToggle.textContent = theme === "dark" ? "Light" : "Dark";
}

function initTheme() {
  const saved = localStorage.getItem("intellitraffic-theme");
  const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  setTheme(saved || preferred);
}

function setPreview(file) {
  selectedFile = file;
  els.predictButton.disabled = !file;
  if (!file) {
    els.previewFrame.classList.add("empty");
    els.previewImage.removeAttribute("src");
    return;
  }
  const url = URL.createObjectURL(file);
  els.previewImage.onload = () => URL.revokeObjectURL(url);
  els.previewImage.src = url;
  els.previewFrame.classList.remove("empty");
}

function resetSelection() {
  selectedFile = null;
  els.imageInput.value = "";
  setPreview(null);
  els.resultContent.classList.add("hidden");
  els.resultEmpty.classList.remove("hidden");
  els.latencyPill.textContent = "Ready";
}

function renderStatus(status) {
  const runtime = status.runtime || {};
  const rows = [
    ["Model", status.model_exists ? "Found" : "Missing"],
    ["Classes", `${status.num_classes || 0}`],
    ["Image size", `${status.img_size || 128} x ${status.img_size || 128}`],
    ["TensorFlow", runtime.tensorflow ? "Available" : "Unavailable"],
    ["NumPy", runtime.numpy ? "Available" : "Unavailable"],
    ["SQLite log", status.database_path || "-"]
  ];
  els.statusList.innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
}

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    const samples = data.samples || [];
    if (!samples.length) {
      els.sampleGrid.innerHTML = "";
      return;
    }
    els.sampleGrid.innerHTML = samples
      .slice(0, 6)
      .map(sample => `
        <button class="sample-button" type="button" data-url="${sample.url}" data-name="${sample.filename}" title="${sample.filename}">
          <img src="${sample.url}" alt="${sample.filename}">
        </button>
      `)
      .join("");
  } catch {
    els.sampleGrid.innerHTML = "";
  }
}

async function useSample(button) {
  const url = button.dataset.url;
  const name = button.dataset.name;
  const res = await fetch(url);
  const blob = await res.blob();
  const file = new File([blob], name, { type: blob.type || "image/jpeg" });
  setPreview(file);
}

async function predict() {
  if (!selectedFile) return;
  els.predictButton.disabled = true;
  els.predictButton.textContent = "Classifying";
  els.latencyPill.textContent = "Running";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.detail || data.error || "Prediction failed.");
    }
    renderResult(data);
    await loadHistory();
  } catch (error) {
    showToast(error.message);
    els.latencyPill.textContent = "Failed";
    await loadHistory();
  } finally {
    els.predictButton.disabled = false;
    els.predictButton.textContent = "Classify";
  }
}

function renderResult(data) {
  const prediction = data.prediction;
  const confidence = Number(prediction.confidence || 0);
  els.resultEmpty.classList.add("hidden");
  els.resultContent.classList.remove("hidden");
  els.latencyPill.textContent = `${data.latency_ms} ms`;
  els.topLabel.textContent = prediction.display_label;
  els.imageMeta.textContent = `${data.image.filename} - ${data.image.width} x ${data.image.height} - ${formatBytes(data.image.size_bytes)}`;
  els.confidenceText.textContent = formatPercent(confidence, 1);
  els.confidenceRing.style.background = `conic-gradient(var(--mapua-red) ${confidence * 100}%, var(--line) 0)`;

  els.topList.innerHTML = prediction.top_k.slice(0, 3).map((item, index) => `
    <div class="rank-item">
      <span class="eyebrow">Rank ${index + 1}</span>
      <strong>${item.display_name}</strong>
      <span class="muted">${formatPercent(item.probability, 2)}</span>
    </div>
  `).join("");

  const probabilities = [...data.probabilities].sort((a, b) => b.probability - a.probability);
  els.classCount.textContent = `${probabilities.length} classes`;
  els.probabilityBars.innerHTML = probabilities.map(item => `
    <div class="prob-row">
      <span class="prob-label">${item.display_name}</span>
      <span class="prob-track"><span class="prob-fill" style="width: ${Math.max(item.probability * 100, 0.3)}%"></span></span>
      <span class="prob-value">${formatPercent(item.probability, 1)}</span>
    </div>
  `).join("");
}

async function loadHistory() {
  const params = new URLSearchParams();
  const q = els.historySearch.value.trim();
  if (q) params.set("q", q);
  params.set("limit", "50");
  try {
    const res = await fetch(`/api/history?${params.toString()}`);
    const data = await res.json();
    renderHistory(data.logs || []);
  } catch (error) {
    showToast(error.message);
  }
}

function renderHistory(logs) {
  if (!logs.length) {
    els.historyBody.innerHTML = `
      <tr>
        <td colspan="6" class="muted">No inference records yet.</td>
      </tr>
    `;
    return;
  }
  els.historyBody.innerHTML = logs.map(log => {
    const time = log.timestamp ? new Date(log.timestamp).toLocaleString() : "-";
    const predicted = log.predicted_class ? log.predicted_class.replaceAll("_", " ") : "-";
    const confidence = log.confidence === null || log.confidence === undefined ? "-" : formatPercent(log.confidence, 1);
    const latency = log.latency_ms === null || log.latency_ms === undefined ? "-" : `${log.latency_ms} ms`;
    return `
      <tr>
        <td>${time}</td>
        <td>${log.filename || "-"}</td>
        <td>${predicted}</td>
        <td>${confidence}</td>
        <td>${latency}</td>
        <td><span class="status-badge ${log.status}">${log.status}</span></td>
      </tr>
    `;
  }).join("");
}

function bindEvents() {
  els.themeToggle.addEventListener("click", () => {
    setTheme(els.html.dataset.theme === "dark" ? "light" : "dark");
  });

  els.imageInput.addEventListener("change", event => {
    const file = event.target.files && event.target.files[0];
    setPreview(file || null);
  });

  ["dragenter", "dragover"].forEach(name => {
    els.dropZone.addEventListener(name, event => {
      event.preventDefault();
      els.dropZone.classList.add("dragging");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    els.dropZone.addEventListener(name, event => {
      event.preventDefault();
      els.dropZone.classList.remove("dragging");
    });
  });

  els.dropZone.addEventListener("drop", event => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setPreview(file);
  });

  els.predictButton.addEventListener("click", predict);
  els.resetButton.addEventListener("click", resetSelection);
  els.sampleGrid.addEventListener("click", event => {
    const button = event.target.closest(".sample-button");
    if (button) useSample(button).catch(error => showToast(error.message));
  });
  els.historySearch.addEventListener("input", () => {
    window.clearTimeout(els.historySearch._timer);
    els.historySearch._timer = window.setTimeout(loadHistory, 200);
  });
}

initTheme();
bindEvents();
loadStatus();
loadSamples();
loadHistory();
