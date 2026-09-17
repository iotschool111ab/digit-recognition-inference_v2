const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const fileInput = document.getElementById("fileInput");
const processedPreview = document.getElementById("processedPreview");
const resultsList = document.getElementById("resultsList");
const modelStatus = document.getElementById("modelStatus");
const modelSelect = document.getElementById("modelSelect");

// CPU 上很慢的 ViT-base 系列，預設不勾，避免每次預測都等好幾秒
const SLOW_MODELS = new Set(["vit_finetune_head", "vit_finetune_backbone", "v_3rd"]);

let drawing = false;
let uploadedFile = null; // 若使用者上傳圖片，優先於畫布內容送出

// 畫布用「白紙黑字」的慣例（跟前處理管線 preprocess_for_mnist 的假設一致：
// 它是為手機拍紙上手寫數字的照片設計的，會自動反轉成 MNIST 的黑底白字格式）
function resetCanvas() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
resetCanvas();

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const point = evt.touches ? evt.touches[0] : evt;
  return {
    x: (point.clientX - rect.left) * (canvas.width / rect.width),
    y: (point.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function startDraw(evt) {
  evt.preventDefault();
  uploadedFile = null;
  drawing = true;
  const { x, y } = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function moveDraw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const { x, y } = getPos(evt);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
window.addEventListener("mouseup", endDraw);
canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", moveDraw);
canvas.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", () => {
  resetCanvas();
  uploadedFile = null;
  fileInput.value = "";
  processedPreview.removeAttribute("src");
  resultsList.innerHTML = '<p class="placeholder">畫完數字後點「預測」查看結果</p>';
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    uploadedFile = fileInput.files[0];
  }
});

async function loadModelStatus() {
  try {
    const res = await fetch("/api/models");
    const status = await res.json();
    modelStatus.innerHTML = "";
    modelSelect.innerHTML = "";
    Object.entries(status).forEach(([key, info]) => {
      const chip = document.createElement("span");
      chip.className = `status-chip ${info.loaded ? "ok" : "missing"}`;
      chip.textContent = info.loaded ? `${key} ✓` : `${key} 缺`;
      chip.title = info.reason || "";
      modelStatus.appendChild(chip);

      if (info.loaded) {
        const label = document.createElement("label");
        label.className = "model-check";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = key;
        cb.checked = !SLOW_MODELS.has(key);
        label.appendChild(cb);
        label.appendChild(document.createTextNode(key));
        modelSelect.appendChild(label);
      }
    });
    if (!modelSelect.children.length) {
      modelSelect.innerHTML = '<span class="placeholder">沒有可用的模型</span>';
    }
  } catch (e) {
    modelStatus.innerHTML = '<span class="status-chip missing">無法取得模型狀態</span>';
  }
}

function selectedModelKeys() {
  return Array.from(modelSelect.querySelectorAll("input:checked")).map((cb) => cb.value);
}

function renderResults(results) {
  if (!results.length) {
    resultsList.innerHTML = '<p class="placeholder">沒有任何模型可用，請確認模型權重是否存在</p>';
    return;
  }
  const bestKey = results[0].key;
  resultsList.innerHTML = "";
  results.forEach((r) => {
    const card = document.createElement("div");
    card.className = "result-card" + (r.key === bestKey ? " best" : "");

    const head = document.createElement("div");
    head.className = "result-head";
    head.innerHTML = `
      <div>
        <div class="result-name">${r.display_name}</div>
        <div class="result-meta">信心值 ${(r.confidence * 100).toFixed(1)}% · ${r.inference_ms} ms</div>
      </div>
      <div class="result-pred">${r.prediction}</div>
    `;
    card.appendChild(head);

    const bars = document.createElement("div");
    bars.className = "prob-bars";
    r.probabilities.forEach((p, digit) => {
      const col = document.createElement("div");
      col.className = "prob-bar-col" + (digit === r.prediction ? " top" : "");
      const bar = document.createElement("div");
      bar.className = "prob-bar";
      bar.style.height = `${Math.max(2, p * 100)}%`;
      const label = document.createElement("div");
      label.className = "prob-bar-label";
      label.textContent = digit;
      col.appendChild(bar);
      col.appendChild(label);
      bars.appendChild(col);
    });
    card.appendChild(bars);
    resultsList.appendChild(card);
  });
}

async function submitPredict() {
  predictBtn.disabled = true;
  predictBtn.textContent = "預測中...";
  resultsList.innerHTML = '<p class="placeholder">推論中，請稍候...</p>';

  try {
    const keys = selectedModelKeys();
    if (!keys.length) {
      resultsList.innerHTML = '<p class="placeholder">請至少勾選一個模型</p>';
      return;
    }
    const blob = uploadedFile || (await new Promise((resolve) => canvas.toBlob(resolve, "image/png")));
    const formData = new FormData();
    formData.append("file", blob, "input.png");
    formData.append("models", keys.join(","));

    const res = await fetch("/api/predict", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "推論失敗");
    }
    const data = await res.json();
    processedPreview.src = data.processed_image_base64;
    renderResults(data.results);
  } catch (e) {
    resultsList.innerHTML = `<p class="placeholder">發生錯誤：${e.message}</p>`;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "預測";
  }
}

predictBtn.addEventListener("click", submitPredict);

loadModelStatus();
