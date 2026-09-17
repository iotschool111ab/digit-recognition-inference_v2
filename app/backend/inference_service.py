import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

from architectures.arch_cnn import SimpleCNN
from architectures.arch_nn import SimpleNN
from architectures.arch_vit import ViT_MNIST

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", APP_DIR.parent.parent / "models"))

# 必備模型：訓練流程一定會產出，缺檔代表部署設定錯誤
REQUIRED_FILES = {
    "nn": "model_nn.pt",
    "cnn": "model_cnn.pt",
    "vit_scratch": "model_vit_scratch.pt",
}

# 選用的微調模型：沒有就在 /api/models 標記缺失，不讓服務掛掉
OPTIONAL_FILES = {
    "vit_finetune_head": "model_vit_finetune_head.pt",
    "vit_finetune_backbone": "model_vit_finetune_backbone.pt",
}

DISPLAY_NAMES = {
    "nn": "SimpleNN",
    "cnn": "SimpleCNN",
    "vit_scratch": "MyViT (from scratch)",
    "vit_finetune_head": "ViT Finetune (head only)",
    "vit_finetune_backbone": "ViT Finetune (backbone)",
    "v_3rd": "HF Benchmark (vit-base-mnist)",
}

# 走 224x224 HF ViT 前處理流程的模型（其餘走 28x28 pipeline）
HF_STYLE_MODELS = {"vit_finetune_head", "vit_finetune_backbone", "v_3rd"}

_models: dict = {}
_status: dict = {}


def _mnist_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def preprocess_for_mnist(pil_img: Image.Image) -> Image.Image:
    """把任意輸入圖片轉成 MNIST 風格的 28x28 置中灰階數字（複用 inference_all.py 的邏輯）。"""
    gray = np.array(pil_img.convert("L"))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        digit = thresh[y:y + h, x:x + w]

        size = 20
        if w > h:
            digit = cv2.resize(digit, (size, max(1, int(h * size / w))))
        else:
            digit = cv2.resize(digit, (max(1, int(w * size / h)), size))

        new_h, new_w = digit.shape
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[
            (28 - new_h) // 2: (28 - new_h) // 2 + new_h,
            (28 - new_w) // 2: (28 - new_w) // 2 + new_w,
        ] = digit
        return Image.fromarray(canvas)

    return Image.fromarray(thresh).resize((28, 28))


def load_all_models() -> dict:
    """伺服器啟動時呼叫一次。缺檔/載入失敗的模型記錄在回傳的 status 裡，不讓服務掛掉。"""
    global _models, _status
    _models = {}
    _status = {}

    arch_map = {"nn": SimpleNN, "cnn": SimpleCNN, "vit_scratch": ViT_MNIST}
    for key, filename in REQUIRED_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            _status[key] = {"loaded": False, "reason": f"找不到權重檔：{path}"}
            continue
        try:
            m = arch_map[key]()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            _models[key] = m
            _status[key] = {"loaded": True, "reason": None}
        except Exception as e:
            _status[key] = {"loaded": False, "reason": f"載入失敗：{e}"}

    for key, filename in OPTIONAL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            _status[key] = {"loaded": False, "reason": "尚未訓練（找不到對應權重檔）"}
            continue
        try:
            m = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224", num_labels=10, ignore_mismatched_sizes=True
            )
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            _models[key] = m
            _status[key] = {"loaded": True, "reason": None}
        except Exception as e:
            _status[key] = {"loaded": False, "reason": f"載入失敗：{e}"}

    try:
        _models["v_3rd_proc"] = ViTImageProcessor.from_pretrained(
            "farleyknight-org-username/vit-base-mnist"
        )
        m = ViTForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")
        m.eval()
        _models["v_3rd"] = m
        _status["v_3rd"] = {"loaded": True, "reason": None}
    except Exception as e:
        _status["v_3rd"] = {"loaded": False, "reason": f"無法從 Hugging Face 下載基準模型：{e}"}

    return _status


def get_status() -> dict:
    return _status


def available_keys() -> list:
    """目前可實際推論的模型 key（排除內部用的 processor）。"""
    return [k for k in _models if k != "v_3rd_proc"]


def predict(pil_img: Image.Image, keys=None) -> dict:
    """回傳每個已載入模型的預測結果，以及前處理後的 28x28 圖片。

    keys 為 None 時跑全部已載入模型；給定 list 時只跑其中有載入的那幾個
    （前端可藉此只挑 nn/cnn 這種毫秒級模型，跳過 CPU 上很慢的 ViT-base）。
    """
    if not _models:
        raise RuntimeError("模型尚未載入完成，請稍後再試")

    selected = set(available_keys())
    if keys is not None:
        selected &= set(keys)
        if not selected:
            raise RuntimeError("沒有選到任何已載入的模型")

    raw = pil_img.convert("RGB")
    processed = preprocess_for_mnist(raw)
    t28 = _mnist_transform()(processed).unsqueeze(0)

    hf_inputs = None
    if "v_3rd_proc" in _models and (selected & HF_STYLE_MODELS):
        hf_inputs = _models["v_3rd_proc"](images=raw, return_tensors="pt")

    results = []
    for key, model in _models.items():
        if key == "v_3rd_proc" or key not in selected:
            continue
        start = time.perf_counter()
        with torch.no_grad():
            if key in HF_STYLE_MODELS:
                if hf_inputs is None:
                    continue
                logits = model(**hf_inputs).logits
            else:
                logits = model(t28)
            probs = torch.softmax(logits, dim=1)[0].numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append({
            "key": key,
            "display_name": DISPLAY_NAMES.get(key, key),
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "probabilities": [float(p) for p in probs],
            "inference_ms": round(elapsed_ms, 2),
        })

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return {"results": results, "processed_image": processed}


def warmup(rounds: int = 2) -> dict:
    """啟動後對所有已載入模型跑幾次 dummy 推論。

    把 PyTorch/MKL 執行緒池初始化、lazy 記憶體配置等一次性成本吃在啟動階段，
    避免第一個真實請求慢上好幾秒。失敗不擋服務啟動。
    """
    if not _models:
        return {"skipped": "沒有已載入的模型"}

    dummy = Image.new("RGB", (28, 28), color="black")
    timings = {}
    for i in range(max(1, rounds)):
        start = time.perf_counter()
        try:
            predict(dummy)
        except Exception as e:
            return {"error": str(e)}
        timings[f"round_{i + 1}_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return timings
