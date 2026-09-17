# 後端推論邏輯模組
# 處理系統環境變數、計算推論耗時與跨平台檔案路徑模組
import os
import time
from pathlib import Path

# import cv2, numpy as np, Image：影像處理三劍客。OpenCV 用於圖像二值化與輪廓尋找；NumPy 處理矩陣操作；PIL (Image) 是 PyTorch 與 FastAPI 間標準的影像傳輸格式。
import cv2
import numpy as np
from PIL import Image

# import torch, torchvision.transforms：PyTorch 核心框架與影像轉換/正規化工具。
import torch
from torchvision import transforms

# from transformers import ...：使用 Hugging Face 的 Vision Transformer (ViT) 模型架構與影像預處理器 (ViTImageProcessor)。
from transformers import ViTForImageClassification, ViTImageProcessor

# 匯入自訂的網路架構檔
from arch_cnn import SimpleCNN
from arch_nn import SimpleNN
from arch_vit import ViT_MNIST

# 取得當前 python 檔案所在的絕對路徑。
APP_DIR = Path(__file__).resolve().parent
# 優先讀取環境變數 MODEL_DIR。若未設定，預設為當前目錄上一層的 models 資料夾（../models）。
MODEL_DIR = Path(os.environ.get("MODEL_DIR", APP_DIR.parent / "models"))

'''
模型 Key                顯示名稱 (display_name)      模型分類/來源        輸入尺寸    狀態屬性
nn                      SimpleNN                    傳統全連接網路      28x28必備   (REQUIRED)
cnn                     SimpleCNN                   傳統卷積網路        28x28必備   (REQUIRED)
vit_scratch             MyViT (from scratch)        自建輕量 ViT        28x28必備   (REQUIRED)
vit_finetune_head       ViT Finetune (head only)    HF ViT (微調分類頭) 224x224選用 (OPTIONAL)
vit_finetune_backbone   ViT Finetune (backbone)     HF ViT (全微調)     224x224選用 (OPTIONAL)
v_3rd                   HF Benchmark                HF 雲端基準模型      224x224    預設自動下載

v_3rd_proc 是用來做 224x224 前處理的 ViTImageProcessor 元件，會在 available_keys() 中被過濾掉，不屬於預測模型)
所有走 224x224 流程的模型（即 HF_STYLE_MODELS 集合中的 vit_finetune_head、vit_finetune_backbone 與 v_3rd），
都是共用這個預處理器將原始圖片轉換為 $1 \times 3 \times 224 \times 224$ 的 Tensor。
'''

# 模型檔案清單與全域變數定義
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

# 寫死模型 key 對應到前端顯示的友善名稱。
DISPLAY_NAMES = {
    "nn": "SimpleNN",
    "cnn": "SimpleCNN",
    "vit_scratch": "MyViT (from scratch)",
    "vit_finetune_head": "ViT Finetune (head only)",
    "vit_finetune_backbone": "ViT Finetune (backbone)",
    "v_3rd": "HF Benchmark (vit-base-mnist)",
}

# 記錄哪些模型需要使用 Hugging Face 的 224x224 分辨率預處理，與傳統 MNIST 28x28 做分流。
# 走 224x224 HF ViT 前處理流程的模型（其餘走 28x28 pipeline）
HF_STYLE_MODELS = {"vit_finetune_head", "vit_finetune_backbone", "v_3rd"}


# _models 存放載入記憶體的模型實例；_status 存放各模型的載入狀態與失敗原因。
_models: dict = {}
_status: dict = {}


# 影像前處理邏輯
# 標準 PyTorch Transform。將 PIL 圖片轉為 Tensor（數值 0.0~1.0），並做單通道正規化至 [-1, 1] 範圍（計算式：$(x - 0.5) / 0.5$）。
def _mnist_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def preprocess_for_mnist(pil_img: Image.Image) -> Image.Image:
    """把任意輸入圖片轉成 MNIST 風格的 28x28 置中灰階數字（複用 inference_all.py 的邏輯）。
        轉灰階與高斯模糊去噪。
        使用 Otsu 演算法自動二值化，將圖片轉為黑底白字。
        尋找最大輪廓，抓出手寫數字的位置並裁切出來。

        保持寬高比 (Aspect Ratio) 將數字縮放到約 20x20 像素。

        貼在 28x28 畫布正中央。
        輸出結果：回傳一張尺寸為 28x28 像素、單通道灰階（MNIST 風格） 的 PIL Image 物件 processed。
    """
    # 轉為 8-bit 灰階圖。
    gray = np.array(pil_img.convert("L"))
    # cv2.GaussianBlur：5x5 核心的高斯模糊，用於去除手寫畫板或上傳圖片的雜訊。
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.threshold + THRESH_OTSU：使用 Otsu 自動閾值演算法進行二值化，並配合 THRESH_BINARY_INV 將文字轉為黑底白字（符合 MNIST 標準格式）。
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 尋找外接矩形。cv2.RETR_EXTERNAL 僅保留最外層輪廓；cv2.CHAIN_APPROX_SIMPLE 压缩輪廓邊界的點數，節省記憶體。
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找出面積最大的輪廓（通常就是數字本身）。
        cnt = max(contours, key=cv2.contourArea)
        # 計算該輪廓的外接矩形（x, y, width, height）。
        x, y, w, h = cv2.boundingRect(cnt)
        # 裁切出數字區域。
        digit = thresh[y:y + h, x:x + w]

        # 統一縮放到 20x20，維持長寬比。此步驟是為了讓後續的置中補齊（zero-padding）動作更精準。
        size = 20
        # 若圖片比目標寬，則以寬度為基準縮放；否則以高度為基準。
        if w > h:
            digit = cv2.resize(digit, (size, max(1, int(h * size / w))))
        else:
            digit = cv2.resize(digit, (max(1, int(w * size / h)), size))

        # 取得縮放後的新尺寸。
        new_h, new_w = digit.shape
        # 建立 28x28 全黑畫布。
        canvas = np.zeros((28, 28), dtype=np.uint8)
        # 計算置中位置（整數除法）。
        canvas[
            (28 - new_h) // 2: (28 - new_h) // 2 + new_h,
            (28 - new_w) // 2: (28 - new_w) // 2 + new_w,
        ] = digit
        # 轉回 PIL 圖片格式並回傳。
        return Image.fromarray(canvas)

    # 兜底：若沒找到任何輪廓（例如全黑圖），就直接將原二值化圖縮放到 28x28 並回傳。
    return Image.fromarray(thresh).resize((28, 28))


# 4. 模型初始化與載入
def load_all_models() -> dict:
    """伺服器啟動時呼叫一次。缺檔/載入失敗的模型記錄在回傳的 status 裡，不讓服務掛掉。"""
    global _models, _status
    _models = {}
    _status = {}

    # 建立模型實例（含必要預處理器）
    arch_map = {"nn": SimpleNN, "cnn": SimpleCNN, "vit_scratch": ViT_MNIST}
    
    # 走 REQUIRED_FILES 的必備模型
    for key, filename in REQUIRED_FILES.items():
        path = MODEL_DIR / filename
        
        # 检查模型文件是否存在
        if not path.exists():
            _status[key] = {"loaded": False, "reason": f"找不到權重檔：{path}"}
            continue
        
        try:
            # 根據模型的 key，選擇對應的模型類別（SimpleNN、SimpleCNN 或 ViT_MNIST）。
            m = arch_map[key]()

            # 載入 PyTorch 模型權重（map_location="cpu"確保 PyTorch 不會試圖把模型搬到沒有 GPU 的環境）
            m.load_state_dict(torch.load(path, map_location="cpu"))

            # 推論模式（關閉 Dropout、BatchNorm 的訓練行為）
            m.eval()
            
            # 將模型加入 _models 字典
            _models[key] = m
            # 成功載入後，在 _status 記錄為 True
            _status[key] = {"loaded": True, "reason": None}
        except Exception as e:
            # 發生任何錯誤（檔損、格式不對、key mismatch…），都記錄錯誤
            _status[key] = {"loaded": False, "reason": f"載入失敗：{e}"}

    
    # 走 OPTIONAL_FILES 的選用模型
    for key, filename in OPTIONAL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            # 模型檔案不存在，標示為未訓練/未找到
            _status[key] = {"loaded": False, "reason": "尚未訓練（找不到對應權重檔）"}
            continue
        try:
            # 建立 Hugging Face ViT 模型（224x224輸入）
            m = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224", num_labels=10, ignore_mismatched_sizes=True
            )

            # 載入 PyTorch 模型權重
            m.load_state_dict(torch.load(path, map_location="cpu"))

            # 推論模式（關閉 Dropout、BatchNorm 的訓練行為）
            m.eval()
            _models[key] = m
            _status[key] = {"loaded": True, "reason": None}
        except Exception as e:
            _status[key] = {"loaded": False, "reason": f"載入失敗：{e}"}


    # 試著載入 GitHub 上的基準模型（vit-base-mnist）
    try:
        # 載入 Hugging Face ImageProcessor（處理 224x224 圖片的標準前處理器）
        _models["v_3rd_proc"] = ViTImageProcessor.from_pretrained(
            "farleyknight-org-username/vit-base-mnist" # 建立一個基準模型
        )

        # 建立 Hugging Face ViT 模型（224x224輸入）
        # 這裡會自動從 Hugging Face Hub 下載預訓練的 ViT 模型
        m = ViTForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")
        # 推論模式（關閉 Dropout、BatchNorm 的訓練行為）
        m.eval()
        _models["v_3rd"] = m
        _status["v_3rd"] = {"loaded": True, "reason": None}
    except Exception as e:
        _status["v_3rd"] = {"loaded": False, "reason": f"無法從 Hugging Face 下載基準模型：{e}"}

    return _status


def get_status() -> dict:
    # 取得目前所有模型的狀態
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

    # 取得目前所有模型的 key（排除內部用的 processor）
    selected = set(available_keys())
    
    # 檢查使用者是否有傳入特定的模型指定清單
    if keys is not None:
        # 取交集，找出同時存在於 available_keys 和使用者要求中的模型
        selected &= set(keys)
        # 如果選出來的清單是空的，就拋出錯誤
        if not selected:
            raise RuntimeError("沒有選到任何已載入的模型")

    # 1. 統一圖像格式為 RGB 三通道
    raw = pil_img.convert("RGB")
    # 2. 模擬 MNIST 官方資料集的特徵轉換（重要特徵工程）:28x28 像素。黑底白字（背景數值為 0，線條數值高）。數字經過外接矩形裁切、縮放並置中於畫布。
    processed = preprocess_for_mnist(raw)
    # 3. 轉換成張量並加入 batch dimension（批次維度），即從 (28, 28) 變成 (1, 28, 28)
    '''
    _mnist_transform 的轉換步驟如下：
    transforms.ToTensor()：將 28x28 的 PIL 圖片轉為 PyTorch 的 torch.Tensor 型態。
    自動將像素數值從 $[0, 255]$ 縮放到 $[0.0, 1.0]$。將維度從 (H, W) 調整為 PyTorch 的通道優先格式 (C, H, W)，此時 Shape 為 (1, 28, 28)。
    transforms.Normalize((0.5,), (0.5,))：執行零均值正規化（Zero-center Normalization），計算公式：$$\text{output} = \frac{\text{input} - 0.5}{0.5}$$
    將像素數值區間從 $[0.0, 1.0]$ 映射到 $[-1.0, 1.0]$，這有助於神經網路的收斂與穩定計算。
    Step 3-2：.unsqueeze(0) 增加 Batch 維度原因：所有 PyTorch 神經網路模型（如 nn.Conv2d）都強制要求輸入包含 Batch 維度，
    標準輸入格式為 (Batch_Size, Channels, Height, Width)，簡稱 N C H W。
    作用：在第 0 個維度插入一個長度為 1 的維度。
    轉換前 Shape：torch.Size([1, 28, 28])（通道, 高, 寬）轉換後 Shape：torch.Size([1, 1, 28, 28])（批次大小=1, 通道=1, 高=28, 寬=28）
    '''
    t28 = _mnist_transform()(processed).unsqueeze(0)

    # 第一階段：Hugging Face 專屬前處理 (hf_inputs)
    hf_inputs = None
    # 檢查「預處理器元件是否載入成功」且「使用者選取的模型中，包含需要走 HF 前處理流程的模型（如 ViT 系列）」。
    if "v_3rd_proc" in _models and (selected & HF_STYLE_MODELS):
        # 將 PIL 圖片轉為張量並加入 batch dimension（批次維度），即從 (28, 28) 變成 (1, 28, 28), torch.Size([1, 3, 224, 224])（Batch=1, Channels=3, Height=224, Width=224）。
        hf_inputs = _models["v_3rd_proc"](images=raw, return_tensors="pt")

    # 第二階段：遍歷模型進行推論 (for 迴圈)
    results = []
    for key, model in _models.items():
        # 跳過 ViT 模型的標準前處理器（使用 Hugging Face 的預設設定）
        if key == "v_3rd_proc" or key not in selected:
            continue

        # 記錄推論開始時間
        start = time.perf_counter()
        # 關閉 gradient 計算，避免浪費記憶體與計算資源
        with torch.no_grad():
            # 檢查模型類型
            if key in HF_STYLE_MODELS:
                # Hugging Face 模型
                if hf_inputs is None:
                    continue
                # 取得 logits（未正規化輸出），Shape 為 (1, 10)
                # 使用 model(**hf_inputs) 解包字典傳入張量。
                # HF 模型的回傳值是一個物件（如 SequenceClassifierOutput），其分類未正規化分數存放在 .logits 中，Shape 為 [1, 10]。
                logits = model(**hf_inputs).logits
            else:
                # 傳統模型 (NN / CNN / 自建 ViT)：直接傳入前面處理好的 28x28 單通道張量 model(t28)，輸出同樣為 Shape [1, 10] 的 Logits。
                logits = model(t28)

            # torch.softmax(..., dim=1)：將未正規化的 Logits 轉換為和為 1.0 的機率分佈（Probability Distribution）。
            # [0]：取第一個 Batch 的結果（因為 Batch Size 為 1），將維度從 [1, 10] 降為 [10]。
            # .numpy()：將 PyTorch Tensor 轉為 NumPy Array 方便後續數值處理。
            probs = torch.softmax(logits, dim=1)[0].numpy()

        # elapsed_ms = ...：計算單個模型的純推論耗時，並乘以 1000 換算成毫秒 (ms)。
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 第四階段：打包單個模型結果
        # "prediction": int(np.argmax(probs))：找出機率最大的索引（0~9），即為模型預測的手寫數字。int() 轉換確保能被 JSON 序列化。
        # "confidence": float(np.max(probs))：取出最高機率值（信心度，例如 0.985 代表 98.5% 把握）。
        # "probabilities": [float(p) for p in probs]：將 10 個類別各自的機率轉為 Python float 串列，供前端畫長條圖（Bar Chart）或 Radar 圖。
        # "inference_ms": round(elapsed_ms, 2)：耗時四捨五入至小數點後兩位。
        results.append({
            "key": key,
            "display_name": DISPLAY_NAMES.get(key, key),
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "probabilities": [float(p) for p in probs],
            "inference_ms": round(elapsed_ms, 2),
        })


    # 第五階段：排序與格式化回傳
    # results.sort(..., reverse=True)：根據信心度 (confidence) 由高至低排序。這樣前端 UI 可以直接將預測最準確/最自信的模型顯示在最頂端。
    # 回傳一個包含「所有模型推論結果清單」與「OpenCV 前處理後的 28x28 PIL 圖像物件」的字典，讓 API 層能夠轉成 Base64 圖片並回傳給前端。
    # 總結資料結構變化
    '''
        [
        {
            "key": "cnn",
            "display_name": "SimpleCNN",
            "prediction": 5,
            "confidence": 0.9982,
            "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.9982, 0.0, 0.0, 0.0, 0.0018],
            "inference_ms": 1.42
        },
        {
            "key": "nn",
            "display_name": "SimpleNN",
            "prediction": 5,
            "confidence": 0.9120,
            "probabilities": [0.01, 0.0, 0.02, 0.03, 0.0, 0.912, 0.01, 0.0, 0.018, 0.0],
            "inference_ms": 0.35
        }
        ]
    '''
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return {"results": results, "processed_image": processed}


# 服務預熱（Model Warm-up）機制:在伺服器啟動、開放接收真實 HTTP 請求之前，先用「假的圖片」強制執行幾次推論，將所有一次性（Cold-start / Lazy Initialization）的延遲成本在啟動階段消化完畢。
'''
Lazy Memory Allocation（惰性記憶體配置）：PyTorch / C++ 底層在模型載入時，通常不會立刻分配推論過程所需的暫存記憶體（Intermediate Activation Buffers），而是在第一次執行 forward() 推論時才動態配置。

Execution Thread Pool Initialization（執行緒池初始化）：PyTorch 依賴底層的 OpenMP / MKL / OneDNN 等多線程加速庫。這些執行緒池（Thread Pools）是在第一次計算矩陣乘法時才創建與綁定的。

CUDNN / CPU Graph Auto-Tuning（自動優化編譯）：首次執行時，深度學習引擎會試圖尋找當前 CPU/GPU 架構下最快的卷積與矩陣運算演算法（Kernel Autotuning）。
'''
def warmup(rounds: int = 2) -> dict:
    """啟動後對所有已載入模型跑幾次 dummy 推論。

    把 PyTorch/MKL 執行緒池初始化、lazy 記憶體配置等一次性成本吃在啟動階段，
    避免第一個真實請求慢上好幾秒。失敗不擋服務啟動。
    """
    # 檢查全域變數 _models 字典是否為空。如果先前的模型載入步驟全數失敗或根本沒模型，則跳過預熱，回傳說明訊息，避免後續觸發 Exception。
    if not _models:
        return {"skipped": "沒有已載入的模型"}

    # 創建假圖片 (Dummy Input)
    # 使用 Pillow 建立一張 28x28 像素、全黑（color="black"）的 RGB PIL Image 物件。模擬使用者上傳的圖片。因為 predict() 函式要求傳入 PIL 圖像，這裡用一張合規的假圖片來滿足前處理與推論的輸入格式需求。
    dummy = Image.new("RGB", (28, 28), color="black")
    # 設定計時與迴圈次數
    timings = {}
    # rounds 參數：預設為 2。max(1, rounds)：防呆機制，確保預熱至少會執行 1 次（即使傳入 rounds=0 或負數）。
    # Round 1：吸收絕大多數一次性成本（記憶體配置、Thread Pool 建立、底層動態連結庫載入）。
    # Round 2+：確認所有 CPU Cache、MKL 權重快取與 Thread Pool 已經穩定運作，並可作為冷熱機耗時的比對基準。
    for i in range(max(1, rounds)):
        # 執行預熱推論與時間量測
        # 記錄該輪預熱開始的高精度時間戳。
        start = time.perf_counter()
        try:
            '''
            predict(dummy)：核心關鍵。直接呼叫後端的 predict 函式，這會觸發：

            OpenCV 影像預處理 (preprocess_for_mnist)。

            傳統 PyTorch 模型推論 (SimpleNN, SimpleCNN, MyViT)。

            Hugging Face ViT 模型的預處理與 forward() 推論。
            '''
            predict(dummy)
        except Exception as e:
            return {"error": str(e)}

        # 計算該輪推論總共花了多少毫秒 (ms)，保留小數點後 1 位，記錄到字典中。
        timings[f"round_{i + 1}_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return timings
