# PS D:\class\projects\AI\MNIST_train_model\app>  uv run uvicorn main:app --host 0.0.0.0 --port 8001 --reload
# 將後端的模型推論邏輯（inference_service.py）封裝成標準的 RESTful API、管理服務啟動與關閉的生命週期，
# 並處理前後端之間的檔案傳輸與資料格式轉換。
# base64 & io：處理圖片記憶體緩衝區（io.BytesIO）以及將處理後的圖片轉為 Base64 字串，方便直接嵌入 JSON 回傳給前端。
import base64
import io
# Python 的內建裝飾器，用於定義非同步的上下文管理器（Context Manager），在此用來實作 FastAPI 的生命週期管控。
from contextlib import asynccontextmanager
from pathlib import Path

'''
FastAPI：Web 應用程式的主體。

File, UploadFile：處理前端透過 multipart/form-data 上傳的圖片檔案。

Form：接收表單文字欄位（例如指定的模型名稱）。

HTTPException：拋出標準的 HTTP 錯誤代碼與訊息（如 400 Bad Request、503 Service Unavailable）。
'''
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
# FastAPI 提供的非同步執行工具，能將 CPU 密集型（CPU-bound）的同步函式丟到獨立的 ThreadPool 中執行，避免阻塞 Asyncio 的 Event Loop。
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
# 匯入先前撰寫好的模型載入與推論服務模組。
import inference_service as svc

# 取得當前 main.py 檔案所在的絕對路徑。
APP_DIR = Path(__file__).resolve().parent

# 二、 服務生命週期管理 (lifespan):處理 Web 伺服器啟動與關閉時的行為
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 伺服器啟動階段 (Startup)：
    print(f"正在從 {svc.MODEL_DIR} 載入模型...")
    # svc.load_all_models：呼叫後端模組載入所有 .pt 權重檔與 Hugging Face 模型。
    # 因為載入模型屬於重度 I/O 與 CPU 運算，使用 await run_in_threadpool(...) 確保不會卡住伺服器的啟動程序。
    status = await run_in_threadpool(svc.load_all_models)
    for key, info in status.items():
        state = "OK" if info["loaded"] else f"跳過（{info['reason']}）"
        print(f"  [{key}] {state}")

    print("正在暖機（warm-up）...")
    # svc.warmup：執行服務預熱，先跑一次假圖片推論，吃掉內存配置與 PyTorch 底層運算的冷啟動成本。
    timings = await run_in_threadpool(svc.warmup)
    print(f"  warm-up: {timings}")
    # 當 yield 執行後，代表伺服器已正式準備完畢，開放入口接收外來的 HTTP 請求。若在 yield 後寫程式碼，則會在伺服器關閉時（Shutdown）執行（例如釋放記憶體）。
    yield

# 實例化 API 應用與靜態檔案掛載
# FastAPI(..., lifespan=lifespan)：建立主應用程式，並註冊剛剛寫好的生命週期管理器。
app = FastAPI(title="MNIST Multi-Model Inference API", lifespan=lifespan)
# app.mount(...)：將本地的 static 資料夾掛載到 /static 路由，用來提供前端所需的靜態檔案（CSS、JavaScript、圖片等）。
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

# HTTP GET 路由定義
# 1. 前端首頁 (GET /)
# 功能：當使用者訪問根目錄時，直接讀取 static/index.html 檔案內容並以 HTML 格式回傳，實現單頁應用程式（SPA）的載入。
@app.get("/", response_class=HTMLResponse)
async def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")

# 健康檢查 (GET /api/health)
# 功能：提供負載平衡器（Load Balancer）、Docker 或 K8s 進行 Health Check 探針檢測，確認容器服務是否存活。
@app.get("/api/health")
async def health():
    return {"status": "ok"}

# 模型狀態查詢 (GET /api/models)
# 功能：回傳全域變數 _status 字典，讓前端知道有哪些模型載入成功、哪些缺檔，以便在介面上動態呈現模型切換選單。
@app.get("/api/models")
async def list_models():
    return svc.get_status()

# 核心預測 API (POST /api/predict)
# 接收參數：
# file: 前端傳入的圖片檔案（UploadFile 支援非同步流式讀取）。
# models: 可選的表單字串（如 "cnn,nn"），指定要跑哪幾個模型。
@app.post("/api/predict")
# async def predict(file: UploadFile = File(...), models: str = Form(default=None)):
async def predict(file: UploadFile = File(...), models: str = Form(default=None)):
    # 圖像解析：使用 file.read() 讀取二進位 byte 資料，
    contents = await file.read()
    try:
        # 並透過 io.BytesIO 轉換後給 Pillow (Image.open) 開啟。
        img = Image.open(io.BytesIO(contents))
    except Exception:
        # 若解析失敗則拋出 400 Bad Request。
        raise HTTPException(status_code=400, detail="無法解析圖片檔案")

    # 解析指定模型：將逗號分隔的字串切割成串列 keys（若無傳入則為 None）。
    keys = [k for k in (models or "").split(",") if k] or None

    try:
        # 推論是重 CPU 工作，丟到 threadpool 才不會卡住 event loop（多人同時用時很有感）
        # 執行異步推論：呼叫 svc.predict(img, keys)。
        # 重點：將計算負擔極重的 PyTorch 推論丟給 run_in_threadpool 處理。
        # 這樣當有多個使用者同時呼叫 API 時，FastAPI 仍然可以順暢接收新請求，而不會造成介面無回應。
        result = await run_in_threadpool(svc.predict, img, keys)
    except RuntimeError as e:
        # 若出錯則拋出 503 Service Unavailable。
        raise HTTPException(status_code=503, detail=str(e))

    # 圖片轉碼與 JSON 包裝：
    buf = io.BytesIO()
    # 取出經過 OpenCV 切割與置中處理後的 28x28 圖像（processed_image）。將其儲存至記憶體緩衝區 buf
    result["processed_image"].save(buf, format="PNG")
    # 再編碼為 Base64 字串。
    processed_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        # "results"：包含各模型的預測數字、信心度、10 個類別機率與推論毫秒數。
        "results": result["results"], 
        # "processed_image_base64": data:image/png;base64,... 格式字串，前端可以直接設定給 HTML <img src="..."> 標籤顯示前處理結果。
        "processed_image_base64": f"data:image/png;base64,{processed_b64}",
    }
