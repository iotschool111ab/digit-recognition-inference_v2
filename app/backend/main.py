import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import backend.inference_service as svc

APP_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"正在從 {svc.MODEL_DIR} 載入模型...")
    status = await run_in_threadpool(svc.load_all_models)
    for key, info in status.items():
        state = "OK" if info["loaded"] else f"跳過（{info['reason']}）"
        print(f"  [{key}] {state}")

    print("正在暖機（warm-up）...")
    timings = await run_in_threadpool(svc.warmup)
    print(f"  warm-up: {timings}")
    yield


app = FastAPI(title="MNIST Multi-Model Inference API", lifespan=lifespan)
app.mount("/frontend", StaticFiles(directory=APP_DIR / "frontend"), name="frontend")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (APP_DIR / "frontend" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models")
async def list_models():
    return svc.get_status()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), models: str = Form(default=None)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="無法解析圖片檔案")

    keys = [k for k in (models or "").split(",") if k] or None

    try:
        # 推論是重 CPU 工作，丟到 threadpool 才不會卡住 event loop（多人同時用時很有感）
        result = await run_in_threadpool(svc.predict, img, keys)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    buf = io.BytesIO()
    result["processed_image"].save(buf, format="PNG")
    processed_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "results": result["results"],
        "processed_image_base64": f"data:image/png;base64,{processed_b64}",
    }
