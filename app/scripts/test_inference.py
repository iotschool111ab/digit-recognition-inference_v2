"""
MNIST 推論測試腳本（不需 Web UI）
=================================

路徑：app/scripts/test_inference.py
執行目錄：app/（與 pyproject.toml 同層）

兩種模式：
  local  直接 import backend.inference_service，在本行程內載入權重並推論（不需啟動 API）
  api    以 HTTP 呼叫已啟動的 FastAPI 服務（/api/health、/api/models、/api/predict）

三種輸入來源（可擇一或併用）：
  --image a.png b.jpg     指定圖片檔（白紙黑字、手機照片、畫布截圖皆可）
  --image-dir DIR         讀取資料夾內所有圖片；檔名開頭若是 0-9（例 7_xxx.png）會當成正確答案計算準確率
  --mnist N               取 MNIST 測試集前 N 張（torchvision 自動下載到 --data-dir），計算準確率
  --synthetic             用字型畫出 0-9 十張圖，做無外部檔案的功能煙霧測試

常用範例（PowerShell 或 Bash 皆同）：
  uv run python scripts/test_inference.py --synthetic
  uv run python scripts/test_inference.py --mnist 500 --models nn,cnn,vit_scratch --min-acc 0.9
  uv run python scripts/test_inference.py --mnist 500 --bypass-preprocess      # 不經前處理，量模型本身
  uv run python scripts/test_inference.py --image ../samples/7.png --save-processed out/
  uv run python scripts/test_inference.py --mode api --url http://localhost:8000 --mnist 100
  uv run python scripts/test_inference.py --mode api --url http://<EC2_HOST> --synthetic

Exit code：0 = 全部通過；1 = 推論或連線錯誤；2 = 參數錯誤（argparse）；3 = 沒有可用模型；4 = 準確率低於 --min-acc
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

APP_DIR = Path(__file__).resolve().parent.parent  # .../app
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------- 樣本來源
class Sample:
    def __init__(self, name: str, image: Image.Image, label: int | None, raw28: Image.Image | None = None):
        self.name, self.image, self.label = name, image, label
        self.raw28 = raw28  # 只有 MNIST 樣本有：原始 28x28 黑底白字，供 --bypass-preprocess 使用


def label_from_name(path: Path) -> int | None:
    c = path.name[:1]
    return int(c) if c.isdigit() else None


def load_images(paths: list[Path]) -> list[Sample]:
    out = []
    for p in paths:
        try:
            img = Image.open(p)
            img.load()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 無法讀取 {p}：{e}")
            continue
        out.append(Sample(p.name, img, label_from_name(p)))
    return out


def to_canvas_style(mnist_img: Image.Image) -> Image.Image:
    """MNIST 是黑底白字；服務的前處理假設「白紙黑字」（Otsu 反轉二值化），
    所以先反轉成白底黑字並放大到 280x280，模擬網頁畫布送出的圖。"""
    inv = Image.eval(mnist_img.convert("L"), lambda v: 255 - v)
    return inv.resize((280, 280), Image.BILINEAR)


def load_mnist(n: int, data_dir: Path) -> list[Sample]:
    from torchvision import datasets  # 專案相依已包含 torchvision

    ds = datasets.MNIST(root=str(data_dir), train=False, download=True)
    n = min(n, len(ds))
    return [Sample(f"mnist_test_{i:05d}", to_canvas_style(ds[i][0]), int(ds[i][1]), ds[i][0]) for i in range(n)]


def synthetic_digits() -> list[Sample]:
    """用 Pillow 內建字型畫 0-9（Pillow >= 10.1 支援 load_default(size)）。
    印刷體與手寫體分布不同，這組只當「流程能跑」的煙霧測試，不設準確率門檻。"""
    try:
        font = ImageFont.load_default(size=200)
    except TypeError:  # 舊版 Pillow
        font = ImageFont.load_default()
    out = []
    for d in range(10):
        img = Image.new("L", (280, 280), 255)
        draw = ImageDraw.Draw(img)
        x0, y0, x1, y1 = draw.textbbox((0, 0), str(d), font=font)
        draw.text(((280 - (x1 - x0)) / 2 - x0, (280 - (y1 - y0)) / 2 - y0), str(d), fill=0, font=font)
        out.append(Sample(f"synthetic_{d}", img, d))
    return out


# ---------------------------------------------------------------- 後端：local
class LocalBackend:
    def __init__(self, model_dir: str | None, offline: bool):
        if model_dir:
            os.environ["MODEL_DIR"] = str(Path(model_dir).resolve())
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(APP_DIR))
        import backend.inference_service as svc  # MODEL_DIR 在 import 時讀取，所以要先設環境變數

        self.svc = svc
        print(f"MODEL_DIR = {svc.MODEL_DIR}")
        t0 = time.perf_counter()
        self.status = svc.load_all_models()
        self.load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.warmup = svc.warmup()
        self.warmup_s = time.perf_counter() - t0

    def predict(self, img: Image.Image, keys: list[str] | None) -> tuple[list[dict], Image.Image]:
        r = self.svc.predict(img, keys)
        return r["results"], r["processed_image"]

    def predict_raw28(self, raw28: Image.Image, keys: list[str] | None) -> tuple[list[dict], Image.Image]:
        """跳過 OpenCV 前處理，把 MNIST 原圖直接送進 28x28 模型，用來量「模型本身」的準確率。
        與一般模式比較，差距就是前處理造成的影響。HF 224x224 模型不適用，會略過。"""
        import numpy as np
        import torch

        svc = self.svc
        t = svc._mnist_transform()(raw28.convert("L")).unsqueeze(0)
        results = []
        for key in svc.available_keys():
            if key in svc.HF_STYLE_MODELS or (keys and key not in keys):
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                probs = torch.softmax(svc._models[key](t), dim=1)[0].numpy()
            results.append({"key": key, "display_name": svc.DISPLAY_NAMES.get(key, key),
                            "prediction": int(np.argmax(probs)), "confidence": float(np.max(probs)),
                            "probabilities": [float(p) for p in probs],
                            "inference_ms": round((time.perf_counter() - t0) * 1000, 2)})
        if not results:
            raise RuntimeError("--bypass-preprocess 只適用 nn / cnn / vit_scratch")
        return results, raw28


# ---------------------------------------------------------------- 後端：api
class ApiBackend:
    def __init__(self, url: str, timeout: float):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.load_s = self.warmup_s = None
        self.warmup = None
        health = self._get("/api/health")
        if health.get("status") != "ok":
            raise RuntimeError(f"/api/health 回應異常：{health}")
        print(f"API = {self.url}  health = {health}")
        self.status = self._get("/api/models")

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def predict(self, img: Image.Image, keys: list[str] | None) -> tuple[list[dict], Image.Image | None]:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        boundary = uuid.uuid4().hex
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.png\"\r\n"
            f"Content-Type: image/png\r\n\r\n".encode() + buf.getvalue() + b"\r\n"
        ]
        if keys:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"models\"\r\n\r\n{','.join(keys)}\r\n".encode())
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.url + "/api/predict", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from None
        processed = None
        b64 = data.get("processed_image_base64", "")
        if b64.startswith("data:image/png;base64,"):
            processed = Image.open(io.BytesIO(base64.b64decode(b64.split(",", 1)[1])))
        return data["results"], processed


# ---------------------------------------------------------------- 主流程
def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser(description="MNIST 推論測試（不需 UI）")
    ap.add_argument("--mode", choices=["local", "api"], default="local")
    ap.add_argument("--url", default="http://localhost:8000", help="api 模式的服務位址")
    ap.add_argument("--timeout", type=float, default=120, help="api 模式每個請求逾時秒數")
    ap.add_argument("--model-dir", help="local 模式權重資料夾（預設 <repo>/models 或環境變數 MODEL_DIR）")
    ap.add_argument("--offline", action="store_true", help="local 模式不連 Hugging Face（HF 模型若無快取會被略過）")
    ap.add_argument("--models", help="逗號分隔，例 nn,cnn,vit_scratch；不給則跑全部已載入模型")
    ap.add_argument("--image", nargs="*", default=[], type=Path)
    ap.add_argument("--image-dir", type=Path)
    ap.add_argument("--mnist", type=int, default=0, help="使用 MNIST 測試集前 N 張")
    ap.add_argument("--data-dir", type=Path, default=APP_DIR / "data", help="MNIST 下載位置")
    ap.add_argument("--synthetic", action="store_true", help="字型產生 0-9 十張圖做煙霧測試")
    ap.add_argument("--bypass-preprocess", action="store_true",
                    help="local + --mnist 專用：跳過 OpenCV 前處理，直接以 MNIST 原圖測 28x28 模型本身的準確率")
    ap.add_argument("--min-acc", type=float, help="準確率門檻（只對有標籤樣本計算）；任一模型低於門檻 exit 4")
    ap.add_argument("--save-processed", type=Path, help="把前處理後的 28x28 圖存到此資料夾")
    ap.add_argument("--json", type=Path, help="把完整結果輸出成 JSON")
    ap.add_argument("--verbose", "-v", action="store_true", help="逐張列出每個模型的預測")
    args = ap.parse_args()

    if args.bypass_preprocess and (args.mode != "local" or not args.mnist):
        ap.error("--bypass-preprocess 只能用於 --mode local 並搭配 --mnist")

    # 1) 收集樣本
    samples: list[Sample] = load_images(args.image)
    if args.image_dir:
        samples += load_images(sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    if args.synthetic:
        samples += synthetic_digits()
    if args.mnist:
        try:
            samples += load_mnist(args.mnist, args.data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] MNIST 取得失敗：{str(e).strip().splitlines()[-1]}")
            print(f"       請確認可連外網，或手動把 MNIST 原始檔放到 {args.data_dir / 'MNIST' / 'raw'}")
            return 1
    if not samples:
        ap.error("沒有任何輸入，請至少指定 --image / --image-dir / --mnist / --synthetic 其中之一")
    if args.bypass_preprocess:
        samples = [s for s in samples if s.raw28 is not None]  # 只保留 MNIST 樣本

    # 2) 建立後端並列出模型狀態
    try:
        backend = LocalBackend(args.model_dir, args.offline) if args.mode == "local" else ApiBackend(args.url, args.timeout)
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        print(f"[FAIL] 無法連線或初始化：{e}")
        return 1
    print("\n模型狀態：")
    for k, v in backend.status.items():
        print(f"  {k:<22} {'OK' if v['loaded'] else '未載入 - ' + str(v['reason'])[:90]}")
    loaded = [k for k, v in backend.status.items() if v["loaded"]]
    keys = [k for k in args.models.split(",") if k] if args.models else None
    if keys:
        missing = [k for k in keys if k not in loaded]
        if missing:
            print(f"[WARN] 指定但未載入的模型會被略過：{missing}")
    if not (set(keys) & set(loaded) if keys else loaded):
        print("[FAIL] 沒有可用模型")
        return 3
    if backend.load_s is not None:
        print(f"\n模型載入 {backend.load_s:.2f}s；warm-up {backend.warmup_s:.2f}s {backend.warmup}")

    # 3) 逐張推論
    if args.save_processed:
        args.save_processed.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    records, errors = [], 0
    t_all = time.perf_counter()
    for s in samples:
        t0 = time.perf_counter()
        try:
            if args.bypass_preprocess:
                results, processed = backend.predict_raw28(s.raw28, keys)
            else:
                results, processed = backend.predict(s.image, keys)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"[ERROR] {s.name}: {e}")
            continue
        rtt_ms = (time.perf_counter() - t0) * 1000
        if args.save_processed and processed is not None:
            processed.save(args.save_processed / f"{Path(s.name).stem}_28x28.png")
        rec = {"name": s.name, "label": s.label, "request_ms": round(rtt_ms, 2), "results": []}
        for r in results:
            st = stats.setdefault(r["key"], {"name": r["display_name"], "n": 0, "correct": 0, "labeled": 0, "ms": []})
            st["n"] += 1
            st["ms"].append(r["inference_ms"])
            if s.label is not None:
                st["labeled"] += 1
                st["correct"] += int(r["prediction"] == s.label)
            rec["results"].append({k: r[k] for k in ("key", "prediction", "confidence", "inference_ms")})
        records.append(rec)
        if args.verbose:
            preds = "  ".join(f"{r['key']}={r['prediction']}({r['confidence']:.2f})" for r in results)
            print(f"  {s.name:<24} label={s.label if s.label is not None else '-'}  {preds}")
    total_s = time.perf_counter() - t_all

    # 4) 匯總
    print(f"\n樣本數 {len(samples)}，成功 {len(records)}，錯誤 {errors}，總耗時 {total_s:.2f}s")
    print(f"{'model':<22}{'accuracy':>12}{'avg ms':>10}{'p50 ms':>10}{'p95 ms':>10}")
    failed_acc = []
    summary = {}
    for k, st in stats.items():
        acc = st["correct"] / st["labeled"] if st["labeled"] else None
        acc_txt = f"{acc:.4f} ({st['correct']}/{st['labeled']})" if acc is not None else "n/a"
        avg = statistics.fmean(st["ms"])
        print(f"{k:<22}{acc_txt:>12}{avg:>10.2f}{pct(st['ms'], .5):>10.2f}{pct(st['ms'], .95):>10.2f}")
        summary[k] = {"display_name": st["name"], "accuracy": acc, "correct": st["correct"], "labeled": st["labeled"],
                      "avg_ms": round(avg, 3), "p50_ms": pct(st["ms"], .5), "p95_ms": pct(st["ms"], .95)}
        if args.min_acc is not None and acc is not None and acc < args.min_acc:
            failed_acc.append(k)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "mode": args.mode, "url": args.url if args.mode == "api" else None,
            "model_status": backend.status, "load_s": backend.load_s, "warmup": backend.warmup,
            "summary": summary, "errors": errors, "samples": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 結果：{args.json}")

    if errors:
        print("\n結果：FAIL（有推論錯誤）")
        return 1
    if failed_acc:
        print(f"\n結果：FAIL（準確率低於 {args.min_acc}：{failed_acc}）")
        return 4
    print("\n結果：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
