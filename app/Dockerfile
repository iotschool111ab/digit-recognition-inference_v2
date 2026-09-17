FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# opencv-python-headless 在部分 slim base image 上仍需要 libglib2.0-0
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 uv（用來依 pyproject.toml / uv.lock 建立可重現的環境）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# 預先下載固定不變的第三方 HF 模型，避免每次容器啟動都要連網路重下載
# （v_3rd 基準模型 + 微調版 ViT 承接用的基底架構，合計約 690MB；
#   跟前一個專題用的是同一組模型，判準與做法完全沿用）
ENV HF_HOME=/app/.cache/huggingface
RUN uv run python -c "\
from transformers import ViTForImageClassification, ViTImageProcessor; \
ViTForImageClassification.from_pretrained('farleyknight-org-username/vit-base-mnist'); \
ViTImageProcessor.from_pretrained('farleyknight-org-username/vit-base-mnist'); \
ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'); \
ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY backend ./backend
COPY frontend ./frontend
COPY architectures ./architectures

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]