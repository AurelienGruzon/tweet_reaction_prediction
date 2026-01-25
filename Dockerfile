FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deps système minimales (certifs SSL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# 1) Installer PyTorch CPU (sinon pip récupère la version CUDA)
# IMPORTANT: index CPU PyTorch
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# 2) Installer deps API
COPY requirements.api.txt ./
RUN pip install --no-cache-dir -r requirements.api.txt

# 3) Copier code + modèles
COPY src ./src
COPY models/best ./models/best
COPY models/best_tokenizer ./models/best_tokenizer

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

