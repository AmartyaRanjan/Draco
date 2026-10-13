# API image. Not built or run in the session that wrote it; see production.md "Not verified".
# GPU embeddings/reranking need the NVIDIA runtime (docker run --gpus all). Ingestion workers are a separate process.
FROM python:3.14-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
RUN useradd --system --uid 10001 draco && mkdir -p app/storage && chown -R draco app/storage
USER draco
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"
# One worker: every worker loads BGE-M3 + ColBERT onto the GPU. Scale with replicas behind a proxy (rate limits are Redis-shared).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30", "--no-access-log"]
