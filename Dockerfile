# API image: FastAPI app (`app/`) plus the Draco framework package (`src/draco`, installed with pip, not from source path).
#   docker build -t rag-api .                                                        # default PyPI torch (CUDA wheels)
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t rag-api .   # CPU-only, much smaller
# GPU embeddings/reranking need the NVIDIA runtime (docker run --gpus all). Ingestion runs in the separate worker image
# (deploy/worker/Dockerfile).
FROM python:3.14-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv
ARG TORCH_INDEX_URL=""
COPY requirements.txt .
# Optional: take torch from another index (e.g. the CPU wheel index) before resolving the rest, so pip keeps that build.
# Docling (PDF conversion) is left out: the API never converts documents, the worker image does (and docling 2.137 also
# conflicts with the pillow pin in requirements.txt; the worker image resolves that).
RUN if [ -n "$TORCH_INDEX_URL" ]; then pip install --index-url "$TORCH_INDEX_URL" "$(grep -E '^torch==' requirements.txt)"; fi \
 && grep -v -E '^docling==' requirements.txt > /tmp/requirements-api.txt \
 && pip install -r /tmp/requirements-api.txt && rm /tmp/requirements-api.txt
# The ASGI server is not in requirements.txt; pinned to the version the app is developed against.
RUN pip install "uvicorn[standard]==0.54.0"
# The framework package. The app imports draco.* (e.g. draco.tools.safety); install it as a regular (non-editable) package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf src build && python -c "import draco, draco.tools.safety"
COPY app ./app
COPY scripts ./scripts
RUN useradd --system --uid 10001 --home-dir /srv draco && mkdir -p app/storage /srv/.cache/huggingface && chown -R draco app/storage /srv/.cache
ENV HF_HOME=/srv/.cache/huggingface
USER draco
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"
# One worker: every worker loads BGE-M3 + ColBERT onto the GPU. Scale with replicas behind a proxy (rate limits are Redis-shared).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30", "--no-access-log"]
