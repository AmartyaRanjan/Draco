# Ingestion worker image: RQ worker that converts uploaded PDFs with Docling, chunks, embeds (BGE-M3) and indexes into
# Qdrant + OpenSearch. Build from the repository root:
#   docker build -f deploy/worker/Dockerfile -t rag-worker .
#   docker build -f deploy/worker/Dockerfile --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t rag-worker .
# Models (Docling layout/OCR, BGE-M3) are downloaded on first use into $HF_HOME; mount a volume there so restarts do not
# re-download them, or pre-seed it.
FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Shared libraries used by Docling's image/OCR stack (OpenCV, pdfium).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
ARG TORCH_INDEX_URL=""
# Docling's layout model needs torchvision. It must come from the same index as torch: a PyPI torchvision on a
# CPU-index torch fails at runtime with "operator torchvision::nms does not exist". 0.29.x pairs with torch 2.14.x.
ARG TORCHVISION_VERSION=0.29.2
COPY requirements.txt .
RUN if [ -n "$TORCH_INDEX_URL" ]; then pip install --index-url "$TORCH_INDEX_URL" "$(grep -E '^torch==' requirements.txt)" "torchvision==$TORCHVISION_VERSION"; fi \
 && pip install -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps . && rm -rf src build \
 && python -c "import draco, docling, torchvision; from transformers import AutoImageProcessor; import torchvision.ops as o; o.nms"
COPY app ./app
RUN useradd --system --uid 10001 --home-dir /srv draco && mkdir -p app/storage /srv/.cache/huggingface && chown -R draco app/storage /srv/.cache
ENV HF_HOME=/srv/.cache/huggingface
USER draco
# Liveness for orchestrators: the worker registers itself in Redis; `rq info` is too heavy for a probe, so none is set.
CMD ["sh", "-c", "exec rqworker ingestion --url \"${REDIS_URL:?set REDIS_URL}\""]
