# KnowYourRights — CPU image for free-tier hosting.
#
# The corpus is embedded with bge-m3, so that model must run locally and ships in the image.
# Reranking is the expensive half on a CPU (measured ~10s for 24 documents), so the default
# profile is `cpu_lean`: embed locally, rerank on NIM. That takes retrieval back to ~1s.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    KYR_PROFILE=cpu_lean \
    KYR_RAM_FLOOR_MB=800 \
    KYR_CRAWL_USE_BROWSER=false \
    KYR_HOST=0.0.0.0 \
    KYR_PORT=7860

# `curl` is for the healthcheck; `git` is needed to pull the data bundle from a HF dataset.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git git-lfs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch is ~200 MB instead of ~2.5 GB with the bundled CUDA runtime.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

# Bake the embedder in. Free hosts wipe their disk on rebuild, so downloading 2.3 GB at first
# request would make every cold start unusable.
ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; \
                 SentenceTransformer('BAAI/bge-m3', trust_remote_code=True)"; \
    fi

COPY knowyourrights/ ./knowyourrights/
COPY scripts/ ./scripts/

# The ~330 MB database is not in git. Provide it at build time by either:
#   (a) COPY data/ ./data/                       — if you vendored it, or
#   (b) setting KYR_DATA_REPO to a Hugging Face dataset repo and letting entrypoint.sh pull it.
COPY data/ ./data/

RUN mkdir -p /app/.runtime && chmod -R 777 /app/.runtime /opt/hf

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD curl -fsS http://localhost:${KYR_PORT}/api/health || exit 1

CMD ["python", "-m", "knowyourrights.server"]
