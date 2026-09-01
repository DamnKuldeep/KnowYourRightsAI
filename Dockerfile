# KnowYourRightsAI — CPU image, works on x86_64 and ARM64 (Graviton).
#
# Deliberately does NOT bake in the corpus or the model weights:
#
#   * the corpus is ~350 MB and already on the host after `git clone` — mount it read-only
#   * bge-m3 is ~2.3 GB and belongs in a named volume, so it survives image rebuilds and is
#     downloaded once rather than on every `docker build`
#
# Measured result: **3.34 GB**. Baking both in would put it near 6 GB. Most of what remains is
# unavoidable — torch, transformers and sentence-transformers are large even in CPU form.
#
# Windows note: `docker run -v /d/path:/data:ro` from Git Bash silently mangles the mount into
# `C:\Program Files\Git\data`. Use `docker compose` (which does no path translation) or prefix
# the command with MSYS_NO_PATHCONV=1.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    KYR_HOST=0.0.0.0 \
    KYR_PORT=8000 \
    KYR_DATA_DIR=/data \
    KYR_RUNTIME_DIR=/runtime

# curl for the healthcheck; libgomp1 is required by torch's CPU kernels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── dependencies ──────────────────────────────────────────────────────────────────────
# CPU-only torch first: the default wheel drags in ~2.5 GB of CUDA runtime that is useless
# in a CPU container. Its own layer so code changes don't reinstall it.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
# torch is already installed above; strip it so pip doesn't pull the CUDA wheel over the top.
RUN grep -v '^torch' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

# ── application ───────────────────────────────────────────────────────────────────────
COPY knowyourrights/ ./knowyourrights/
COPY scripts/ ./scripts/

# Run as a non-root user; the mounted volumes need to be writable by it.
RUN useradd --create-home --uid 10001 kyr \
    && mkdir -p /models /runtime /data \
    && chown -R kyr:kyr /app /models /runtime
USER kyr

EXPOSE 8000

# Generous start period: on a cold CPU box the models take up to ~140 s to load, and the
# server deliberately reports ready:false until they are warm.
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=5 \
  CMD curl -fsS "http://localhost:${KYR_PORT}/api/health" || exit 1

CMD ["python", "-m", "knowyourrights.server"]
