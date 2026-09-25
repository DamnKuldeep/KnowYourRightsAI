# KnowYourRights — one container, no GPU, no model weights: every model runs behind an API.
#
# The ~400 MB corpus is not baked in. It is already on the host after `git lfs pull`, so it is
# mounted read-only (see docker-compose.yml) and a code change never re-copies it.
#
# Chromium is installed for the pages that only render with JavaScript. Build with
# --build-arg INSTALL_BROWSER=false for a smaller image that reads static pages only, and set
# KYR_CRAWL_USE_BROWSER=false to match.

FROM python:3.12-slim

ARG INSTALL_BROWSER=true

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    KYR_HOST=0.0.0.0 \
    KYR_PORT=8000 \
    KYR_DATA_DIR=/data \
    KYR_RUNTIME_DIR=/runtime

WORKDIR /app

# Dependencies first, in their own layer, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_BROWSER" = "true" ]; then \
         python -m playwright install --with-deps chromium; \
       fi \
    && rm -rf /var/lib/apt/lists/*

COPY knowyourrights/ ./knowyourrights/

# Run as an unprivileged user; the runtime volume must be writable by it.
RUN useradd --create-home --uid 10001 kyr \
    && mkdir -p /runtime /data \
    && chown -R kyr:kyr /app /runtime
USER kyr

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/api/health', timeout=4).status != 200)"

CMD ["python", "-m", "knowyourrights.server"]
