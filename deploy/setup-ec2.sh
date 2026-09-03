#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04 or 24.04 box, ARM or x86.
#
# 24.04 ships Python 3.12 with PEP 668 enforced, so a bare `pip install` is refused. Every
# install here goes through a venv, which is unaffected — verified on 24.04.4 / 3.12.3.
#
# Paste this into the EC2 browser terminal — download first, then run:
#
#   curl -fsSL https://raw.githubusercontent.com/DamnKuldeep/KnowYourRightsAI/main/deploy/setup-ec2.sh -o setup.sh
#   bash setup.sh
#
# Not `curl … | bash`. Piping makes this script bash's stdin, so the `read` prompts below would
# consume the script's own text instead of what you type. The prompts now read /dev/tty directly
# so the pipe form works too, but downloading first is the form to prefer: it is one file you can
# read before running, and it cannot be truncated halfway by a dropped connection.
#
# It asks for your two API keys, installs everything, builds the vector index, calibrates the
# abstention thresholds, and leaves the app running as a service. Roughly 20 minutes, most of
# it downloading the embedding model.
#
# Safe to re-run: it skips work that is already done.

set -euo pipefail

REPO="https://github.com/DamnKuldeep/KnowYourRightsAI.git"
APP_DIR="$HOME/KnowYourRightsAI"
STEP=0

say() { STEP=$((STEP + 1)); echo; echo "═══ [$STEP/10] $* ═══"; }
ok()  { echo "  ✓ $*"; }
die() { echo; echo "  ✗ $*" >&2; exit 1; }

# ── 0. sanity ─────────────────────────────────────────────────────────────────────────
[ "$(id -u)" -ne 0 ] || die "Run this as the 'ubuntu' user, not root."
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
if [ "$MEM_MB" -lt 3500 ]; then
  echo
  echo "  This box has ${MEM_MB} MB of RAM and the embedding model needs ~2.3 GB free."
  echo "  Two ways forward:"
  echo "    · use a 4 GB+ instance (m7i-flex.large, c7i-flex.large, t4g.medium), or"
  echo "    · run the lite profile here — no models at all, BM25 only, Recall@5 90.5%"
  echo "      instead of 100%, in under 1 GB. Re-run with:  KYR_FORCE_LITE=1"
  echo
  [ "${KYR_FORCE_LITE:-0}" = "1" ] || die "Stopping. Pick a bigger box, or set KYR_FORCE_LITE=1."
fi
# `cpu` keeps the cross-encoder, because it is what makes the system refuse questions it cannot
# answer: with it, 10 of the 11 stress questions are caught; without it, only 5. On a tool that
# cites statute at people, losing that is worse than being slow. The cost is bounded by reranking
# only 8 candidates instead of 24 (see KYR_RERANK_POOL below).
# If answers still take too long on this box, `KYR_PROFILE=cpu_lean` drops the cross-encoder for
# ~90 ms retrieval at Recall@5 93% — and weaker abstention. One line in .env, then restart.
PROFILE="cpu"
[ "${KYR_FORCE_LITE:-0}" = "1" ] && PROFILE="lite"
ok "$(nproc) CPU(s), ${MEM_MB} MB RAM, $(uname -m)"

# ── 1. keys ───────────────────────────────────────────────────────────────────────────
say "API keys"
if [ -f "$APP_DIR/.env" ] && grep -q "nvapi-\|sk-or-" "$APP_DIR/.env" 2>/dev/null; then
  ok "keeping the .env already on this box"
  SKIP_ENV=1
else
  SKIP_ENV=0
  echo "  Both are free. Paste and press Enter (input is hidden)."
  echo "  NVIDIA:     https://build.nvidia.com   OpenRouter: https://openrouter.ai/keys"
  echo

  # Terminals with bracketed paste on (the EC2 browser console among them) wrap a paste in
  # ESC[200~ … ESC[201~. `read` captures those markers as part of the value, so the key silently
  # becomes unusable — the provider then reports "no key", or httpx refuses to encode the
  # Authorization header at all. Turn the mode off, and keep only characters an API key can
  # contain, so nothing invisible survives into .env.
  printf '\033[?2004l' 2>/dev/null || true
  # Strip whole escape sequences before filtering characters. Removing the punctuation first
  # would leave the digits behind — ESC[200~ would become a literal "200" glued to the key,
  # which is corruption that looks like a valid key.
  clean_key() {
    printf '%s' "${1:-}" \
      | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z~]//g' \
      | tr -cd 'A-Za-z0-9._-'
  }

  # Read the keyboard, not stdin. Under `curl … | bash` the script *is* bash's stdin, so a bare
  # `read` consumes the script's own bytes: the key silently became a fragment of this file
  # ("10 chars, starts 2.package"), every API call failed 401, and the installer stopped early
  # because the lines `read` had eaten never executed. /dev/tty is the terminal regardless of
  # what stdin is piped from.
  if [ -r /dev/tty ]; then
    read -rsp "  NVIDIA_API_KEY (nvapi-…): " NVIDIA_KEY < /dev/tty; echo
    read -rsp "  OPENROUTER_API_KEY (sk-or-…, optional, Enter to skip): " OR_KEY < /dev/tty; echo
  else
    # No terminal at all — a CI runner or `bash < script`. Take them from the environment
    # rather than reading garbage and pretending it worked.
    NVIDIA_KEY="${NVIDIA_API_KEY:-}"
    OR_KEY="${OPENROUTER_API_KEY:-}"
    [ -n "$NVIDIA_KEY$OR_KEY" ] || die "No terminal to read keys from. Either run this from a
  terminal, or pass them in:
      NVIDIA_API_KEY=nvapi-… OPENROUTER_API_KEY=sk-or-… bash setup-ec2.sh"
  fi
  NVIDIA_KEY=$(clean_key "${NVIDIA_KEY:-}")
  OR_KEY=$(clean_key "${OR_KEY:-}")
  [ -n "$NVIDIA_KEY" ] || [ -n "$OR_KEY" ] || die "At least one key is required."

  # Say what was actually captured. A truncated or empty key is far cheaper to notice here than
  # twenty minutes later when every model probe fails.
  # Shape check. Both providers use a fixed prefix, so a value that lacks it is not a key that
  # will fail later — it is something else entirely, and saying so now costs one line instead of
  # twenty minutes and a 401 for every model.
  if [ -n "$NVIDIA_KEY" ] && [ "${NVIDIA_KEY#nvapi-}" = "$NVIDIA_KEY" ]; then
    die "That NVIDIA key does not start with 'nvapi-' (got ${#NVIDIA_KEY} chars starting
  '${NVIDIA_KEY:0:12}'). Nothing was written. Re-run and paste the key from
  https://build.nvidia.com — or press Enter to skip NVIDIA and use OpenRouter alone."
  fi
  if [ -n "$OR_KEY" ] && [ "${OR_KEY#sk-or-}" = "$OR_KEY" ]; then
    die "That OpenRouter key does not start with 'sk-or-' (got ${#OR_KEY} chars starting
  '${OR_KEY:0:12}'). Nothing was written. Re-run and paste the key from
  https://openrouter.ai/keys — or press Enter to skip OpenRouter and use NVIDIA alone."
  fi

  [ -n "$NVIDIA_KEY" ] && ok "NVIDIA key: ${#NVIDIA_KEY} chars, starts ${NVIDIA_KEY:0:6}" \
                       || echo "  · no NVIDIA key — OpenRouter only"
  [ -n "$OR_KEY" ] && ok "OpenRouter key: ${#OR_KEY} chars, starts ${OR_KEY:0:9}" \
                   || echo "  · no OpenRouter key — NVIDIA only"
fi

# ── 2. packages ───────────────────────────────────────────────────────────────────────
say "Installing system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pip python3-venv git git-lfs curl >/dev/null
git lfs install --skip-repo >/dev/null
ok "python, git, git-lfs"

# ── 3. swap ───────────────────────────────────────────────────────────────────────────
# The model load spikes host RAM to ~2.3 GB. On a 4 GB box that is the likeliest way to get
# OOM-killed mid-startup, and swap is cheap insurance.
say "Adding swap"
if [ "$MEM_MB" -ge 6000 ]; then
  ok "skipped — ${MEM_MB} MB is comfortably above the ~2.3 GB load spike"
elif swapon --show | grep -q swapfile; then
  ok "swap already present"
else
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  ok "2 GB swap enabled"
fi

# A hard ceiling so a runaway process cannot take the machine down, set from what this box
# actually has rather than a constant — 3500M would needlessly cap an 8 GB instance.
# The floor is 3500M because that is the ceiling actually validated on a 4 GB box: the app
# peaks near 3 GB while bge-m3 loads, and a tighter limit would have systemd kill a healthy
# process. Bigger boxes simply get more headroom.
MEM_LIMIT=$(( MEM_MB * 70 / 100 ))
[ "$MEM_LIMIT" -lt 3500 ] && MEM_LIMIT=3500

# ── 4. code + corpus ──────────────────────────────────────────────────────────────────
# git-lfs must be initialised BEFORE cloning or the 350 MB corpus arrives as 134-byte pointer
# files and the app fails later with a confusing "database not found".
say "Cloning the repository and the ~350 MB corpus (a few minutes)"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only && ok "updated existing clone"
else
  git clone --quiet "$REPO" "$APP_DIR" && ok "cloned"
fi
cd "$APP_DIR"
git lfs pull

DB_SIZE=$(du -sm data/legal_db 2>/dev/null | cut -f1 || echo 0)
[ "$DB_SIZE" -gt 100 ] || die "The corpus is only ${DB_SIZE} MB — Git LFS did not fetch it.
  Run:  cd $APP_DIR && git lfs install && git lfs pull"
ok "corpus present (${DB_SIZE} MB)"

# ── 5. python deps ────────────────────────────────────────────────────────────────────
say "Installing Python dependencies (several minutes)"
[ -d .venv ] || python3 -m venv .venv
# CPU-only torch: the default wheel drags in ~2.5 GB of CUDA runtime that is useless here.
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
./.venv/bin/pip install -q -r requirements.txt
ok "dependencies installed"

# Size is not integrity. A clone can be the full 299 MB and still be unopenable if one small
# bookkeeping file is missing — a stale .gitignore rule caused exactly that, and the failure only
# surfaced at calibration, after a ten-minute model download, as an unreadable Lance error
# repeated once per query. Opening the table costs a second and turns that into a clear message
# before anything expensive happens. This is the first point where lancedb exists to do it.
if ! ./.venv/bin/python - <<'PY'
import sys
try:
    import lancedb
    rows = lancedb.connect("data/legal_db").open_table("laws").count_rows()
except Exception as exc:                        # noqa: BLE001 — any failure here is fatal
    print(f"    {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
    sys.exit(1)
print(f"    {rows:,} rows readable")
sys.exit(0 if rows > 30000 else 1)
PY
then
  die "The corpus is the right size but will not open — the clone is incomplete.
  Try:  cd $APP_DIR && git pull && git lfs pull
  If it still fails, delete $APP_DIR and re-run this script for a clean clone."
fi
ok "corpus opens and is complete"

# ── 6. configuration ──────────────────────────────────────────────────────────────────
say "Writing configuration"
if [ "$SKIP_ENV" -eq 0 ]; then
  cat > .env <<EOF
NVIDIA_API_KEY=${NVIDIA_KEY:-}
OPENROUTER_API_KEY=${OR_KEY:-}

# Retrieval profile. cpu_lean = dense + BM25, no cross-encoder: Recall@5 93% in ~90 ms.
# KYR_PROFILE=cpu adds the cross-encoder for Recall@5 100%, at ~20 s a question on 2 vCPU.
KYR_PROFILE=${PROFILE}
# 8, not the 12 this used to set. Measured on the gold set, 8 is better than 12 on every axis:
#   pool 24  Recall@5 100%    MRR 0.873   top-1 79%   off-topic caught 11/11   1.00x time
#   pool 12  Recall@5  95.2%  MRR 0.849   top-1 76%   off-topic caught 10/11   0.57x
#   pool  8  Recall@5  95.2%  MRR 0.861   top-1 79%   off-topic caught 10/11   0.32x
# Same recall as 12, better ranking, and nearly twice as fast — 12 was simply a bad pick. The
# remaining 4.8 points to reach 100% cost 3x the reranking time, which a 2-vCPU box cannot spare.
KYR_RERANK_POOL=8
KYR_HOST=127.0.0.1
KYR_CRAWL_USE_BROWSER=false
KYR_DEFAULT_DEPTH=standard
KYR_SESSION_CREDIT_BUDGET=800
KYR_MODEL_IDLE_EVICT_S=1800
KYR_LOG_LEVEL=WARNING
EOF
  chmod 600 .env
fi
ok ".env written"

# Everything under .runtime/cache is derived — query embeddings, search results, crawled pages.
# On a re-run it is stale by definition: the profile may have changed, the reranker may have
# changed, and a cached search result computed under the old settings would quietly survive and
# make the new configuration look like the old one. Thresholds and the usage ledger are kept.
if [ -d .runtime/cache ]; then
  CACHE_MB=$(du -sm .runtime/cache 2>/dev/null | cut -f1 || echo 0)
  rm -rf .runtime/cache
  ok "cleared ${CACHE_MB} MB of stale cache (thresholds and usage ledger kept)"
fi
rm -rf .crawl4ai/cache 2>/dev/null || true

# ── 7. prepare the index and thresholds ───────────────────────────────────────────────
say "Building the vector index (~30 s) and downloading the embedding model (~10 min)"
./.venv/bin/python scripts/build_index.py
ok "vector index ready — dense search ~23 ms instead of ~300 ms"

./.venv/bin/python scripts/probe_models.py || echo "  (model probe had trouble; continuing)"

# Thresholds are specific to the reranker in use. Skipping this leaves inherited defaults,
# which is how off-topic questions start getting confident answers.
./.venv/bin/python scripts/calibrate.py
ok "abstention thresholds calibrated"

# ── 8. run as a service ───────────────────────────────────────────────────────────────
say "Installing the service"
sudo tee /etc/systemd/system/kyr.service >/dev/null <<EOF
[Unit]
Description=KnowYourRightsAI
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment=HF_HOME=$HOME/.cache/huggingface
ExecStart=$APP_DIR/.venv/bin/python -m knowyourrights.server
Restart=always
RestartSec=15
TimeoutStartSec=300
MemoryMax=${MEM_LIMIT}M
OOMPolicy=stop

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now kyr >/dev/null
ok "service installed and started"

# ── 9. idle shutdown ──────────────────────────────────────────────────────────────────
say "Installing the idle timer"
sudo cp deploy/idle-shutdown.sh /usr/local/bin/ && sudo chmod +x /usr/local/bin/idle-shutdown.sh
( crontab -l 2>/dev/null | grep -v idle-shutdown; \
  echo "*/5 * * * * /usr/local/bin/idle-shutdown.sh" ) | crontab -
ok "the box will stop itself 30 minutes after the last question"

# ── cloudflared ───────────────────────────────────────────────────────────────────────
# Architecture matters here and nowhere else in this script: t4g is ARM, c7i/m7i/t3 are x86.
say "Installing cloudflared (for a public URL)"
if command -v cloudflared >/dev/null; then
  ok "already installed"
else
  CF_ARCH=$([ "$(uname -m)" = "aarch64" ] && echo arm64 || echo amd64)
  if curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}"        -o /tmp/cloudflared; then
    sudo install -m 755 /tmp/cloudflared /usr/local/bin/cloudflared && ok "cloudflared (${CF_ARCH})"
  else
    echo "  (download failed — install it later if you want a public URL)"
  fi
fi

# ── wait for readiness ────────────────────────────────────────────────────────────────
echo
echo "Waiting for the models to load (up to 3 minutes on a cold box)…"
for _ in $(seq 1 36); do
  if curl -fsS http://127.0.0.1:8000/api/health 2>/dev/null | grep -q '"ready": *true'; then
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  KnowYourRightsAI is running on this machine."
    echo
    echo "  Next: expose it with a public URL —"
    echo "      cloudflared tunnel --url http://localhost:8000"
    echo "  (cloudflared is already installed for this machine's architecture)"
    echo
    echo "  Check it:   curl -s localhost:8000/api/health | head -c 200"
    echo "  Logs:       journalctl -u kyr -f"
    echo "  Restart:    sudo systemctl restart kyr"
    echo
    echo "  Measure THIS machine (free — no API calls, ~15-25 min on 2 vCPU):"
    echo "      ./.venv/bin/python scripts/benchmark.py --all"
    echo "      ./.venv/bin/python scripts/deploy_report.py --out DEPLOYED.md"
    echo "  Accuracy should match the published numbers exactly; only latency should move."
    echo "════════════════════════════════════════════════════════════════"
    exit 0
  fi
  sleep 5
done

echo
echo "  Still loading. That is normal on a small box — check with:"
echo "      journalctl -u kyr -f"
