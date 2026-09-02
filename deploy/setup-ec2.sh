#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04 box (ARM or x86).
#
# Paste this into the EC2 browser terminal:
#
#   curl -fsSL https://raw.githubusercontent.com/DamnKuldeep/KnowYourRightsAI/main/deploy/setup-ec2.sh | bash
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

say() { STEP=$((STEP + 1)); echo; echo "═══ [$STEP/9] $* ═══"; }
ok()  { echo "  ✓ $*"; }
die() { echo; echo "  ✗ $*" >&2; exit 1; }

# ── 0. sanity ─────────────────────────────────────────────────────────────────────────
[ "$(id -u)" -ne 0 ] || die "Run this as the 'ubuntu' user, not root."
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
[ "$MEM_MB" -ge 3500 ] || die "This box has ${MEM_MB} MB of RAM. The embedding model needs ~2.3 GB
  free; use a t4g.medium (4 GB) or larger."
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
  read -rsp "  NVIDIA_API_KEY (nvapi-…): " NVIDIA_KEY; echo
  read -rsp "  OPENROUTER_API_KEY (sk-or-…, optional, Enter to skip): " OR_KEY; echo
  [ -n "${NVIDIA_KEY:-}" ] || [ -n "${OR_KEY:-}" ] || die "At least one key is required."
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
if swapon --show | grep -q swapfile; then
  ok "swap already present"
else
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  ok "2 GB swap enabled"
fi

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

# ── 6. configuration ──────────────────────────────────────────────────────────────────
say "Writing configuration"
if [ "$SKIP_ENV" -eq 0 ]; then
  cat > .env <<EOF
NVIDIA_API_KEY=${NVIDIA_KEY:-}
OPENROUTER_API_KEY=${OR_KEY:-}

# Local reranking. Do not use cpu_lean: every remote reranking endpoint currently 410s.
KYR_PROFILE=cpu
# Reranking is ~75% of retrieval time and scales with this. 12 keeps a 2-core box usable.
KYR_RERANK_POOL=12
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
MemoryMax=3500M
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
    echo
    echo "  Check it:   curl -s localhost:8000/api/health | head -c 200"
    echo "  Logs:       journalctl -u kyr -f"
    echo "  Restart:    sudo systemctl restart kyr"
    echo "════════════════════════════════════════════════════════════════"
    exit 0
  fi
  sleep 5
done

echo
echo "  Still loading. That is normal on a small box — check with:"
echo "      journalctl -u kyr -f"
