# Deploying KnowYourRights for free

## What has to fit

| | Size | Can it move off the box? |
|---|---|---|
| `BAAI/bge-m3` embedder | ~2.3 GB download, ~2.3 GB RAM (fp32 CPU) | **No.** The corpus is embedded with it; a different model means re-embedding all 38,890 chunks. |
| Reranker | ~1.1 GB, and ~10s per query on a CPU | **Yes** — it reads text, not vectors. Push it to NIM. |
| LanceDB + parquet | ~330 MB | No, but it is just files. |
| LLMs | — | Already remote (NVIDIA NIM). |
| Chromium (crawl4ai) | ~400 MB RSS | Optional. `KYR_CRAWL_USE_BROWSER=false` keeps HTTP-only crawling. |

**Measured on 2 CPU cores:** local reranking → 10–12s per retrieval. Reranking on NIM →
**0.5–1.4s**. So every CPU deployment should use `KYR_PROFILE=cpu_lean`. Minimum viable box is
about **4 GB RAM**; 8 GB is comfortable.

---

## Option 1 — Google Colab + a tunnel (free GPU, ephemeral)

Best for showing it working. Free T4, so it runs at full speed, but the session dies after a
few hours and the URL changes each time.

```python
!git clone https://github.com/<you>/KnowYourRights.git && cd KnowYourRights
%cd KnowYourRights
!pip install -q -r requirements.txt
!pip install -q huggingface_hub

import os
os.environ["NVIDIA_API_KEY"] = "nvapi-..."      # or use Colab Secrets
os.environ["KYR_DATA_REPO"]  = "<you>/knowyourrights-db"
!python scripts/fetch_data.py
!python scripts/calibrate.py                     # thresholds for this machine's reranker

# free tunnel, no signup
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
import subprocess
subprocess.Popen(["python", "-m", "uvicorn", "knowyourrights.server:app",
                  "--host", "0.0.0.0", "--port", "8000"])
!./cloudflared tunnel --url http://localhost:8000
```

Cloudflare prints a `https://<random>.trycloudflare.com` URL. Kaggle works the same way and
gives 30 GPU-hours a week.

---

## Option 2 — Oracle Cloud Always Free (persistent, free forever)

The only genuinely free *persistent* box with enough memory. Note Oracle **halved** the Always
Free ARM allowance in June 2026 — it is now **2 OCPU / 12 GB**, not 4/24. Still plenty here.

1. Sign up (a card is needed for a $1 verification hold), pick a home region with Ampere A1
   capacity — Ashburn or London tend to have it. Capacity errors are common; retry.
2. Create an **Ampere A1 (ARM)** VM, Ubuntu 22.04, 2 OCPU / 12 GB, and open port 80/443 in
   both the VCN security list and `iptables`.
3. Install and run:

```bash
sudo apt update && sudo apt install -y python3-pip git
git clone https://github.com/<you>/KnowYourRights.git && cd KnowYourRights
pip install -r requirements.txt              # ARM64 wheels exist for everything here

echo "NVIDIA_API_KEY=nvapi-..." > .env
echo "KYR_PROFILE=cpu_lean"    >> .env
echo "KYR_CRAWL_USE_BROWSER=false" >> .env
echo "KYR_HOST=0.0.0.0"        >> .env

KYR_DATA_REPO=<you>/knowyourrights-db python scripts/fetch_data.py
python scripts/probe_nim.py
python scripts/calibrate.py

sudo tee /etc/systemd/system/kyr.service >/dev/null <<'EOF'
[Unit]
Description=KnowYourRights
After=network.target
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/KnowYourRights
ExecStart=/usr/bin/python3 -m knowyourrights.server
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now kyr
```

Put Caddy in front for HTTPS on a real domain (`caddy reverse-proxy --from your.domain --to :8000`).

---

## Option 3 — Hugging Face Spaces

Convenient and close to the model weights, **but check the current terms first**: as of this
writing HF's pricing page indicates Docker Spaces require PRO ($9/month), so this may no longer
be a free path. If your account does allow it:

- SDK `docker`, hardware **CPU basic** (2 vCPU / 16 GB).
- Add `NVIDIA_API_KEY` as a **Secret**, and `KYR_DATA_REPO` as a variable.
- The 50 GB disk is **wiped on rebuild**, which is why the Dockerfile bakes the embedder in.
- Set `KYR_PORT=7860` (already the Dockerfile default).
- Free Spaces sleep when idle; the first request after a sleep pays the model load (~30s).

---

## Option 4 — AWS (free credits, not free forever)

AWS changed its free tier: accounts opened after **15 July 2025** no longer get the 12-month
free EC2 allowance. New accounts get **$200 in credits** ($100 on signup, $100 more for using
services) which draw down against normal pricing.

`t3.micro` (the old free instance) has **1 GB RAM** and cannot load bge-m3. You need
`t3.medium` (4 GB, ~$30/mo) or `t3.small` (2 GB, ~$15/mo, tight). With $200 of credits a
`t3.medium` runs for roughly **six months**, then starts billing — set a budget alert.

```bash
# Ubuntu 22.04, t3.medium, 20 GB gp3, security group open on 80/443
sudo apt update && sudo apt install -y python3-pip git
git clone https://github.com/<you>/KnowYourRightsAI.git && cd KnowYourRightsAI
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
# then identical to the Oracle steps below: .env, fetch_data, probe, calibrate, systemd
```

Use Lightsail ($5/mo, 1 GB — too small) only if you move the embedder off-box.

---

## Why Render (and Vercel, Railway, Fly) cannot host this

Render's free web service is **512 MB RAM and 0.1 CPU**, and spins down after 15 minutes idle.
`BAAI/bge-m3` needs about **2.3 GB** resident. Quantising to int8 lands near 600 MB of weights
plus runtime overhead — still over the limit, and the corpus is embedded with the full-precision
model, so a quantised encoder would drift from the indexed vectors anyway.

There is no configuration in which this app starts on 512 MB. Render's **Standard** plan
(2 GB, ~$25/mo) is the first tier that works, and Oracle gives you 12 GB for nothing.

Vercel and Netlify are serverless with short execution limits and no persistent process, so a
30-second deep-research turn with resident models does not fit at all.

| Host | Free RAM | Runs this? |
|---|---|---|
| **Oracle Always Free** | 12 GB (2 OCPU ARM) | **Yes — free forever** |
| **Colab / Kaggle** | 12–16 GB + T4 GPU | **Yes — but ephemeral** |
| AWS (new account) | $200 credits, ~6 mo on t3.medium | Yes, then billed |
| Hugging Face Spaces | 16 GB CPU basic | Docker may need PRO — verify |
| Render free | 512 MB | No |
| Railway / Fly.io free | 256–512 MB | No |
| Vercel / Netlify | serverless | No |

---

## After deploying, always

```bash
python scripts/probe_nim.py     # pin the models this key can reach
python scripts/calibrate.py     # REQUIRED after any profile change
```

Thresholds are stored per backend *and* model, so the values calibrated for your local GPU
reranker do not apply to the NIM one. Skipping this leaves the deployment on uncalibrated
defaults, which is how off-topic questions start getting confident answers.

## Sensible production settings

```bash
KYR_PROFILE=cpu_lean
KYR_CRAWL_USE_BROWSER=false        # saves ~400 MB; loses JS-only portals
KYR_DEFAULT_DEPTH=standard         # cap research depth for shared instances
KYR_SESSION_CREDIT_BUDGET=800      # auto-downshift depth as free credits run down
KYR_MODEL_IDLE_EVICT_S=1800        # give memory back when idle
KYR_LOG_LEVEL=WARNING
```

One worker only — a second would load a second copy of bge-m3.

## Cost reality

Everything above is free, but the **NVIDIA free tier is metered**: roughly 1,000 credits, about
one credit per request, and a legal question costs 4–8. That is a few hundred questions. Watch
`/api/usage`, and set `KYR_SESSION_CREDIT_BUDGET` so the agent downshifts before it runs dry
rather than failing afterwards.
