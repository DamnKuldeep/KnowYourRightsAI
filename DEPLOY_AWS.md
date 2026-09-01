# Deploying on AWS — a demo that wakes on demand

Goal: a link you can share, that costs almost nothing between demos, and does not fall over
during one.

**Your situation:** $100 of credit, expiring in **182 days**. That is the number that shapes
everything below. There is no point stretching the money over two years — it expires first.
$100 / 182 days is about **$0.55 a day** of headroom, which is far more than this needs.

---

## The short version

| | |
|---|---|
| Instance | `t4g.medium` (ARM Graviton, 2 vCPU, 4 GB) — **stopped by default** |
| Wake | A Lambda Function URL that starts it when someone opens the link |
| Sleep | A cron job on the box that stops it after 30 minutes with no questions |
| Cost between demos | **~$2.40/month** (the EBS volume, nothing else) |
| Cost during demos | **$0.034/hour** |
| **Total over 182 days** | **≈ $25–40** — comfortably inside the credit |

You get a permanent URL. Visiting it when the box is asleep shows a "booting, ~2 minutes"
page that refreshes itself and then hands over to the app.

---

## Why not something that truly scales to zero

Worth stating plainly, because it is the obvious question:

| | Why it does not work |
|---|---|
| **Lambda** | 15-minute cap, no memory retained between invocations, and a 2.3 GB model reload per cold start. A deep research turn alone can exceed the limit. |
| **Fargate + ALB** | ECS will not scale a service to zero behind a load balancer, and the ALB is ~$16/month on its own — more than the instance. |
| **App Runner** | Has a scale-to-zero mode, but provisioned memory is still billed and every wake reloads the model. |
| **EC2 `t3.micro`/`small`** | 1–2 GB RAM. `bge-m3` alone needs ~2.3 GB. |

Nothing on AWS gives request-driven scaling for a workload that must keep 2.3 GB of weights
resident. Wake-on-demand is the honest substitute: the *link* is always up, the *machine* is not.

---

## What has to fit

| Component | Cost | Can it move off the box? |
|---|---|---|
| `BAAI/bge-m3` embedder | ~2.3 GB RAM | **No.** The corpus is embedded with it — a different model means re-embedding all 38,890 chunks. |
| Reranker | ~1.1 GB, **~10 s/query on 2 CPU cores** | In principle, but see the warning. |
| Corpus | ~350 MB disk | No, but it is just files. |
| LLMs | — | Already remote (NVIDIA NIM). |
| Chromium | 300–500 MB RSS | Optional — `KYR_CRAWL_USE_BROWSER=false`. |

> **Do not plan around the remote reranker.** The `cpu_lean` profile offloads reranking to
> NVIDIA to dodge the CPU cost. During evaluation **every NIM reranking endpoint returned 410
> or 404** — `llama-3.2-nv-rerankqa-1b-v2`, `llama-nemotron-rerank-1b-v2` and
> `nv-rerankqa-mistral-4b-v3` all failed inside one probe. The app degrades correctly and says
> so, but deploy with `KYR_PROFILE=cpu` and tune `KYR_RERANK_POOL` instead.

---

## Part 1 — the instance

### Launch

- **AMI** Ubuntu 22.04 **ARM64**, **type** `t4g.medium`, **storage** 30 GB gp3
- **Security group** inbound 22 from your IP only, and 443. Do **not** open 8000.
- **No Elastic IP.** It is free only while attached to a *running* instance and billed when
  stopped — exactly backwards for this design. Use a Cloudflare Tunnel for a stable address.

### Install

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git git-lfs
git lfs install                       # BEFORE cloning, or you get 134-byte pointer files

git clone https://github.com/DamnKuldeep/KnowYourRightsAI.git
cd KnowYourRightsAI
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # ~200 MB, not 2.5 GB
pip install -r requirements.txt
```

Forgetting `git lfs install` before the clone is the single most common way to get this wrong —
the app starts, then fails confusingly because `data/legal_db` is pointer text.

### Configure

```bash
cat > .env <<'EOF'
NVIDIA_API_KEY=nvapi-your-key-here

KYR_PROFILE=cpu                  # local reranking; do not depend on the remote one
KYR_RERANK_POOL=12               # halves the CPU rerank cost — the key knob on 2 cores
KYR_HOST=127.0.0.1               # Caddy/tunnel faces the internet, not uvicorn
KYR_CRAWL_USE_BROWSER=false      # saves ~400 MB; loses JS-only portals
KYR_DEFAULT_DEPTH=standard       # cap research depth on a shared demo box
KYR_SESSION_CREDIT_BUDGET=800    # downshift depth before NIM credits run out
KYR_MODEL_IDLE_EVICT_S=1800
KYR_LOG_LEVEL=WARNING
EOF
chmod 600 .env
```

### Prepare — both steps are required

```bash
python scripts/build_index.py     # 23 s. Without it every query scans 159 MB.
python scripts/probe_nim.py       # pin reachable models
python scripts/calibrate.py       # thresholds do not transfer between rerankers
python scripts/benchmark.py --all # confirm Recall@5 on this machine
```

Skipping `calibrate.py` leaves uncalibrated defaults, which is how off-topic questions start
getting confident answers.

### Run it as a service

```bash
sudo tee /etc/systemd/system/kyr.service >/dev/null <<'EOF'
[Unit]
Description=KnowYourRightsAI
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/KnowYourRightsAI
Environment=HF_HOME=/home/ubuntu/.cache/huggingface
ExecStart=/home/ubuntu/KnowYourRightsAI/.venv/bin/python -m knowyourrights.server
Restart=always
RestartSec=15
TimeoutStartSec=300          # models take up to ~140 s on a cold CPU box
MemoryMax=3500M              # hard ceiling; bge-m3 spikes to ~2.3 GB while loading
OOMPolicy=stop

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload && sudo systemctl enable --now kyr
journalctl -u kyr -f
```

Add swap as insurance against the load spike:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### A public address

```bash
# Cloudflare Tunnel — free, no inbound ports, survives the instance changing IP
cloudflared tunnel --url http://localhost:8000
```

Or with your own domain and Caddy:

```bash
sudo apt install -y caddy
echo 'demo.yourdomain.com { reverse_proxy 127.0.0.1:8000 }' | sudo tee /etc/caddy/Caddyfile
sudo systemctl restart caddy      # certificates are automatic
```

---

## Part 2 — sleeping when idle

Install the timer that stops the box when nobody is asking anything:

```bash
sudo cp deploy/idle-shutdown.sh /usr/local/bin/ && sudo chmod +x /usr/local/bin/idle-shutdown.sh
( crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/idle-shutdown.sh" ) | crontab -
```

It watches the app's own LLM-call counter rather than CPU load, because the box is *supposed*
to sit warm-but-quiet between questions in a demo. It also refuses to act within 15 minutes of
boot, so a visitor arriving during startup is never killed mid-load.

`shutdown -h` on an EBS-backed instance **stops** it rather than terminating — the disk and
everything on it survives until the next wake.

---

## Part 3 — waking on demand

A Lambda Function URL (no API Gateway, no cost at demo volumes) that starts the instance when
someone opens the link and shows a waiting page until the app reports ready.

### IAM role

```bash
cat > trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF

aws iam create-role --role-name kyr-waker --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name kyr-waker \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

INSTANCE_ID=i-0123456789abcdef0
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)

cat > ec2-policy.json <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ec2:StartInstances"],
  "Resource":"arn:aws:ec2:${REGION}:${ACCOUNT}:instance/${INSTANCE_ID}"},
 {"Effect":"Allow","Action":["ec2:DescribeInstances"],"Resource":"*"}]}
EOF

aws iam put-role-policy --role-name kyr-waker \
  --policy-name kyr-start-instance --policy-document file://ec2-policy.json
```

Scoped to one instance id — the waker can start that box and nothing else.

### The function

```bash
cd deploy && zip waker.zip wake_lambda.py && cd ..

aws lambda create-function --function-name kyr-waker \
  --runtime python3.12 --handler wake_lambda.handler \
  --role arn:aws:iam::${ACCOUNT}:role/kyr-waker \
  --zip-file fileb://deploy/waker.zip --timeout 15 --memory-size 256 \
  --environment "Variables={INSTANCE_ID=${INSTANCE_ID},APP_URL=https://demo.yourdomain.com,REGION=${REGION}}"

aws lambda create-function-url-config --function-name kyr-waker --auth-type NONE
aws lambda add-permission --function-name kyr-waker --statement-id public \
  --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE
aws lambda get-function-url-config --function-name kyr-waker --query FunctionUrl --output text
```

That URL is what you share. It is always live, costs nothing at demo volumes (Lambda's 1M
free requests a month never expire), and it boots the demo when someone arrives.

**Flow:** visitor opens the link → Lambda sees the instance stopped → starts it, returns a
waiting page that refreshes every 10 s → once `/api/health` reports `ready: true`, the page
redirects to the app. Roughly two minutes, most of it loading the models.

---

## Docker, if you prefer it

The repo ships a `Dockerfile` and `docker-compose.yml`. The image deliberately contains
**neither the corpus nor the model weights** — the corpus is mounted read-only from the clone,
and the models live in a named volume so they download once and survive rebuilds. That keeps
the image at ~1.5 GB rather than ~4 GB, which matters when you are pulling it onto a
billed-by-the-second instance.

```bash
cp .env.example .env      # add NVIDIA_API_KEY
docker compose up --build

# first run only: build the index and calibrate, inside the container
docker compose exec app python scripts/build_index.py
docker compose exec app python scripts/calibrate.py
docker compose restart app
```

On the EC2 box the only difference is installing Docker first:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && newgrp docker
git lfs install && git clone https://github.com/DamnKuldeep/KnowYourRightsAI.git
cd KnowYourRightsAI && cp .env.example .env   # edit it
docker compose up -d --build
```

`restart: unless-stopped` in the compose file means the container comes back by itself after
the instance wakes — no systemd unit needed if you go this route. Use one or the other, not
both.

---

## What it actually costs

`t4g.medium` is $0.0336/hour; 30 GB of gp3 is $2.40/month whether the instance runs or not.

| Pattern | Compute (182 days) | Storage | **Total** |
|---|---:|---:|---:|
| **Wake-on-demand, ~1 h/day** | $6.11 | $14.40 | **$20.51** |
| Wake-on-demand, ~3 h/day | $18.34 | $14.40 | **$32.74** |
| Weekdays 9–9 on a schedule | $35.28 | $14.40 | **$49.68** |
| Always on | $146.75 | $14.40 | **$161.15** ✗ over budget |

Always-on does not fit in $100 for 182 days. Wake-on-demand uses about a fifth of the credit
and leaves the rest spare.

Lambda is free at this volume. Egress is well inside the 100 GB/month free allowance. Skip the
Elastic IP and there is nothing else billing.

**If you want the demo to feel fast**, `g4dn.xlarge` (T4 GPU) at $0.526/hour runs the GPU
profile — 399 ms retrieval instead of ~5 s. Two hours a week for 26 weeks is ~$27 of compute,
which fits. It needs a Deep Learning AMI and more setup, and the language model dominates
end-to-end time anyway, so I would only bother if the retrieval delay is visibly annoying in
front of an audience.

---

## Not breaking in production

Each of these is a failure actually observed, not a hypothetical.

**Set a billing alarm first.** Credits run out silently and then billing just… continues.

```bash
aws budgets create-budget --account-id ${ACCOUNT} \
  --budget '{"BudgetName":"kyr","BudgetLimit":{"Amount":"20","Unit":"USD"},
             "TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL",
    "ComparisonOperator":"GREATER_THAN","Threshold":50},
    "Subscribers":[{"SubscriptionType":"EMAIL","Address":"you@example.com"}]}]'
```

**Your NIM credits will run out before your AWS credits.** ~1,000 credits, 3–9 per legal
question — a few hundred questions total. `KYR_SESSION_CREDIT_BUDGET=800` makes the agent
downshift research depth as the budget depletes rather than failing after. Watch `/api/usage`.

**Memory is what kills it.** The bge-m3 load spikes to ~2.3 GB. On a 4 GB box with anything
else running, that is the failure. Already mitigated: a RAM floor check that refuses to load
and degrades to keyword-only search rather than being OOM-killed, plus `MemoryMax` and swap.

**One worker, always.** A second uvicorn worker loads a second copy of bge-m3 for no benefit —
model access is already serialised internally.

**Models get retired mid-run.** During evaluation `nemotron-3-nano-30b-a3b` returned 410, and
every reranking endpoint returned 410 or 404. The registry fails over automatically, and after
a bug found exactly this way it treats a single 410 as *transient* — sidelined for the process,
retried later, only recorded as unavailable after 3 failures in an hour. Check what is actually
resolved with `curl -s localhost:8000/api/health`.

**Back up the calibration.** `thresholds.json` is the difference between abstaining properly
and answering off-topic questions confidently.

```bash
tar czf ~/kyr-runtime-$(date +%F).tgz .runtime/thresholds.json .runtime/nim_probe.json
```

**Rotate the logs.** `usage.jsonl` grows with every API call.

```bash
sudo tee /etc/logrotate.d/kyr >/dev/null <<'EOF'
/home/ubuntu/KnowYourRightsAI/.runtime/*.jsonl {
  weekly
  rotate 4
  compress
  missingok
  notifempty
  copytruncate
}
EOF
```

---

## Pre-flight checklist

- [ ] Billing alarm set
- [ ] `git lfs install` run **before** cloning; `data/legal_db/` is ~350 MB, not pointer files
- [ ] `scripts/build_index.py` run — vector search ~23 ms, not ~300 ms
- [ ] `scripts/calibrate.py` run — `thresholds.json` exists and does not say "config default"
- [ ] `scripts/benchmark.py --all` gives Recall@5 ≈ 100% on this machine
- [ ] `KYR_PROFILE=cpu`, `KYR_RERANK_POOL=12`
- [ ] `KYR_HOST=127.0.0.1` with Caddy or a tunnel in front; 8000 not public
- [ ] `.env` is `chmod 600` and not in git
- [ ] Swap added; `MemoryMax` set
- [ ] `/api/health` returns `ready: true` before you share the link
- [ ] Idle-shutdown installed **and tested** — stop it once and confirm the Lambda wakes it
- [ ] No Elastic IP attached (it bills while the instance is stopped)
