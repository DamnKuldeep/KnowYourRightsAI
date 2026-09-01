# Deploying on AWS with $100 of credit

Two goals: it should not break in production, and it should cost nothing while nobody is using
it. Those pull in different directions — the honest answer is below.

---

## What has to fit, and what that rules out

| Component | Cost | Can it move off the box? |
|---|---|---|
| `BAAI/bge-m3` embedder | ~2.3 GB RAM (fp32 CPU) | **No.** The corpus is embedded with it. A different model means re-embedding all 38,890 chunks. |
| Reranker | ~1.1 GB, **~10 s/query on CPU** | In principle yes — but see the warning below. |
| Corpus (LanceDB + parquet) | ~350 MB disk | No, but it is just files. |
| LLMs | — | Already remote (NVIDIA NIM). |
| Chromium (crawl4ai) | 300–500 MB RSS | Optional: `KYR_CRAWL_USE_BROWSER=false`. |

**Minimum viable instance: 4 GB RAM.** This rules out `t3.micro` (1 GB) and `t3.small` (2 GB).

> ### Warning — do not rely on remote reranking
> The `cpu_lean` profile pushes reranking to NVIDIA to avoid the 10-second CPU cost. During
> evaluation **every NIM reranking endpoint returned 410 or 404** — `llama-3.2-nv-rerankqa-1b-v2`,
> `llama-nemotron-rerank-1b-v2` and `nv-rerankqa-mistral-4b-v3` all failed within one probe.
> The system degrades correctly (it falls back to fusion-only ranking and says so), but you
> should not *plan* a deployment around that endpoint being there.
>
> Deploy with local reranking and size the instance for it.

---

## The AWS free tier is not what it used to be

Accounts opened after **15 July 2025** no longer get the 12-month free EC2 allowance. New
accounts get **$200 in credits** ($100 on signup, $100 more for usage) that draw down against
normal pricing. Your $100 is a **budget, not a free tier** — when it is gone, billing starts.

So the design goal is: make $100 last, and make sure it cannot silently become $300.

---

## Recommended: one EC2 instance you switch on and off

Simple, debuggable, and the cost is genuinely proportional to uptime.

### Instance choice

| Instance | vCPU | RAM | $/hour | Notes |
|---|---:|---:|---:|---|
| **`t4g.medium`** (ARM Graviton) | 2 | 4 GB | **~$0.0336** | **Recommended.** ~20% cheaper than x86. |
| `t3.medium` (x86) | 2 | 4 GB | ~$0.0416 | Use if an ARM wheel is ever missing. |
| `t4g.large` | 2 | 8 GB | ~$0.0672 | Comfortable if you also want Chromium. |

### What $100 actually buys

Storage is billed whether the instance runs or not: **30 GB gp3 ≈ $2.40/month**. That is the
floor. Everything else scales with uptime.

| Pattern | Compute/month | + storage | **Total/month** | $100 lasts |
|---|---:|---:|---:|---:|
| Always on | $24.20 | $2.40 | **$26.60** | ~3.8 months |
| 12 h/day | $12.10 | $2.40 | **$14.50** | ~6.9 months |
| 8 h/day weekdays | $5.80 | $2.40 | **$8.20** | ~12 months |
| On demand only (~2 h/day) | $2.00 | $2.40 | **$4.40** | ~22 months |
| Stopped, storage only | $0 | $2.40 | **$2.40** | ~41 months |

Add a **Spot instance** and compute drops ~70% again — `t4g.medium` spot is around $0.010/hr,
so always-on becomes ~$9.60/month. Spot can be reclaimed with a 2-minute warning, which is fine
for a demo and not for anything anyone depends on.

**Skip the load balancer.** An ALB is ~$16/month on its own — more than the instance. Use
Caddy on the box for HTTPS, or a Cloudflare Tunnel (free, and no inbound ports at all).

---

## Setting it up

### 1. Launch

- AMI: **Ubuntu 22.04 ARM64**, instance `t4g.medium`, storage **30 GB gp3**
- Security group: inbound **22** (your IP only) and **443**. Do not open 8000 publicly.
- Allocate an **Elastic IP** only if you need a stable address — it is free while attached to a
  *running* instance but **charged when the instance is stopped**. If you will stop the box
  often, skip it and use a Cloudflare Tunnel instead.

### 2. Install

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git git-lfs
git lfs install

git clone https://github.com/DamnKuldeep/KnowYourRightsAI.git
cd KnowYourRightsAI
python3 -m venv .venv && source .venv/bin/activate

# CPU-only torch: ~200 MB instead of ~2.5 GB with the CUDA runtime
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

Without `git lfs install` **before** cloning you get 134-byte pointer files instead of the
database, and a confusing startup failure. This is the single most common way to get this wrong.

### 3. Configure

```bash
cat > .env <<'EOF'
NVIDIA_API_KEY=nvapi-your-key-here

KYR_PROFILE=cpu                  # local reranking — do not depend on the remote reranker
KYR_HOST=127.0.0.1               # Caddy/tunnel faces the internet, not uvicorn
KYR_CRAWL_USE_BROWSER=false      # saves ~400 MB; loses JS-only portals
KYR_DEFAULT_DEPTH=standard       # cap research depth on a shared instance
KYR_SESSION_CREDIT_BUDGET=800    # downshift depth before NIM credits run out
KYR_MODEL_IDLE_EVICT_S=1800      # release model memory after 30 min idle
KYR_RERANK_POOL=12               # halves the 10 s CPU rerank; costs a little recall
KYR_LOG_LEVEL=WARNING
EOF
chmod 600 .env
```

`KYR_RERANK_POOL=12` is the important CPU tuning knob: reranking is ~75% of retrieval time and
scales with pool size.

### 4. Build the index and calibrate — both required

```bash
python scripts/build_index.py     # 23 s. Without it every query scans 159 MB.
python scripts/probe_nim.py       # pin reachable models
python scripts/calibrate.py       # REQUIRED — thresholds don't transfer between rerankers
python scripts/benchmark.py --all # confirm the numbers on this machine
```

Skipping `calibrate.py` leaves uncalibrated defaults, which is how off-topic questions start
getting confident answers.

### 5. Run it as a service

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
# Models take up to ~140 s to load on a cold CPU box; don't let systemd kill the boot.
TimeoutStartSec=300
# Hard ceiling so a runaway process cannot take the machine down with it.
MemoryMax=3500M
OOMPolicy=stop

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload && sudo systemctl enable --now kyr
sudo systemctl status kyr
journalctl -u kyr -f
```

### 6. HTTPS

```bash
sudo apt install -y caddy
echo 'your.domain { reverse_proxy 127.0.0.1:8000 }' | sudo tee /etc/caddy/Caddyfile
sudo systemctl restart caddy      # certificates are automatic
```

No domain? A Cloudflare Tunnel gives you a public HTTPS URL with no open inbound ports:

```bash
cloudflared tunnel --url http://localhost:8000
```

---

## Making it cost only what you use

### Option A — a schedule (simplest, predictable)

EventBridge Scheduler rules that start and stop the instance. Two rules, no code:

```bash
INSTANCE_ID=i-0123456789abcdef0
ROLE_ARN=arn:aws:iam::<account>:role/EventBridgeEC2Role   # needs ec2:StartInstances/StopInstances

aws scheduler create-schedule --name kyr-start \
  --schedule-expression "cron(0 9 ? * MON-FRI *)" --schedule-expression-timezone "Asia/Kolkata" \
  --flexible-time-window '{"Mode":"OFF"}' \
  --target "{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:startInstances\",\"RoleArn\":\"$ROLE_ARN\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"$INSTANCE_ID\\\"]}\"}"

aws scheduler create-schedule --name kyr-stop \
  --schedule-expression "cron(0 21 ? * MON-FRI *)" --schedule-expression-timezone "Asia/Kolkata" \
  --flexible-time-window '{"Mode":"OFF"}' \
  --target "{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:stopInstances\",\"RoleArn\":\"$ROLE_ARN\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"$INSTANCE_ID\\\"]}\"}"
```

9am–9pm on weekdays is ~$8/month all-in — **$100 lasts about a year**.

### Option B — idle auto-shutdown (pay for actual use)

Let the box turn *itself* off when nobody has asked anything. The server already tracks this.

```bash
cat > /home/ubuntu/idle-shutdown.sh <<'EOF'
#!/bin/bash
# Stop the instance after 30 minutes with no questions asked.
IDLE_LIMIT=1800
STATE=/tmp/kyr-last-activity
CALLS=$(curl -s http://127.0.0.1:8000/api/usage | python3 -c \
        'import sys,json; print(json.load(sys.stdin).get("total_calls",0))' 2>/dev/null || echo 0)
PREV=$(cut -d' ' -f1 "$STATE" 2>/dev/null || echo -1)
NOW=$(date +%s)
if [ "$CALLS" != "$PREV" ]; then
  echo "$CALLS $NOW" > "$STATE"; exit 0
fi
LAST=$(cut -d' ' -f2 "$STATE" 2>/dev/null || echo "$NOW")
if [ $((NOW - LAST)) -gt "$IDLE_LIMIT" ]; then
  logger "kyr: idle for $IDLE_LIMIT s, shutting down"
  sudo shutdown -h now
fi
EOF
chmod +x /home/ubuntu/idle-shutdown.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/ubuntu/idle-shutdown.sh") | crontab -
```

Then start it on demand from your phone or laptop:

```bash
aws ec2 start-instances --instance-ids i-0123456789abcdef0
```

Two caveats worth being clear about: a stopped instance takes **~2–3 minutes** to boot and load
models, and this is *scale-to-zero-by-schedule*, not true request-driven autoscaling. Nothing on
AWS gives you the latter for a workload that must keep 2.3 GB of model weights resident.

### Option C — Spot for the cheapest always-on

A persistent Spot request at `t4g.medium` runs ~$0.010/hr — **~$9.60/month always on**. AWS can
reclaim it with a 2-minute warning. Fine for a portfolio demo; not for anything with users who
would notice.

### What does *not* work

| | Why |
|---|---|
| **Lambda** | 15-minute cap, no persistent memory between invocations, and a 2.3 GB model reload per cold start. A deep turn already exceeds this. |
| **Fargate scale-to-zero** | ECS does not scale to zero for a service behind a load balancer, and the ALB alone is ~$16/month. |
| **App Runner** | Has a scale-to-zero mode, but a provisioned-memory charge continues and cold starts reload the model. |
| **`t3.micro` / `t3.small`** | 1–2 GB RAM. bge-m3 alone needs ~2.3 GB. |

---

## Not breaking in production

The failures below were all observed during evaluation, not imagined.

### Set a billing alarm before anything else

```bash
aws budgets create-budget --account-id <account> \
  --budget '{"BudgetName":"kyr","BudgetLimit":{"Amount":"25","Unit":"USD"},
             "TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL",
    "ComparisonOperator":"GREATER_THAN","Threshold":50},
    "Subscribers":[{"SubscriptionType":"EMAIL","Address":"you@example.com"}]}]'
```

Credits run out silently. This is the difference between a $100 project and a surprise bill.

### The NIM free tier is metered too

~1,000 credits, roughly one per request, and a legal question costs **3–9**. That is a few
hundred questions. `KYR_SESSION_CREDIT_BUDGET=800` makes the agent downshift research depth as
the budget depletes rather than failing once it is gone. Watch `/api/usage`.

### Health checks

`/api/health` reports readiness, the active profile, live VRAM/RAM, model resolution and cache
stats. It returns `ready: false` while models load — do not route traffic on process liveness
alone, or the first users hit a cold, unwarmed instance.

```bash
curl -s localhost:8000/api/health | python3 -m json.tool | head -20
```

### Memory is the thing that will kill it

The bge-m3 load spikes host RAM to ~2.3 GB. On a 4 GB box with anything else running, that is
the failure. Mitigations already in place: a RAM floor check that refuses to load and degrades
to keyword-only search rather than being OOM-killed, and `MemoryMax` in the unit file as a hard
ceiling. Add swap as insurance:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### One worker, always

A second uvicorn worker loads a second copy of bge-m3 and doubles memory for no benefit — GPU
and model access are already serialised internally. The server defaults to one; do not "tune"
this.

### Model retirement is a live event, not a hypothetical

During evaluation `nemotron-3-nano-30b-a3b` returned 410 and every reranking endpoint returned
410 or 404. The registry fails over to alternates automatically. It also — after a bug found
exactly this way — treats a single 410 as *transient*: the model is sidelined for the process
and retried later, and only recorded as unavailable after 3 failures within an hour. One blip
no longer permanently retires a healthy model.

Check what is actually resolved: `curl -s localhost:8000/api/health | grep -A5 models`.

### Back up what you cannot rebuild cheaply

```bash
tar czf ~/kyr-runtime-$(date +%F).tgz .runtime/thresholds.json .runtime/nim_probe.json
```

`thresholds.json` is the calibration. Losing it silently degrades abstention — regenerate with
`scripts/calibrate.py` rather than guessing.

### Log rotation

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

`usage.jsonl` grows with every API call and will otherwise fill the disk eventually.

---

## Pre-flight checklist

- [ ] Billing alarm set at $25
- [ ] `git lfs install` run **before** cloning; `data/legal_db/` is ~350 MB, not 134-byte pointers
- [ ] `scripts/build_index.py` run — vector search ~23 ms, not ~300 ms
- [ ] `scripts/calibrate.py` run — `thresholds.json` exists and is not "config default"
- [ ] `scripts/benchmark.py --all` gives Recall@5 ≈ 100% on this machine
- [ ] `KYR_PROFILE=cpu` (not `cpu_lean` — remote reranking is unreliable)
- [ ] `KYR_HOST=127.0.0.1` with Caddy or a tunnel in front; port 8000 not public
- [ ] `.env` is `chmod 600` and not in git
- [ ] Swap added; `MemoryMax` set in the unit file
- [ ] `/api/health` returns `ready: true` before you share the URL
- [ ] Schedule or idle-shutdown configured, and you have tested that it restarts cleanly

---

## What I would actually do with $100

Run `t4g.medium` on a **weekday 9am–9pm schedule** with idle-shutdown as a backstop: about
**$8/month**, so the credit lasts roughly a year, and the site is up whenever anyone is likely
to look at it. Keep a Cloudflare Tunnel rather than an Elastic IP so a stopped instance costs
nothing but its disk.

If you need it up permanently, Spot at ~$9.60/month is the cheaper route — accept that AWS can
reclaim it with two minutes' notice.
