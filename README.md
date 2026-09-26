<div align="center">

# ⚖️ KnowYourRights

**Plain-language answers about Indian law, with the exact section each one comes from.**

Ask in English, Hindi or Hinglish.

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![retrieval](https://img.shields.io/badge/Recall%405-100%25-brightgreen)
![off-topic](https://img.shields.io/badge/off--topic%20refused-8%2F8-brightgreen)

*General legal information, not legal advice.*

![KnowYourRights answering "How do I file an RTI, and what does it cost?"](docs/screenshot.png)

</div>

## Overview

Ask a general-purpose chatbot *"can the police arrest me without a warrant?"* and you get a
fluent paragraph you cannot check, often citing the Code of Criminal Procedure, which was
repealed on 1 July 2024. KnowYourRights is built to do the opposite: every factual claim traces
to a specific section of a specific Act, and when it has nothing on point it says so.

- **Cited answers.** Each `[S1]` in an answer links to the section it came from, and every
  citation is checked against the retrieved text before the reader sees it.
- **The current law.** Questions about the IPC, CrPC or Evidence Act are translated to the 2023
  codes that replaced them (Section 420 IPC → Section 318 BNS), never answered from memory.
- **Jurisdiction.** Every source is labelled central, state or Union Territory law, and the
  answer follows where the matter is, not where the reader lives.
- **Procedures.** For "how do I…" questions it reads official government pages for the current
  fee, deadline, portal and appeal route.
- **Safety first.** A disclosure of violence, self-harm or an arrest in progress gets helpline
  numbers before any research starts.
- **Live and honest.** Answers stream as they are written; the reader sees each research step,
  and is told when part of the pipeline ran in a reduced mode.

## How it works

```text
question ─► safety gate ─► planner ─► research (statute search · official web pages)
         ─► relevance grading ─► writer (streamed) ─► citation check ─► answer
```

The model never decides to call a tool: it produces a validated plan and plain Python runs it, so
a hostile web page cannot trigger anything. Statute search is hybrid (semantic + keyword),
fused, reranked and diversified over a LanceDB corpus of about 38,600 chunks covering the
Constitution, ~1,000 central Acts and the 2023 criminal codes. Details are in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Tech stack

| Layer | Technology |
|---|---|
| Server | Python 3.11+, FastAPI, Uvicorn, server-sent events |
| Models | [OpenRouter](https://openrouter.ai) (Gemini 2.5 Flash-Lite, Qwen 3.7 Flash, …); NVIDIA NIM as failover |
| Retrieval | LanceDB (vector + BM25), `baai/bge-m3` embeddings, `cohere/rerank-v3.5` reranking, via OpenRouter |
| Web research | crawl4ai (HTTP, Chromium when needed), ddgs search, Wikipedia API |
| Frontend | Plain HTML, CSS and JavaScript: no framework, no build step |
| Quality | pytest (228 tests, fully mocked), ruff, GitHub Actions |

No model runs locally, so there is no GPU requirement and the process needs a few hundred MB of
memory.

## Getting started

**You need:** Python 3.11+, [Git LFS](https://git-lfs.com) (the corpus is stored with it), and an
[OpenRouter API key](https://openrouter.ai/keys).

```bash
git clone https://github.com/DamnKuldeep/KnowYourRightsAI.git
cd KnowYourRightsAI
git lfs pull                                # the ~400 MB corpus

python -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium       # optional: reads pages that need JavaScript

cp .env.example .env                        # then set OPENROUTER_API_KEY in .env
python -m knowyourrights.server             # http://127.0.0.1:8000
```

Or with Docker:

```bash
cp .env.example .env                        # set OPENROUTER_API_KEY
docker compose up --build                   # http://127.0.0.1:8000
```

Docker reads `.env` strictly: write `KEY=value` with no spaces around `=`. Build with
`--build-arg INSTALL_BROWSER=false` for a 1.9 GB image without Chromium (3.4 GB with it), and set
`KYR_CRAWL_USE_BROWSER=false` to match.

From the terminal, without the server:

```bash
python scripts/ask.py "can police search my phone"
python scripts/ask.py --search "right to information appeal"    # statute search only
```

## Configuration

Settings are environment variables (or `.env`); every one is listed with its default in
[`.env.example`](.env.example). The ones most deployments touch:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Required.** Chat, embeddings and reranking. |
| `NVIDIA_API_KEY` | — | Optional second chat provider, used as failover. |
| `KYR_MAX_ACTIVE_TURNS` | `5` | Answers researched at once; later questions queue. |
| `KYR_CLIENT_BUDGET_USD` | `1.0` | Spend allowed per client (IP address) before the free limit applies. |
| `KYR_DAILY_BUDGET_USD` | `5.0` | Ceiling on the whole service's spend per day. |
| `KYR_TRUST_PROXY_HEADERS` | `false` | Set `true` behind a reverse proxy or tunnel. |
| `KYR_ADMIN_TOKEN` | — | Bearer token for `/api/status`. |
| `KYR_LOGIN_USERS` | — | `name:password,…`: require sign-in for everything but `/api/health`. |
| `KYR_HOST` / `KYR_PORT` | `127.0.0.1` / `8000` | Where the server listens. |

**Running it publicly.** At most five answers are researched at once. Later questions wait in a
first-come queue and are shown their place in line. Each client may spend $1 before being told
the free allowance is used up, and the service stops for the day at $5. A typical answer costs
$0.002–0.004; a deep one about $0.015. To keep the site to people you choose, set
`KYR_LOGIN_USERS`: visitors then see a sign-in page, sessions last 30 days, and ten wrong
passwords from one address lock it out for 15 minutes.

## API

| Method | Path | |
|---|---|---|
| `GET` `POST` | `/login`, `POST` `/logout` | Sign in and out, when `KYR_LOGIN_USERS` is set |
| `POST` | `/api/chat` | Ask a question; the answer streams back as server-sent events |
| `POST` | `/api/stop` | Stop the answer being written |
| `POST` | `/api/reset` | Forget the conversation |
| `POST` | `/api/feedback` | Rate an answer |
| `GET` | `/api/config` | States, disclaimer and depth settings for the UI |
| `GET` | `/api/quota` | This client's remaining allowance |
| `GET` | `/api/health` | `ok`, `degraded` or `unavailable` (HTTP 503) |
| `GET` | `/api/status` | Full diagnostics; needs `KYR_ADMIN_TOKEN` or a local request |

A chat stream carries typed events: `stage` and `tool` (progress), `source` and
`sources_final` (citations), `notice` (warnings and rate-limit countdowns), `safety`,
`procedure`, `token` (the answer), `verdict` (citation check), `queue`, `limit`, `usage` and
`done`. Refusals (validation, rate or budget limits, a full queue) return JSON
`{"error": {"kind", "message", "retry_after_s"}}`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                      # 228 tests, ~10 s: no network, keys or corpus needed
ruff check knowyourrights scripts tests
```

Measurement scripts call the real APIs and cost a few cents each:

| Script | Measures |
|---|---|
| `scripts/evaluate.py` | Retrieval: Recall@5, MRR, abstention, exact lookups; `--degraded` for outage modes |
| `scripts/e2e_check.py` | Whole answers: time to first word, stages, cost |
| `scripts/ui_check.py` | 15 kinds of question in a real browser, with screenshots |
| `scripts/calibrate.py` | Abstention and citation thresholds for a ranking method |
| `scripts/calibrate_safety.py` | The safety gate's threshold on its labelled set |
| `scripts/race_models.py` | Candidate models for each role, on the real prompts |
| `scripts/verify_embeddings.py` | That the embedding API still matches the corpus's vectors |

## Results

| | |
|---|---|
| Statute retrieval, Recall@5 / MRR | **100%** (42/42) / **0.929** |
| Off-topic questions refused | **8 / 8** |
| Exact provision lookups | **4 / 4** |
| Safety gate: disclosures caught / false alarms | **33 / 33** / **0 / 26** |
| Browser sweep of 15 question types | **15 / 15** |
| Cost per answer | $0.002–0.004 (deep mode ~$0.015) |

Method, the reduced modes and the limitations are in [EVALUATION.md](EVALUATION.md).

## Project structure

```text
knowyourrights/
  server/          FastAPI app: routes, validation, admission queue, per-client limits
  orchestrator/    one turn: plan → research → write → verify → commit
  agents/          model stages (planning, grading, extraction) and answer post-processing
  retrieval/       hybrid search, ranking, embeddings and reranking over the corpus
  llm/             chat client, model routing, failover, rate limits, spend tracking
  tools/           statute search, web search, page reading, portal navigation, URL safety
  context/         conversation memory, token budgets, what reaches the prompt
  web/             the browser UI
  config.py        every setting, overridable from the environment
scripts/           evaluation, calibration, corpus maintenance, terminal client
tests/             the test suite
data/              the LanceDB corpus (Git LFS) and its documentation
notebooks/         how the corpus was built
```

## Limitations

- **Central law only.** State laws are covered incidentally and labelled; for state subjects
  (tenancy, stamp duty) the answer says so and asks which state.
- **A snapshot.** Amendments after the corpus was built are missing. Deep mode checks fees and
  deadlines against the live web, but anything time-sensitive is worth verifying.
- **The evaluation set is small** (42 questions) and was written by the author.
- **Not legal advice.** For your situation, consult a lawyer. Free legal aid: NALSA, **15100**.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): how a question becomes an answer, and why.
- [EVALUATION.md](EVALUATION.md): what was measured, how, and what it showed.
- [CHANGELOG.md](CHANGELOG.md): what changed, release by release.
- [data/KnowYourRights_DB_README.md](data/KnowYourRights_DB_README.md): the corpus.

## License

[MIT](LICENSE)
