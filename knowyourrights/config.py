"""Every tuning knob lives here. Nothing in this module may import torch, lancedb or
transformers — it is imported by scripts that must stay cheap.

Values are overridable from the environment (and therefore from ``.env``); the defaults are
tuned for the machine described in the plan: RTX 3050 4 GB / 16 GB RAM with ~3 GB free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, but this is how the API key normally arrives
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:  # pragma: no cover - dotenv is in requirements
    pass


# ── paths ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KYR_DATA_DIR", ROOT / "data"))
RUNTIME_DIR = Path(os.environ.get("KYR_RUNTIME_DIR", ROOT / ".runtime"))
CACHE_DIR = RUNTIME_DIR / "cache"

DB_PATH = Path(os.environ.get("LEGAL_DB_PATH", DATA_DIR / "legal_db"))
TABLE = "laws"
PARQUET = Path(os.environ.get("LEGAL_PARQUET", DATA_DIR / "chunks_metadata.parquet"))
ENRICH_CACHE = Path(os.environ.get("LEGAL_CACHE", DATA_DIR / "enrichment_cache.json"))

WEB_DIR = Path(__file__).resolve().parent / "web"


# ── env helpers ───────────────────────────────────────────────────────────────────────
def env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return default if v is None or v.strip() == "" else v.strip()


def env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ── providers ─────────────────────────────────────────────────────────────────────────
# Two OpenAI-compatible endpoints, used together. NVIDIA has been unreliable in practice —
# live 410s on healthy models, 503s, and calls swinging from 1s to 2.6s — so every role can
# fail over to the other provider rather than to nothing.
NIM_BASE_URL = env_str("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_RETRIEVAL_BASE = env_str("NIM_RETRIEVAL_BASE", "https://ai.api.nvidia.com/v1/retrieval")
NVIDIA_API_KEY = env_str("NVIDIA_API_KEY", "")
NIM_TIMEOUT_S = env_float("NIM_TIMEOUT_S", 90.0)

OPENROUTER_BASE_URL = env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = env_str("OPENROUTER_API_KEY", "")
# Sent as HTTP-Referer/X-Title; OpenRouter uses them for attribution on free models.
OPENROUTER_APP_URL = env_str("OPENROUTER_APP_URL", "https://github.com/DamnKuldeep/KnowYourRightsAI")
OPENROUTER_APP_NAME = env_str("OPENROUTER_APP_NAME", "KnowYourRightsAI")
# Free tier: 20 requests/minute, and a daily cap. Both are enforced client-side so we degrade
# on our own terms rather than being cut off mid-answer.
OPENROUTER_RPM = env_int("OPENROUTER_RPM", 15)
OPENROUTER_DAILY_LIMIT = env_int("OPENROUTER_DAILY_LIMIT", 1000)
# Stop well short of the ceiling so a demo never dies on the last few requests.
OPENROUTER_DAILY_RESERVE = env_int("OPENROUTER_DAILY_RESERVE", 60)

PROVIDERS = ("nim", "openrouter")


def provider_available(name: str) -> bool:
    return bool(NVIDIA_API_KEY) if name == "nim" else bool(OPENROUTER_API_KEY)


@dataclass(frozen=True)
class ModelSpec:
    """One model on one provider, and the limits we hold ourselves to when calling it.

    ``rpm`` is per-model. NVIDIA caps around 40 requests/minute *per model*, so distinct models
    get genuinely independent buckets; OpenRouter caps per-account instead, which the daily
    ledger handles separately.

    ``thinking`` controls Nemotron's reasoning pass. Measured on nemotron-3-nano: disabling it
    via ``chat_template_kwargs`` cut a small structured reply from 63 completion tokens to 11.
    Reasoning arrives in a separate ``reasoning_content`` field rather than mixed into the
    answer, so this is purely a latency/credit decision — and for the writer also a UX one,
    since a thinking pass delays the first visible token of a streamed answer.
    """

    id: str
    provider: str = "nim"
    rpm: int = 30
    ctx: int = 128_000
    max_out: int = 1024
    temperature: float = 0.2
    thinking: bool = False

    @property
    def key(self) -> str:
        """Provider-qualified id — both providers serve some of the same model names."""
        return f"{self.provider}:{self.id}"


# Order is measured, not assumed — see scripts/race_models.py. Two findings drove it:
#
#   * OpenRouter's free tier shares roughly 20 requests/minute across *all* free models, and
#     the fast role fires 4-6 times per question. Racing them produced 429s almost immediately,
#     so NVIDIA leads the fast list: its limit is 40/min **per model**, which is a much better
#     fit for a chatty stage. OpenRouter is the fallback, which is exactly what it is good at.
#   * Bigger is not better here. nemotron-3-ultra-550b is the largest free model available and
#     took **21.5 s** for a two-sentence answer; nemotron-3-super-120b on NIM did the same job
#     in 2.8 s. The 550B model is kept last as a availability backstop, not as a first choice.
#
# Measured medians (2 calls each, realistic prompts):
#   fast   nim-lightning-30b 4.8 s · openrouter-lightning 3.6 s · nim-nano intermittent 410
#   writer nim-super-120b 2.8 s · openrouter-super-120b 4.1 s · openrouter-ultra-550b 21.6 s

# Cheap, high-frequency structured stages: planning, query writing, grading, gap analysis,
# verification. Runs several times per question, so throughput matters more than eloquence.
FAST_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("nvidia/nemotron-3-nano-30b-a3b", "nim", rpm=30, max_out=1400, temperature=0.1),
    ModelSpec("nvidia/nemotron-3.5-lightning-30b-a3b", "nim", rpm=30, max_out=1400,
              temperature=0.1),
    ModelSpec("nvidia/nemotron-3.5-lightning:free", "openrouter", rpm=OPENROUTER_RPM,
              ctx=1_000_000, max_out=1400, temperature=0.1),
    ModelSpec("google/gemma-4-26b-a4b-it:free", "openrouter", rpm=OPENROUTER_RPM,
              ctx=262_144, max_out=1400, temperature=0.1),
)

# The user-facing answer: one or two calls a turn, so quality is worth more than speed — but
# not at 21 seconds. NIM's 120B leads on measured latency and OpenRouter covers the outage case.
WRITER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("nvidia/nemotron-3-super-120b-a12b", "nim", rpm=25, max_out=1600,
              temperature=0.3),
    ModelSpec("google/gemma-4-31b-it:free", "openrouter", rpm=OPENROUTER_RPM,
              ctx=262_144, max_out=1800, temperature=0.3),
    ModelSpec("nvidia/nemotron-3-super-120b-a12b:free", "openrouter", rpm=OPENROUTER_RPM,
              ctx=262_144, max_out=1800, temperature=0.3),
    ModelSpec("z-ai/glm-5.2:free", "openrouter", rpm=OPENROUTER_RPM,
              ctx=256_000, max_out=1800, temperature=0.3),
    ModelSpec("nvidia/nemotron-3-ultra-550b-a55b:free", "openrouter", rpm=OPENROUTER_RPM,
              ctx=1_000_000, max_out=1800, temperature=0.3),
)

# Force a single model for experiments, as "provider:model-id".
FAST_MODEL_OVERRIDE = env_str("KYR_FAST_MODEL", "")
WRITER_MODEL_OVERRIDE = env_str("KYR_WRITER_MODEL", "")
FAST_MODEL = FAST_MODELS[0]
WRITER_MODEL = WRITER_MODELS[0]


# Optional remote reranker — used when the `lean` profile offloads reranking off-GPU.
# Independent of the embedder (a cross-encoder reads text, not vectors), so it stays valid
# even after `baai/bge-m3` leaves the hosted catalog.
NIM_RERANK_MODEL = env_str("KYR_NIM_RERANK_MODEL", "nvidia/llama-3.2-nv-rerankqa-1b-v2")
NIM_RERANK_ALTERNATES = (
    "nvidia/llama-nemotron-rerank-1b-v2",
    "nvidia/nv-rerankqa-mistral-4b-v3",
)
NIM_RERANK_RPM = env_int("KYR_NIM_RERANK_RPM", 30)

# 429 / transient-failure policy. Deadlines, not retry counts, decide when to give up:
# a rate limit must degrade the answer, never kill the turn.
RETRY_INITIAL_DELAY = env_float("KYR_RETRY_INITIAL_DELAY", 2.0)
RETRY_MAX_DELAY = env_float("KYR_RETRY_MAX_DELAY", 45.0)
RETRY_MULTIPLIER = env_float("KYR_RETRY_MULTIPLIER", 2.0)
RETRY_MAX_ATTEMPTS = env_int("KYR_RETRY_MAX_ATTEMPTS", 8)

# AIMD self-tuning of the per-model buckets.
AIMD_DECREASE = env_float("KYR_AIMD_DECREASE", 0.7)   # multiply rpm by this on a 429
AIMD_INCREASE = env_float("KYR_AIMD_INCREASE", 2.0)   # add this per clean minute
AIMD_FLOOR_RPM = env_int("KYR_AIMD_FLOOR_RPM", 5)

# Rough credit accounting so the UI can warn before the free tier runs dry.
SESSION_CREDIT_BUDGET = env_int("KYR_SESSION_CREDIT_BUDGET", 0)  # 0 = unlimited/unknown


# ── local models (embedder is corpus-locked; reranker is swappable) ───────────────────
# Changing EMBED_MODEL invalidates the whole database — see DB README §8.
EMBED_MODEL = env_str("KYR_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024
EMBED_MAX_SEQ = env_int("KYR_EMBED_MAX_SEQ", 1024)
EMBED_QUERY_PREFIX = ""  # bge-m3 encodes queries and passages symmetrically

RERANK_QUALITY = env_str("KYR_RERANK_QUALITY", "BAAI/bge-reranker-v2-m3")
RERANK_BALANCED = env_str("KYR_RERANK_BALANCED", "BAAI/bge-reranker-base")
RERANK_CPU = env_str("KYR_RERANK_CPU", "BAAI/bge-reranker-base")


@dataclass(frozen=True)
class Profile:
    """A measured resource plan. ``model_vram_mb`` is what the *models* occupy (weights plus
    working activations), measured on an RTX 3050 with fp16 weights:

        bge-m3            +1090 MiB
        bge-reranker-base  +760 MiB   -> balanced = 1850
        bge-reranker-v2-m3 +1394 MiB  -> quality  = 2484

    A profile is selectable when ``free_vram >= model_vram_mb + VRAM_RESERVE_MB``, so the
    reserve is what stays available to the desktop and everything else on the machine.
    """

    name: str
    rerank_backend: str            # "local" | "nim" | "none"
    rerank_model: str | None
    model_vram_mb: int
    embed_batch: int
    rerank_batch: int
    note: str = ""
    # `lite` turns the embedder off entirely and runs on BM25 alone. Measured on the gold set:
    # Recall@5 90.5% and MRR 0.769 against 100% / 0.873 for the full pipeline, at ~60 ms instead
    # of ~400 ms, in under 1 GB of RAM. It works this well because the BM25 index covers
    # `embed_text`, which carries the LLM-generated citizen questions — so keyword search is
    # querying the way people ask, not raw statutory language.
    # The real cost is abstention, not recall: without a reranker only 5 of the 11 stress
    # questions are caught at retrieval level, so the planner's out_of_scope check carries it.
    use_embedder: bool = True

    @property
    def needs_models(self) -> bool:
        return self.use_embedder or self.rerank_backend == "local"


# Ordered best-first; the first profile that fits the probed machine wins.
PROFILES: tuple[Profile, ...] = (
    Profile("quality",  "local", RERANK_QUALITY,  model_vram_mb=2484, embed_batch=8, rerank_batch=16,
            note="strongest multilingual reranker; ~0.85s per rerank batch"),
    Profile("balanced", "local", RERANK_BALANCED, model_vram_mb=1850, embed_batch=8, rerank_batch=25,
            note="multilingual reranker, ~0.39s per batch, keeps ~1.4 GB VRAM free"),
    Profile("lean",     "nim",   None,            model_vram_mb=1090, embed_batch=4, rerank_batch=25,
            note="embedding stays local (corpus-locked); reranking offloaded to NIM"),
    Profile("cpu",      "local", RERANK_CPU,      model_vram_mb=0,    embed_batch=2, rerank_batch=8,
            note="no usable CUDA; expect multi-second retrieval"),
)

# For CPU-only hosting: keep dense retrieval, drop the cross-encoder.
#
# This used to send reranking to NIM. That is not a real option — every NIM reranking endpoint
# returns 410/404, so the profile silently degraded to fused RRF while still loading the
# *uncalibrated* NIM thresholds, which is strictly worse than asking for no reranker at all.
#
# Measured on the gold set (42 questions), all four CPU-viable configurations:
#
#   embedder + cross-encoder, pool 24   Recall@5 100%    MRR 0.873   ~20 s   <- unusable on 2 vCPU
#   embedder + cross-encoder, pool 12   Recall@5  95.2%  MRR 0.849   ~10 s   <- still too slow
#   embedder, no cross-encoder (this)   Recall@5  93%    MRR 0.787   ~90 ms
#   neither (`lite`)                    Recall@5  90.5%  MRR 0.769   ~60 ms
#
# The cross-encoder is worth 7 points of Recall@5 and it is the right default wherever there is
# a GPU. On two shared vCPUs it costs 20 seconds a question, which no demo survives, so this
# profile trades those 7 points for a response that arrives while somebody is still watching.
CPU_LEAN = Profile("cpu_lean", "none", None, model_vram_mb=0, embed_batch=2, rerank_batch=25,
                   note="dense + BM25 on CPU, no cross-encoder — Recall@5 93% in ~90 ms")

# For a box too small to hold bge-m3 at all — a 1-2 GB free-tier instance. Never selected
# automatically: it trades real retrieval quality for fitting, and that should be a decision
# somebody makes, not something that quietly happens.
LITE = Profile("lite", "none", None, model_vram_mb=0, embed_batch=1, rerank_batch=1,
               use_embedder=False,
               note="BM25 only — no models, <1 GB RAM, Recall@5 90.5% at ~60 ms")

PROFILES = PROFILES + (CPU_LEAN,)

PROFILE_REQUEST = env_str("KYR_PROFILE", "auto")          # auto | quality | balanced | lean | cpu
# 1 GB left for the desktop. This is what makes `balanced` rather than `quality` the default
# on a 4 GB laptop card that is also driving a display.
VRAM_RESERVE_MB = env_int("KYR_VRAM_RESERVE_MB", 1024)
RAM_FLOOR_MB = env_int("KYR_RAM_FLOOR_MB", 1200)          # refuse to load below this
RAM_LOAD_HEADROOM_MB = env_int("KYR_RAM_LOAD_HEADROOM_MB", 2600)  # bge-m3 load spike (measured 2329 MB)
MODEL_IDLE_EVICT_S = env_int("KYR_MODEL_IDLE_EVICT_S", 0)  # 0 = never evict
GPU_OOM_RETRIES = env_int("KYR_GPU_OOM_RETRIES", 2)       # batch halvings before falling back


# ── retrieval ─────────────────────────────────────────────────────────────────────────
FETCH_K = env_int("KYR_FETCH_K", 25)          # per ranked list, before fusion
TOP_K = env_int("KYR_TOP_K", 5)               # sections returned to the answer layer
RERANK_POOL = env_int("KYR_RERANK_POOL", 24)  # candidates that reach the cross-encoder
RRF_K = env_int("KYR_RRF_K", 60)
# BM25 score treated as "certainly relevant" when no reranker is available. Measured on
# this corpus: on-topic legal queries peak around 24-31, off-topic ones around 13-19.
BM25_FULL_SCORE = env_float("KYR_BM25_FULL_SCORE", 40.0)
MMR_LAMBDA = env_float("KYR_MMR_LAMBDA", 0.6)
# When the question names a specific Act, several sections *of that Act* is the right answer,
# so diversity is dialled down rather than spreading results across unrelated statutes.
MMR_LAMBDA_FOCUSED = env_float("KYR_MMR_LAMBDA_FOCUSED", 0.85)
# Without a reranker the base ordering is weaker, so diversity costs more than it returns.
MMR_LAMBDA_NO_RERANK = env_float("KYR_MMR_LAMBDA_NO_RERANK", 0.97)
# How much extra weight a ranked list restricted to a named Act carries in the fusion.
ACT_FILTER_WEIGHT = env_float("KYR_ACT_FILTER_WEIGHT", 2.5)

# The general law of the land. Dozens of sectoral statutes grant *someone* a power of arrest —
# forest officers, naval authorities, railway police — and they rank well for a query like
# "can the police arrest me" while being useless to the person asking. When a question is
# plainly about crime or policing and names no particular Act, these four get their own
# weighted ranked lists so the general code outranks the specialist one.
GENERAL_CODES = (
    "Constitution of India",
    "Bharatiya Nyaya Sanhita, 2023",
    "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "Bharatiya Sakshya Adhiniyam, 2023",
)
GENERAL_CODE_WEIGHT = env_float("KYR_GENERAL_CODE_WEIGHT", 2.0)
# Applied to the *ordering* after reranking, not to the reported score. Large enough to beat
# a near-tie, since sectoral and general provisions are often worded almost identically.
GENERAL_CODE_BOOST = env_float("KYR_GENERAL_CODE_BOOST", 0.25)
# Words that mean "this is a general criminal-law or policing question".
CRIMINAL_TRIGGERS = (
    "police", "arrest", "arrested", "custody", "detain", "detention", "bail", "fir",
    "offence", "offense", "crime", "criminal", "punishment", "penalty", "imprison",
    "jail", "magistrate", "accused", "charge", "prosecut", "remand", "interrogat",
    "search warrant", "seizure", "handcuff", "lock-up", "lockup",
)
TOPK_MIN, TOPK_MAX = 2, 12

# Thresholds are reranker-specific: a local sigmoid score and a NIM logit live on different
# scales. scripts/calibrate.py re-derives these per profile and writes them to
# .runtime/thresholds.json, which overrides these defaults at load time.
LOW_SCORE = env_float("KYR_LOW_SCORE", 0.05)          # below this -> abstain, go to the web
CITE_MIN_SCORE = env_float("KYR_CITE_MIN_SCORE", 0.20)  # pre-filter before the LLM grader
THRESHOLDS_FILE = RUNTIME_DIR / "thresholds.json"


# ── research depth ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DepthBudget:
    name: str
    max_rounds: int
    max_crawls: int
    nav_depth: int
    max_llm_calls: int
    deadline_s: float
    max_queries_per_subq: int


DEPTHS: dict[str, DepthBudget] = {
    "quick":    DepthBudget("quick",    max_rounds=1, max_crawls=0,  nav_depth=0, max_llm_calls=4,  deadline_s=25,  max_queries_per_subq=1),
    "standard": DepthBudget("standard", max_rounds=1, max_crawls=3,  nav_depth=1, max_llm_calls=8,  deadline_s=75,  max_queries_per_subq=3),
    "deep":     DepthBudget("deep",     max_rounds=4, max_crawls=10, nav_depth=2, max_llm_calls=20, deadline_s=240, max_queries_per_subq=4),
}
DEFAULT_DEPTH = env_str("KYR_DEFAULT_DEPTH", "auto")


# ── context window management ─────────────────────────────────────────────────────────
# Deliberately far below the model's advertised window: latency, credits and
# lost-in-the-middle all degrade long before the context limit does.
WRITER_INPUT_BUDGET_TOKENS = env_int("KYR_WRITER_INPUT_BUDGET", 14_000)
FAST_INPUT_BUDGET_TOKENS = env_int("KYR_FAST_INPUT_BUDGET", 8_000)
CONTEXT_SAFETY_TOKENS = env_int("KYR_CONTEXT_SAFETY", 512)

STATUTE_TEXT_CAP = env_int("KYR_STATUTE_TEXT_CAP", 2600)   # chars per statute section
WEB_TEXT_CAP = env_int("KYR_WEB_TEXT_CAP", 1800)           # chars per web/crawl source
WIKI_TEXT_CAP = env_int("KYR_WIKI_TEXT_CAP", 1200)
PAGE_CHUNK_CHARS = env_int("KYR_PAGE_CHUNK_CHARS", 1400)   # crawled-page chunk size
PAGE_CHUNKS_KEPT = env_int("KYR_PAGE_CHUNKS_KEPT", 3)      # top chunks kept per page

HISTORY_TURNS_VERBATIM = env_int("KYR_HISTORY_TURNS", 4)
HISTORY_SUMMARY_TRIGGER = env_int("KYR_HISTORY_SUMMARY_TRIGGER", 8)


# ── web search & crawling ─────────────────────────────────────────────────────────────
WEB_MAX_RESULTS = env_int("KYR_WEB_MAX_RESULTS", 5)
WEB_TIMEOUT = env_float("KYR_WEB_TIMEOUT", 12.0)
WEB_CACHE_TTL = env_int("KYR_WEB_CACHE_TTL", 1800)
WEB_MAX_PER_MIN = env_int("KYR_WEB_MAX_PER_MIN", 10)

WIKI_MAX_RESULTS = env_int("KYR_WIKI_MAX_RESULTS", 2)
WIKI_TIMEOUT = env_float("KYR_WIKI_TIMEOUT", 10.0)

CRAWL_TIMEOUT_S = env_float("KYR_CRAWL_TIMEOUT_S", 25.0)
CRAWL_CACHE_TTL = env_int("KYR_CRAWL_CACHE_TTL", 86_400)
CRAWL_MAX_CONCURRENT = env_int("KYR_CRAWL_MAX_CONCURRENT", 3)
CRAWL_USE_BROWSER = env_bool("KYR_CRAWL_USE_BROWSER", True)   # escalate to Chromium when needed
CRAWL_BROWSER_IDLE_S = env_int("KYR_CRAWL_BROWSER_IDLE_S", 180)
CRAWL_MIN_CHARS = env_int("KYR_CRAWL_MIN_CHARS", 400)         # below this, retry with a browser
CRAWL_RESPECT_ROBOTS = env_bool("KYR_CRAWL_RESPECT_ROBOTS", True)
CRAWL_USER_AGENT = env_str(
    "KYR_CRAWL_USER_AGENT",
    "KnowYourRights/0.1 (public legal-information assistant; +https://github.com/)",
)

# Trust tiers. Higher wins when the writer must choose between conflicting sources.
TIER_STATUTE = 100
TIER_OFFICIAL = 80
TIER_LEGAL_PORTAL = 60
TIER_WIKIPEDIA = 40
TIER_WEB = 20

OFFICIAL_DOMAINS = (
    "indiacode.nic.in", "gov.in", "nic.in", "sci.gov.in", "egazette.gov.in",
    "eci.gov.in", "rti.gov.in", "rtionline.gov.in", "consumerhelpline.gov.in",
    "doj.gov.in", "mha.gov.in", "labour.gov.in", "india.gov.in",
)
LEGAL_PORTAL_DOMAINS = ("indiankanoon.org", "prsindia.org", "barandbench.com", "livelaw.in")


# ── safety ────────────────────────────────────────────────────────────────────────────
HELPLINES = (
    ("Emergency (police / fire / ambulance)", "112"),
    ("Women's helpline", "1091"),
    ("Women's helpline (domestic abuse)", "181"),
    ("Childline", "1098"),
    ("Free legal aid (NALSA)", "15100"),
    ("Mental health (Tele-MANAS)", "14416"),
)

DISCLAIMER = (
    "General information about central Indian law, with citations. This is not legal advice — "
    "for your situation consult a qualified lawyer, or call NALSA on 15100 for free legal aid."
)

CATEGORIES = (
    "Fundamental Rights", "Criminal & Police", "Consumer & Services",
    "Employment & Labour", "Family & Marriage", "Property & Housing",
    "Women & Children", "Privacy & Data", "Health & Medicine", "Education",
    "Environment", "Taxation & Finance", "Business & Companies", "Information & RTI",
    "Civil Procedure & Courts", "Transport & Motor", "Government & Administration", "Other",
)

# The corpus is central law but a few state acts leak in, and the `jurisdiction` column is
# unreliable — the act title is the trustworthy signal. See DB README §9.
STATE_PREFIXES = (
    "Andhra Pradesh", "Arunachal", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Orissa",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Jammu", "Puducherry", "Pondicherry",
)

INDIAN_STATES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Ladakh", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
)


# ── server ────────────────────────────────────────────────────────────────────────────
HOST = env_str("KYR_HOST", "127.0.0.1")
PORT = env_int("KYR_PORT", 8000)
LOG_LEVEL = env_str("KYR_LOG_LEVEL", "INFO")
TRACE_ENABLED = env_bool("KYR_TRACE", True)


def ensure_runtime_dirs() -> None:
    """Create the writable runtime tree. Safe to call repeatedly."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
