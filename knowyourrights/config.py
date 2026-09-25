"""Every tuning knob in one place.

Values can be overridden from the environment, or from a ``.env`` file in development (real
environment variables win over ``.env``). Nothing here imports a heavy library, so scripts and
tests can import it cheaply. See ``.env.example`` for the settings an operator is likely to set.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover - python-dotenv is a runtime requirement
    pass


# ── env helpers ───────────────────────────────────────────────────────────────────────
def env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return default if value is None or not value.strip() else value.strip()


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
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Whole escape sequences are stripped before character filtering: removing the punctuation first
# would leave the digits of ESC[200~ behind, glued to a pasted value.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z~]")
_NOT_KEY_CHAR = re.compile(r"[^A-Za-z0-9._-]")


def env_key(name: str) -> str:
    """Read an API key, keeping only characters an API key can contain.

    Keys arrive by paste, and a paste picks things up: bracketed-paste escapes, a non-breaking
    space, a smart dash. httpx encodes headers as ASCII, so one invisible character makes every
    call fail with an encoding error while the key looks fine in the file.
    """
    raw = os.environ.get(name, "")
    cleaned = _NOT_KEY_CHAR.sub("", _ANSI_ESCAPE.sub("", raw))
    if raw.strip() and cleaned != raw.strip():
        print(f"warning: {name} contained characters an API key cannot have; "
              f"{len(raw.strip()) - len(cleaned)} removed. Re-paste it if authentication fails.",
              file=sys.stderr)
    return cleaned


# ── paths ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KYR_DATA_DIR", ROOT / "data"))
RUNTIME_DIR = Path(os.environ.get("KYR_RUNTIME_DIR", ROOT / ".runtime"))
CACHE_DIR = RUNTIME_DIR / "cache"
DB_PATH = Path(os.environ.get("LEGAL_DB_PATH", DATA_DIR / "legal_db"))
TABLE = "laws"
WEB_DIR = Path(__file__).resolve().parent / "web"


def ensure_runtime_dirs() -> None:
    """Create the writable runtime tree. Safe to call repeatedly."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── providers ─────────────────────────────────────────────────────────────────────────
# OpenRouter serves the chat models and the retrieval models. NVIDIA NIM is an optional second
# chat provider with its own rate limits, used only when OpenRouter's models cannot answer.
OPENROUTER_BASE_URL = env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = env_key("OPENROUTER_API_KEY")
# Sent as HTTP-Referer and X-Title; OpenRouter uses them for attribution.
OPENROUTER_APP_URL = env_str("OPENROUTER_APP_URL",
                             "https://github.com/DamnKuldeep/KnowYourRightsAI")
OPENROUTER_APP_NAME = env_str("OPENROUTER_APP_NAME", "KnowYourRightsAI")
# Free models share ~20 requests a minute and 1,000 a day; both are enforced here, so the app
# degrades on its own terms rather than being cut off mid-answer. Paid models are metered in
# money, not by those caps, and get a realistic ceiling instead. 429s back off either way.
OPENROUTER_RPM = env_int("OPENROUTER_RPM", 15)
OPENROUTER_PAID_RPM = env_int("OPENROUTER_PAID_RPM", 60)
OPENROUTER_DAILY_LIMIT = env_int("OPENROUTER_DAILY_LIMIT", 1000)
OPENROUTER_DAILY_RESERVE = env_int("OPENROUTER_DAILY_RESERVE", 60)

NIM_BASE_URL = env_str("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = env_key("NVIDIA_API_KEY")

LLM_TIMEOUT_S = env_float("KYR_LLM_TIMEOUT_S", 90.0)
PROVIDERS = ("openrouter", "nim")


def provider_available(name: str) -> bool:
    return bool(OPENROUTER_API_KEY) if name == "openrouter" else bool(NVIDIA_API_KEY)


def is_free_model(model_id: str) -> bool:
    """Only ``:free`` variants draw on OpenRouter's daily allowance; paid calls cost money."""
    return model_id.endswith(":free") or model_id == "openrouter/free"


@dataclass(frozen=True)
class ModelSpec:
    """One model on one provider, and the limits we hold ourselves to when calling it.

    ``thinking`` controls a model's reasoning pass. Off by default: on short structured stages it
    multiplies completion tokens several times over, and for the writer it delays the first
    visible word of the answer.
    """

    id: str
    provider: str = "openrouter"
    rpm: int = 30
    ctx: int = 128_000
    max_out: int = 1024
    temperature: float = 0.2
    thinking: bool = False

    @property
    def key(self) -> str:
        """Provider-qualified id: both providers serve some of the same model names."""
        return f"{self.provider}:{self.id}"


# Routing is measured, not assumed: `python scripts/race_models.py` reproduces it for about a
# cent, timing the real planner prompt (output must validate as a Plan) and the writer by first
# token, which is what a reader feels.
#
#   fast role (planner prompt)            median     valid   cost/call
#     google/gemini-2.5-flash-lite        1,430 ms   2/2     $0.00020
#     inception/mercury-2.5               1,475 ms   2/2     $0.00010
#     nvidia/nemotron-3-nano-30b-a3b      1,912 ms   2/2     $0.00011
#     nemotron-3.5-lightning:free        29,132 ms   2/2     free, and unusable
#
# The fast role runs 4-6 times a question, so a free model there would both take ~30 s a stage
# and spend the shared allowance. A paid model costs about a tenth of a cent per question.
FAST_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("google/gemini-2.5-flash-lite", rpm=OPENROUTER_PAID_RPM,
              ctx=1_048_576, max_out=1400, temperature=0.1),
    ModelSpec("inception/mercury-2.5", rpm=OPENROUTER_PAID_RPM,
              ctx=260_000, max_out=1400, temperature=0.1),
    ModelSpec("nvidia/nemotron-3-nano-30b-a3b", rpm=OPENROUTER_PAID_RPM,
              ctx=262_144, max_out=1400, temperature=0.1),
    # failover on independent limits
    ModelSpec("nvidia/nemotron-3-nano-30b-a3b", "nim", rpm=30, max_out=1400, temperature=0.1),
    ModelSpec("nvidia/nemotron-3.5-lightning-30b-a3b", "nim", rpm=30, max_out=1400,
              temperature=0.1),
)

# The user-facing answer: one streamed call a turn. qwen3.7-flash leads because it streams
# reliably: the free 120B was faster to its first token but, over one session, failed 12 of 22
# streams, 5 of them after text had started, which a reader sees as an answer cutting off.
WRITER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("qwen/qwen3.7-flash", rpm=OPENROUTER_PAID_RPM,
              ctx=1_000_000, max_out=1800, temperature=0.3),
    ModelSpec("google/gemini-2.5-flash-lite", rpm=OPENROUTER_PAID_RPM,
              ctx=1_048_576, max_out=1800, temperature=0.3),
    ModelSpec("nvidia/nemotron-3-super-120b-a12b:free", rpm=OPENROUTER_RPM,
              ctx=262_144, max_out=1800, temperature=0.3),
    ModelSpec("nvidia/nemotron-3-super-120b-a12b", "nim", rpm=25, max_out=1600,
              temperature=0.3),
)

# Force a single model for an experiment, as "provider:model-id".
FAST_MODEL_OVERRIDE = env_str("KYR_FAST_MODEL", "")
WRITER_MODEL_OVERRIDE = env_str("KYR_WRITER_MODEL", "")

# 429 and transient-failure policy. Deadlines, not attempt counts, decide when to give up: a
# rate limit should cost an answer some depth, never the whole turn.
RETRY_INITIAL_DELAY = env_float("KYR_RETRY_INITIAL_DELAY", 2.0)
RETRY_MAX_DELAY = env_float("KYR_RETRY_MAX_DELAY", 45.0)
RETRY_MULTIPLIER = env_float("KYR_RETRY_MULTIPLIER", 2.0)
RETRY_MAX_ATTEMPTS = env_int("KYR_RETRY_MAX_ATTEMPTS", 8)

# AIMD self-tuning of the per-model request buckets.
AIMD_DECREASE = env_float("KYR_AIMD_DECREASE", 0.7)   # multiply the rate by this on a 429
AIMD_INCREASE = env_float("KYR_AIMD_INCREASE", 2.0)   # add this per clean minute
AIMD_FLOOR_RPM = env_int("KYR_AIMD_FLOOR_RPM", 5)


# ── retrieval models (over OpenRouter) ────────────────────────────────────────────────
# The embedder is corpus-locked: the database was built with bge-m3, and OpenRouter's copy was
# verified against the stored vectors at cosine 1.0000 (scripts/verify_embeddings.py).
EMBED_API_MODEL = env_str("KYR_EMBED_API_MODEL", "baai/bge-m3")
EMBED_DIM = 1024
# The reranker is swappable, but its scores have their own scale, so a new one needs
# `python scripts/calibrate.py`.
#   cohere/rerank-v3.5       ~830 ms   $0.001 a search
#   qwen/qwen3-reranker-8b  ~1190 ms   $0.000275 a search
RERANK_API_MODEL = env_str("KYR_RERANK_API_MODEL", "cohere/rerank-v3.5")
# Short on purpose: a slow rerank should fall back to fused ranking, not hold a turn open.
RETRIEVAL_API_TIMEOUT_S = env_float("KYR_RETRIEVAL_API_TIMEOUT_S", 20.0)
RETRIEVAL_API_EMBED_BATCH = env_int("KYR_RETRIEVAL_API_EMBED_BATCH", 64)
# Paid endpoints, so the free tier's ~20/min does not apply. A deep turn makes ~16 retrieval
# calls, and a limit of 12/min once cost it a minute of waiting on its own throttle.
RETRIEVAL_API_RPM = env_int("KYR_RETRIEVAL_API_RPM", 120)


# ── retrieval ─────────────────────────────────────────────────────────────────────────
FETCH_K = env_int("KYR_FETCH_K", 25)          # per ranked list, before fusion
TOP_K = env_int("KYR_TOP_K", 5)               # sections returned to the answer layer
RERANK_POOL = env_int("KYR_RERANK_POOL", 24)  # candidates that reach the reranker
RRF_K = env_int("KYR_RRF_K", 60)
# A BM25 score treated as "certainly relevant" when ranking without a reranker. Measured on
# this corpus: on-topic legal queries peak around 24-31, off-topic ones around 13-19.
BM25_FULL_SCORE = env_float("KYR_BM25_FULL_SCORE", 40.0)
MMR_LAMBDA = env_float("KYR_MMR_LAMBDA", 0.6)
# When the question names an Act, several sections of that Act is the right answer, so
# diversity is dialled down rather than spread across unrelated statutes.
MMR_LAMBDA_FOCUSED = env_float("KYR_MMR_LAMBDA_FOCUSED", 0.85)
# Without a reranker the base ordering is weaker, so diversity costs more than it returns.
MMR_LAMBDA_NO_RERANK = env_float("KYR_MMR_LAMBDA_NO_RERANK", 0.97)
# Extra weight for ranked lists restricted to an Act the question names.
ACT_FILTER_WEIGHT = env_float("KYR_ACT_FILTER_WEIGHT", 2.5)

# The general law of the land. Dozens of sectoral statutes grant someone a power of arrest, and
# they rank well for "can the police arrest me" while being useless to the person asking. When a
# question is plainly about crime or policing and names no Act, these get weighted lists.
GENERAL_CODES = (
    "Constitution of India",
    "Bharatiya Nyaya Sanhita, 2023",
    "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "Bharatiya Sakshya Adhiniyam, 2023",
)
GENERAL_CODE_WEIGHT = env_float("KYR_GENERAL_CODE_WEIGHT", 2.0)
# Applied to the ordering after reranking, as fractions of the query's best score so they mean
# the same thing on any reranker's scale: a general code within the eligible fraction of the best
# candidate is lifted by the boost fraction. It settles near-ties; it never rescues bad rows.
GENERAL_CODE_BOOST = env_float("KYR_GENERAL_CODE_BOOST", 0.25)
GENERAL_CODE_ELIGIBLE = env_float("KYR_GENERAL_CODE_ELIGIBLE", 0.5)
# Show the reranker each section's citizen questions as well as its text (ranking.py explains).
RERANK_WITH_QUESTIONS = env_bool("KYR_RERANK_WITH_QUESTIONS", True)
# Words that mark a general criminal-law or policing question.
CRIMINAL_TRIGGERS = (
    "police", "arrest", "arrested", "custody", "detain", "detention", "bail", "fir",
    "offence", "offense", "crime", "criminal", "punishment", "penalty", "imprison",
    "jail", "magistrate", "accused", "charge", "prosecut", "remand", "interrogat",
    "search warrant", "seizure", "handcuff", "lock-up", "lockup",
)

# Abstention and citation cut-offs. Scores are ranking-method specific, so these are only the
# fallback: calibrated values ship in knowyourrights/thresholds.json, and a local run of
# scripts/calibrate.py writes .runtime/thresholds.json, which wins.
LOW_SCORE = env_float("KYR_LOW_SCORE", 0.05)            # below this, abstain
CITE_MIN_SCORE = env_float("KYR_CITE_MIN_SCORE", 0.20)  # below this, never cited
THRESHOLDS_FILE = RUNTIME_DIR / "thresholds.json"
PACKAGED_THRESHOLDS = Path(__file__).resolve().parent / "thresholds.json"


# ── research depth ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DepthBudget:
    name: str
    max_rounds: int      # research rounds (gap analysis decides whether another is needed)
    max_crawls: int      # web pages read in full
    nav_depth: int       # link levels followed inside a portal
    deadline_s: float    # wall-clock budget; research stops and the answer is written from
                         # what was found


DEPTHS: dict[str, DepthBudget] = {
    "quick": DepthBudget("quick", max_rounds=1, max_crawls=0, nav_depth=0, deadline_s=25),
    "standard": DepthBudget("standard", max_rounds=1, max_crawls=3, nav_depth=1, deadline_s=75),
    "deep": DepthBudget("deep", max_rounds=4, max_crawls=10, nav_depth=2, deadline_s=240),
}


# ── context window management ─────────────────────────────────────────────────────────
# Far below the models' advertised windows: latency, cost and lost-in-the-middle all degrade
# long before the context limit does.
WRITER_INPUT_BUDGET_TOKENS = env_int("KYR_WRITER_INPUT_BUDGET", 14_000)
CONTEXT_SAFETY_TOKENS = env_int("KYR_CONTEXT_SAFETY", 512)

STATUTE_TEXT_CAP = env_int("KYR_STATUTE_TEXT_CAP", 2600)   # chars per statute section
WEB_TEXT_CAP = env_int("KYR_WEB_TEXT_CAP", 1800)           # chars per web or crawled source
WIKI_TEXT_CAP = env_int("KYR_WIKI_TEXT_CAP", 1200)
PAGE_CHUNK_CHARS = env_int("KYR_PAGE_CHUNK_CHARS", 1400)   # crawled-page chunk size
PAGE_CHUNKS_KEPT = env_int("KYR_PAGE_CHUNKS_KEPT", 3)      # best chunks kept per page

HISTORY_TURNS_VERBATIM = env_int("KYR_HISTORY_TURNS", 4)
HISTORY_SUMMARY_TRIGGER = env_int("KYR_HISTORY_SUMMARY_TRIGGER", 8)
# Each past answer is capped on its own before the history budget is spent: a follow-up needs to
# know what was answered, not all of it, and one long answer must not evict every earlier turn.
HISTORY_ANSWER_CAP_TOKENS = env_int("KYR_HISTORY_ANSWER_CAP_TOKENS", 220)
# A past source is offered back to a new question only if it shares this fraction of the
# question's content words. A pre-filter: the grader then judges everything recalled.
RECALL_MIN_SHARE = env_float("KYR_RECALL_MIN_SHARE", 0.34)
# Sources remembered per conversation for follow-ups; the oldest are dropped past this.
SESSION_POOL_MAX = env_int("KYR_SESSION_POOL_MAX", 40)


# ── web search and crawling ───────────────────────────────────────────────────────────
WEB_MAX_RESULTS = env_int("KYR_WEB_MAX_RESULTS", 5)
WEB_TIMEOUT = env_float("KYR_WEB_TIMEOUT", 8.0)
# Keyless search engines, tried in this order. ddgs's "auto" picks at random and falls through
# slowly when one rate-limits: measured 6-8 s per search against 1-2 s for this list.
WEB_BACKENDS = env_str("KYR_WEB_BACKENDS", "duckduckgo,yahoo,yandex")
WEB_CACHE_TTL = env_int("KYR_WEB_CACHE_TTL", 1800)
WEB_MAX_PER_MIN = env_int("KYR_WEB_MAX_PER_MIN", 10)

WIKI_MAX_RESULTS = env_int("KYR_WIKI_MAX_RESULTS", 2)
WIKI_TIMEOUT = env_float("KYR_WIKI_TIMEOUT", 10.0)

# Per page. Government pages load three at a time in ~4.4 s; a page that has not answered in
# 10 s is dropped and the answer is written from the pages that did.
CRAWL_TIMEOUT_S = env_float("KYR_CRAWL_TIMEOUT_S", 10.0)
# A batch keeps whatever has arrived within this budget, so a slow page costs only itself.
CRAWL_BATCH_BUDGET_S = env_float("KYR_CRAWL_BATCH_BUDGET_S", 14.0)
CRAWL_CACHE_TTL = env_int("KYR_CRAWL_CACHE_TTL", 86_400)
CRAWL_MAX_CONCURRENT = env_int("KYR_CRAWL_MAX_CONCURRENT", 3)
CRAWL_USE_BROWSER = env_bool("KYR_CRAWL_USE_BROWSER", True)   # escalate to Chromium if needed
CRAWL_BROWSER_IDLE_S = env_int("KYR_CRAWL_BROWSER_IDLE_S", 180)
CRAWL_MIN_CHARS = env_int("KYR_CRAWL_MIN_CHARS", 400)         # below this, retry with a browser
CRAWL_RESPECT_ROBOTS = env_bool("KYR_CRAWL_RESPECT_ROBOTS", True)
CRAWL_USER_AGENT = env_str(
    "KYR_CRAWL_USER_AGENT",
    "KnowYourRights/1.0 (public legal-information assistant; "
    "+https://github.com/DamnKuldeep/KnowYourRightsAI)",
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


# ── safety and jurisdiction ───────────────────────────────────────────────────────────
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

# The corpus is central law, but a few state Acts leaked in and its `jurisdiction` column is
# unreliable, so an Act's title is the trustworthy signal (DB README §9).
STATE_PREFIXES = (
    "Andhra Pradesh", "Arunachal", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Orissa",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Jammu", "Puducherry", "Pondicherry",
)

# Acts Parliament passed for a Union Territory: not state law, but not all-India law either. A
# Delhi Act applies only in Delhi whoever enacted it. (prefix, place), longest prefix first; the
# places must match INDIAN_STATES, which is what the user picks in the UI.
TERRITORY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("National Capital Territory of Delhi", "Delhi"),
    ("New Delhi", "Delhi"),
    ("Delhi", "Delhi"),
    ("Chandigarh", "Chandigarh"),
    ("Dadra and Nagar Haveli", "Dadra and Nagar Haveli and Daman and Diu"),
    ("Daman and Diu", "Dadra and Nagar Haveli and Daman and Diu"),
    ("Goa, Daman and Diu", "Goa"),
    ("Andaman and Nicobar", "Andaman and Nicobar Islands"),
    ("Lakshadweep", "Lakshadweep"),
    ("Ladakh", "Ladakh"),
)

# What the user can pick as "where I am": states and Union Territories alike.
INDIAN_STATES = (
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
)


# ── server ────────────────────────────────────────────────────────────────────────────
HOST = env_str("KYR_HOST", "127.0.0.1")
PORT = env_int("KYR_PORT", 8000)
LOG_LEVEL = env_str("KYR_LOG_LEVEL", "INFO")

SESSION_MAX = env_int("KYR_SESSION_MAX", 200)            # conversations held in memory
SESSION_TTL_S = env_int("KYR_SESSION_TTL_S", 6 * 3600)   # idle time before one is dropped

# ── public deployment guards ──────────────────────────────────────────────────────────
# At most this many answers are researched at once; later questions wait in a first-come queue
# and are shown their place in line. Past the queue's length, or its wait, they are turned away.
MAX_ACTIVE_TURNS = env_int("KYR_MAX_ACTIVE_TURNS", 5)
MAX_QUEUED_TURNS = env_int("KYR_MAX_QUEUED_TURNS", 20)
QUEUE_TIMEOUT_S = env_float("KYR_QUEUE_TIMEOUT_S", 180.0)
# Per client (IP address): questions in progress or queued, and questions per minute.
CLIENT_MAX_PENDING = env_int("KYR_CLIENT_MAX_PENDING", 2)
CLIENT_RPM = env_int("KYR_CLIENT_RPM", 10)
# Dollars each client may spend before being told the free allowance is used up. The window is
# how long before that allowance resets; 0 means it never does. 0 dollars disables the limit.
CLIENT_BUDGET_USD = env_float("KYR_CLIENT_BUDGET_USD", 1.0)
CLIENT_BUDGET_WINDOW_H = env_float("KYR_CLIENT_BUDGET_WINDOW_H", 0.0)
# A ceiling on the whole service's spend per UTC day, whoever spends it. 0 disables it.
DAILY_BUDGET_USD = env_float("KYR_DAILY_BUDGET_USD", 5.0)
# Behind a reverse proxy or tunnel every request arrives from the proxy's address. Set this only
# when one is in front of the app, or clients could claim any address they like.
TRUST_PROXY_HEADERS = env_bool("KYR_TRUST_PROXY_HEADERS", False)
# Bearer token for /api/status. Without one, that endpoint answers only from this machine.
ADMIN_TOKEN = env_key("KYR_ADMIN_TOKEN")
