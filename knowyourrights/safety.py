"""Deciding, before anything else happens, whether someone is describing an emergency.

Two tiers, because the two failure directions have very different costs.

**Tier 1 — patterns.** Instant, no model, no network. A person typing "he is hitting me right
now" gets 112 before the planner has been asked anything, and that holds even if every provider
is rate-limited. High precision by construction: the phrasings are literal.

**Tier 2 — meaning.** Patterns cannot generalise, and paraphrase is exactly what people produce
under stress: *"my partner keeps hurting me and I'm scared to go home"* matches nothing in tier
one. So the message is compared against curated exemplars of each kind of crisis using the
embedder that is already loaded — the corpus is embedded with bge-m3, it is multilingual, and
the query is embedded again moments later for retrieval where the cache serves it for free. The
practical added cost of this tier is one cache miss, once per turn.

**The guard that makes tier 2 usable.** *"what is the punishment for rape"* is semantically very
close to a report of rape, and firing a helpline card at someone reading about the law is not a
harmless error — it is how people learn to ignore the card that matters. So tier 2 is suppressed
for messages that read as questions *about* the law rather than reports of something happening.
Tier 1 is never suppressed: if the literal phrasing is a disclosure, it is a disclosure.

Both tiers are tuned against a labelled set in ``tests/test_safety.py``. Run
``python scripts/calibrate_safety.py`` to re-measure after changing exemplars or thresholds.
"""

from __future__ import annotations

import logging
import re

from .agents.schemas import SafetyCheck

log = logging.getLogger(__name__)

# ── tier 1: literal phrasings ─────────────────────────────────────────────────────────
# Match the act and its object rather than the actor. Requiring a specific subject ("he is
# hitting") missed "my husband is hitting me", which is precisely the disclosure this exists
# for. Past tense matters as much as present: "I was beaten" is a report, not history.
# Each pattern is marked STRONG or WEAK, and the distinction is the whole reason tier 1 is
# usable at all.
#
# STRONG phrasings carry a subject or object — "hitting me", "I was raped", "kill myself". They
# cannot plausibly be a question about the law, so nothing suppresses them.
#
# WEAK phrasings are bare topic nouns: "suicide", "child labour", "forced labour". These are the
# vocabulary of the crime *and* of every legal question about it, and matching them naively is
# exactly how "is suicide a crime in India" and "child labour laws in India" ended up producing
# helpline cards. They only fire when the message does not read as a question.
STRONG, WEAK = True, False

_URGENT_PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"(?i)\b((hit|hitt|beat|beating|attack|assault|abus|threaten)\w*\s+"
                r"(me|us|her|him|my \w+)|(being|was|were|been|got)\s+"
                r"(beaten|hit|attacked|abused|thrashed)|"
                r"maar\s+(rahe|raha|rahi)|peet\s+(rahe|raha|rahi)|ghar\s+mein\s+maar)\b"),
     "violence", STRONG),

    (re.compile(r"(?i)\b(kill myself|end my life|want to die|khudkushi|atmahatya|self harm|"
                r"(thinking|think|planning|plan|considering|attempted|attempting|commit|"
                r"committing)\s+(about\s+|of\s+)?suicide)\b"), "self_harm", STRONG),
    (re.compile(r"(?i)\bsuicide\b"), "self_harm", WEAK),

    # The bare noun is never enough — "punishment for rape" is a legal question. The verb has to
    # carry a subject marker for this to read as a report.
    (re.compile(r"(?i)\b((being|was|were|been|got)\s+(raped|molested|violated)|"
                r"sexual(ly)? assault(ed|ing)?|raped me|molested me|"
                r"rape ho|chhed(chhad|khani))\b"), "sexual_violence", STRONG),

    (re.compile(r"(?i)\b(kidnapp?ed|abducted|trafficked|bandhak|"
                r"(held|holding|keep|keeping|hold)\s+(me|him|her|us|\w+)\s+"
                r"(captive|hostage|locked))\b"), "trafficking", STRONG),
    (re.compile(r"(?i)\bforced (labour|labor|prostitution)\b"), "trafficking", WEAK),

    (re.compile(r"(?i)\b(missing child|child is missing|bacha gum)\b"), "child", STRONG),
    (re.compile(r"(?i)\b(child labour|child marriage|bal vivah)\b"), "child", WEAK),

    (re.compile(r"(?i)\b(arrest(ing|ed)? (me|him|her) (right )?now|police (are|is) here|"
                r"being detained|abhi arrest)\b"), "arrest_in_progress", STRONG),
)

URGENT_ADVICE: dict[str, str] = {
    "violence": "If you are in immediate danger, call 112 now, or 181 for the women's helpline.",
    "self_harm": "If you are thinking about harming yourself, please call Tele-MANAS on 14416 — "
                 "someone is there right now.",
    "sexual_violence": "If this is happening now or just happened, call 112, or 1091 for the "
                       "women's helpline. You can report at any police station regardless of "
                       "where it happened.",
    "trafficking": "Call 112 immediately. Childline is 1098 if a child is involved.",
    "child": "Call Childline on 1098 — it operates 24 hours.",
    "arrest_in_progress": "Call 112 if you need help now. You have the right to know the "
                          "grounds of arrest, to inform someone, and to a lawyer — free legal "
                          "aid is on 15100.",
}

# ── tier 2: exemplars of each crisis, for meaning-based matching ──────────────────────
# Written the way a frightened person types, not the way a statute reads. Hindi and Hinglish are
# included deliberately: bge-m3 is multilingual, and this is the population that most needs the
# gate to work in the language they panic in.
CRISIS_EXEMPLARS: dict[str, tuple[str, ...]] = {
    "violence": (
        "my partner keeps hurting me and I am scared to go home",
        "my father gets drunk and hurts my mother every night",
        "I am not safe in my own house, he gets violent",
        "in-laws torture me for dowry and it is getting worse",
        "mujhe ghar par maarte hain aur main darr gayi hoon",
        "pati roz maarta hai",
        "he threw things at me and I have bruises",
    ),
    "self_harm": (
        "I do not want to live anymore",
        "everything feels pointless and I am thinking of ending it",
        "I have been thinking about hurting myself",
        "main jeena nahi chahta",
        "mujhe marne ka mann karta hai",
    ),
    "sexual_violence": (
        "someone forced himself on me",
        "my boss touched me inappropriately and I froze",
        "I was attacked sexually last night",
        "he did something to me without my consent",
        "mere saath galat kaam hua",
        "kisi ne meri marzi ke bina chhua",
    ),
    "trafficking": (
        "they took my passport and will not let me leave the building",
        "I was promised a job but now they will not let me go",
        "someone is forcing me to work without pay and I cannot escape",
        "mujhe bandhak bana ke rakha hai",
    ),
    "child": (
        "my daughter has not come home and I cannot find her",
        "a child in my neighbourhood is being made to work",
        "they are marrying off a girl who is only fourteen",
        "mera beta gum ho gaya hai",
    ),
    "arrest_in_progress": (
        "officers are at my door right now and want to take me",
        "they have picked up my brother and will not say why",
        "police station mein baithaya hua hai abhi",
        "they are taking me away and I do not know my rights",
    ),
}

# Messages that read as questions *about* the law rather than reports of something happening.
# Only tier 2 is suppressed by this — a literal disclosure still fires however it is framed.
_INFORMATIONAL = re.compile(
    r"(?i)("
    r"^\s*(what|which|when|where|how|who|is|are|can|does|do|should|why)\b"
    r"|\b(punishment|penalty|sentence|definition|meaning|difference)\s+(for|of|between)\b"
    r"|\bunder (which|what) (act|section|law)\b"
    r"|\b(section|article|chapter)\s+\d"
    r"|\b(bns|bnss|bsa|ipc|crpc|rti)\b"
    r"|\bhow (do|can|would) i\b"
    r"|\b(procedure|process|steps|apply|file|register|draft)\b"
    # Topic queries carry no question word at all — "child labour laws in India" is a request
    # for the law, not a report of it. Nobody disclosing an emergency reaches for this register.
    r"|\b(laws|legislation|regulations|statutes)\b"
    r"|\b(act|rules|provisions?|rights)\s+(of|for|about|regarding|under|in India)\b"
    r")")

# Cosine similarity above which a message is treated as the same kind of thing as an exemplar.
# Measured on the labelled set in tests/test_safety.py; see scripts/calibrate_safety.py.
SEMANTIC_THRESHOLD = 0.64

_exemplar_cache: tuple[list[str], list[str], object] | None = None


def looks_informational(message: str) -> bool:
    """Is this a question about the law rather than a report of something happening?"""
    return bool(_INFORMATIONAL.search(message or ""))


def check_patterns(message: str) -> SafetyCheck:
    """Tier 1 alone. Synchronous, no model, no network — always safe to call.

    Strong patterns fire unconditionally. Weak ones are bare topic nouns and fire only when the
    message is not phrased as a question about the law.
    """
    text = message or ""
    informational = looks_informational(text)
    for pattern, kind, strong in _URGENT_PATTERNS:
        if not strong and informational:
            continue
        if pattern.search(text):
            return SafetyCheck(urgent=True, kind=kind, reason=URGENT_ADVICE.get(kind, ""))
    return SafetyCheck()


async def _exemplar_matrix(embedder):
    """Embed the exemplars once. ~30 short strings, so this is a one-off of well under a second."""
    global _exemplar_cache
    if _exemplar_cache is not None:
        return _exemplar_cache
    import numpy as np

    kinds: list[str] = []
    texts: list[str] = []
    for kind, examples in CRISIS_EXEMPLARS.items():
        for text in examples:
            kinds.append(kind)
            texts.append(text)
    vectors = await embedder.encode(texts)
    matrix = np.asarray(vectors, dtype="float32")
    # bge-m3 output is L2-normalised, so a dot product is already the cosine. Normalise anyway:
    # it costs nothing here and makes the function correct for any embedder.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-9, None)
    _exemplar_cache = (kinds, texts, matrix)
    return _exemplar_cache


async def check(message: str, embedder=None, *, allow_semantic: bool = True) -> SafetyCheck:
    """Both tiers. Patterns first, then meaning, with the informational guard on tier 2 only.

    Never raises: a safety gate that can fail closed on an embedder problem would be worse than
    one that occasionally misses a paraphrase, so any failure degrades to tier 1's verdict.
    """
    literal = check_patterns(message)
    if literal.urgent or not allow_semantic:
        return literal
    if not (message or "").strip() or looks_informational(message):
        return literal

    try:
        import numpy as np

        if embedder is None:
            from .retrieval.embedder import get_embedder

            embedder = get_embedder()
        if not getattr(embedder.plan, "use_embedder", True):
            return literal                      # `lite` has no embedder; tier 1 still applies

        kinds, texts, matrix = await _exemplar_matrix(embedder)
        # encode_one is cached, and retrieval embeds this same string moments later, so in the
        # normal path this costs one embedding for the whole turn rather than one extra.
        vector = np.asarray(await embedder.encode_one(message), dtype="float32")
        vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
        scores = matrix @ vector
        best = int(scores.argmax())
        if float(scores[best]) >= SEMANTIC_THRESHOLD:
            kind = kinds[best]
            log.info("safety gate fired semantically (%s, %.3f) on %r",
                     kind, float(scores[best]), message[:80])
            return SafetyCheck(urgent=True, kind=kind, reason=URGENT_ADVICE.get(kind, ""))
    except Exception as exc:                    # noqa: BLE001 — never let this path fail a turn
        log.warning("semantic safety tier unavailable (%s) — patterns only", exc)

    return literal


def reset_cache() -> None:
    """Drop the exemplar embeddings. For tests that swap the embedder."""
    global _exemplar_cache
    _exemplar_cache = None
