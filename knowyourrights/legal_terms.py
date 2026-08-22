"""Indian legal vocabulary: acronyms, repeals, and what this corpus does *not* contain.

Three jobs, all of which a deterministic table does better than an LLM:

1. **Expand acronyms** before retrieval. "RTI" shares no tokens with "Right to Information
   Act, 2005", so without expansion the search simply misses.

2. **Redirect repealed codes.** The IPC, CrPC and Indian Evidence Act were replaced on
   2024-07-01 and are deliberately absent from the corpus (DB README §3). A question about
   "IPC 302" must be answered from the BNS, with the substitution made explicit.

3. **Know the gaps.** Several statutes people ask about are simply not here. That matters
   because fuzzy matching fails *loudly wrong* rather than quietly: "Information Technology
   Act" matches the *Indian Institutes of Information Technology* Act — a real, unrelated
   statute. Naming the gaps lets the agent go to the web instead of citing nonsense.

Every canonical title below was verified against the live corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── acronym / alias -> exact act title present in the corpus ──────────────────────────
ACRONYMS: dict[str, str] = {
    # information & governance
    "rti": "Right to Information Act, 2005",
    "rti act": "Right to Information Act, 2005",
    # criminal law (post-2024 codes)
    "bns": "Bharatiya Nyaya Sanhita, 2023",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "bsa": "Bharatiya Sakshya Adhiniyam, 2023",
    "nyaya sanhita": "Bharatiya Nyaya Sanhita, 2023",
    "nagarik suraksha": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "sakshya adhiniyam": "Bharatiya Sakshya Adhiniyam, 2023",
    # other criminal statutes
    "ndps": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "ndps act": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "pocso": "Protection of Children from Sexual Offences Act, 2012",
    "uapa": "Unlawful Activities (Prevention) Act, 1967",
    "sc st act": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
    "sc/st act": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
    "atrocities act": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
    "jj act": "Juvenile Justice (Care and Protection of Children) Act, 2015",
    "juvenile justice act": "Juvenile Justice (Care and Protection of Children) Act, 2015",
    # women & family
    "dv act": "Protection of Women from Domestic Violence Act, 2005",
    "pwdva": "Protection of Women from Domestic Violence Act, 2005",
    "domestic violence act": "Protection of Women from Domestic Violence Act, 2005",
    "posh": "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013",
    "posh act": "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013",
    "hma": "Hindu Marriage Act, 1955",
    "hindu marriage act": "Hindu Marriage Act, 1955",
    "special marriage act": "Special Marriage Act, 1954",
    "mtp act": "Medical Termination of Pregnancy Act, 1971",
    "dowry act": "Dowry Prohibition Act, 1961",
    # consumer, transport, property
    "cpa": "Consumer Protection Act, 2019",
    "consumer act": "Consumer Protection Act, 2019",
    "mv act": "Motor Vehicles Act, 1988",
    "motor vehicle act": "Motor Vehicles Act, 1988",
    "rera": "Real Estate (Regulation and Development) Act, 2016",
    "tp act": "Transfer of Property Act, 1882",
    "transfer of property act": "Transfer of Property Act, 1882",
    # labour
    "epf": "Employees Provident Funds and Miscellaneous Provisions Act, 1952",
    "epfo": "Employees Provident Funds and Miscellaneous Provisions Act, 1952",
    "esi": "Employees State Insurance Act, 1948",
    "esic": "Employees State Insurance Act, 1948",
    "mgnrega": "Mahatma Gandhi National Rural Employment Guarantee Act, 2005",
    "nrega": "Mahatma Gandhi National Rural Employment Guarantee Act, 2005",
    "maternity benefit act": "Maternity Benefit Act, 1961",
    "minimum wages act": "Minimum Wages Act, 1948",
    "factories act": "Factories Act, 1948",
    # civil, commercial, tax
    "cpc": "Code of Civil Procedure, 1908",
    "civil procedure code": "Code of Civil Procedure, 1908",
    "ni act": "Negotiable Instruments Act, 1881",
    "cheque bounce": "Negotiable Instruments Act, 1881 dishonour of cheque",
    "ibc": "Insolvency and Bankruptcy Code Act, 2016",
    "contract act": "Indian Contract Act, 1872",
    "specific relief act": "Specific Relief Act, 1963",
    "limitation act": "Limitation Act, 1963",
    "companies act": "Companies Act, 2013",
    # rights & welfare
    "rte": "Right of Children to Free and Compulsory Education Act, 2009",
    "rte act": "Right of Children to Free and Compulsory Education Act, 2009",
    "aadhaar act": "Aadhaar (Targeted Delivery of Financial and other Subsidies, Benefits and Services) Act, 2016",
    "transgender act": "Transgender Persons (Protection of Rights) Act, 2019",
    "mental healthcare act": "Mental Healthcare Act, 2017",
    "it act": "Information Technology Act, 2000",
    "ita": "Information Technology Act, 2000",
    "information technology act": "Information Technology Act, 2000",
    "cyber law": "Information Technology Act, 2000",
    "epa": "Environment (Protection) Act, 1986",
    "wildlife act": "Wild Life (Protection) Act, 1972",
    "citizenship act": "Citizenship Act, 1955",
    "passport act": "Passports Act, 1967",
}

# Concepts that are not act titles but still need spelling out for retrieval.
CONCEPTS: dict[str, str] = {
    "fir": "First Information Report (FIR) registration of a cognizable offence by police",
    "pil": "Public Interest Litigation writ petition",
    "nbw": "non-bailable warrant",
    "cji": "Chief Justice of India",
    # Bare "sc"/"hc" are deliberately absent: expanding them would mangle "SC/ST" and they
    # carry little retrieval value anyway.
    "dlsa": "District Legal Services Authority free legal aid",
    "nalsa": "National Legal Services Authority free legal aid",
    "pio": "Public Information Officer under the Right to Information Act, 2005",
    "cic": "Central Information Commission under the Right to Information Act, 2005",
    "faa": "First Appellate Authority under the Right to Information Act, 2005",
    "ncdrc": "National Consumer Disputes Redressal Commission",
    "anticipatory bail": "anticipatory bail before arrest",
}


@dataclass(frozen=True)
class Repeal:
    old: str
    new: str
    effective: str
    note: str


# The colonial criminal codes are absent from the corpus by design. Anyone asking about them
# needs the replacement provision plus an explicit statement that the substitution happened.
REPEALED: dict[str, Repeal] = {
    "ipc": Repeal(
        "Indian Penal Code, 1860", "Bharatiya Nyaya Sanhita, 2023", "2024-07-01",
        "The IPC was replaced by the Bharatiya Nyaya Sanhita (BNS) on 1 July 2024. Offences "
        "committed before that date are still tried under the IPC, but section numbers differ.",
    ),
    "indian penal code": Repeal(
        "Indian Penal Code, 1860", "Bharatiya Nyaya Sanhita, 2023", "2024-07-01",
        "The IPC was replaced by the Bharatiya Nyaya Sanhita (BNS) on 1 July 2024.",
    ),
    "crpc": Repeal(
        "Code of Criminal Procedure, 1973", "Bharatiya Nagarik Suraksha Sanhita, 2023", "2024-07-01",
        "The CrPC was replaced by the Bharatiya Nagarik Suraksha Sanhita (BNSS) on 1 July 2024.",
    ),
    "cr.p.c": Repeal(
        "Code of Criminal Procedure, 1973", "Bharatiya Nagarik Suraksha Sanhita, 2023", "2024-07-01",
        "The CrPC was replaced by the Bharatiya Nagarik Suraksha Sanhita (BNSS) on 1 July 2024.",
    ),
    "criminal procedure code": Repeal(
        "Code of Criminal Procedure, 1973", "Bharatiya Nagarik Suraksha Sanhita, 2023", "2024-07-01",
        "The CrPC was replaced by the Bharatiya Nagarik Suraksha Sanhita (BNSS) on 1 July 2024.",
    ),
    "indian evidence act": Repeal(
        "Indian Evidence Act, 1872", "Bharatiya Sakshya Adhiniyam, 2023", "2024-07-01",
        "The Indian Evidence Act was replaced by the Bharatiya Sakshya Adhiniyam (BSA) on 1 July 2024.",
    ),
    "evidence act": Repeal(
        "Indian Evidence Act, 1872", "Bharatiya Sakshya Adhiniyam, 2023", "2024-07-01",
        "The Indian Evidence Act was replaced by the Bharatiya Sakshya Adhiniyam (BSA) on 1 July 2024.",
    ),
}

# Statutes people ask about that this corpus does not contain — each verified absent against
# the live index. Naming them lets the agent go to the web instead of citing a same-sounding
# but unrelated act ("Information Technology Act" competes with the Indian Institutes of
# Information Technology Acts, which are a real and completely different statute).
NOT_IN_CORPUS: dict[str, str] = {
    "pmla": "The Prevention of Money Laundering Act, 2002 is not in this database.",
    "prevention of money laundering": "The Prevention of Money Laundering Act, 2002 is not in this database.",
    "dpdp": "The Digital Personal Data Protection Act, 2023 is not in this database.",
    "dpdp act": "The Digital Personal Data Protection Act, 2023 is not in this database.",
    "digital personal data protection": "The Digital Personal Data Protection Act, 2023 is not in this database.",
}


def _pattern(keys) -> re.Pattern:
    """Longest-first alternation so 'rti act' wins over 'rti'."""
    ordered = sorted(keys, key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in ordered) + r")(?!\w)",
                      re.IGNORECASE)


_ACRONYM_RE = _pattern(ACRONYMS)
_CONCEPT_RE = _pattern(CONCEPTS)
_REPEALED_RE = _pattern(REPEALED)
_NOT_IN_CORPUS_RE = _pattern(NOT_IN_CORPUS)

# One combined table so expansion is a *single* pass. Running the three tables in sequence
# re-expands its own output: "IPC" becomes "Bharatiya Nyaya Sanhita, 2023", whose "nyaya
# sanhita" is itself an alias, giving "Bharatiya Bharatiya Nyaya Sanhita, 2023, 2023".
# Priority on key collision: repeal > acronym > concept.
_EXPANSIONS: dict[str, str] = {
    **{k.lower(): v for k, v in CONCEPTS.items()},
    **{k.lower(): v for k, v in ACRONYMS.items()},
    **{k.lower(): v.new for k, v in REPEALED.items()},
}
_EXPAND_RE = _pattern(_EXPANSIONS)


def expand(text: str) -> str:
    """Rewrite a citizen's phrasing into terms the statute text actually uses.

    Single pass by construction, so a replacement can never be rewritten again.
    """
    if not text:
        return ""
    out = _EXPAND_RE.sub(lambda m: _EXPANSIONS[m.group(1).lower()], text)
    return re.sub(r"\s+", " ", out).strip()


def detect_acts(text: str) -> list[str]:
    """Canonical act titles explicitly named or implied by the text."""
    found: list[str] = []
    for match in _ACRONYM_RE.finditer(text or ""):
        title = ACRONYMS[match.group(1).lower()]
        if title not in found:
            found.append(title)
    for match in _REPEALED_RE.finditer(text or ""):
        title = REPEALED[match.group(1).lower()].new
        if title not in found:
            found.append(title)
    return found


def detect_repeals(text: str) -> list[Repeal]:
    """Repealed codes mentioned, so the answer can say what replaced them."""
    seen: dict[str, Repeal] = {}
    for match in _REPEALED_RE.finditer(text or ""):
        repeal = REPEALED[match.group(1).lower()]
        seen.setdefault(repeal.old, repeal)
    return list(seen.values())


def detect_gaps(text: str) -> list[str]:
    """Warnings about statutes this corpus is known not to hold."""
    seen: list[str] = []
    for match in _NOT_IN_CORPUS_RE.finditer(text or ""):
        note = NOT_IN_CORPUS[match.group(1).lower()]
        if note not in seen:
            seen.append(note)
    return seen


# ── section references ────────────────────────────────────────────────────────────────
# The label suffix must be attached to the digits ("6A", "6-A") — allowing a space would
# swallow the next word, turning "Section 6 of the RTI Act" into section "6OF".
_LABEL = r"([0-9]{1,4}(?:-?[A-Za-z]{1,2})?)"
_SECTION_RE = re.compile(rf"(?<!\w)(?:sections?|secs?\.?|s\.|§)\s*{_LABEL}(?!\w)", re.IGNORECASE)
_ARTICLE_RE = re.compile(rf"(?<!\w)(?:articles?|arts?\.?)\s*{_LABEL}(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class SectionRef:
    kind: str          # "section" | "article"
    label: str
    act: str | None


def _clean_label(raw: str) -> str:
    """Normalise "6-a" / "6 A" / "006A" to the corpus's own `section_label` spelling."""
    label = re.sub(r"[\s-]+", "", str(raw)).upper()
    digits = re.match(r"^0*(\d+)([A-Z]*)$", label)
    return f"{digits.group(1)}{digits.group(2)}" if digits else label


def detect_section_refs(text: str) -> list[SectionRef]:
    """Find "Section 6 of the RTI Act" / "Article 21" style references.

    These get answered by exact lookup rather than similarity search — asking what a named
    provision says deserves the provision, not its nearest neighbour.
    """
    if not text:
        return []
    acts = detect_acts(text)
    act = acts[0] if acts else None
    refs: list[SectionRef] = []

    for match in _ARTICLE_RE.finditer(text):
        refs.append(SectionRef("article", _clean_label(match.group(1)), "Constitution of India"))
    for match in _SECTION_RE.finditer(text):
        refs.append(SectionRef("section", _clean_label(match.group(1)), act))

    deduped: list[SectionRef] = []
    for ref in refs:
        if ref not in deduped:
            deduped.append(ref)
    return deduped


_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Function words that mark romanised Hindi. Chosen to be unambiguous in an English sentence:
# "hai", "nahi", "kya", "mujhe" do not occur in ordinary English legal questions.
# Deliberately excludes anything that is also an English word. Hindi "the" (were), "main"
# (I), "par" (on) and "to" all collide, and including "the" alone was enough to classify
# "what is the punishment for cheating" as Hinglish.
_HINGLISH_MARKERS = frozenset("""
kya kyu kyun kaise kahan kaun kitna kitni nahi nahin mat hai hain tha thi hoga hogi
karo kare karna kiya raha rahi rahe mujhe mera meri mere tum tumhara aap aapka aapko
uska uski unka humara hamara kuch bhi toh phir agar lekin magar sakta sakti
mein ka ki ke ko se wala wali bina saath liye gaya gayi diya bola bole chahiye
""".split())

# Unambiguous enough that one is signal even in a very short message.
_HINGLISH_STRONG = frozenset("""
kya kyun kaise kahan kaun nahi nahin hai hain mujhe mera meri chahiye karo raha rahi rahe
""".split())


def detect_language(text: str) -> str:
    """Which language to answer in: ``hi``, ``hinglish`` or ``en``.

    Done in code rather than asked of the planner, which proved unreliable — it labelled a
    plainly English question "hi" and the answer came back in Hindi. Script is decisive, and
    romanised Hindi is recognised by function words that simply do not appear in an English
    sentence.
    """
    sample = (text or "").strip()
    if not sample:
        return "en"
    if _DEVANAGARI_RE.search(sample):
        return "hi"
    words = re.findall(r"[a-zA-Z]+", sample.lower())
    if not words:
        return "en"
    markers = sum(1 for w in words if w in _HINGLISH_MARKERS)
    strong = sum(1 for w in words if w in _HINGLISH_STRONG)
    if markers >= 2 or (strong >= 1 and len(words) <= 5):
        return "hinglish"
    return "en"


LANGUAGE_INSTRUCTION = {
    "hi": "The user wrote in Hindi (Devanagari). Reply in Hindi, in Devanagari script.",
    "hinglish": ("The user wrote in Hinglish — Hindi in Latin script, mixed with English. "
                 "Reply the same way: Latin script, natural Hindi-English mix, never Devanagari."),
    "en": "The user wrote in English. Reply in English.",
}


def is_state_law(act_title: str, state_prefixes) -> str | None:
    """Return the state a title belongs to, if it looks like state law.

    'Delhi ...' acts are genuinely central — Parliament legislates for the Delhi UT — so the
    caller's prefix list deliberately excludes it (DB README §9).
    """
    title = (act_title or "").strip()
    for state in state_prefixes:
        if title.startswith(state):
            return state
    return None
