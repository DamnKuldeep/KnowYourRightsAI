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
    # RTI roles. Unexpanded, "PIO" shares no token with the statute, which says "Central Public
    # Information Officer" throughout.
    "pio": "Public Information Officer under the Right to Information Act",
    "cpio": "Central Public Information Officer under the Right to Information Act",
    "spio": "State Public Information Officer under the Right to Information Act",
    "faa": "First Appellate Authority under the Right to Information Act",
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
_ANNOTATE_RE = _pattern({**CONCEPTS, **ACRONYMS})


def expand(text: str) -> str:
    """Rewrite a citizen's phrasing into terms the statute text actually uses.

    Single pass by construction, so a replacement can never be rewritten again.
    """
    if not text:
        return ""
    out = _EXPAND_RE.sub(lambda m: _EXPANSIONS[m.group(1).lower()], text)
    return re.sub(r"\s+", " ", out).strip()


# Roles and bodies that exist under exactly one Act. Naming one names the Act, even though the
# word itself is not an acronym for it: "penalty on the PIO" is an RTI question, and without this
# it was treated as a general criminal one and the BNS was boosted over RTI Section 20 — which
# the reranker had scored highest.
IMPLIES_ACT: dict[str, str] = {
    "pio": "Right to Information Act, 2005",
    "cpio": "Right to Information Act, 2005",
    "spio": "Right to Information Act, 2005",
    "public information officer": "Right to Information Act, 2005",
    "first appellate authority": "Right to Information Act, 2005",
    "information commission": "Right to Information Act, 2005",
    "information commissioner": "Right to Information Act, 2005",
}
_IMPLIES_RE = _pattern(IMPLIES_ACT)


def annotate(text: str) -> str:
    """Name what an acronym stands for *beside* it, for the cross-encoder.

    The reranker used to see the raw query, because replacing "RTI" with "Right to Information
    Act, 2005" reads as broken English ("my Right to Information Act, 2005 was rejected"). But a
    bare "RTI" meant it could not tell the RTI Act from the Credit Information Companies Act:
    "my RTI was rejected, how do I appeal" ranked the latter first. A parenthetical keeps the
    sentence natural and gives the model the name: "my RTI (Right to Information Act, 2005)".
    """
    if not text:
        return ""
    def add(m: re.Match) -> str:
        key = m.group(1).lower()
        full = ACRONYMS.get(key) or CONCEPTS.get(key)
        if not full or full.lower() == m.group(0).lower():
            return m.group(0)
        # "FIR (First Information Report (FIR) …)" — drop the acronym's own echo inside.
        full = re.sub(rf"\s*\({re.escape(m.group(0))}\)", "", full, flags=re.I)
        return f"{m.group(0)} ({full})"
    return _ANNOTATE_RE.sub(add, text)


def detect_acts(text: str) -> list[str]:
    """Canonical act titles explicitly named or implied by the text."""
    found: list[str] = []
    for match in _ACRONYM_RE.finditer(text or ""):
        title = ACRONYMS[match.group(1).lower()]
        if title not in found:
            found.append(title)
    for match in _IMPLIES_RE.finditer(text or ""):
        title = IMPLIES_ACT[match.group(1).lower()]
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


@dataclass(frozen=True)
class SectionMapping:
    """One section of a repealed code and where its content now lives."""

    code: str            # "IPC" | "CrPC" | "IEA"
    old: str             # "420"
    new_act: str         # "Bharatiya Nyaya Sanhita, 2023"
    new: str             # "318"   — the section, for lookup
    new_ref: str         # "318(4)" — what to cite, where the content is one sub-section
    subject: str

    @property
    def note(self) -> str:
        return (f"Section {self.old} of the {_OLD_CODE_NAMES[self.code]} is now Section "
                f"{self.new_ref} of the {self.new_act} ({self.subject}).")


_OLD_CODE_NAMES = {"IPC": "Indian Penal Code", "CrPC": "Code of Criminal Procedure",
                   "IEA": "Indian Evidence Act"}
_NEW_CODE_SHORT = {"Bharatiya Nyaya Sanhita, 2023": "BNS",
                   "Bharatiya Nagarik Suraksha Sanhita, 2023": "BNSS",
                   "Bharatiya Sakshya Adhiniyam, 2023": "BSA"}
_BNS, _BNSS, _BSA = ("Bharatiya Nyaya Sanhita, 2023", "Bharatiya Nagarik Suraksha Sanhita, 2023",
                     "Bharatiya Sakshya Adhiniyam, 2023")

# The sections people actually ask about by their old number. Every entry was checked against
# the corpus's own heading for the new section — BNS 318 is "Cheating", BNSS 482 is "Direction
# for grant of bail to person apprehending arrest" — and tests/test_jurisdiction.py re-checks
# them against the database, so a wrong row fails the build rather than reaching a user.
#
# Without this, "Section 420 IPC" was looked up as BNS Section 420 — carrying the old number
# into the new code. That section does not exist, so the writer improvised "Section 420 of the
# BNS", which is false. Where an old number *does* exist in the new code, the same bug would
# have silently returned a different offence.
_MAP_ROWS = [
    ("IPC", "34", _BNS, "3", "3(5)", "acts done by several persons in furtherance of common intention"),
    ("IPC", "120B", _BNS, "61", "61(2)", "criminal conspiracy"),
    ("IPC", "153A", _BNS, "196", "196", "promoting enmity between groups"),
    ("IPC", "279", _BNS, "281", "281", "rash driving on a public way"),
    ("IPC", "295A", _BNS, "299", "299", "outraging religious feelings"),
    ("IPC", "302", _BNS, "103", "103", "punishment for murder"),
    ("IPC", "304A", _BNS, "106", "106(1)", "causing death by negligence"),
    ("IPC", "304B", _BNS, "80", "80", "dowry death"),
    ("IPC", "307", _BNS, "109", "109", "attempt to murder"),
    ("IPC", "323", _BNS, "115", "115(2)", "voluntarily causing hurt"),
    ("IPC", "354", _BNS, "74", "74", "assault to outrage a woman's modesty"),
    ("IPC", "363", _BNS, "137", "137(2)", "kidnapping"),
    ("IPC", "376", _BNS, "64", "64", "punishment for rape"),
    ("IPC", "379", _BNS, "303", "303(2)", "theft"),
    ("IPC", "406", _BNS, "316", "316(2)", "criminal breach of trust"),
    ("IPC", "420", _BNS, "318", "318(4)", "cheating and dishonestly inducing delivery of property"),
    ("IPC", "498A", _BNS, "85", "85", "cruelty by husband or his relatives"),
    ("IPC", "499", _BNS, "356", "356", "defamation"),
    ("IPC", "500", _BNS, "356", "356(2)", "punishment for defamation"),
    ("IPC", "506", _BNS, "351", "351(2)", "criminal intimidation"),
    ("IPC", "509", _BNS, "79", "79", "word, gesture or act to insult a woman's modesty"),
    ("CrPC", "41", _BNSS, "35", "35", "when police may arrest without warrant"),
    ("CrPC", "50", _BNSS, "47", "47", "grounds of arrest and right to bail"),
    ("CrPC", "57", _BNSS, "58", "58", "not to be detained more than twenty-four hours"),
    ("CrPC", "125", _BNSS, "144", "144", "maintenance of wives, children and parents"),
    ("CrPC", "144", _BNSS, "163", "163", "orders in urgent cases of nuisance or apprehended danger"),
    ("CrPC", "154", _BNSS, "173", "173", "information in cognizable cases (FIR)"),
    ("CrPC", "156", _BNSS, "175", "175", "police power to investigate cognizable cases"),
    ("CrPC", "167", _BNSS, "187", "187", "procedure when investigation exceeds twenty-four hours"),
    ("CrPC", "200", _BNSS, "223", "223", "examination of complainant"),
    ("CrPC", "436", _BNSS, "478", "478", "bail in bailable offences"),
    ("CrPC", "437", _BNSS, "480", "480", "bail in non-bailable offences"),
    ("CrPC", "438", _BNSS, "482", "482", "anticipatory bail"),
    ("CrPC", "439", _BNSS, "483", "483", "special powers of High Court or Sessions Court on bail"),
    ("CrPC", "482", _BNSS, "528", "528", "inherent powers of the High Court"),
    ("IEA", "24", _BSA, "22", "22", "confession caused by inducement, threat or promise"),
    ("IEA", "25", _BSA, "23", "23", "confession to a police officer"),
    ("IEA", "32", _BSA, "26", "26", "statements of persons who are dead (dying declarations)"),
    ("IEA", "65B", _BSA, "63", "63", "admissibility of electronic records"),
]
SECTION_MAP: dict[tuple[str, str], SectionMapping] = {
    (code, old.upper()): SectionMapping(code, old, act, new, ref, subject)
    for code, old, act, new, ref, subject in _MAP_ROWS
}

_CODE_ALIAS = (r"(?P<code>i\.?\s?p\.?\s?c\.?|indian\s+penal\s+code|cr\.?\s?p\.?\s?c\.?|"
               r"code\s+of\s+criminal\s+procedure|i\.?e\.?a\.?|(?:indian\s+)?evidence\s+act)")
_OLD_NUM = r"(?P<num>\d{1,3}\s?[a-z]{0,2})"
_OLD_SECTION_RES = (
    # "Section 420 IPC", "sec 420 of the IPC", "420 IPC", "s. 498A of the Indian Penal Code"
    re.compile(rf"(?:\b(?:section|sec\.?|s\.)\s*)?{_OLD_NUM}\s*(?:of\s+(?:the\s+)?)?{_CODE_ALIAS}\b",
               re.I),
    # "IPC 420", "IPC section 302", "CrPC s. 438"
    re.compile(rf"\b{_CODE_ALIAS}\s*(?:(?:section|sec\.?|s\.)\s*)?{_OLD_NUM}\b", re.I),
)


def _code_of(alias: str) -> str:
    a = alias.lower().replace(" ", "").replace(".", "")
    if a in ("ipc", "indianpenalcode"):
        return "IPC"
    if a in ("crpc", "codeofcriminalprocedure"):
        return "CrPC"
    return "IEA"


def map_repealed_sections(text: str) -> tuple[list[SectionMapping], list[tuple[str, str]]]:
    """Old-code sections named in the text: ``(mapped, unmapped)``.

    ``unmapped`` lists (code, section) pairs we have no verified mapping for. Those must NOT
    be looked up by their old number in the new code; the caller searches instead and the
    writer is told the number is unknown.
    """
    mapped: list[SectionMapping] = []
    unmapped: list[tuple[str, str]] = []
    for pattern in _OLD_SECTION_RES:
        for m in pattern.finditer(text or ""):
            code = _code_of(m.group("code"))
            num = re.sub(r"\s+", "", m.group("num")).upper()
            hit = SECTION_MAP.get((code, num))
            if hit and hit not in mapped:
                mapped.append(hit)
            elif not hit and (code, num) not in unmapped:
                unmapped.append((code, num))
    return mapped, unmapped


def detect_section_refs(text: str) -> list[SectionRef]:
    """Find "Section 6 of the RTI Act" / "Article 21" style references.

    These get answered by exact lookup rather than similarity search — asking what a named
    provision says deserves the provision, not its nearest neighbour.

    A section of a *repealed* code is translated through SECTION_MAP, never carried across by
    number: "Section 420 IPC" is BNS 318, not BNS 420.
    """
    if not text:
        return []
    refs: list[SectionRef] = []
    mapped, unmapped = map_repealed_sections(text)
    for m in mapped:
        refs.append(SectionRef("section", m.new, m.new_act))
    old_numbers = {m.old.upper() for m in mapped} | {num for _, num in unmapped}

    acts = [a for a in detect_acts(text)
            if not (mapped or unmapped) or a not in _NEW_CODE_SHORT]
    act = acts[0] if acts else None

    for match in _ARTICLE_RE.finditer(text):
        refs.append(SectionRef("article", _clean_label(match.group(1)), "Constitution of India"))
    for match in _SECTION_RE.finditer(text):
        label = _clean_label(match.group(1))
        if label.upper() in old_numbers:
            continue            # an old-code number: handled above, or deliberately not guessed
        refs.append(SectionRef("section", label, act))

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
    """Return the state a title belongs to, if it looks like state-legislature law."""
    title = (act_title or "").strip()
    for state in state_prefixes:
        if title.startswith(state):
            return state
    return None


def territory_of(act_title: str, territory_prefixes) -> str | None:
    """Return the Union Territory an Act is limited to, if Parliament passed it for one.

    Kept separate from :func:`is_state_law` because the two are different facts about the same
    reader-facing question. A Delhi Act *was* passed by Parliament (DB README §9) — so it is not
    state law — but it still applies only in Delhi, so it is not all-India law either. Collapsing
    the two into "central" is what put the Delhi Rent Act in front of someone asking about Mumbai
    with a label saying it applied across India.
    """
    title = (act_title or "").strip()
    for prefix, place in territory_prefixes:
        if title.startswith(prefix):
            return place
    return None


# Cities people name instead of their state. Only the unambiguous large ones: a wrong guess
# here would send someone to the wrong state's law.
CITY_STATE = {
    "mumbai": "Maharashtra", "bombay": "Maharashtra", "pune": "Maharashtra",
    "nagpur": "Maharashtra", "thane": "Maharashtra", "new delhi": "Delhi",
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "mysuru": "Karnataka",
    "chennai": "Tamil Nadu", "madras": "Tamil Nadu", "coimbatore": "Tamil Nadu",
    "kolkata": "West Bengal", "calcutta": "West Bengal", "hyderabad": "Telangana",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "jaipur": "Rajasthan", "lucknow": "Uttar Pradesh",
    "noida": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh", "gurgaon": "Haryana",
    "gurugram": "Haryana", "kochi": "Kerala", "thiruvananthapuram": "Kerala",
    "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh", "patna": "Bihar",
    "bhubaneswar": "Odisha", "guwahati": "Assam", "srinagar": "Jammu & Kashmir",
    "मुंबई": "Maharashtra", "दिल्ली": "Delhi", "बेंगलुरु": "Karnataka", "कोलकाता": "West Bengal",
    "चेन्नई": "Tamil Nadu", "लखनऊ": "Uttar Pradesh", "जयपुर": "Rajasthan", "पटना": "Bihar",
}


def place_named(text: str, states) -> str | None:
    """The state or UT a message is about, if it names one or one of its big cities.

    The planner flags "this depends on your state" from the topic alone, so "What does the
    Delhi Rent Control Act say about eviction?" was answered and then asked which state the
    user was in.
    """
    lowered = (text or "").lower()
    for state in sorted(states, key=len, reverse=True):
        names = {state.lower(), state.lower().replace("&", "and")}
        if any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", lowered) for n in names):
            return state
    for city, state in CITY_STATE.items():
        if re.search(rf"(?<!\w){re.escape(city)}(?!\w)", lowered):
            return state
    return None
