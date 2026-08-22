"""The retrieval gold set.

The build notebook shipped ten illustrative queries. This grows that to a set wide enough to
tune against — every domain of the 18-category taxonomy, plus the two failure modes that
matter for a public legal tool: questions whose answer is *procedure* (BNSS) rather than
*offence* (BNS), and questions that are state subjects the central corpus cannot answer.

Each expectation is a substring matched case-insensitively against the returned ``citation``,
so it can pin an Act loosely ("Consumer Protection") or a specific provision tightly
("Article 21"). Every target was verified to exist in the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldCase:
    """``expect`` may list several acceptable answers, and often must.

    Indian law frequently gives more than one right answer: the 24-hour custody limit lives in
    both Article 22 and the BNSS, and the Code on Wages, 2019 subsumed the Minimum Wages Act,
    1948. Insisting on a single citation would score correct retrieval as a miss and push
    tuning in the wrong direction.
    """

    query: str
    expect: str | tuple[str, ...]
    category: str
    note: str = ""

    @property
    def accepted(self) -> tuple[str, ...]:
        return (self.expect,) if isinstance(self.expect, str) else self.expect

    def matches(self, citation: str) -> bool:
        low = (citation or "").lower()
        return any(option.lower() in low for option in self.accepted)


GOLD: tuple[GoldCase, ...] = (
    # ── constitutional / fundamental rights ──────────────────────────────────────────
    GoldCase("right to equality before the law", "Article 14", "Fundamental Rights"),
    GoldCase("right to life and personal liberty", "Article 21", "Fundamental Rights"),
    GoldCase("freedom to practise my religion", "Article 25", "Fundamental Rights"),
    GoldCase("is untouchability banned", "Article 17", "Fundamental Rights"),
    GoldCase("freedom of speech and expression", "Article 19", "Fundamental Rights"),
    GoldCase("can I move the Supreme Court if my fundamental rights are violated",
             "Article 32", "Fundamental Rights"),
    GoldCase("protection against arrest and detention", "Article 22", "Fundamental Rights"),
    GoldCase("no person shall be prosecuted twice for the same offence", "Article 20",
             "Fundamental Rights", "double jeopardy"),

    # ── criminal: offences (BNS) vs procedure (BNSS) ─────────────────────────────────
    GoldCase("what is the punishment for cheating", "Bharatiya Nyaya", "Criminal & Police"),
    GoldCase("punishment for murder", "Bharatiya Nyaya", "Criminal & Police"),
    GoldCase("what counts as theft", "Bharatiya Nyaya", "Criminal & Police"),
    GoldCase("when can a police officer arrest without a warrant", "Bharatiya Nagarik",
             "Criminal & Police", "procedure, not offence"),
    # The 24-hour limit is stated in both the Constitution and the procedure code.
    GoldCase("how long can police keep me in custody before producing me before a magistrate",
             ("Bharatiya Nagarik", "Article 22"), "Criminal & Police"),
    GoldCase("how do I get anticipatory bail", "Bharatiya Nagarik", "Criminal & Police"),
    GoldCase("police refuse to register my FIR what can I do", "Bharatiya Nagarik",
             "Criminal & Police"),
    GoldCase("what evidence is admissible in court", "Bharatiya Sakshya", "Criminal & Police"),

    # ── information & RTI ────────────────────────────────────────────────────────────
    GoldCase("how do I file an RTI request for information", "Right to Information",
             "Information & RTI"),
    GoldCase("how long does a public authority have to answer my RTI",
             "Right to Information", "Information & RTI"),
    GoldCase("my RTI was rejected, how do I appeal", "Right to Information", "Information & RTI"),

    # ── consumer ─────────────────────────────────────────────────────────────────────
    GoldCase("my consumer complaint against a defective product", "Consumer",
             "Consumer & Services"),
    GoldCase("compensation for deficiency in service", "Consumer", "Consumer & Services"),
    GoldCase("what is an unfair trade practice", "Consumer", "Consumer & Services"),

    # ── employment & labour ──────────────────────────────────────────────────────────
    GoldCase("maternity leave entitlement at work", "Maternity", "Employment & Labour"),
    # The Code on Wages, 2019 consolidated and replaced the Minimum Wages Act, 1948 — both
    # are in the corpus and either is a defensible citation.
    GoldCase("my employer is not paying minimum wages",
             ("Minimum Wages", "Code on Wages"), "Employment & Labour"),
    GoldCase("sexual harassment at my workplace complaint committee", "Sexual Harassment",
             "Employment & Labour"),
    GoldCase("gratuity after five years of service",
             ("Gratuity", "Code on Social Security"), "Employment & Labour"),
    GoldCase("provident fund deduction from my salary",
             ("Provident Fund", "Code on Social Security"), "Employment & Labour"),

    # ── women, children, family ──────────────────────────────────────────────────────
    GoldCase("protection from domestic violence", "Domestic Violence", "Women & Children"),
    GoldCase("dowry demand by my in-laws", "Dowry", "Women & Children"),
    GoldCase("sexual offences against a child", "Protection of Children", "Women & Children"),
    GoldCase("grounds for divorce under Hindu law", "Hindu Marriage", "Family & Marriage"),
    GoldCase("marriage between people of different religions", "Special Marriage",
             "Family & Marriage"),

    # ── transport, property, education, environment, tax, business ───────────────────
    GoldCase("driving without a licence penalty", "Motor Vehicles", "Transport & Motor"),
    GoldCase("compensation for a road accident death", "Motor Vehicles", "Transport & Motor"),
    GoldCase("builder delayed possession of my flat", "Real Estate", "Property & Housing"),
    GoldCase("free education for children between six and fourteen",
             "Compulsory Education", "Education"),
    GoldCase("penalty for polluting the environment", "Environment", "Environment"),
    GoldCase("cheque bounced what is the remedy", "Negotiable Instruments", "Taxation & Finance"),
    GoldCase("rights of a person with mental illness", "Mental Healthcare", "Health & Medicine"),
    GoldCase("caste based discrimination and atrocities", "Scheduled Castes", "Fundamental Rights"),
    GoldCase("rights of transgender persons", "Transgender", "Fundamental Rights"),
    GoldCase("when can a woman legally terminate a pregnancy", "Termination of Pregnancy",
             "Health & Medicine"),
)


@dataclass(frozen=True)
class StressCase:
    query: str
    reason: str
    expect_state: bool = False


# Questions the central corpus should decline rather than answer confidently. A wrong-but-
# plausible citation is worse than an honest "this is a state subject, let me search".
STRESS: tuple[StressCase, ...] = (
    StressCase("my landlord in Mumbai won't return my deposit",
               "rent control is a state subject", expect_state=True),
    StressCase("rules for my housing society in Karnataka",
               "cooperative societies are a state subject", expect_state=True),
    StressCase("stamp duty on property registration in Tamil Nadu",
               "stamp duty rates are set by states", expect_state=True),
    # Off-topic queries set the floor for the abstention threshold, so there need to be
    # enough of them to estimate it from — two samples is not a distribution.
    StressCase("what is the best pizza recipe", "not a legal question"),
    StressCase("write me a poem about the monsoon", "not a legal question"),
    StressCase("who won the cricket world cup in 2011", "not a legal question"),
    StressCase("how do I fix a memory leak in my python program", "not a legal question"),
    StressCase("what is the capital of Australia", "not a legal question"),
    StressCase("recommend a good movie to watch tonight", "not a legal question"),
    StressCase("how tall is Mount Everest", "not a legal question"),
    StressCase("translate good morning into French", "not a legal question"),
)


def offtopic_cases() -> tuple[StressCase, ...]:
    """Stress cases that are genuinely not legal questions (excludes state-subject ones)."""
    return tuple(c for c in STRESS if not c.expect_state)


# Questions that must resolve by *exact lookup*, not similarity search.
EXACT: tuple[tuple[str, str, str], ...] = (
    ("what does Article 21 say", "Article 21", "Constitution of India"),
    ("show me Section 6 of the RTI Act", "Section 6", "Right to Information Act, 2005"),
    ("Article 14 text", "Article 14", "Constitution of India"),
    ("Section 35 BNSS", "Section 35", "Bharatiya Nagarik Suraksha Sanhita, 2023"),
)
