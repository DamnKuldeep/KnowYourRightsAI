"""Replace one mis-segmented Act in the corpus with a clean rebuild from its official text.

Why this exists: the source dataset behind notebook 01 flattened some Acts badly. The worst is
the Right to Information Act, 2005 — 31 sections in law, 14 in the corpus, with the numbers
shifted (corpus "Section 7" held Section 8's text) and Chapter V, including the Section 19
appeal, merged into a single "Section 12". Answers then cited the right words to the wrong
section: the 30-day reply rule came back as "Section 6".

The rebuild reproduces notebook 01's pipeline exactly, so repaired rows are indistinguishable
from the rest of the corpus to both retrievers:

* the same enrichment prompt — three citizen questions, 6-10 keywords, one category;
* the same ``embed_text`` layout — heading, questions, keywords, then the chunk;
* the same chunking — 480 words, 80 overlap, at most 25 chunks a section;
* the same embedder — ``baai/bge-m3``, verified against stored vectors at cosine 1.0000.

It refuses to write unless it finds every section, in order, 1 to N. A partial parse that
silently replaced a whole Act with half of it would be worse than the bug it fixes.

    python scripts/repair_act.py --dry-run                 # parse and print, write nothing
    python scripts/repair_act.py                           # rebuild RTI, then re-index
    python scripts/repair_act.py --restore VERSION         # roll the table back

Needs ``pypdf`` only to turn a PDF into text (not an app dependency):  pip install pypdf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field                                  # noqa: E402

from knowyourrights import config                                     # noqa: E402
from knowyourrights.llm import retrieval_api                          # noqa: E402
from knowyourrights.llm.client import get_client                      # noqa: E402
from knowyourrights.runtime.console import bold, rule, setup_console  # noqa: E402

setup_console()

# ── the Acts we know how to repair ────────────────────────────────────────────────────
# Each entry says where the official text is, where its body begins and ends, and how many
# sections the Act has — the count is the safety check that stops a bad parse from being
# written. Add an Act by adding an entry.
ACTS = {
    "Right to Information Act, 2005": {
        "url": "https://cic.gov.in/sites/default/files/RTI-Act_English.pdf",
        "pdf": "data/repair/rti_act_2005_cic.pdf",
        "body_start": "An Act to provide for setting out",
        "body_end": "THE FIRST SCHEDULE",
        "sections": 31,
        "act_number": "22",
        # From the Act's own Arrangement of Sections. The PDF's headings are OCR-damaged
        # ("Ftight to information", "Govemment"), and a heading is the first thing a reader
        # sees on a citation, so the published names are used and the PDF supplies only text.
        "names": [
            "Short title, extent and commencement", "Definitions", "Right to information",
            "Obligations of public authorities", "Designation of Public Information Officers",
            "Request for obtaining information", "Disposal of request",
            "Exemption from disclosure of information",
            "Grounds for rejection to access in certain cases", "Severability",
            "Third party information", "Constitution of Central Information Commission",
            "Term of office and conditions of service",
            "Removal of Chief Information Commissioner or Information Commissioner",
            "Constitution of State Information Commission",
            "Term of office and conditions of service",
            "Removal of State Chief Information Commissioner or State Information Commissioner",
            "Powers and functions of Information Commissions", "Appeal", "Penalties",
            "Protection of action taken in good faith", "Act to have overriding effect",
            "Bar of jurisdiction of courts", "Act not to apply to certain organizations",
            "Monitoring and reporting", "Appropriate Government to prepare programmes",
            "Power to make rules by appropriate Government",
            "Power to make rules by competent authority", "Laying of rules",
            "Power to remove difficulties", "Repeal",
        ],
        # (first section, chapter) — the PDF's Chapter VI heading did not survive extraction.
        "chapters": [
            (1, "CHAPTER I Preliminary"),
            (3, "CHAPTER II Right to information and obligations of public authorities"),
            (12, "CHAPTER III The Central Information Commission"),
            (15, "CHAPTER IV The State Information Commission"),
            (18, "CHAPTER V Powers and functions of the Information Commissions, appeal "
                 "and penalties"),
            (21, "CHAPTER VI Miscellaneous"),
        ],
        "source_note": ("Right to Information Act, 2005, official text as modified up to "
                        "1 Feb 2011 (Central Information Commission); section-level repair "
                        "of the corpus, 2026-09"),
    },
}

# Notebook 01, cell 4 — the taxonomy the rest of the corpus was categorised with.
CATEGORIES = ["Fundamental Rights", "Criminal & Police", "Consumer & Services",
              "Employment & Labour", "Family & Marriage", "Property & Housing",
              "Women & Children", "Privacy & Data", "Health & Medicine", "Education",
              "Environment", "Taxation & Finance", "Business & Companies", "Information & RTI",
              "Civil Procedure & Courts", "Transport & Motor", "Government & Administration",
              "Other"]

# Notebook 01, cell 19 — verbatim, so repaired rows are enriched exactly like the rest.
ENRICH_SYS = ("You write search metadata for a section of Indian central law, for an app that helps "
              "ordinary citizens. Base everything ONLY on the provided text; never invent legal "
              "facts. Reply with a single JSON object and nothing else: {\"questions\":[3 everyday "
              "questions a normal person might ask that this provision answers],\"keywords\":[6-10 "
              "short topic keywords],\"category\":\"one of " + " | ".join(CATEGORIES) + "\"}.")
ENRICH_MAX_CHARS = 3000
MAX_WORDS, OVERLAP, MAX_CHUNKS = 480, 80, 25          # notebook 01, cell 4


class Enrichment(BaseModel):
    questions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    category: str = "Other"


# ── parsing ───────────────────────────────────────────────────────────────────────────
def pdf_text(path: Path) -> str:
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def _clean(text: str, act_title: str) -> str:
    """Remove page furniture and the OCR damage seen in the official PDF."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == act_title:                                   # running header, 18x in RTI
            continue
        if re.fullmatch(r"\(?(Chapter|Sec\.?|Sections?)\b[^)]{0,60}\)?\.?", s, re.I):
            continue                                         # "(Chapter I —Preliminary.)"
        if re.fullmatch(r"[\divxlcIVXLC.\s]{1,4}", s) or len(s) <= 2:
            continue                                         # page numbers, stray marks
        kept.append(s)
    text = " ".join(kept)
    # "(1)" as OCR reads it: "(/)", "(I)", "(l)", and a bare "0)" after the heading dash.
    text = re.sub(r"\((?:/|I|l)\)", "(1)", text)
    text = re.sub(r"([.—–-])\s*0\)", r"\1(1)", text)
    # Case slips inside a word — "informatiOn", "acceSs", "serVice", "laW": a lowercase word
    # with a stray capital is always OCR here, never statute style.
    text = re.sub(r"\b[a-z]+[A-Z][a-zA-Z]*\b",
                  lambda m: m.group(0).lower() if not m.group(0)[1:].isupper() else m.group(0),
                  text)
    text = re.sub(r"\bof(?=[A-Z][a-z])", "of ", text)        # "ofJammu"
    text = re.sub(r"\b4ct\b", "Act", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_sections(raw: str, spec: dict, act_title: str) -> list[dict]:
    """Split the body into sections, accepting a heading only if it is the NEXT number.

    Enforcing the sequence is what makes this safe on real statute text, which is full of
    things that look like "12. Something—": clause lists, cross-references, schedules. A number
    out of sequence is text, not a heading.
    """
    start = raw.find(spec["body_start"])
    end = raw.find(spec["body_end"], start)
    if start < 0 or end < 0:
        raise SystemExit(f"could not find the body markers in the text")
    body = raw[start:end]

    heading = re.compile(r"(?m)^\s*(\d{1,3})\.\s*([A-Z][^\n]{1,180}?)\s*\.?\s*[—–-]\s*")
    chapter = re.compile(r"(?m)^\s*CHAPTER\s+([IVXL]+)\s*\n\s*([A-Z][A-Z ,'&-]{3,})\s*$")
    chapters = [(m.start(), f"CHAPTER {m.group(1)} {m.group(2).strip().title()}")
                for m in chapter.finditer(body)]

    found, expected = [], 1
    for m in heading.finditer(body):
        if int(m.group(1)) != expected:
            continue
        found.append((m.start(), m.end(), expected, m.group(2).strip().rstrip(".")))
        expected += 1

    names = spec.get("names") or []
    known_chapters = spec.get("chapters") or []
    sections = []
    for i, (s, e, n, name) in enumerate(found):
        stop = found[i + 1][0] if i + 1 < len(found) else len(body)
        text = _clean(body[e:stop], act_title)
        if known_chapters:
            chap = next((c for first, c in reversed(known_chapters) if first <= n), "")
        else:
            chap = next((c for pos, c in reversed(chapters) if pos <= s), "")
        if len(names) >= n:
            name = names[n - 1]
        sections.append({"label": str(n), "num": float(n), "name": name, "chapter": chap,
                         "text": text})
    return sections


def split_words(t: str) -> list[str]:
    """Notebook 01's chunker, unchanged."""
    w = t.split()
    if len(w) <= MAX_WORDS:
        return [t]
    out, i = [], 0
    while i < len(w) and len(out) < MAX_CHUNKS:
        out.append(" ".join(w[i:i + MAX_WORDS]))
        i += MAX_WORDS - OVERLAP
    return out


# ── enrichment, rows, vectors ─────────────────────────────────────────────────────────
async def enrich(act_title: str, sec: dict) -> Enrichment:
    head = f"{act_title} — Section {sec['label']}"
    if sec["name"]:
        head += f" ({sec['name']})"
    user = head + ":\n" + sec["text"][:ENRICH_MAX_CHARS]
    result = await get_client().chat_json(
        [{"role": "system", "content": ENRICH_SYS}, {"role": "user", "content": user}],
        Enrichment, Enrichment(), role="fast", stage="repair-enrich", max_tokens=500)
    if result.category not in CATEGORIES:
        result.category = "Information & RTI" if "Information" in act_title else "Other"
    return result


def build_rows(act_title: str, template: dict, sections: list[dict],
               enrichments: list[Enrichment], spec: dict) -> list[dict]:
    rows = []
    for sec, e in zip(sections, enrichments):
        unit_id = f"central_act|{act_title}|{sec['label']}"
        head = f"{act_title} — Section {sec['label']}"
        if sec["name"]:
            head += f" ({sec['name']})"
        for j, chunk in enumerate(split_words(sec["text"])):
            r = dict(template)
            r.update({
                "act_number": spec["act_number"],
                "section_label": sec["label"], "section_num": sec["num"],
                "section_name": sec["name"], "chapter": sec["chapter"],
                "full_text": sec["text"], "source_snapshot": spec["source_note"],
                "unit_id": unit_id, "chunk_id": f"{unit_id}#{j}", "chunk_text": chunk,
                "category": e.category,
                "citation": f"Section {sec['label']}, {act_title}",
                # notebook 01, cell 25: heading, questions, keywords, chunk
                "embed_text": "\n".join([head, *e.questions,
                                         *([" ".join(e.keywords)] if e.keywords else []),
                                         chunk]),
            })
            r.pop("vector", None)
            rows.append(r)
    return rows


# ── main ──────────────────────────────────────────────────────────────────────────────
async def main_async(args) -> int:
    import lancedb
    import numpy as np

    act_title = args.act
    spec = ACTS[act_title]
    db = lancedb.connect(str(config.DB_PATH))
    table = db.open_table(config.TABLE)

    if args.restore is not None:
        table.restore(args.restore)
        print(f"  restored {config.TABLE} to version {args.restore}")
        return 0

    rule(f"parse — {act_title}")
    pdf = Path(spec["pdf"])
    if not pdf.exists():
        import httpx
        pdf.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(spec["url"], timeout=60, follow_redirects=True, verify=False,
                      headers={"User-Agent": "KnowYourRights corpus repair"})
        r.raise_for_status()
        pdf.write_bytes(r.content)
    sections = parse_sections(pdf_text(pdf), spec, act_title)
    labels = [int(s["label"]) for s in sections]
    print(f"  sections found: {len(sections)} of {spec['sections']}")
    if labels != list(range(1, spec["sections"] + 1)):
        missing = sorted(set(range(1, spec["sections"] + 1)) - set(labels))
        print(f"  {bold('REFUSING TO WRITE')}: sections missing or out of order: {missing}")
        return 1
    for s in sections:
        print(f"    §{s['label']:<3} {s['name'][:48]:<50} {len(s['text'].split()):>5} words"
              f"  {s['chapter'][:34]}")

    old = table.search().where(f"act_title = '{act_title}'").limit(10_000).to_pandas()
    print(f"\n  replaces {len(old)} existing row(s) across "
          f"{old['section_label'].nunique()} mis-numbered section(s)")
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    rule("enrich — citizen questions and keywords, as notebook 01 did")
    enrichments = []
    for s in sections:
        e = await enrich(act_title, s)
        enrichments.append(e)
        print(f"    §{s['label']:<3} {e.category:<22} {(e.questions or ['—'])[0][:60]}")

    template = {k: v for k, v in old.iloc[0].to_dict().items() if k != "vector"}
    rows = build_rows(act_title, template, sections, enrichments, spec)

    rule("embed — baai/bge-m3, the corpus's own model")
    vectors = await retrieval_api.embed([r["embed_text"] for r in rows])
    for r, v in zip(rows, vectors):
        r["vector"] = np.asarray(v, dtype="float32")
    print(f"  {len(rows)} chunks embedded, spend so far {retrieval_api.session().stats()}")

    rule("write")
    before = table.version
    table.delete(f"act_title = '{act_title}'")
    table.add(rows)
    table.create_fts_index("embed_text", use_tantivy=False, replace=True)
    print(f"  table version {before} -> {table.version}; {table.count_rows():,} rows")
    print(f"  to undo:  python scripts/repair_act.py --restore {before}")
    print("  next:     python scripts/build_index.py --rebuild   (the vector index)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="Right to Information Act, 2005", choices=list(ACTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", type=int, metavar="VERSION")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
