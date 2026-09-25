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

from pydantic import BaseModel, Field

from knowyourrights import config
from knowyourrights.llm import retrieval_api
from knowyourrights.llm.client import get_client
from knowyourrights.retrieval.store import sql_quote
from knowyourrights.runtime.console import bold, rule, setup_console

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
ENRICH_SYS = ("You write search metadata for a section of Indian central law, for an app that "
              "helps ordinary citizens. Base everything ONLY on the provided text; never invent "
              "legal facts. Reply with a single JSON object and nothing else: "
              "{\"questions\":[3 everyday "
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


_HEADING = re.compile(r"(?m)^\s*(\d{1,3})\.\s*([A-Z][^\n]{1,180}?)\s*\.?\s*[—–-]\s*")
_CHAPTER = re.compile(r"(?m)^\s*CHAPTER\s+([IVXL]+)\s*\n\s*([A-Z][A-Z ,'&-]{3,})\s*$")


def _numbered_headings(body: str) -> list[tuple[int, int, int, str]]:
    """``(start, end, number, name)`` for each heading that is the next number in sequence."""
    found, expected = [], 1
    for m in _HEADING.finditer(body):
        if int(m.group(1)) == expected:
            found.append((m.start(), m.end(), expected, m.group(2).strip().rstrip(".")))
            expected += 1
    return found


def parse_sections(raw: str, spec: dict, act_title: str) -> list[dict]:
    """Split the body into sections, accepting a heading only if it is the NEXT number.

    Enforcing the sequence is what makes this safe on real statute text, which is full of
    things that look like "12. Something—": clause lists, cross-references, schedules. A number
    out of sequence is text, not a heading.
    """
    start = raw.find(spec["body_start"])
    end = raw.find(spec["body_end"], start)
    if start < 0 or end < 0:
        raise SystemExit("could not find the body markers in the text")
    body = raw[start:end]

    chapters = [(m.start(), f"CHAPTER {m.group(1)} {m.group(2).strip().title()}")
                for m in _CHAPTER.finditer(body)]
    found = _numbered_headings(body)
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
    for sec, e in zip(sections, enrichments, strict=True):
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


# ── actions ───────────────────────────────────────────────────────────────────────────
REMOVAL_LOG = Path("data/repair/removed.json")


def remove_act(table, act_title: str, reason: str | None) -> int:
    """Delete an Act that should not be in the corpus, and record why in the removal log."""
    where = f"act_title = {sql_quote(act_title)}"
    n = table.count_rows(where)
    if not n or not reason:
        print(f"  no rows for {act_title!r}" if not n else
              "  --remove needs --reason: a removal nobody can explain later is a bug")
        return 1
    before = table.version
    table.delete(where)
    table.create_fts_index("embed_text", use_tantivy=False, replace=True)
    REMOVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(REMOVAL_LOG.read_text(encoding="utf-8")) if REMOVAL_LOG.exists() else []
    entries.append({"act_title": act_title, "rows": n, "reason": reason,
                    "table_version_before": before, "date": time.strftime("%Y-%m-%d")})
    REMOVAL_LOG.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  removed {n} row(s); version {before} -> {table.version}; logged in {REMOVAL_LOG}")
    print(f"  undo with --restore {before}; next: python scripts/build_index.py --rebuild")
    return 0


def fetch_pdf(spec: dict, insecure: bool) -> Path:
    """The official text, downloaded once. Some government hosts serve an incomplete TLS
    certificate chain; ``--insecure-download`` exists for them and must be asked for."""
    import httpx

    pdf = Path(spec["pdf"])
    if not pdf.exists():
        pdf.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(spec["url"], timeout=60, follow_redirects=True,
                             verify=not insecure,
                             headers={"User-Agent": "KnowYourRights corpus repair"})
        response.raise_for_status()
        pdf.write_bytes(response.content)
    return pdf


def parse_checked(spec: dict, act_title: str, pdf: Path) -> list[dict] | None:
    """Every section, in sequence, or None (with the reason printed) if any is missing."""
    sections = parse_sections(pdf_text(pdf), spec, act_title)
    print(f"  sections found: {len(sections)} of {spec['sections']}")
    missing = sorted(set(range(1, spec["sections"] + 1)) - {int(s["label"]) for s in sections})
    if missing or len(sections) != spec["sections"]:
        print(f"  {bold('REFUSING TO WRITE')}: sections missing or out of order: {missing}")
        return None
    for s in sections:
        print(f"    §{s['label']:<3} {s['name'][:48]:<50} {len(s['text'].split()):>5} words"
              f"  {s['chapter'][:34]}")
    return sections


async def rebuild_rows(act_title: str, spec: dict, sections: list[dict], old) -> list[dict]:
    """Enrich, lay out and embed the new rows exactly as notebook 01 built the rest."""
    import numpy as np

    rule("enrich — citizen questions and keywords, as notebook 01 did")
    enrichments = []
    for s in sections:
        enrichments.append(await enrich(act_title, s))
        print(f"    §{s['label']:<3} {enrichments[-1].category:<22} "
              f"{(enrichments[-1].questions or ['—'])[0][:60]}")
    template = {k: v for k, v in old.iloc[0].to_dict().items() if k != "vector"}
    rows = build_rows(act_title, template, sections, enrichments, spec)
    rule("embed — baai/bge-m3, the corpus's own model")
    vectors = await retrieval_api.embed([r["embed_text"] for r in rows])
    for row, vector in zip(rows, vectors, strict=True):
        row["vector"] = np.asarray(vector, dtype="float32")
    print(f"  {len(rows)} chunks embedded, spend so far {retrieval_api.session().stats()}")
    return rows


def replace_act(table, act_title: str, rows: list[dict]) -> None:
    before = table.version
    table.delete(f"act_title = {sql_quote(act_title)}")
    table.add(rows)
    table.create_fts_index("embed_text", use_tantivy=False, replace=True)
    print(f"  table version {before} -> {table.version}; {table.count_rows():,} rows")
    print(f"  to undo:  python scripts/repair_act.py --restore {before}")
    print("  next:     python scripts/build_index.py --rebuild   (the vector index)")


# ── main ──────────────────────────────────────────────────────────────────────────────
async def main_async(args) -> int:
    import lancedb

    table = lancedb.connect(str(config.DB_PATH)).open_table(config.TABLE)
    if args.restore is not None:
        table.restore(args.restore)
        print(f"  restored {config.TABLE} to version {args.restore}")
        return 0
    if args.remove:
        return remove_act(table, args.remove, args.reason)

    act_title, spec = args.act, ACTS[args.act]
    rule(f"parse — {act_title}")
    sections = parse_checked(spec, act_title, fetch_pdf(spec, args.insecure_download))
    if sections is None:
        return 1
    old = table.search().where(f"act_title = {sql_quote(act_title)}").limit(10_000).to_pandas()
    print(f"\n  replaces {len(old)} existing row(s) across "
          f"{old['section_label'].nunique()} mis-numbered section(s)")
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    rows = await rebuild_rows(act_title, spec, sections, old)
    rule("write")
    replace_act(table, act_title, rows)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--act", default="Right to Information Act, 2005", choices=list(ACTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", type=int, metavar="VERSION")
    ap.add_argument("--remove", metavar="ACT_TITLE", help="delete an Act that should not be here")
    ap.add_argument("--reason", help="why, recorded in data/repair/removed.json")
    ap.add_argument("--insecure-download", action="store_true",
                    help="skip TLS verification when downloading the official PDF")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
