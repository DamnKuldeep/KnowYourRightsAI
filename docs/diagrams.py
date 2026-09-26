"""Builds the SVG diagrams in docs/. Run: python docs/diagrams.py

Hand-placed on a grid rather than auto-laid-out, so each figure shows one mechanism cleanly.
GitHub shows SVGs as images, where page colours do not reach, so every figure carries its own
light card and fixed palette (the app's own) and reads the same on light and dark themes.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"

INK, DIM, LINE, CARD, PAPER = "#23201c", "#6c665e", "#b9b2a7", "#fbfaf8", "#ffffff"
STYLES = {  # kind: (fill, stroke, dash)
    "code":   (PAPER,     "#b9b2a7", ""),
    "model":  ("#e7f1ee", "#1f6f5c", ""),
    "api":    ("#e9eff8", "#2f5d9e", ""),
    "safety": ("#fbeceb", "#b3261e", ""),
    "deep":   ("#f3eefa", "#7a5ea8", "5 4"),
    "store":  ("#f4f2ee", "#8a8378", ""),
    "group":  ("none",    "#d9d3c9", "4 4"),
}
ARROWS = {"ink": "#6c665e", "deep": "#7a5ea8", "safety": "#b3261e", "api": "#2f5d9e"}


def text(x, y, s, size=13, weight=400, fill=INK, anchor="middle", rotate=None):
    turn = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{turn}>{escape(s)}</text>')


def box(x, y, w, h, title, sub="", kind="code", pill=False, title_size=13):
    fill, stroke, dash = STYLES[kind]
    rx = h / 2 if pill else 9
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
           f'stroke="{stroke}" stroke-width="1.4"{dash_attr}/>']
    cx = x + w / 2
    if sub:
        out.append(text(cx, y + h / 2 - 3, title, title_size, 620))
        out.append(text(cx, y + h / 2 + 13, sub, 11, 400, DIM))
    else:
        out.append(text(cx, y + h / 2 + 4.5, title, title_size, 620))
    return "".join(out)


def arrow(points, label="", at=None, color="ink", dashed=False, anchor="middle"):
    """A polyline ending in an arrowhead. `at` is where the label goes (defaults to the middle
    of the first segment, nudged above it)."""
    stroke = ARROWS[color]
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = [f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="1.5"'
           f'{dash} marker-end="url(#head-{color})"/>']
    if label:
        (x1, y1), (x2, y2) = points[0], points[1]
        lx, ly = at or ((x1 + x2) / 2, (y1 + y2) / 2 - 7)
        out.append(text(lx, ly, label, 11, 500, stroke, anchor))
    return "".join(out)


def legend(x, y, items):
    out, cx = [], x
    for kind, label in items:
        fill, stroke, dash = STYLES[kind]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<rect x="{cx}" y="{y - 10}" width="22" height="13" rx="3" fill="{fill}" '
                   f'stroke="{stroke}" stroke-width="1.3"{dash_attr}/>')
        out.append(text(cx + 28, y, label, 11.5, 400, DIM, "start"))
        cx += 36 + 6.3 * len(label) + 18
    return "".join(out)


def figure(name, w, h, label, body, top=0):
    heads = "".join(
        f'<marker id="head-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{c}"/></marker>' for k, c in ARROWS.items())
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
           f'height="{h}" role="img" aria-label="{escape(label)}" font-family="{FONT}">'
           f'<title>{escape(label)}</title><defs>{heads}</defs>'
           f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="16" fill="{CARD}" '
           f'stroke="#e3ded6" stroke-width="1.5"/>'
           f'<g transform="translate(0 {top})">{body}</g></svg>\n')
    (OUT / name).write_text(svg, encoding="utf-8")
    print(f"wrote docs/{name}")


# ── 1. how a question is answered ──────────────────────────────────────────────────────────
def pipeline():
    L, R, CX = 90, 390, 240          # main column
    BX, BW = 494, 262                # branch column
    b = [
        box(140, 24, 200, 40, "Question", pill=True),
        box(L, 98, 300, 56, "Safety gate", "phrase patterns + meaning, before anything else"),
        box(L, 188, 300, 56, "Planner", "intent · depth · search queries · language", "model"),
        # research round
        f'<rect x="{L}" y="278" width="300" height="132" rx="11" fill="none" stroke="#b9b2a7" '
        f'stroke-width="1.4"/>',
        text(CX, 301, "Research round", 13, 620),
        text(CX, 301 + 0, "", 11),
        box(104, 316, 136, 36, "Statute search", title_size=12),
        box(250, 316, 126, 36, "Official sites", title_size=12),
        box(104, 362, 136, 36, "Web pages", title_size=12),
        box(250, 362, 126, 36, "Wikipedia · portals", title_size=12),
        box(L, 444, 300, 56, "Grader", "drops sources that only share words", "model"),
        box(L, 534, 300, 56, "Writer", "streams the answer as it is written", "model"),
        box(L, 624, 300, 56, "Citation check", "every [S1] must match a source it was given"),
        box(140, 714, 200, 40, "Answer + sources", pill=True),
        # main flow
        arrow([(CX, 64), (CX, 96)]),
        arrow([(CX, 154), (CX, 186)]),
        arrow([(CX, 244), (CX, 276)]),
        arrow([(CX, 410), (CX, 442)]),
        arrow([(CX, 500), (CX, 532)]),
        arrow([(CX, 590), (CX, 622)]),
        arrow([(CX, 680), (CX, 712)]),
        # branches
        box(BX, 98, BW, 56, "Helpline card, shown first",
            "violence · self-harm · arrest in progress", "safety"),
        arrow([(R, 126), (BX - 2, 126)], "emergency", color="safety"),
        box(BX + 30, 196, BW - 60, 40, "Short reply, nothing cited", pill=True, title_size=12.5),
        arrow([(R, 216), (BX + 28, 216)], "off-topic"),
        box(BX, 300, BW, 56, "Exact lookup", "“Section 420 IPC” → BNS 318(4); quick mode stops"),
        arrow([(BX, 328), (R + 2, 328)], "named section"),
        box(BX, 444, BW, 56, "Procedure card", "fee · time limit · appeal · portal", "model"),
        arrow([(R, 472), (BX - 2, 472)], "how-to"),
        arrow([(BX + BW / 2, 500), (BX + BW / 2, 562), (R + 2, 562)]),
        box(BX, 624, BW, 56, "Self-verify", "web-check fees and deadlines, then rewrite", "deep"),
        arrow([(R, 652), (BX - 2, 652)], "deep mode", color="deep", dashed=True),
        arrow([(BX + BW / 2, 680), (BX + BW / 2, 734), (342, 734)], color="deep", dashed=True),
        # deep research loop, on the left
        arrow([(L, 472), (48, 472), (48, 344), (L - 2, 344)], color="deep", dashed=True),
        text(36, 408, "deep: another round while gaps remain", 11, 500, ARROWS["deep"],
             rotate=-90),
        legend(92, 792, [("model", "model call"), ("code", "plain code"),
                         ("deep", "deep mode only"), ("safety", "safety")]),
    ]
    figure("pipeline.svg", 780, 816, "How a question becomes a cited answer. Models plan, grade "
           "and write; code runs the search, checks every citation and decides what happens "
           "next.", "".join(b))


# ── 2. statute search ──────────────────────────────────────────────────────────────────────
def retrieval():
    CX = 380
    lists = [(24, "Vector search", "bge-m3 embeddings", "api"),
             (206, "Keyword search", "BM25 full text", "code"),
             (388, "Named Act only", "weight ×2.5 when one is named", "deep"),
             (570, "Constitution + 2023 codes", "×2.0 for policing questions", "deep")]
    b = [box(280, 20, 200, 40, "Question", pill=True),
         box(230, 90, 300, 52, "Acronym expansion", "RTI → Right to Information Act, 2005"),
         arrow([(CX, 60), (CX, 88)])]
    for x, title, sub, kind in lists:
        b.append(box(x, 182, 166, 60, title, sub, kind, title_size=12.5))
        b.append(arrow([(CX, 142), (CX, 160), (x + 83, 160), (x + 83, 180)]))
        b.append(arrow([(x + 83, 242), (x + 83, 262), (CX, 262), (CX, 280)]))
    b += [
        box(215, 282, 330, 52, "Weighted rank fusion", "merged, then one row per section"),
        box(215, 368, 330, 52, "Rerank", "Cohere v3.5 reads each section and its citizen questions",
            "api"),
        box(215, 454, 330, 52, "General law wins near-ties",
            "the BNSS, not the Navy Act, for “can police arrest me”"),
        box(215, 540, 330, 52, "Diversify", "maximal marginal relevance on the stored vectors"),
        box(215, 626, 330, 52, "Confident enough?", "best score vs. the calibrated threshold"),
        arrow([(CX, 334), (CX, 366)]),
        arrow([(CX, 420), (CX, 452)]),
        arrow([(CX, 506), (CX, 538)]),
        arrow([(CX, 592), (CX, 624)]),
        box(60, 716, 250, 44, "Top 5 sections → the writer", kind="model", pill=True,
            title_size=12.5),
        box(450, 716, 250, 44, "Abstain: “nothing on point”", kind="safety", pill=True,
            title_size=12.5),
        arrow([(300, 678), (300, 696), (185, 696), (185, 714)], "yes", at=(240, 690)),
        arrow([(460, 678), (460, 696), (575, 696), (575, 714)], "no", at=(520, 690)),
        # what happens when the APIs are down
        text(560, 386, "if down: fused order,", 11, 500, ARROWS["api"], "start"),
        text(560, 400, "with its own threshold", 11, 500, ARROWS["api"], "start"),
        legend(40, 800, [("api", "API call (OpenRouter)"), ("code", "local, on the corpus"),
                         ("deep", "added only when it applies")]),
    ]
    figure("retrieval.svg", 760, 824, "Statute search: two to four ranked lists are fused, "
           "reranked and diversified; below a calibrated score it abstains instead of guessing.",
           "".join(b))


# ── 3. the system ─────────────────────────────────────────────────────────────────────────
def system():
    GX, GW = 244, 200               # request gates
    MX, MW = 498, 192               # modules, inside their dashed group
    EX, EW = 790, 156               # outside services
    b = [
        box(24, 72, 170, 56, "Browser", "plain HTML + JS"),
        '<rect x="224" y="24" width="500" height="472" rx="13" fill="none" stroke="#b9b2a7" '
        'stroke-width="1.4"/>',
        text(240, 48, "Server · FastAPI, one process", 12, 650, DIM, "start"),
        box(GX, 72, GW, 56, "Sign-in", "signed cookie, 30 days"),
        box(GX, 150, GW, 56, "Validate", "every field bounded"),
        box(GX, 228, GW, 56, "Guards", "rate · $ per visitor · $ per day"),
        box(GX, 306, GW, 56, "Admission queue", "5 answers at once, then a line"),
        box(GX, 404, GW, 56, "Spend book", "per visitor and per day", "store"),
        arrow([(GX + 100, 128), (GX + 100, 148)]),
        arrow([(GX + 100, 206), (GX + 100, 226)]),
        arrow([(GX + 100, 284), (GX + 100, 304)]),
        arrow([(194, 100), (GX - 2, 100)], "POST", at=(219, 92)),
        box(484, 72, 220, 56, "Orchestrator", "plan → research → write → verify", "model"),
        arrow([(GX + GW, 334), (464, 334), (464, 100), (482, 100)]),
        # modules the orchestrator calls
        '<rect x="484" y="150" width="220" height="264" rx="11" fill="none" stroke="#d9d3c9" '
        'stroke-width="1.3" stroke-dasharray="4 4"/>',
        arrow([(594, 128), (594, 148)]),
        box(MX, 166, MW, 48, "Model stages", "plan · grade · write", "model", title_size=12.5),
        box(MX, 226, MW, 48, "Statute search", "hybrid, reranked", title_size=12.5),
        box(MX, 286, MW, 48, "Web tools", "search · read · navigate", title_size=12.5),
        box(MX, 346, MW, 48, "Model client", "routing · failover · cost", title_size=12.5),
        # the answer streams back
        arrow([(594, 72), (594, 12), (109, 12), (109, 70)], "SSE: steps, sources, then the answer "
              "word by word", at=(350, 4)),
        # outside
        box(EX, 226, EW, 48, "Legal corpus", "LanceDB on disk", "store", title_size=12.5),
        box(EX, 286, EW, 48, "The web", "public URLs only", "api", title_size=12.5),
        box(EX, 346, EW, 48, "OpenRouter", "chat · embed · rerank", "api", title_size=12.5),
        box(EX, 424, EW, 48, "NVIDIA NIM", "backup chat", "api", title_size=12.5),
        arrow([(MX + MW, 250), (EX - 2, 250)], "vector + BM25", at=(747, 242)),
        arrow([(MX + MW, 310), (EX - 2, 310)]),
        arrow([(MX + MW, 370), (EX - 2, 370)], "HTTPS", at=(747, 362)),
        arrow([(MX + MW, 382), (740, 382), (740, 448), (EX - 2, 448)], "if it fails",
              at=(746, 418), anchor="start", dashed=True),
        arrow([(594, 414), (594, 432), (GX + GW + 2, 432)], "charges each call",
              at=(520, 450), dashed=True),
        legend(40, 520, [("model", "uses a model"), ("code", "plain code"),
                         ("api", "outside service"), ("store", "stored state")]),
    ]
    figure("system.svg", 970, 572, "The system: requests pass sign-in, validation, spending "
           "guards and a queue before the orchestrator runs a turn and streams it back.",
           "".join(b), top=26)


if __name__ == "__main__":
    pipeline()
    retrieval()
    system()
