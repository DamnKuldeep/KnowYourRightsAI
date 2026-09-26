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


def step(x, y, n):
    """A numbered badge, so prose can refer to a step by number."""
    return (f'<circle cx="{x}" cy="{y}" r="11" fill="#1f6f5c"/>'
            + text(x, y + 4, str(n), 11.5, 700, "#ffffff"))


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
        text(CX, 301, "Research round · the plan picks, all in parallel", 12.5, 620),
        box(104, 316, 136, 36, "Statute search", title_size=12),
        box(250, 316, 126, 36, "Official sites", title_size=12),
        box(104, 362, 136, 36, "Web pages", title_size=12),
        box(250, 362, 126, 36, "Wikipedia · portals", title_size=12),
        box(L, 444, 300, 56, "Grader", "drops sources that only share words", "model"),
        box(L, 534, 300, 56, "Writer", "streams the answer as it is written", "model"),
        box(L, 624, 300, 56, "Citation check", "every [S1] must match a source it was given"),
        box(140, 714, 200, 40, "Answer + sources", pill=True),
        # numbered steps, outside the boxes so they never collide with centred text
        step(68, 126, 1), step(68, 216, 2), step(68, 297, 3), step(68, 452, 4),
        step(68, 562, 5), step(68, 652, 6),
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
    """Top to bottom, in the order things happen: the question comes in, passes the checks that
    come before any spending, one turn runs, the turn uses three kinds of work, and the answer
    streams back. Outside services sit below the server they are called from."""
    C1, C2, C3, W = 40, 350, 660, 260          # three columns
    mid = [x + W / 2 for x in (C1, C2, C3)]

    def band(y, n, label):
        return step(66, y - 4, n) + text(84, y, label, 12, 650, DIM, "start")

    b = [
        box(C1, 24, W, 56, "Browser", "plain HTML + JavaScript"),
        arrow([(C1 + W - 40, 80), (C1 + W - 40, 182)]),
        step(C1 + W - 18, 104, 1), text(C1 + W + 2, 108, "your question", 11.5, 500, DIM, "start"),
        # the server
        '<rect x="28" y="124" width="904" height="484" rx="14" fill="none" stroke="#b9b2a7" '
        'stroke-width="1.4"/>',
        text(912, 146, "Server · FastAPI, one process", 12, 650, DIM, "end"),
        band(170, 2, "Checks before spending"),
        box(C1, 184, W, 58, "Sign-in", "signed cookie, when enabled"),
        box(C2, 184, W, 58, "Limits", "rate · $ per visitor · $ per day"),
        box(C3, 184, W, 58, "Queue", "5 answers at once, then a line"),
        arrow([(C1 + W, 213), (C2 - 2, 213)]),
        arrow([(C2 + W, 213), (C3 - 2, 213)]),
        box(C2, 296, W, 58, "Orchestrator", "plan → research → write → check"),
        step(C2 + 22, 325, 3),
        arrow([(mid[2], 242), (mid[2], 325), (C2 + W + 2, 325)]),
        f'<polyline points="{mid[1]},354 {mid[1]},382" fill="none" stroke="{ARROWS["ink"]}" '
        f'stroke-width="1.5"/>',
        f'<polyline points="{mid[0]},382 {mid[2]},382" fill="none" stroke="{ARROWS["ink"]}" '
        f'stroke-width="1.5"/>',
        arrow([(mid[0], 382), (mid[0], 406)]),
        arrow([(mid[1], 382), (mid[1], 406)]),
        arrow([(mid[2], 382), (mid[2], 406)]),
        step(mid[1], 382, 4),
        box(C1, 408, W, 58, "Model stages", "plan · grade · write · fact-check", "model"),
        box(C2, 408, W, 58, "Web tools", "search · read pages · follow portals"),
        box(C3, 408, W, 58, "Statute search", "meaning + keywords, reranked"),
        box(C1, 516, W, 58, "Model client", "picks a model · fails over · counts cost"),
        box(C3, 516, W, 58, "Legal corpus", "LanceDB on disk · 38,609 chunks", "store"),
        arrow([(mid[0], 466), (mid[0], 514)]),
        arrow([(mid[2], 466), (mid[2], 514)]),
        text(mid[2] + 10, 494, "reads", 11, 500, DIM, "start"),
        # the answer streams back
        arrow([(C2, 340), (14, 340), (14, 52), (C1 - 2, 52)]),
        step(190, 327, 5), text(208, 331, "the answer streams back", 11.5, 500, DIM, "start"),
        # outside
        text(40, 634, "Outside services", 12, 650, DIM, "start"),
        box(C1, 646, W, 58, "Model providers", "OpenRouter · NVIDIA NIM as backup", "api"),
        box(C2, 646, W, 58, "The public web", "government sites · search · Wikipedia", "api"),
        arrow([(mid[0], 574), (mid[0], 644)], "chat · embeddings · rerank", at=(mid[0] + 8, 618),
              anchor="start"),
        arrow([(mid[1], 466), (mid[1], 644)], "public addresses only", at=(mid[1] + 8, 618),
              anchor="start"),
        legend(40, 742, [("model", "uses a model"), ("code", "plain code"),
                         ("api", "outside service"), ("store", "stored data")]),
    ]
    figure("system.svg", 960, 766, "The system, in order: a question passes sign-in, spending "
           "limits and a queue; the orchestrator runs one turn using model stages, web tools and "
           "statute search; the answer streams back.", "".join(b))


if __name__ == "__main__":
    pipeline()
    retrieval()
    system()
