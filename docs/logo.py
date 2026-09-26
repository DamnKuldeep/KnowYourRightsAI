"""Builds the KnowYourRights brand assets.

    python docs/logo.py path/to/LibreBaskerville[wght].ttf

The mark is a speech bubble (a plain-language answer) holding balanced scales (the law), whose
saffron finial, with the app's green and white, nods to India without using any national emblem.
The favicon drops the bubble, which blurs at 16 px, and keeps the scales.

The scales are plain geometry. Letters are drawn as paths from Libre Baskerville (SIL Open Font
License, github.com/google/fonts/tree/main/ofl/librebaskerville), so the files look the same
everywhere and need no font. PNGs are rendered with Playwright, already a dependency of the app.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "docs" / "brand"
WEB = ROOT / "knowyourrights" / "web"

GREEN, GREEN_TOP, GREEN_BOTTOM = "#0e6b5a", "#13806b", "#0a4a3f"
SAFFRON, INK, INK_DARK, DIM, DIM_DARK = "#f2a93b", "#1c1a17", "#ebe8e2", "#6c665e", "#aba69d"
GRADIENT = (f'<linearGradient id="kyr-g" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{GREEN_TOP}"/>'
            f'<stop offset="1" stop-color="{GREEN_BOTTOM}"/>'
            f'</linearGradient>')


class Type:
    """Text set as SVG paths from one weight of the font."""

    def __init__(self, font_path: Path, weight: int) -> None:
        self.font = instancer.instantiateVariableFont(TTFont(font_path), {"wght": weight})
        self.glyphs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm

    def _name(self, ch: str) -> str:
        return self.cmap[ord(ch)]

    def line(self, text: str, x: float, baseline: float, size: float,
             tracking: float = 0) -> tuple[str, float]:
        """A run of text starting at x; returns (paths, width). `tracking` is in em."""
        s, pen_x, out = size / self.upm, x, []
        for ch in text:
            name = self._name(ch)
            out.append(self._path(name, (s, 0, 0, -s, pen_x, baseline)))
            pen_x += self.font["hmtx"][name][0] * s + tracking * size
        return "".join(out), pen_x - x - tracking * size

    def _path(self, name: str, matrix) -> str:
        pen = SVGPathPen(self.glyphs)
        self.glyphs[name].draw(TransformPen(pen, matrix))
        return f'<path d="{pen.getCommands()}"/>'


def svg(w, h, body, label, defs=GRADIENT):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" role="img" aria-label="{label}"><title>{label}</title>'
            f'<defs>{defs}</defs>{body}</svg>\n')


def scales(cx: float, top: float, span: float, ink: str, *, stroke: float, pan: float,
           drop: float, post: float, base: float) -> str:
    """Balanced scales: a saffron finial, a beam, a post on a base, and two strung pans."""
    beam_y, left, right = top + 22, cx - span / 2, cx + span / 2
    pan_y = beam_y + drop

    def hanging(x: float) -> str:
        return (f'<path d="M{x - pan} {pan_y}h{2 * pan}a{pan} {pan * .78} 0 0 1-{2 * pan} 0z" '
                f'fill="{ink}"/><path d="M{x} {beam_y}L{x - pan + 6} {pan_y}M{x} {beam_y}'
                f'L{x + pan - 6} {pan_y}" stroke="{ink}" stroke-width="{stroke * .55}" '
                'stroke-linecap="round" fill="none"/>')

    return (f'<rect x="{cx - stroke / 2}" y="{beam_y}" width="{stroke}" height="{post}" '
            f'rx="{stroke / 2}" fill="{ink}"/>'
            f'<rect x="{cx - base / 2}" y="{beam_y + post - 4}" width="{base}" '
            f'height="{stroke * 1.2}" rx="{stroke * .6}" fill="{ink}"/>'
            f'<rect x="{left}" y="{beam_y - stroke / 2}" width="{span}" height="{stroke}" '
            f'rx="{stroke / 2}" fill="{ink}"/>' + hanging(left) + hanging(right)
            + f'<circle cx="{cx}" cy="{top}" r="{stroke * 1.15}" fill="{SAFFRON}"/>')


def mark_body(x=0, y=0, size=512) -> str:
    """The app icon, drawn on a 512 grid and placed at (x, y) with the given size."""
    return (f'<g transform="translate({x} {y}) scale({size / 512})">'
            '<rect width="512" height="512" rx="116" fill="url(#kyr-g)"/>'
            '<path fill="#fff" d="M160 84h192a88 88 0 0 1 88 88v132a88 88 0 0 1-88 88H244l-96 70 '
            '16-72a88 88 0 0 1-84-88V172a88 88 0 0 1 88-88z"/>'
            + scales(256, 150, 222, GREEN, stroke=19, pan=46, drop=86, post=148, base=112)
            + '</g>')


def favicon_body() -> str:
    return ('<rect width="512" height="512" rx="116" fill="url(#kyr-g)"/>'
            + scales(256, 116, 316, "#ffffff", stroke=28, pan=62, drop=116, post=214, base=164))


def lockup(bold: Type, regular: Type, dark: bool) -> str:
    """Mark + wordmark + tagline, for the README header."""
    ink, dim = (INK_DARK, DIM_DARK) if dark else (INK, DIM)
    rights = "#5cc6ab" if dark else GREEN
    know, w1 = regular.line("KnowYour", 150, 84, 58)
    right, w2 = bold.line("Rights", 150 + w1, 84, 58)
    tag, w3 = regular.line("INDIAN LAW, CITED TO THE SECTION", 153, 124, 17.5, tracking=0.16)
    width = int(150 + max(w1 + w2, w3) + 12)
    body = (mark_body(0, 8, 124) + f'<g fill="{ink}">{know}</g>'
            f'<g fill="{rights}">{right}</g><g fill="{dim}">{tag}</g>')
    return svg(width, 140, body, "KnowYourRights: Indian law, cited to the section")


def social(bold: Type, regular: Type) -> str:
    """1280×640 image for link previews and the GitHub social preview."""
    know, w1 = regular.line("KnowYour", 0, 0, 92)
    right, _ = bold.line("Rights", w1, 0, 92)
    tag, _ = regular.line("Plain-language answers about Indian law,", 0, 0, 34)
    tag2, _ = regular.line("each claim cited to the section.", 0, 0, 34)
    x = 420
    body = ('<rect width="1280" height="640" fill="url(#kyr-g)"/>'
            '<circle cx="1180" cy="-40" r="330" fill="#ffffff" opacity=".05"/>'
            '<circle cx="1260" cy="700" r="260" fill="#ffffff" opacity=".04"/>'
            + mark_body(110, 190, 260).replace('fill="url(#kyr-g)"', 'fill="#0b5448"')
            + f'<g transform="translate({x} 300)" fill="#ffffff">{know}</g>'
            f'<g transform="translate({x} 300)" fill="{SAFFRON}">{right}</g>'
            f'<g transform="translate({x + 3} 372)" fill="#d7eee7">{tag}</g>'
            f'<g transform="translate({x + 3} 418)" fill="#d7eee7">{tag2}</g>'
            f'<rect x="{x + 3}" y="462" width="120" height="6" rx="3" fill="{SAFFRON}"/>')
    return svg(1280, 640, body, "KnowYourRights")


async def render_pngs(jobs: list[tuple[Path, Path, int, int]]) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for source, target, w, h in jobs:
            page = await browser.new_page(viewport={"width": w, "height": h})
            data = base64.b64encode(source.read_bytes()).decode()
            await page.set_content(f'<body style="margin:0"><img width="{w}" height="{h}" '
                                   f'src="data:image/svg+xml;base64,{data}" '
                                   'style="display:block"></body>')
            await page.wait_for_timeout(150)
            await page.screenshot(path=str(target), omit_background=True)
            print(f"wrote {target.relative_to(ROOT)}")
        await browser.close()


def main(font_path: str) -> None:
    bold, regular = Type(Path(font_path), 700), Type(Path(font_path), 400)
    BRAND.mkdir(parents=True, exist_ok=True)
    files = {
        "mark.svg": svg(512, 512, mark_body(), "KnowYourRights"),
        "favicon.svg": svg(512, 512, favicon_body(), "KnowYourRights"),
        "logo-light.svg": lockup(bold, regular, dark=False),
        "logo-dark.svg": lockup(bold, regular, dark=True),
        "social.svg": social(bold, regular),
    }
    for name, content in files.items():
        (BRAND / name).write_text(content, encoding="utf-8")
        print(f"wrote docs/brand/{name}")
    for name in ("mark.svg", "favicon.svg"):
        shutil.copyfile(BRAND / name, WEB / name)
        print(f"wrote knowyourrights/web/{name}")
    asyncio.run(render_pngs([
        (BRAND / "social.svg", BRAND / "social-preview.png", 1280, 640),
        (BRAND / "mark.svg", WEB / "apple-touch-icon.png", 180, 180),
        (BRAND / "favicon.svg", WEB / "favicon-32.png", 32, 32),
    ]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
