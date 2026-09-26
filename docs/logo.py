"""Builds the KnowYourRights brand assets from glyph outlines.

    python docs/logo.py path/to/LibreBaskerville[wght].ttf

The mark is a speech bubble (a plain-language answer) holding a § (the section it rests on), with
a saffron bookmark ribbon (the citation that marks the place). The favicon keeps the § and the
ribbon without the bubble, which blurs at 16 px.

Letters are drawn as paths from Libre Baskerville (SIL Open Font License,
github.com/google/fonts/tree/main/ofl/librebaskerville), so the files look the same everywhere and
need no font. PNGs are rendered with Playwright, already a dependency of the app.
"""

from __future__ import annotations

import asyncio
import base64
import shutil
import sys
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
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

    def glyph_centered(self, ch: str, cx: float, cy: float, height: float) -> str:
        """One character scaled so its ink is `height` tall, centred on (cx, cy)."""
        name = self._name(ch)
        bounds = BoundsPen(self.glyphs)
        self.glyphs[name].draw(bounds)
        x0, y0, x1, y1 = bounds.bounds
        s = height / (y1 - y0)
        return self._path(name, (s, 0, 0, -s, cx - (x0 + x1) / 2 * s, cy + (y0 + y1) / 2 * s))

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


def mark_body(bold: Type, x=0, y=0, size=512) -> str:
    """The app icon, drawn on a 512 grid and placed at (x, y) with the given size."""
    k = size / 512
    return (f'<g transform="translate({x} {y}) scale({k})">'
            '<rect width="512" height="512" rx="116" fill="url(#kyr-g)"/>'
            '<path fill="#fff" d="M164 92h184a84 84 0 0 1 84 84v120a84 84 0 0 1-84 84H240l-92 70 '
            '16-72a84 84 0 0 1-80-84V176a84 84 0 0 1 84-84z"/>'
            f'<path fill="{SAFFRON}" d="M352 92h44v96l-22-18-22 18z"/>'
            f'<g fill="{GREEN}">{bold.glyph_centered("§", 248, 236, 200)}</g></g>')


def favicon_body(bold: Type) -> str:
    return ('<rect width="512" height="512" rx="116" fill="url(#kyr-g)"/>'
            f'<path fill="{SAFFRON}" d="M356 0h64v150l-32-26-32 26z"/>'
            f'<g fill="#fff">{bold.glyph_centered("§", 236, 262, 340)}</g>')


def lockup(bold: Type, regular: Type, dark: bool) -> str:
    """Mark + wordmark + tagline, for the README header."""
    ink, dim = (INK_DARK, DIM_DARK) if dark else (INK, DIM)
    rights = "#5cc6ab" if dark else GREEN
    know, w1 = regular.line("KnowYour", 150, 84, 58)
    right, w2 = bold.line("Rights", 150 + w1, 84, 58)
    tag, w3 = regular.line("INDIAN LAW, CITED TO THE SECTION", 153, 124, 17.5, tracking=0.16)
    width = int(150 + max(w1 + w2, w3) + 12)
    body = (mark_body(bold, 0, 8, 124) + f'<g fill="{ink}">{know}</g>'
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
            + mark_body(bold, 110, 190, 260).replace('fill="url(#kyr-g)"', 'fill="#0b5448"')
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
        "mark.svg": svg(512, 512, mark_body(bold), "KnowYourRights"),
        "favicon.svg": svg(512, 512, favicon_body(bold), "KnowYourRights"),
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
