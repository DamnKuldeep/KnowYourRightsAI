"""An answer assembled without a model, for when the writer cannot be reached.

It quotes rather than paraphrases: with no model there is nothing to do the summarising, and a
verbatim provision with its citation is still a real answer.
"""

from __future__ import annotations

from ..evidence import Evidence

MAX_STATUTES = 4
MAX_OTHERS = 4
EXCERPT_CHARS = 420


def source_digest(items: list[Evidence], question: str) -> str:
    if not items:
        return ("I could not reach the writing model, and no sources were found for this "
                "question. Please try again in a moment.")
    lines = [f"**Provisions found for:** {question}", ""]
    for item in [i for i in items if i.is_statute][:MAX_STATUTES]:
        lines += [f"**{item.label()}**{_flags(item)}",
                  f"> {' '.join(item.text.split())[:EXCERPT_CHARS]}…", ""]
    others = [i for i in items if not i.is_statute][:MAX_OTHERS]
    if others:
        lines.append("**Also found:**")
        lines += [f"- [{i.title[:70]}]({i.url})" if i.url else f"- {i.title[:70]}"
                  for i in others]
        lines.append("")
    lines.append("_This is the raw statutory text, not an explanation — the model that writes "
                 "the plain-language answer was unavailable. Try again shortly._")
    return "\n".join(lines)


def _flags(item: Evidence) -> str:
    flags = []
    if item.state:
        flags.append(f"applies only in {item.state}")
    if item.is_omitted:
        flags.append("this provision has been omitted")
    return f"  _({'; '.join(flags)})_" if flags else ""
