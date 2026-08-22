"""Terminal output that survives a Windows console.

The default Windows code page is cp1252, so the box-drawing and emoji this project prints
raise ``UnicodeEncodeError`` before anything useful reaches the user. Call
:func:`setup_console` first thing in any entry point.
"""

from __future__ import annotations

import os
import sys


def setup_console() -> None:
    """Force UTF-8 on stdout/stderr and enable ANSI escapes where possible."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if os.name == "nt":
        try:  # opt conhost into VT processing; Windows Terminal already does this
            import ctypes

            kernel32 = ctypes.windll.kernel32
            for handle in (-11, -12):  # stdout, stderr
                mode = ctypes.c_uint32()
                if kernel32.GetConsoleMode(kernel32.GetStdHandle(handle), ctypes.byref(mode)):
                    kernel32.SetConsoleMode(kernel32.GetStdHandle(handle), mode.value | 0x0004)
        except Exception:
            pass


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if supports_color() else text


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if supports_color() else text


def rule(title: str, width: int = 74) -> None:
    print(f"\n{bold(title)}\n" + "─" * width)
