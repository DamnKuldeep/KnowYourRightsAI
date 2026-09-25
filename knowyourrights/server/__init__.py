"""The web server. Run it with ``python -m knowyourrights.server``."""

from .api import app, main

__all__ = ["app", "main"]
