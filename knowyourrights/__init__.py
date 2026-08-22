"""KnowYourRights — a deep legal research agent over Indian central law.

The retrieval substrate (LanceDB + bge-m3 vectors) is built by ``01-building-database.ipynb``
and documented in ``data/KnowYourRights_DB_README.md``. This package is the answer layer:
plan -> research (multi-round) -> grade -> navigate -> write, with citations.
"""

__version__ = "0.1.0"
