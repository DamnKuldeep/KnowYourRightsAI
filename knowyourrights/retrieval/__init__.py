"""Retrieval over the LanceDB corpus of Indian central law.

Hybrid dense + BM25 -> reciprocal rank fusion -> one row per section -> rerank -> MMR, following
the contract in ``data/KnowYourRights_DB_README.md`` §11.
"""
