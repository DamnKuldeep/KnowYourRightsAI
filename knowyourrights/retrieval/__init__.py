"""Retrieval over the LanceDB corpus of Indian central law.

Contract preserved from ``data/KnowYourRights_DB_README.md`` §11:
hybrid dense+BM25 -> RRF -> de-duplicate to one row per section -> cross-encoder rerank -> MMR.
"""
