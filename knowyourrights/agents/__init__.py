"""Small single-purpose model stages, coordinated by plain Python.

The orchestration is code, not an agent loop. A model emits a validated plan and code executes
it, so there is no such thing as a hallucinated tool call, and no way for text fetched off the
web to trigger one.
"""
