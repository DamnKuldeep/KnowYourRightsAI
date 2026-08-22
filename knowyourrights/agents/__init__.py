"""Small single-purpose agents, coordinated by plain Python.

The orchestration is code, not an agent loop. The model emits a *validated plan* and code
executes it, so there is no such thing as a hallucinated tool call — and, just as importantly,
no way for text fetched off the web to trigger one.
"""

from .schemas import (
    Coverage, Gap, Grades, Plan, Procedure, ResearchStep, SourceGrade, SubQuestion,
)

__all__ = ["Plan", "ResearchStep", "SubQuestion", "Grades", "SourceGrade",
           "Gap", "Coverage", "Procedure"]
