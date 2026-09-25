"""The research pipeline for one question: plan, research, write, verify, commit.

``core`` runs a turn, ``research`` gathers and grades evidence, ``writer`` produces the answer,
``verify`` checks it in deep mode, and ``turn`` holds the state they share.
"""

from .core import Orchestrator, get_orchestrator
from .turn import TurnBudget, TurnState

__all__ = ["Orchestrator", "TurnBudget", "TurnState", "get_orchestrator"]
