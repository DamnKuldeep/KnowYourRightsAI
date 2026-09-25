"""What can go wrong when calling a model, as the rest of the system needs to see it.

The distinction that matters to a caller is whether to degrade or to stop:

* :class:`DeadlineExceeded` — the turn ran out of time. Stop gathering, answer with what exists.
* :class:`ModelUnavailable` — no model for the role could be reached. Skip or degrade the stage.
* :class:`ProviderAuthError` — a provider rejected its API key. Nothing will work until the
  operator fixes the configuration, so the turn stops and says so.
"""

from __future__ import annotations


class LLMError(RuntimeError):
    """Any failure to get an answer from a model."""


class DeadlineExceeded(LLMError):
    """The turn ran out of time while waiting for a provider."""


class ModelUnavailable(LLMError):
    """No candidate model for the role could be reached."""


class ProviderAuthError(LLMError):
    """A provider rejected the configured API key. An operator problem, not a transient one."""
