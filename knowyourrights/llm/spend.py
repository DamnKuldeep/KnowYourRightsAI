"""Attributing every billed call to the turn that caused it.

Chat, embedding and reranking calls all report what the provider billed (``usage.cost``). Those
figures used to be summed into process-wide totals, and a turn's cost was read as the change in
the total between its start and end. With two people asking at once each saw the other's spend
in their own figure, and nothing could say which client had spent what.

A :class:`SpendMeter` is bound to the running turn through a context variable. asyncio copies
the context into every task a turn creates, so calls made deep inside research steps, and the
background summary a turn schedules, all charge the right meter without passing it around.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class SpendMeter:
    """Dollars and calls charged to one turn. ``on_charge`` forwards each charge elsewhere."""

    cost_usd: float = 0.0
    calls: int = 0
    on_charge: Callable[[float], None] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, cost_usd: float, *, call: bool = True) -> None:
        cost = max(0.0, float(cost_usd or 0.0))
        with self._lock:
            self.cost_usd += cost
            self.calls += 1 if call else 0
        if cost and self.on_charge is not None:
            self.on_charge(cost)


_current: ContextVar[SpendMeter | None] = ContextVar("kyr_spend_meter", default=None)


def charge(cost_usd: float, *, call: bool = True) -> None:
    """Charge the meter of the turn this code is running in, if there is one."""
    meter = _current.get()
    if meter is not None:
        meter.add(cost_usd, call=call)


@contextmanager
def metered(meter: SpendMeter) -> Iterator[SpendMeter]:
    """Charge everything inside this block, and every task it starts, to ``meter``."""
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)
