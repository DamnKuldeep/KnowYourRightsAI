"""Admission control: a few answers at a time, everyone else in a fair queue.

Each answer holds several model calls, retrieval calls and possibly a browser for up to four
minutes, so running every request at once would slow them all and risk the provider's rate
limits. At most ``capacity`` turns run together. Others wait first-come, first-served and are
told their place in line as it changes. The queue itself is bounded, as is each wait and each
client's share, so a burst of traffic or one busy client cannot take the service over.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .. import config


class QueueFull(Exception):
    """The waiting line is at its limit."""


class ClientBusy(Exception):
    """This client already has its share of questions running or waiting."""


class QueueTimeout(Exception):
    """The wait for a free slot exceeded the limit."""


@dataclass(eq=False)
class Ticket:
    client: str
    admitted: bool = False
    released: bool = False


class AdmissionQueue:
    def __init__(self, capacity: int | None = None, max_waiting: int | None = None,
                 per_client: int | None = None) -> None:
        self.capacity = max(1, capacity or config.MAX_ACTIVE_TURNS)
        self.max_waiting = max(0, config.MAX_QUEUED_TURNS if max_waiting is None
                               else max_waiting)
        self.per_client = max(1, per_client or config.CLIENT_MAX_PENDING)
        self.active = 0
        self._waiting: deque[Ticket] = deque()
        self._by_client: Counter[str] = Counter()
        self._changed = asyncio.Condition()

    def join(self, client: str) -> Ticket:
        """Take a place in line, or raise if the line or this client's share is full."""
        if self._by_client[client] >= self.per_client:
            raise ClientBusy
        if len(self._waiting) >= self.max_waiting and self.active >= self.capacity:
            raise QueueFull
        ticket = Ticket(client)
        self._waiting.append(ticket)
        self._by_client[client] += 1
        return ticket

    async def wait(self, ticket: Ticket, timeout_s: float | None = None) -> AsyncIterator[int]:
        """Yield the ticket's place in line each time it changes; return once admitted.

        Nothing is yielded when a slot is free immediately. The ticket is released if the caller
        stops iterating early (the reader left) or the wait times out.
        """
        deadline = time.monotonic() + (config.QUEUE_TIMEOUT_S if timeout_s is None
                                       else timeout_s)
        last = None
        try:
            while True:
                position = await self._next_position(ticket, last, deadline)
                if position == 0:
                    return
                last = position
                yield position
        except BaseException:
            await self.release(ticket)
            raise

    async def _next_position(self, ticket: Ticket, last: int | None, deadline: float) -> int:
        """0 once admitted; otherwise the place in line, waiting until it differs from ``last``."""
        async with self._changed:
            while True:
                if self._admit_if_first(ticket):
                    return 0
                position = self._waiting.index(ticket) + 1
                if position != last:
                    return position
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueueTimeout
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=remaining)
                except TimeoutError:
                    raise QueueTimeout from None

    def _admit_if_first(self, ticket: Ticket) -> bool:
        if self.active < self.capacity and self._waiting and self._waiting[0] is ticket:
            self._waiting.popleft()
            self.active += 1
            ticket.admitted = True
            return True
        return False

    async def release(self, ticket: Ticket) -> None:
        """Give back the ticket's slot or place in line. Safe to call more than once."""
        if ticket.released:
            return
        ticket.released = True
        async with self._changed:
            if ticket.admitted:
                self.active -= 1
            elif ticket in self._waiting:
                self._waiting.remove(ticket)
            self._by_client[ticket.client] -= 1
            if self._by_client[ticket.client] <= 0:
                del self._by_client[ticket.client]
            self._changed.notify_all()

    def stats(self) -> dict:
        return {"active": self.active, "capacity": self.capacity,
                "waiting": len(self._waiting), "max_waiting": self.max_waiting}
