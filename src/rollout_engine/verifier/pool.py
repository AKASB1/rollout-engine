"""The verifier pool: one queue, ``servers`` identical servers, an order hook.

The pool knows nothing about service times (they are hidden in the trace): a driver tells it
when a verification ends. Whenever a server is free and the queue is not empty, the order
hook picks the queued entry it serves next; free servers are filled in id order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class QEntry:
    sid: int
    gidx: int
    task: str
    arrived: int  # ms
    seq: int  # arrival order (ties)


class VerifierPool:
    def __init__(self, servers: int):
        self.n = servers
        self.serving: list[int | None] = [None] * servers  # sid in service per server
        self.started: list[int] = [0] * servers
        self.queue: list[QEntry] = []
        self._seq = 0

    def enqueue(self, sid: int, gidx: int, task: str, t: int) -> None:
        self._seq += 1
        self.queue.append(QEntry(sid, gidx, task, t, self._seq))

    def free_servers(self) -> list[int]:
        return [i for i, s in enumerate(self.serving) if s is None]

    def assign(self, t: int, pick: Callable[[list[QEntry]], int]) -> list[tuple[int, int]]:
        """Fill free servers (id order); ``pick(queue)`` returns the index served next."""
        out = []
        for i in range(self.n):
            if not self.queue:
                break
            if self.serving[i] is not None:
                continue
            k = pick(self.queue)
            if not (0 <= k < len(self.queue)):
                raise IndexError(f"verifier order returned {k} for a queue of {len(self.queue)}")
            e = self.queue.pop(k)
            self.serving[i] = e.sid
            self.started[i] = t
            out.append((i, e.sid))
        return out

    def complete(self, server: int) -> int:
        sid = self.serving[server]
        if sid is None:
            raise RuntimeError(f"verifier server {server} is idle")
        self.serving[server] = None
        return sid

    def remove_group(self, gidx: int) -> None:
        if any(e.gidx == gidx for e in self.queue):
            self.queue = [e for e in self.queue if e.gidx != gidx]

    def busy(self) -> int:
        return sum(1 for s in self.serving if s is not None)
