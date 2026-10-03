"""Records and interfaces shared by the simulator, the policies, and the live service.

Nothing here imports the simulator, asyncio, or http.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """What the scheduler may know about a group before any sample of it finishes."""

    gid: str
    task: str
    prompt_tokens: int
    max_tokens: int
    n_samples: int
    est_tokens: float


@dataclass(frozen=True, slots=True)
class Place:
    """Place a unit on a worker. ``kind`` is ``group`` (``ref`` = group index: its unplaced
    samples), ``sample`` (``ref`` = sample id), or ``batch`` (``ref`` = tuple of sample ids,
    static engine only)."""

    kind: str
    ref: int | tuple[int, ...]
    worker: int


@dataclass(frozen=True, slots=True)
class Drop:
    gidx: int


class InvalidAction(RuntimeError):
    """A policy returned an action the engine rejects; the run aborts (never skipped)."""

    def __init__(self, policy: str, action, reason: str):
        super().__init__(f"policy {policy!r}: invalid action {action!r}: {reason}")
        self.policy = policy
        self.action = action
        self.reason = reason


class RunError(RuntimeError):
    """A run cannot continue (deadlock, running out of stream); names the policy."""


def specs_from_trace(trace) -> list[GroupSpec]:
    return [
        GroupSpec(g.group_id, g.task, g.prompt_tokens, g.max_tokens, g.n_samples, g.est_tokens)
        for g in trace.groups
    ]
