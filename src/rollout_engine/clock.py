"""Time conventions.

Inside the simulator, time is an integer number of milliseconds since the start of the run
and every engine parameter is an integer number of nanoseconds. A duration becomes an event
time by rounding up to the next millisecond (``ceil_ms``), in integer arithmetic, so event
order never depends on float rounding. Files and reports show decimal seconds with at most
three decimals (``fmt_s``).

Policies never read a clock: they read time from the view. The live service reads a clock
object: ``VirtualClock`` in tests, ``WallClock`` otherwise.
"""

from __future__ import annotations

import time
from typing import Protocol

NS_PER_MS = 1_000_000


def ceil_ms(ns: int) -> int:
    """Round a non-negative integer duration in nanoseconds up to whole milliseconds."""
    return -(-ns // NS_PER_MS)


def fmt_s(ms: int) -> str:
    """Integer milliseconds as decimal seconds with three decimals (exact)."""
    sign = "-" if ms < 0 else ""
    ms = abs(ms)
    return f"{sign}{ms // 1000}.{ms % 1000:03d}"


def parse_s_to_ms(text: str) -> int:
    """Parse a non-negative decimal number of seconds with at most three decimals into ms.

    Exact (no float): ``"1.25"`` → 1250. Raises ValueError otherwise.
    """
    text = text.strip()
    if not text or text[0] in "+-":
        if text.startswith("+"):
            text = text[1:]
        else:
            raise ValueError(f"not a non-negative decimal: {text!r}")
    whole, dot, frac = text.partition(".")
    if dot and not frac:
        raise ValueError(f"not a decimal: {text!r}")
    if not whole:
        whole = "0"
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise ValueError(f"not a decimal: {text!r}")
    if len(frac) > 3:
        if frac[3:].strip("0"):
            raise ValueError(f"more than three decimals: {text!r}")
        frac = frac[:3]
    return int(whole) * 1000 + int((frac + "000")[:3])


class Clock(Protocol):
    def now_ms(self) -> int: ...


class VirtualClock:
    """A clock that only moves when told to (tests and the simulator)."""

    def __init__(self, start_ms: int = 0) -> None:
        self._now = start_ms

    def now_ms(self) -> int:
        return self._now

    def set(self, ms: int) -> None:
        if ms < self._now:
            raise ValueError("virtual time cannot go backwards")
        self._now = ms


class WallClock:
    """Monotonic wall clock in integer milliseconds since construction (live service)."""

    def __init__(self) -> None:
        self._t0 = time.monotonic_ns()

    def now_ms(self) -> int:
        return (time.monotonic_ns() - self._t0) // NS_PER_MS
