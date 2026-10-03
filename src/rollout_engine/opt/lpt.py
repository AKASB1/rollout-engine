"""List scheduling and LPT of a fixed batch plan on identical workers (Graham)."""

from __future__ import annotations

import heapq


def list_schedule(
    costs: list[int], workers: int, order: list[int] | None = None
) -> tuple[int, list[int]]:
    """Each batch, in ``order``, goes to the worker that becomes free first (ties: the
    lowest id). Returns (makespan, worker of each batch)."""
    order = list(range(len(costs))) if order is None else order
    heap = [(0, w) for w in range(workers)]
    assign = [-1] * len(costs)
    for k in order:
        load, w = heapq.heappop(heap)
        assign[k] = w
        heapq.heappush(heap, (load + costs[k], w))
    return max(load for load, _ in heap), assign


def lpt(costs: list[int], workers: int) -> tuple[int, list[int]]:
    order = sorted(range(len(costs)), key=lambda k: (-costs[k], k))
    return list_schedule(costs, workers, order)


def graham_list_bound(costs: list[int], workers: int) -> float:
    return (2 - 1 / workers) * max(sum(costs) / workers, max(costs, default=0))


def graham_lpt_factor(workers: int) -> float:
    return 4 / 3 - 1 / (3 * workers)
