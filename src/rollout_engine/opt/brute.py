"""Brute force over set partitions and worker assignments (small instances, check 2)."""

from __future__ import annotations

from collections.abc import Callable, Iterator


def set_partitions(n: int, cap: int) -> Iterator[list[list[int]]]:
    """All partitions of range(n) into blocks of at most ``cap`` elements."""
    blocks: list[list[int]] = []

    def rec(k: int):
        if k == n:
            yield [list(b) for b in blocks]
            return
        for b in blocks:
            if len(b) < cap:
                b.append(k)
                yield from rec(k + 1)
                b.pop()
        blocks.append([k])
        yield from rec(k + 1)
        blocks.pop()

    yield from rec(0)


def best_total(lengths: list[int], cap: int, cost: Callable[[int, int], float]) -> float:
    return min(
        sum(cost(len(b), max(lengths[i] for i in b)) for b in p)
        for p in set_partitions(len(lengths), cap)
    )


def best_assignment(costs: list[float], workers: int) -> float:
    """Optimal makespan of fixed jobs on identical workers (exhaustive, with pruning)."""
    order = sorted(costs, reverse=True)
    best = [sum(order)]
    loads = [0.0] * workers

    def rec(k: int):
        if k == len(order):
            best[0] = min(best[0], max(loads))
            return
        seen = []
        for w in range(workers):
            if loads[w] in seen:
                continue  # identical workers with the same load give the same subtree
            seen.append(loads[w])
            if loads[w] + order[k] >= best[0]:
                continue
            loads[w] += order[k]
            rec(k + 1)
            loads[w] -= order[k]

    rec(0)
    return best[0]


def best_makespan(
    lengths: list[int], workers: int, cap: int, cost: Callable[[int, int], float]
) -> float:
    return min(
        best_assignment([cost(len(b), max(lengths[i] for i in b)) for b in p], workers)
        for p in set_partitions(len(lengths), cap)
    )
