"""Optimal contiguous partition of sorted lengths into batches (docs/optimization.md §2).

For any batch cost that depends on a batch only through its size b and its longest response
Lmax and is nondecreasing in Lmax, an optimal partition into batches of at most ``cap``
samples is contiguous in sorted order (exchange argument), so the dynamic program

    F[0] = 0,  F[j] = min over i in [max(1, j - cap + 1), j] of F[i-1] + cost(j - i + 1, L_j)

over the ascending lengths L_1 <= ... <= L_n is exact. O(n * cap) cost evaluations.
"""

from __future__ import annotations

from collections.abc import Callable


def dp_partition(
    lengths_asc: list[int], cap: int, cost: Callable[[int, int, int, int], int | float]
) -> tuple[float, list[tuple[int, int]]]:
    """Return (total cost, segments [(i, j)] 0-based inclusive) of the optimal partition.

    ``cost(b, lmax, i, j)`` gets the segment bounds too, so a caller can fold in segment
    properties (for example the longest prompt); the optimality claim holds when the cost
    depends on (b, lmax) only and is nondecreasing in lmax.
    """
    n = len(lengths_asc)
    if any(lengths_asc[k] > lengths_asc[k + 1] for k in range(n - 1)):
        raise ValueError("lengths must be sorted ascending")
    INF = float("inf")
    F = [0.0] + [INF] * n
    arg = [0] * (n + 1)
    for j in range(1, n + 1):
        lmax = lengths_asc[j - 1]
        best, bi = INF, -1
        for i in range(max(1, j - cap + 1), j + 1):
            c = F[i - 1] + cost(j - i + 1, lmax, i - 1, j - 1)
            if c < best:  # strict: ties keep the largest batch ending at j
                best, bi = c, i
        F[j] = best
        arg[j] = bi
    segs = []
    j = n
    while j > 0:
        i = arg[j]
        segs.append((i - 1, j - 1))
        j = i - 1
    segs.reverse()
    return F[n], segs
