"""Lower bounds on the generation time of a single phase (docs/simulator.md section 7).

Exact rational arithmetic (fractions.Fraction) in nanoseconds; compare against the simulated
generation time in milliseconds with ``gen_time_ms * 10**6 >= bound``.
"""

from __future__ import annotations

from fractions import Fraction

from rollout_engine.config import EngineParams


def sample_work_ns(p: EngineParams, P: int, L: int) -> Fraction:
    """Lower bound on the GPU time a sample of prompt P and length L needs on one worker,
    excluding the prefill: L iterations, each sharing a0 among at most max_seqs samples."""
    return Fraction(L * p.a0_ns, p.max_seqs) + L * p.a1_ns + p.a2_ns * (L * P + L * (L - 1) // 2)


def work_bound_ns(p: EngineParams, groups: list[tuple[int, list[int]]], workers: int) -> Fraction:
    """(sum over samples of sample work + sum over groups of one prefill) / W.

    ``groups`` lists (prompt_tokens, [response lengths]) per group."""
    total = Fraction(0)
    for P, lengths in groups:
        total += p.prefill_c0_ns + p.prefill_c1_ns * P
        for L in lengths:
            total += sample_work_ns(p, P, L)
    return total / workers


def alone_time_ns(p: EngineParams, P: int, L: int) -> int:
    """A sample run alone (b = 1), its prefill included."""
    return (
        p.prefill_c0_ns
        + p.prefill_c1_ns * P
        + L * (p.a0_ns + p.a1_ns + p.a2_ns * P)
        + p.a2_ns * L * (L - 1) // 2
    )


def longest_sample_bound_ns(p: EngineParams, groups: list[tuple[int, list[int]]]) -> int:
    return max((alone_time_ns(p, P, L) for P, lengths in groups for L in lengths), default=0)


def continuous_lower_bound_ns(p: EngineParams, groups, workers: int) -> Fraction:
    return max(work_bound_ns(p, groups, workers), Fraction(longest_sample_bound_ns(p, groups)))
