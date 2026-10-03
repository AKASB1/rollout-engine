"""The static-engine batch cost model shared by the engine, the policies, and the exact
references (docs/optimization.md section 1)."""

from __future__ import annotations

from rollout_engine.clock import ceil_ms
from rollout_engine.config import EngineParams


def batch_cost_ns(p: EngineParams, b: int, pmax: int, lmax: int) -> tuple[int, int]:
    """(prefill ns, decode ns) of a static batch: one prefill of the padded prompts, then
    lmax iterations carrying all b samples; iteration j (0-based) takes
    a0 + a1*b + a2*b*(pmax + j)."""
    prefill = p.prefill_c0_ns + p.prefill_c1_ns * b * pmax
    decode = (
        lmax * (p.a0_ns + p.a1_ns * b + p.a2_ns * b * pmax) + p.a2_ns * b * lmax * (lmax - 1) // 2
    )
    return prefill, decode


def batch_cost_ms(p: EngineParams, b: int, pmax: int, lmax: int) -> int:
    """The batch duration the static engine produces: ceil_ms(prefill) + ceil_ms(decode).
    Nondecreasing in b and in lmax."""
    pre, dec = batch_cost_ns(p, b, pmax, lmax)
    return ceil_ms(pre) + ceil_ms(dec)
