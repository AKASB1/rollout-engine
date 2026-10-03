"""Small builders shared by the tests."""

from rollout_engine.config import parse_system
from rollout_engine.trace.schema import GroupRow, Trace

MS = 1_000_000


def tiny_system(
    *,
    eta=0,
    T=2,
    B=2,
    partition="disaggregated",
    F=10,
    S=1,
    W=0,
    switch=2,
    inflight="drain",
    warm=0,
    servers=4,
    engine="continuous",
    workers=2,
    mode="closed",
    **rollout,
):
    gpus = (
        {"total": workers + 1, "partition": "disaggregated", "rollout": workers, "train": 1}
        if partition == "disaggregated"
        else {"total": workers, "partition": "colocated", "switch_ms": switch}
    )
    r = {
        "engine": engine,
        "tp": 1,
        "a0_ns": MS,
        "a1_ns": 0,
        "a2_ns": 0,
        "prefill_c0_ns": 0,
        "prefill_c1_ns": 0,
        "max_seqs": 64,
        "kv_tokens": 10**9,
        "static_batch": 8,
    }
    r.update(rollout)
    return parse_system(
        {
            "schema_version": 1,
            "gpus": gpus,
            "rollout": r,
            "verifier": {
                "servers": servers,
                "tasks": {"math": {"verify_mean_s": 0.0}, "code": {"verify_mean_s": 1.0}},
            },
            "trainer": {"fixed_ms": F, "ns_per_token": 0},
            "sync": {"sync_ms": S, "swap_ms": W, "inflight": inflight},
            "loop": {
                "mode": mode,
                "steps": T,
                "groups_per_step": B,
                "eta": eta,
                "warmup_steps": warm,
            },
        }
    )


def tiny_trace(lengths, verify_ms=None, n=1, prompt=1, max_tokens=100, est=10.0, task="math"):
    """One group per entry of ``lengths`` (an int, or a list for several samples)."""
    groups = []
    for i, L in enumerate(lengths):
        ls = list(L) if isinstance(L, (list, tuple)) else [L] * n
        vs = verify_ms[i] if verify_ms is not None else 0
        vs = list(vs) if isinstance(vs, (list, tuple)) else [vs] * len(ls)
        groups.append(GroupRow(f"g{i}", task, prompt, max_tokens, est, tuple(ls), tuple(vs)))
    return Trace(tuple(groups))
