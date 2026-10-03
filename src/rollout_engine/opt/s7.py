"""Experiment S7: exact references on small static instances (docs/optimization.md §6).

Each instance is a single-phase trace of ``n`` one-sample groups with one prompt length on
the static engine with ``W`` workers and batch cap ``cap``. The MILP and the DP use the
true lengths (clairvoyant references); the policies run in the simulator.
"""

from __future__ import annotations

import copy
import math

from rollout_engine.api import Place
from rollout_engine.batching import batch_cost_ms
from rollout_engine.config import deep_merge, derive_system, parse_system
from rollout_engine.opt.dp import dp_partition
from rollout_engine.opt.milp import solve_makespan
from rollout_engine.policies.composed import make_policy
from rollout_engine.rng import stream
from rollout_engine.sim.driver import simulate
from rollout_engine.trace.generator import DEFAULT_GENERATOR, generate

PROMPT = 512


def instance(seed: int, n: int | None = None, workers: int | None = None, cap: int | None = None):
    """(system, trace, (n, W, cap)) of instance ``seed`` (sizes drawn from the seed unless
    given)."""
    rng = stream(seed, "s7.instance")
    n = n or int(rng.integers(8, 17))
    workers = workers or int(rng.choice([2, 3]))
    cap = cap or int(rng.integers(4, 7))
    gen = copy.deepcopy(DEFAULT_GENERATOR)
    gen.update({"groups": n, "n_samples": 1, "prompt_fixed": PROMPT})
    gen["estimate"] = {"model": "group", "error_sigma": 0.5}
    gen["tasks"] = {"math": dict(DEFAULT_GENERATOR["tasks"]["math"], share=1.0, len_rho=1.0)}
    trace = generate(gen, seed)
    raw = derive_system(
        total_gpus=workers + 1,
        rollout_gpus=workers,
        engine="static",
        mode="single_phase",
        steps=1,
        groups_per_step=n,
        verifier_servers=64,
        verify_mean_s={"math": 0.2},
    )
    raw = deep_merge(raw, {"rollout": {"static_batch": cap}})
    return parse_system(raw), trace, (n, workers, cap)


def cost_fn(system):
    p = system.engine
    return lambda b, lmax: batch_cost_ms(p, b, PROMPT, lmax)


class PlanPolicy:
    """Replays a fixed per-worker batch plan (lists of sample ids) on the static engine."""

    name = "milp_plan"

    def __init__(self, plan: list[list[list[int]]]):
        self.plan = [list(x) for x in plan]

    def act(self, v):
        out = []
        for w in v.workers():
            if w.accepting and self.plan[w.wid]:
                out.append(Place("batch", tuple(self.plan[w.wid].pop(0)), w.wid))
        return out

    def pick_verification(self, v, q):
        return 0


def gen_time_ms(sim) -> int:
    return max(s.gen_t for s in sim.core.samples)


POLICIES = {
    "fifo_chunk": {"name": "composed", "params": {"batch": "fifo_chunk"}},
    "sorted_chunk": {"name": "composed", "params": {"batch": "sorted_chunk"}},
    "sorted_chunk_oracle": {
        "name": "composed",
        "params": {"batch": "sorted_chunk", "estimator": "oracle"},
    },
    "dp": {"name": "dp"},
    "oracle_dp": {"name": "oracle_dp"},
}


def evaluate(seed: int, **limits) -> dict:
    system, trace, (n, W, cap) = instance(seed)
    L = [g.resp_tokens[0] for g in trace.groups]
    order = sorted(range(n), key=lambda k: (L[k], k))
    Ls = [L[k] for k in order]
    cost = cost_fn(system)
    dp_total, _ = dp_partition(Ls, cap, lambda b, lm, i, j: cost(b, lm))
    lb = max(math.ceil(dp_total / W), cost(1, Ls[-1]))
    res = solve_makespan(Ls, W, cap, cost, **limits)
    row = {
        "seed": seed,
        "n": n,
        "W": W,
        "cap": cap,
        "milp_status": res.status,
        # costs are integer ms, so the optimum is an integer; HiGHS returns it with float noise
        "milp_ms": round(res.objective) if res.objective is not None else None,
        "milp_raw": res.objective,
        "milp_bound_ms": res.bound,
        "milp_gap": res.gap,
        "dp_total_ms": dp_total,
        "lower_bound_ms": lb,
        "wall_solve_s": res.wall_s,
        "wall_solve_cpu_s": res.cpu_s,
    }
    # replay the MILP plan on the static engine
    plan = [[[order[k] for k in range(i, j + 1)] for i, j in segs] for segs in res.plan]
    if res.status == "optimal":
        sim = simulate(system, trace, PlanPolicy(plan))
        row["milp_replay_ms"] = gen_time_ms(sim)
    for name, cfg in POLICIES.items():
        sim = simulate(system, trace, make_policy(copy.deepcopy(cfg)))
        g = gen_time_ms(sim)
        row[f"{name}_ms"] = g
        if res.status == "optimal":  # gaps only against a proven optimum
            row[f"{name}_gap"] = g / row["milp_ms"] - 1
    return row
