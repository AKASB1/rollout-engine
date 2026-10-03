"""Decision-cost microbenchmarks: one policy invocation with 64, 512, and 4096 pending
samples, and DP / MILP solve times. Wall-clock numbers on a shared machine: the minimum of
at least five repetitions is reported, with process CPU time beside it; they are never used
in a deterministic comparison."""

from __future__ import annotations

import copy
import time

from rollout_engine.api import specs_from_trace
from rollout_engine.batching import batch_cost_ms
from rollout_engine.config import derive_system, parse_system
from rollout_engine.opt.dp import dp_partition
from rollout_engine.opt.milp import solve_makespan
from rollout_engine.opt.s7 import cost_fn, instance
from rollout_engine.policies.composed import make_policy
from rollout_engine.scheduler.core import Core
from rollout_engine.trace.generator import DEFAULT_GENERATOR, generate

PARTS = {
    "reference (fifo+group+early+round_robin)": {"name": "reference"},
    "fifo+sample+late+first_free": {
        "name": "composed",
        "params": {"dispatch": "sample", "binding": "late", "assign": "first_free"},
    },
    "longest_first+sample+late+lpt": {"name": "lpt"},
    "fifo+group+early+least_loaded": {"name": "composed", "params": {"assign": "least_loaded"}},
    "online_adaptive": {"name": "online_adaptive"},
    "static dp": {"name": "dp"},
}


def _setup(pending: int, cfg: dict, engine: str):
    groups = pending // 8
    gen = copy.deepcopy(DEFAULT_GENERATOR)
    gen["groups"] = groups + 8
    tr = generate(gen, 1)
    raw = derive_system(groups_per_step=groups, engine=engine, mode="single_phase", steps=1)
    sysc = parse_system(raw)
    core = Core(sysc, specs_from_trace(tr)[:groups])
    pol = make_policy(copy.deepcopy(cfg))
    pol.bind(sysc, tr)
    pol.p["window_groups"] = groups
    return core, pol


def time_invocation(pending: int, cfg: dict, reps: int = 5) -> dict:
    engine = "static" if cfg.get("name") == "dp" else "continuous"
    walls, cpus, n_actions = [], [], 0
    for _ in range(reps):
        core, pol = _setup(pending, cfg, engine)
        v = core.view_for(0)
        t0, c0 = time.perf_counter(), time.process_time()
        acts = pol.act(v)
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
        n_actions = len(acts)
    return {
        "wall_min_ms": min(walls) * 1e3,
        "wall_cpu_min_ms": min(cpus) * 1e3,
        "actions": n_actions,
    }


def run(log=print) -> list[dict]:
    rows = []
    for name, cfg in PARTS.items():
        for pending in (64, 512, 4096):
            r = time_invocation(pending, cfg)
            rows.append({"what": "policy_invocation", "part": name, "size": pending, **r})
            log(f"  {name:45s} {pending:5d} pending: {r['wall_min_ms']:.2f} ms wall (min of 5)")
    p = parse_system(derive_system(engine="static")).engine
    for n in (64, 512, 4096):
        L = list(range(1, n + 1))
        walls, cpus = [], []
        for _ in range(5):
            t0, c0 = time.perf_counter(), time.process_time()
            dp_partition(L, 16, lambda b, lm, i, j: batch_cost_ms(p, b, 512, lm))
            walls.append(time.perf_counter() - t0)
            cpus.append(time.process_time() - c0)
        rows.append(
            {
                "what": "dp_solve",
                "part": "dp cap 16",
                "size": n,
                "wall_min_ms": min(walls) * 1e3,
                "wall_cpu_min_ms": min(cpus) * 1e3,
                "actions": 0,
            }
        )
    for n in (16, 32, 64):
        sysc, tr, _ = instance(8100 + n, n=n, workers=3, cap=6)
        L = sorted(g.resp_tokens[0] for g in tr.groups)
        walls, cpus, st = [], [], ""
        for _ in range(5):
            r = solve_makespan(L, 3, 6, cost_fn(sysc))
            walls.append(r.wall_s)
            cpus.append(r.cpu_s)
            st = r.status
        rows.append(
            {
                "what": "milp_solve",
                "part": f"milp W3 cap6 ({st})",
                "size": n,
                "wall_min_ms": min(walls) * 1e3,
                "wall_cpu_min_ms": min(cpus) * 1e3,
                "actions": 0,
            }
        )
    return rows
