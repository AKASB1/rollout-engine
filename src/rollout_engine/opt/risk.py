"""Experiment S9 (Tier 2): risk-aware static batching on the S7 instances.

Plans on the estimates are replayed on the true lengths in the simulator (static engine):

- ``det``: the makespan MILP on the estimated lengths (segments of the estimate order);
- ``cvar(lam)``: the scenario MILP (expected makespan + lam * CVaR_0.9) over S = 16 scenarios
  drawn from the planner's belief (estimate times a log-normal error of level 0.5, the level
  the S7 generator uses), ``lam`` tuned on the tuning instances;
- ``dp_policy``: the online ``dp`` policy (re-plans at every idle worker), for context;
- ``optimum``: the MILP on the true lengths (a clairvoyant reference).

The result is the realized makespan relative to the optimum.
"""

from __future__ import annotations

import math

from rollout_engine.opt.milp import solve_makespan
from rollout_engine.opt.s7 import PlanPolicy, cost_fn, gen_time_ms, instance
from rollout_engine.opt.scenario import draw_scenarios, solve_cvar
from rollout_engine.policies.composed import make_policy
from rollout_engine.sim.driver import simulate

BELIEF_SIGMA = 0.5
SCENARIOS = 16
ALPHA = 0.9
LAMS = (0.0, 0.5, 1.0, 2.0)


def _replay(system, trace, plan_segs, order):
    plan = [[[order[k] for k in range(i, j + 1)] for i, j in segs] for segs in plan_segs]
    return gen_time_ms(simulate(system, trace, PlanPolicy(plan)))


def evaluate(seed: int, lams=LAMS) -> dict:
    system, trace, (n, W, cap) = instance(seed)
    cost = cost_fn(system)
    L = [g.resp_tokens[0] for g in trace.groups]
    est = [g.est_tokens for g in trace.groups]
    mt = trace.groups[0].max_tokens
    row = {"seed": seed, "n": n, "W": W, "cap": cap}
    # clairvoyant optimum on the true lengths
    o_true = sorted(range(n), key=lambda k: (L[k], k))
    opt = solve_makespan([L[k] for k in o_true], W, cap, cost)
    row["optimum_ms"] = round(opt.objective) if opt.objective is not None else None
    row["optimum_status"] = opt.status
    # plans in the estimate order
    o_est = sorted(range(n), key=lambda k: (est[k], k))
    Le = [max(1, min(mt, math.ceil(est[k]))) for k in o_est]
    det = solve_makespan(Le, W, cap, cost)
    row["det_ms"] = _replay(system, trace, det.plan, o_est)
    row["det_status"] = det.status
    scen = draw_scenarios([est[k] for k in o_est], mt, BELIEF_SIGMA, SCENARIOS, seed)
    row["wall_solve_det_s"] = det.wall_s
    for lam in lams:
        r = solve_cvar(scen, W, cap, cost, alpha=ALPHA, lam=lam)
        row[f"cvar{lam:g}_ms"] = _replay(system, trace, r.plan, o_est)
        row[f"cvar{lam:g}_status"] = r.status
        row[f"wall_solve_cvar{lam:g}_s"] = r.wall_s
    row["dp_policy_ms"] = gen_time_ms(simulate(system, trace, make_policy({"name": "dp"})))
    if opt.status == "optimal":
        for k in [x for x in row if x.endswith("_ms") and x != "optimum_ms"]:
            row[k.replace("_ms", "_ratio")] = row[k] / row["optimum_ms"]
    return row
