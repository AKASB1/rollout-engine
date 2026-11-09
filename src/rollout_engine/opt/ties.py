"""Ranges of the realized makespan over model-optimal plans (S9, Tier 2).

The S9 planners (``risk.evaluate``) return one plan per model: the makespan MILP on the estimates
(``det``) or the scenario MILP with a CVaR term (``cvar(lam)``). Several plans can reach the same
model optimum and still end at different times on the true lengths, and which of them the solver
returns depends on its version. For a model with optimum ``f*``, ``realized_range`` solves two
more MILPs over the same plans with the constraint ``model objective <= f* + TOL_MS``: one
minimizes and one maximizes the realized makespan

    R = max_w  sum_seg  C_true[seg] y[seg, w],

where ``C_true[seg]`` is the batch cost of the segment at its longest true length. ``tie_row``
checks on every returned plan that this value equals the plan's replay in the simulator.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from rollout_engine.opt.milp import milp_status, segments, solve_makespan
from rollout_engine.opt.risk import ALPHA, BELIEF_SIGMA, LAMS, SCENARIOS, _replay
from rollout_engine.opt.s7 import cost_fn, instance
from rollout_engine.opt.scenario import draw_scenarios, solve_cvar

# model objectives are in ms on a lattice of 1/16 ms (16 scenarios, alpha 0.9), so distinct plans
# differ by far more than this; it only absorbs the solver's floating-point error
TOL_MS = 1e-3
TIME_LIMIT_S = 60.0


def plan_value(plan, costs, segs) -> float:
    """max over workers of the summed segment costs of a plan (per worker: segments (i, j))."""
    index = {seg: g for g, seg in enumerate(segs)}
    return max((sum(costs[index[seg]] for seg in work) for work in plan), default=0.0)


def realized_range(
    c_model: np.ndarray,
    c_true: np.ndarray,
    n: int,
    workers: int,
    cap: int,
    f_opt: float,
    lam: float | None = None,
    alpha: float = ALPHA,
    *,
    r_upper: float | None = None,
    tol: float = TOL_MS,
    time_limit: float = TIME_LIMIT_S,
) -> tuple[tuple, tuple]:
    """((min, status, retried, plan), (max, status, retried, plan)) of the realized makespan over
    the plans that are optimal for the model (``retried``: solved again without presolve, see
    below; ``plan``: the extremal plan, per worker its segments, None if there is none).

    ``c_model`` is S x G (segments of ``segments(n, cap)``): one row of estimated costs for the
    makespan model (``lam`` None), or the scenario costs of the CVaR model with weight ``lam``.
    The planners' symmetry-breaking rows are left out, so the plans form a set closed under
    relabelling the workers: the worst realized makespan is then the largest true load of worker 0,
    a linear objective. Both values are computed from the plan the solver returns. ``r_upper``, the
    realized makespan of a plan known to be optimal for the model, bounds the minimization (HiGHS
    1.12 declared some of these minimizations infeasible without it).
    """
    segs = segments(n, cap)
    G, W, S = len(segs), workers, c_model.shape[0]
    ny = G * W
    iT, iz, ieta, iR = ny, ny + S, ny + 2 * S, ny + 2 * S + 1
    nv = iR + 1
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    def add(terms, lo, hi):
        r = len(lb)
        for col, val in terms:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        lb.append(lo)
        ub.append(hi)

    for k in range(n):  # every sample covered exactly once
        add(
            [(g * W + w, 1.0) for g, (i, j) in enumerate(segs) if i <= k <= j for w in range(W)],
            1,
            1,
        )
    for s in range(S):  # model load of every worker in every scenario at most T_s
        for w in range(W):
            add([(g * W + w, c_model[s, g]) for g in range(G)] + [(iT + s, -1.0)], -np.inf, 0.0)
    obj = np.zeros(nv)
    if lam is None:
        obj[iT] = 1.0
    else:
        if lam > 0:
            for s in range(S):  # z_s >= T_s - eta
                add([(iz + s, 1.0), (iT + s, -1.0), (ieta, 1.0)], 0.0, np.inf)
        obj[iT : iT + S] = 1.0 / S
        obj[iz : iz + S] = lam / ((1 - alpha) * S)
        obj[ieta] = lam
    add([(c, obj[c]) for c in np.flatnonzero(obj)], -np.inf, f_opt + tol)  # optimal plans only
    for w in range(W):  # R >= every true load (used by the minimization)
        add([(g * W + w, -c_true[g]) for g in range(G)] + [(iR, 1.0)], 0.0, np.inf)
    A = coo_matrix((vals, (rows, cols)), shape=(len(lb), nv)).tocsr()
    lo = np.zeros(nv)
    hi = np.full(nv, np.inf)
    hi[:ny] = 1.0
    lo[ieta] = -np.inf
    if lam is None or lam == 0.0:  # z and eta have no role in the objective
        hi[iz:iR] = 0.0
        lo[ieta] = 0.0
    integrality = np.zeros(nv)
    integrality[:ny] = 1
    c_min = np.zeros(nv)
    c_min[iR] = 1.0
    c_max = np.zeros(nv)
    c_max[0:ny:W] = -c_true  # the true load of worker 0
    hi_min = hi.copy()
    if r_upper is not None:  # only the minimization: in the maximization R is not used
        hi_min[iR] = r_upper
    out = []
    for c, h in ((c_min, hi_min), (c_max, hi)):
        for presolve in (True, False):
            res = milp(
                c,
                integrality=integrality,
                bounds=Bounds(lo, h),
                constraints=LinearConstraint(A, lb, ub),
                options={
                    "disp": False,
                    "presolve": presolve,
                    "mip_rel_gap": 0.0,
                    "time_limit": time_limit,
                },
            )
            status = milp_status(res)
            # the planner's own plan satisfies every row, so "infeasible" (or a solve error) is the
            # solver's, not the model's: HiGHS 1.12's presolve reported it for a few of these
            if status not in ("infeasible", "error"):
                break
        if res.x is None:
            out.append((None, status, not presolve, None))
            continue
        plan = [[seg for g, seg in enumerate(segs) if res.x[g * W + w] > 0.5] for w in range(W)]
        out.append((plan_value(plan, c_true, segs), status, not presolve, plan))
    return out[0], out[1]


def covers(plan, n: int) -> bool:
    """Every sample in exactly one segment of the plan."""
    seen = sorted(k for work in plan for i, j in work for k in range(i, j + 1))
    return seen == list(range(n))


def model_objective(plan, c_model, segs, lam: float | None, alpha: float = ALPHA) -> float:
    """The planner's objective of a plan: the makespan on the estimates (``lam`` None), or the
    mean scenario makespan plus ``lam`` times its CVaR_alpha (the Rockafellar-Uryasev minimum
    over eta, attained at one of the scenario makespans)."""
    T = [plan_value(plan, c_model[s], segs) for s in range(c_model.shape[0])]
    if lam is None:
        return T[0]
    tail = min(e + sum(max(0.0, t - e) for t in T) / ((1 - alpha) * len(T)) for e in T)
    return sum(T) / len(T) + lam * tail


def tie_row(seed: int, lams=LAMS) -> dict:
    """One S9 instance: per model the realized makespan of the plan the solver returns (replayed
    in the simulator, as in ``risk.evaluate``) and its range over the model-optimal plans. The
    range covers the plans no worse than the planner's objective, which is the optimal set only
    when the planner's status is optimal (the script counts other rows as problems)."""
    system, trace, (n, W, cap) = instance(seed)
    cost = cost_fn(system)
    L = [g.resp_tokens[0] for g in trace.groups]
    est = [g.est_tokens for g in trace.groups]
    mt = trace.groups[0].max_tokens
    segs = segments(n, cap)
    row: dict = {"seed": seed, "n": n, "W": W, "cap": cap}
    o_true = sorted(range(n), key=lambda k: (L[k], k))
    opt = solve_makespan([L[k] for k in o_true], W, cap, cost)
    row["optimum_ms"] = round(opt.objective) if opt.objective is not None else None
    row["optimum_status"] = opt.status
    o_est = sorted(range(n), key=lambda k: (est[k], k))
    Le = [max(1, min(mt, math.ceil(est[k]))) for k in o_est]
    Lt = [L[k] for k in o_est]
    c_true = np.array([cost(j - i + 1, max(Lt[i : j + 1])) for i, j in segs], dtype=float)
    models = [("det", None)] + [(f"cvar{lam:g}", float(lam)) for lam in lams]
    scen = draw_scenarios([est[k] for k in o_est], mt, BELIEF_SIGMA, SCENARIOS, seed)
    c_scen = np.empty((scen.shape[0], len(segs)))
    for g, (i, j) in enumerate(segs):
        mx = scen[:, i : j + 1].max(axis=1)
        for s in range(scen.shape[0]):
            c_scen[s, g] = cost(j - i + 1, int(mx[s]))
    for name, lam in models:
        if lam is None:
            res = solve_makespan(Le, W, cap, cost)
            c_model = np.array([[cost(j - i + 1, Le[j]) for i, j in segs]], dtype=float)
        else:
            res = solve_cvar(scen, W, cap, cost, alpha=ALPHA, lam=lam)
            c_model = c_scen
        row[f"{name}_status"] = res.status
        if res.objective is None or not covers(res.plan, n):  # no plan to replay or compare with
            row.update(
                {
                    f"{name}_{k}": None
                    for k in ("ms", "min_ms", "max_ms", "min_status", "max_status")
                }
            )
            row.update(
                {f"{name}_formula_ok": False, f"{name}_retried": 0, f"{name}_extremes_ok": False}
            )
            continue
        replayed = _replay(system, trace, res.plan, o_est)
        value = plan_value(res.plan, c_true, segs)
        row[f"{name}_ms"] = replayed
        row[f"{name}_formula_ok"] = bool(value == replayed)
        ends = realized_range(c_model, c_true, n, W, cap, res.objective, lam, r_upper=value)
        ok = True
        for (v, st, _, plan), k in zip(ends, ("min", "max")):
            row[f"{name}_{k}_ms"] = round(v) if v is not None else None
            row[f"{name}_{k}_status"] = st
            # an extremal plan must cover every sample once, be optimal for the model, and end
            # in the simulator where the cost formula says
            ok = ok and plan is not None and covers(plan, n)
            ok = ok and model_objective(plan, c_model, segs, lam) <= res.objective + TOL_MS
            ok = ok and _replay(system, trace, plan, o_est) == v
        row[f"{name}_retried"] = sum(int(e[2]) for e in ends)
        row[f"{name}_extremes_ok"] = bool(ok)
    if opt.status == "optimal":
        for name, _ in models:
            for k in ("", "_min", "_max"):
                v = row[f"{name}{k}_ms"]
                row[f"{name}{k}_ratio"] = v / row["optimum_ms"] if v is not None else None
    return row
