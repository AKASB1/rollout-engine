"""Scenario MILP with CVaR for static batching (Tier 2; docs/optimization.md section 7).

The true lengths are unknown; the planner draws ``S`` length scenarios from its belief (the
estimate times a log-normal error of a known level, fixed seed) and chooses one plan (batches
as contiguous segments of the *estimate*-sorted order, each on a worker) for all scenarios:

    minimize   (1/S) sum_s T_s + lam * (eta + 1 / ((1 - alpha) S) sum_s z_s)
    subject to every sample covered once;  load_{w,s} = sum_seg C_s[seg] y[seg, w] <= T_s;
               z_s >= T_s - eta, z_s >= 0;  expected loads non-increasing in w (symmetry)

``eta + 1/((1-alpha) S) sum z_s`` is the Rockafellar-Uryasev form of the CVaR_alpha of the
makespan. Unlike section 3, contiguity in the estimate order is a restriction (the scenarios
order the samples differently), so the plan is a heuristic optimum within that family; the
evaluation replays it on the true lengths.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from rollout_engine.opt.milp import MilpResult, segments
from rollout_engine.rng import stream


def draw_scenarios(
    est: list[float], max_tokens: int, sigma: float, S: int, seed: int
) -> np.ndarray:
    rng = stream(seed, "scenario.lengths")
    z = rng.standard_normal((S, len(est)))
    L = np.ceil(np.asarray(est)[None, :] * np.exp(sigma * z))
    return np.clip(L, 1, max_tokens).astype(np.int64)


def solve_cvar(
    scen: np.ndarray,  # S x n, columns in the estimate-sorted order
    workers: int,
    cap: int,
    cost,
    *,
    alpha: float = 0.9,
    lam: float = 1.0,
    node_limit: int = 200_000,
    time_limit: float = 60.0,
) -> MilpResult:
    S, n = scen.shape
    segs = segments(n, cap)
    G, W = len(segs), workers
    C = np.empty((S, G))
    for g, (i, j) in enumerate(segs):
        mx = scen[:, i : j + 1].max(axis=1)
        for s in range(S):
            C[s, g] = cost(j - i + 1, int(mx[s]))
    ny = G * W
    iT, iz, ieta = ny, ny + S, ny + 2 * S
    nv = ny + 2 * S + 1
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for k in range(n):
        for g, (i, j) in enumerate(segs):
            if i <= k <= j:
                for w in range(W):
                    rows.append(r)
                    cols.append(g * W + w)
                    vals.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        r += 1
    for s in range(S):
        for w in range(W):
            for g in range(G):
                rows.append(r)
                cols.append(g * W + w)
                vals.append(C[s, g])
            rows.append(r)
            cols.append(iT + s)
            vals.append(-1.0)
            lb.append(-np.inf)
            ub.append(0.0)
            r += 1
    for s in range(S):  # z_s - T_s + eta >= 0
        rows += [r, r, r]
        cols += [iz + s, iT + s, ieta]
        vals += [1.0, -1.0, 1.0]
        lb.append(0.0)
        ub.append(np.inf)
        r += 1
    Cm = C.mean(axis=0)
    for w in range(W - 1):
        for g in range(G):
            rows += [r, r]
            cols += [g * W + w, g * W + w + 1]
            vals += [Cm[g], -Cm[g]]
        lb.append(0.0)
        ub.append(np.inf)
        r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    c = np.zeros(nv)
    c[iT : iT + S] = 1.0 / S
    c[iz : iz + S] = lam / ((1 - alpha) * S)
    c[ieta] = lam
    integrality = np.zeros(nv)
    integrality[:ny] = 1
    lo = np.zeros(nv)
    lo[ieta] = -np.inf
    hi = np.full(nv, np.inf)
    hi[:ny] = 1.0
    t0, c0 = time.perf_counter(), time.process_time()
    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lo, hi),
        constraints=LinearConstraint(A, lb, ub),
        options={
            "disp": False,
            "presolve": True,
            "mip_rel_gap": 0.0,
            "node_limit": node_limit,
            "time_limit": time_limit,
        },
    )
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    msg = (res.message or "").lower()
    status = (
        "optimal"
        if res.status == 0
        else (
            "solver_capped"
            if (res.status == 1 and "time" in msg)
            else ("node_limit" if res.status == 1 else "error")
        )
    )
    plan: list[list[tuple[int, int]]] = [[] for _ in range(W)]
    if res.x is not None:
        for g, seg in enumerate(segs):
            for w in range(W):
                if res.x[g * W + w] > 0.5:
                    plan[w].append(seg)
    bound = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    return MilpResult(
        status,
        float(res.fun) if res.fun is not None else None,
        bound,
        gap,
        plan,
        wall,
        cpu,
        getattr(res, "mip_node_count", None),
        nv,
        r,
    )
