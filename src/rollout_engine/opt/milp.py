"""MILP for the makespan of static batches on W workers (docs/optimization.md section 3).

Solved with scipy.optimize.milp (HiGHS) under deterministic limits: mip_rel_gap = 0 and a
node limit; the wall-clock time limit is only a safety net (a run that hits it is reported as
``solver_capped``). Costs are integers in milliseconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

NODE_LIMIT = 200_000
TIME_LIMIT_S = 60.0


@dataclass
class MilpResult:
    status: str  # optimal | node_limit | solver_capped | infeasible | error
    objective: float | None
    bound: float | None
    gap: float | None
    plan: list[list[tuple[int, int]]]  # per worker: segments (i, j) of the sorted lengths
    wall_s: float
    cpu_s: float
    nodes: int | None
    n_vars: int
    n_cons: int


def segments(n: int, cap: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i, min(n, i + cap))]


def solve_makespan(
    lengths_asc: list[int],
    workers: int,
    cap: int,
    cost: Callable[[int, int], int],
    *,
    node_limit: int = NODE_LIMIT,
    time_limit: float = TIME_LIMIT_S,
) -> MilpResult:
    n = len(lengths_asc)
    if any(lengths_asc[k] > lengths_asc[k + 1] for k in range(n - 1)):
        raise ValueError("lengths must be sorted ascending")
    segs = segments(n, cap)
    S, W = len(segs), workers
    C = np.array([cost(j - i + 1, lengths_asc[j]) for i, j in segs], dtype=float)
    nv = S * W + 1  # y[s, w] at s * W + w, then T
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    r = 0
    # every sample covered exactly once
    for k in range(n):
        for s, (i, j) in enumerate(segs):
            if i <= k <= j:
                for w in range(W):
                    rows.append(r)
                    cols.append(s * W + w)
                    vals.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        r += 1
    # the load of every worker is at most T
    for w in range(W):
        for s in range(S):
            rows.append(r)
            cols.append(s * W + w)
            vals.append(C[s])
        rows.append(r)
        cols.append(nv - 1)
        vals.append(-1.0)
        lb.append(-np.inf)
        ub.append(0.0)
        r += 1
    # symmetry breaking: loads non-increasing in the worker index
    for w in range(W - 1):
        for s in range(S):
            rows.append(r)
            cols.append(s * W + w)
            vals.append(C[s])
            rows.append(r)
            cols.append(s * W + w + 1)
            vals.append(-C[s])
        lb.append(0.0)
        ub.append(np.inf)
        r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    c = np.zeros(nv)
    c[-1] = 1.0
    integrality = np.ones(nv)
    integrality[-1] = 0
    bounds = Bounds(np.zeros(nv), np.concatenate([np.ones(nv - 1), [np.inf]]))
    opts = {
        "disp": False,
        "presolve": True,
        "mip_rel_gap": 0.0,
        "node_limit": node_limit,
        "time_limit": time_limit,
    }
    t0, c0 = time.perf_counter(), time.process_time()
    res = milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(A, lb, ub),
        options=opts,
    )
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    msg = (res.message or "").lower()
    if res.status == 0:
        status = "optimal"
    elif res.status == 1:
        status = "solver_capped" if "time" in msg else "node_limit"
    elif res.status == 2:
        status = "infeasible"
    else:
        status = "error"
    plan: list[list[tuple[int, int]]] = [[] for _ in range(W)]
    if res.x is not None:
        for s, seg in enumerate(segs):
            for w in range(W):
                if res.x[s * W + w] > 0.5:
                    plan[w].append(seg)
    obj = float(res.fun) if res.fun is not None else None
    bound = getattr(res, "mip_dual_bound", None)
    gap = getattr(res, "mip_gap", None)
    return MilpResult(
        status,
        obj,
        float(bound) if bound is not None else None,
        float(gap) if gap is not None else None,
        plan,
        wall,
        cpu,
        getattr(res, "mip_node_count", None),
        nv,
        r,
    )


def plan_makespan(plan, lengths_asc, cost) -> int:
    return max((sum(cost(j - i + 1, lengths_asc[j]) for i, j in segs) for segs in plan), default=0)


def solver_versions() -> dict:
    return {"scipy": scipy.__version__, "highs": "1.2.0 (vendored by SciPy 1.13.1; banner)"}
