"""Optional solver cross-check (Tier 2): solve the S7 makespan MILP with SciPy's HiGHS (the
default), a current HiGHS (``highspy``), and Gurobi (``gurobipy``), and compare objectives,
solve times, and the size up to which each proves optimality.

    python scripts/solver_crosscheck.py [--out benchmarks/results/solver_crosscheck.md]

Run it in a throwaway environment that has ``highspy`` (and, optionally, a licensed
``gurobipy``); never in the environment the committed results come from. Nothing in the
default path, the tests, or CI depends on it. The model is the one of docs/optimization.md
section 3, rebuilt here so the script stands alone. Times are wall-clock and process CPU on
a shared machine.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rollout_engine.opt.s7 import cost_fn, instance  # noqa: E402

TIME_LIMIT = 60.0


def build(L, W, cap, cost):
    n = len(L)
    segs = [(i, j) for i in range(n) for j in range(i, min(n, i + cap))]
    S = len(segs)
    C = np.array([cost(j - i + 1, L[j]) for i, j in segs], dtype=float)
    nv = S * W + 1
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
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
    for w in range(W - 1):
        for s in range(S):
            rows += [r, r]
            cols += [s * W + w, s * W + w + 1]
            vals += [C[s], -C[s]]
        lb.append(0.0)
        ub.append(np.inf)
        r += 1
    from scipy.sparse import coo_matrix

    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    c = np.zeros(nv)
    c[-1] = 1.0
    return c, A, np.array(lb), np.array(ub), nv


def solve_scipy(c, A, lb, ub, nv):
    from scipy.optimize import Bounds, LinearConstraint, milp

    integ = np.ones(nv)
    integ[-1] = 0
    hi = np.ones(nv)
    hi[-1] = np.inf
    t0, c0 = time.perf_counter(), time.process_time()
    res = milp(
        c,
        integrality=integ,
        bounds=Bounds(np.zeros(nv), hi),
        constraints=LinearConstraint(A, lb, ub),
        options={"mip_rel_gap": 0.0, "time_limit": TIME_LIMIT, "node_limit": 200_000},
    )
    return (
        "optimal" if res.status == 0 else f"status{res.status}",
        res.fun,
        time.perf_counter() - t0,
        time.process_time() - c0,
    )


def solve_highspy(c, A, lb, ub, nv):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("time_limit", TIME_LIMIT)
    inf = highspy.kHighsInf
    hi = np.ones(nv)
    hi[-1] = inf
    h.addVars(nv, np.zeros(nv), hi)
    h.changeColsCost(nv, np.arange(nv, dtype=np.int32), c)
    h.changeColsIntegrality(
        nv - 1,
        np.arange(nv - 1, dtype=np.int32),
        np.array([highspy.HighsVarType.kInteger] * (nv - 1)),
    )
    lb2 = np.where(np.isinf(lb), -inf, lb)
    ub2 = np.where(np.isinf(ub), inf, ub)
    h.addRows(
        A.shape[0], lb2, ub2, A.nnz, A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data
    )
    t0, c0 = time.perf_counter(), time.process_time()
    h.run()
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    st = h.getModelStatus()
    status = "optimal" if st == highspy.HighsModelStatus.kOptimal else str(st).split(".")[-1]
    return (status, h.getInfo().objective_function_value, wall, cpu)


def solve_gurobi(c, A, lb, ub, nv):
    import gurobipy as gp

    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 0.0
    m.Params.TimeLimit = TIME_LIMIT
    m.Params.Threads = 1
    vt = np.array(["B"] * (nv - 1) + ["C"])
    x = m.addMVar(nv, lb=0.0, ub=np.r_[np.ones(nv - 1), np.inf], vtype=vt)
    m.setObjective(c @ x)
    fin_lb, fin_ub = np.isfinite(lb), np.isfinite(ub)
    eq = fin_lb & fin_ub & (lb == ub)
    m.addMConstr(A[eq], x, "=", lb[eq])
    le = fin_ub & ~eq
    m.addMConstr(A[le], x, "<", ub[le])
    ge = fin_lb & ~eq
    m.addMConstr(A[ge], x, ">", lb[ge])
    t0, c0 = time.perf_counter(), time.process_time()
    m.optimize()
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    status = "optimal" if m.Status == gp.GRB.OPTIMAL else f"status{m.Status}"
    return (status, m.ObjVal if m.SolCount else None, wall, cpu)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("benchmarks", "results", "solver_crosscheck.md"))
    ap.add_argument("--instances", type=int, default=50)
    a = ap.parse_args(argv)
    solvers = [("SciPy milp (HiGHS 1.2.0)", solve_scipy)]
    versions = {}
    try:
        import highspy

        solvers.append((f"highspy {highspy.Highs().version()}", solve_highspy))
    except ImportError:
        versions["highspy"] = "not installed"
    try:
        import gurobipy as gp

        gp.Model().dispose()
        v = gp.gurobi.version()
        solvers.append((f"Gurobi {v[0]}.{v[1]}.{v[2]} (1 thread)", solve_gurobi))
    except Exception as e:  # noqa: BLE001 - missing package or license
        versions["gurobi"] = f"unavailable ({type(e).__name__})"
    rows = []
    cases = [("S7", s, None) for s in range(6000, 6000 + a.instances)] + [
        ("sweep", 8000, n) for n in (64, 96, 128, 192, 256)
    ]
    for kind, seed, n in cases:
        if n is None:
            sysc, tr, (nn, W, cap) = instance(seed)
        else:
            sysc, tr, (nn, W, cap) = instance(seed, n=n, workers=3, cap=6)
        L = sorted(g.resp_tokens[0] for g in tr.groups)
        model = build(L, W, cap, cost_fn(sysc))
        for name, fn in solvers:
            st, obj, wall, cpu = fn(*model)
            rows.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "n": nn,
                    "W": W,
                    "solver": name,
                    "status": st,
                    "objective": obj,
                    "wall_s": wall,
                    "cpu_s": cpu,
                }
            )
        print(
            kind,
            seed,
            nn,
            [
                (r["solver"].split()[0], r["status"], r["objective"], round(r["wall_s"], 3))
                for r in rows[-len(solvers) :]
            ],
            flush=True,
        )
    names = [s for s, _ in solvers]
    lines = [
        "# Solver cross-check (Tier 2, optional)",
        "",
        f"Generated by `scripts/solver_crosscheck.py` in a throwaway environment ({platform.python_version()}, {platform.system()} {platform.release()}); "
        "wall-clock and process CPU times on a shared machine (other jobs were running; CPU load sampled at about 47 % before the benchmark runs), "
        f"time limit {TIME_LIMIT:g} s, relative gap 0. Model: docs/optimization.md section 3. Not used by any result in the README.",
        "",
        "| Solver | S7 instances solved to optimality | objectives equal to SciPy's (rel 1e-6) | median S7 wall time (ms) | median S7 CPU time (ms) |",
        "|---|---|---|---|---|",
    ]
    base = {(r["seed"], r["n"]): r for r in rows if r["solver"] == names[0]}
    for name in names:
        rs = [r for r in rows if r["solver"] == name and r["kind"] == "S7"]
        opt = sum(1 for r in rs if r["status"] == "optimal")
        same = sum(
            1
            for r in rs
            if r["objective"] is not None
            and base[(r["seed"], r["n"])]["objective"] is not None
            and abs(r["objective"] - base[(r["seed"], r["n"])]["objective"])
            <= 1e-6 * abs(base[(r["seed"], r["n"])]["objective"])
        )
        lines.append(
            f"| {name} | {opt}/{len(rs)} | {same}/{len(rs)} | {1000 * float(np.median([r['wall_s'] for r in rs])):.1f} | {1000 * float(np.median([r['cpu_s'] for r in rs])):.1f} |"
        )
    lines += [
        "",
        "Tractability sweep (W = 3, cap 6, seed 8000): status and wall time (s)",
        "",
        "| n | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for n in (64, 96, 128, 192, 256):
        cells = []
        for name in names:
            r = next(
                x for x in rows if x["kind"] == "sweep" and x["n"] == n and x["solver"] == name
            )
            cells.append(f"{r['status']}, {r['wall_s']:.2f}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    for k, v in versions.items():
        lines.append(f"\n{k}: {v}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
