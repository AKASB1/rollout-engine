"""Benchmark command line.

    python -m rollout_engine.bench [--quick] [--workers N] [--only S1,S2] [--out DIR]
    python -m rollout_engine.bench tune [--workers N]
    python -m rollout_engine.bench microbench

The evaluation reads the frozen ``configs/tuned/tuned.json`` (``--quick`` falls back to
untuned defaults when it does not exist). Results: ``benchmarks/results/<full|quick>/``.
Everything is simulated with assumed parameters.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time

from rollout_engine.bench import output
from rollout_engine.bench.experiments import build_cells, cells_hash, load_config, load_tuned
from rollout_engine.bench.runner import default_workers, jobs_for, run_jobs
from rollout_engine.bench.stats import mean_ci, objective, paired, weight_vector
from rollout_engine.config import config_hash

FIRST = ["scenario", "variant", "policy", "seed"]


def log(msg: str) -> None:
    print(msg, flush=True)


def _s7_rows(seeds: list[int], workers: int) -> list[dict]:
    from rollout_engine.opt.s7 import evaluate

    if workers <= 1:
        rows = [evaluate(s) for s in seeds]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(evaluate, seeds))
    return sorted(rows, key=lambda r: r["seed"])


def _s9_rows(seeds: list[int], workers: int) -> list[dict]:
    from rollout_engine.opt.risk import evaluate as s9_eval

    if workers <= 1:
        rows = [s9_eval(s) for s in seeds]
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(s9_eval, seeds))
    return sorted(rows, key=lambda r: r["seed"])


def _split_wall(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    det = [{k: v for k, v in r.items() if not k.startswith("wall_")} for r in rows]
    wall = [
        {"seed": r["seed"], **{k: v for k, v in r.items() if k.startswith("wall_")}} for r in rows
    ]
    return det, wall


def _s7_sweep(cfg: dict) -> list[dict]:
    from rollout_engine.opt.milp import solve_makespan
    from rollout_engine.opt.s7 import cost_fn, instance

    c = cfg["S7"]
    rows = []
    for n in c["sweep_n"]:
        capped = False
        for seed in c["sweep_seeds"]:
            sysc, tr, _ = instance(seed, n=n, workers=c["sweep_workers"], cap=c["sweep_cap"])
            L = sorted(g.resp_tokens[0] for g in tr.groups)
            r = solve_makespan(
                L,
                c["sweep_workers"],
                c["sweep_cap"],
                cost_fn(sysc),
                time_limit=c["sweep_time_limit_s"],
            )
            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "status": r.status,
                    "objective_ms": round(r.objective) if r.objective is not None else None,
                    "gap": r.gap,
                    "n_vars": r.n_vars,
                    "wall_solve_s": r.wall_s,
                    "wall_solve_cpu_s": r.cpu_s,
                    "nodes": r.nodes,
                }
            )
            capped = capped or r.status != "optimal"
        log(f"  S7 sweep n={n}: {[x['status'] for x in rows if x['n'] == n]}")
        if capped:
            break
    return rows


def evaluate(args, cfg, tuned) -> int:
    t_start = time.perf_counter()
    git_state = output.git_commit()  # read before any result file is (re)written
    quick = args.quick
    only = args.only.split(",") if args.only else None
    if tuned is None and not quick:
        log("configs/tuned/tuned.json is missing: run `python -m rollout_engine.bench tune` first")
        return 2
    mode = "quick" if quick else "full"
    out = args.out or os.path.join("benchmarks", "results", mode)
    cells = build_cells(cfg, tuned, quick, only)
    scen = set(
        only
        or (
            cfg["quick"]["scenarios"]
            if quick
            else ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "SENS"]
        )
    )
    jobs = jobs_for(cells)
    log(f"{mode} evaluation: {len(cells)} cells, {len(jobs)} runs, {args.workers} workers")
    rows, walls, viols = run_jobs(jobs, args.workers, log)
    cellmap = {c.key: c for c in cells}
    # objective
    r_source = {}
    for key, cell in cellmap.items():
        if tuned and key in tuned.get("R", {}):
            r_source[key] = ("tuned", tuned["R"][key], tuned["weights"][key])
        else:
            ref = [
                r
                for r in rows
                if f"{r['scenario']}/{r['variant']}" == key and r["policy"] == cell.reference
            ]
            k = "phase_time_s" if cell.kind == "single" else "step_time_mean_s"
            R = sum(r[k] for r in ref) / len(ref)
            r_source[key] = (
                "quick-fallback (reference on the evaluated seeds)",
                R,
                weight_vector(cfg["weights"][cell.kind], cell.kind),
            )
    for r in rows:
        key = f"{r['scenario']}/{r['variant']}"
        _, R, w = r_source[key]
        r["J"] = objective(r, cellmap[key].kind, R, w)
    # aggregates (long format) and paired differences
    agg, pair = [], []
    metrics = [k for k in output.columns_of(rows, FIRST) if k not in FIRST and k != "policy_name"]
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["variant"], r["policy"]), []).append(r)
    for (s, v, p), rs in sorted(groups.items()):
        for m in metrics:
            vals = [x[m] for x in rs if isinstance(x.get(m), (int, float))]
            if not vals:
                continue
            mu, ci, n = mean_ci([float(x) for x in vals])
            agg.append(
                {
                    "scenario": s,
                    "variant": v,
                    "policy": p,
                    "metric": m,
                    "mean": mu,
                    "ci95": ci,
                    "n": n,
                }
            )
    for _key, cell in sorted(cellmap.items()):
        ref = {r["seed"]: r for r in groups.get((cell.scenario, cell.variant, cell.reference), [])}
        for spec in cell.policies:
            if spec.label == cell.reference:
                continue
            pol = {r["seed"]: r for r in groups.get((cell.scenario, cell.variant, spec.label), [])}
            for m in ("J", "phase_time_s" if cell.kind == "single" else "step_time_mean_s"):
                d = paired(
                    {k: x[m] for k, x in ref.items()},
                    {k: x[m] for k, x in pol.items()},
                    cfg["tie_band"],
                )
                pair.append(
                    {
                        "scenario": cell.scenario,
                        "variant": cell.variant,
                        "policy": spec.label,
                        "metric": m,
                        **d,
                    }
                )
    output.write_csv(os.path.join(out, "runs.csv"), rows, FIRST + ["policy_name", "J"])
    output.write_csv(os.path.join(out, "wall_runs.csv"), walls, FIRST)
    output.write_csv(
        os.path.join(out, "aggregate.csv"), agg, ["scenario", "variant", "policy", "metric"]
    )
    output.write_csv(
        os.path.join(out, "paired.csv"), pair, ["scenario", "variant", "policy", "metric"]
    )
    s7info = {}
    if "S7" in scen:
        c7 = cfg["seeds"]["s7_evaluation"]
        n7 = cfg["quick"]["s7_count"] if quick else c7["count"]
        s7 = _s7_rows(list(range(c7["start"], c7["start"] + n7)), args.workers)
        output.write_csv(
            os.path.join(out, "s7.csv"),
            [{k: v for k, v in r.items() if not k.startswith("wall_")} for r in s7],
            ["seed"],
        )
        output.write_csv(
            os.path.join(out, "wall_s7.csv"),
            [
                {"seed": r["seed"], **{k: v for k, v in r.items() if k.startswith("wall_")}}
                for r in s7
            ],
            ["seed"],
        )
        if not quick:
            sweep = _s7_sweep(cfg)
            output.write_csv(os.path.join(out, "wall_s7_sweep.csv"), sweep, ["n", "seed"])
        s7info = {"instances": n7, "milp_status": sorted({r["milp_status"] for r in s7})}
        bad = [
            r["seed"]
            for r in s7
            if r["milp_status"] == "optimal" and r.get("milp_replay_ms") != r["milp_ms"]
        ]
        below = [
            (r["seed"], p)
            for r in s7
            for p in ("fifo_chunk", "sorted_chunk", "sorted_chunk_oracle", "dp", "oracle_dp")
            if r["milp_status"] == "optimal" and r[f"{p}_ms"] < r["milp_ms"]
        ]
        if bad or below:
            viols.append(f"S7: replay mismatch {bad}, below the MILP optimum {below}")
    if "S9" in scen and not quick:
        c7 = cfg["seeds"]["s7_evaluation"]
        s9 = _s9_rows(list(range(c7["start"], c7["start"] + c7["count"])), args.workers)
        det, wall = _split_wall(s9)
        output.write_csv(os.path.join(out, "s9.csv"), det, ["seed"])
        output.write_csv(os.path.join(out, "wall_s9.csv"), wall, ["seed"])
    wall_total = time.perf_counter() - t_start
    manifest = {
        "mode": mode,
        **git_state,
        "experiments_hash": config_hash(cfg),
        "tuned_hash": config_hash(tuned) if tuned else None,
        "tuned_commit_note": "configs/tuned/tuned.json as committed at the commit above",
        "cells_hash": cells_hash(cells),
        "seeds": sorted({r["seed"] for r in rows}),
        "runs": len(rows),
        "objective_normalizers": {
            k: {"source": v[0], "R": v[1], "weights": v[2]} for k, v in sorted(r_source.items())
        },
        "software": output.software(),
        "hardware": output.hardware(),
        "workers": args.workers,
        "s7": s7info,
        "violations": viols,
        "wall_note": "wall_* files are wall-clock measurements on a shared machine (other jobs may run); they are excluded from every deterministic comparison",
        "wall_total_s": round(wall_total, 1),
        "label": "simulated, assumed parameters",
    }
    output.write_json(os.path.join(out, "manifest.json"), manifest)
    log(f"wrote {out}: {len(rows)} runs, {len(viols)} violations, {wall_total:.0f} s")
    if viols:
        for v in viols[:10]:
            log(f"  VIOLATION {v}")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rollout_engine.bench", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("command", nargs="?", default="run", choices=["run", "tune", "microbench"])
    ap.add_argument(
        "--quick", action="store_true", help="3 seeds, 12 steps, a subset of scenarios and policies"
    )
    ap.add_argument(
        "--workers", type=int, default=int(os.environ.get("BENCH_WORKERS", default_workers()))
    )
    ap.add_argument("--only", default="", help="comma-separated scenario ids (S1..S7, SENS)")
    ap.add_argument(
        "--out", default="", help="output directory (default benchmarks/results/<mode>)"
    )
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.command == "tune":
        from rollout_engine.bench.tune import tune

        t0 = time.perf_counter()
        tuned = tune(cfg, args.workers, log)
        from rollout_engine.opt.s7 import (
            evaluate as s7_eval,  # noqa: F401  (S7 has no tuned parameters)
        )

        c7 = cfg["seeds"]["s7_tuning"]
        s7t = _s7_rows(list(range(c7["start"], c7["start"] + c7["count"])), args.workers)
        output.write_csv(
            os.path.join("configs", "tuned", "s7_tuning.csv"),
            [{k: v for k, v in r.items() if not k.startswith("wall_")} for r in s7t],
            ["seed"],
        )
        tuned["s7_tuning"] = {
            "instances": len(s7t),
            "milp_status": sorted({r["milp_status"] for r in s7t}),
            "note": "S7 has no tuned parameter; the tuning set is reported in configs/tuned/s7_tuning.csv",
        }
        s9t = _s9_rows(list(range(c7["start"], c7["start"] + c7["count"])), args.workers)
        det, _ = _split_wall(s9t)
        output.write_csv(os.path.join("configs", "tuned", "s9_tuning.csv"), det, ["seed"])
        means = {}
        for lam in cfg["S9"]["lams"]:
            vals = [r[f"cvar{lam:g}_ratio"] for r in s9t if f"cvar{lam:g}_ratio" in r]
            means[f"{lam:g}"] = sum(vals) / len(vals)
        best = min(sorted(means), key=lambda k: means[k])
        tuned["s9"] = {
            "lam": float(best),
            "mean_ratio_by_lam": means,
            "criterion": "mean realized makespan / optimum on the tuning instances",
        }
        tuned["wall_tuning_s"] = round(time.perf_counter() - t0, 1)
        output.write_json(os.path.join("configs", "tuned", "tuned.json"), tuned)
        log(f"wrote configs/tuned/tuned.json in {tuned['wall_tuning_s']} s")
        return 0
    if args.command == "microbench":
        from rollout_engine.bench.microbench import run

        git_state = output.git_commit()  # read before the result files are rewritten
        rows = run(log)
        output.write_csv(
            os.path.join("benchmarks", "results", "wall_decision_cost.csv"),
            rows,
            ["what", "part", "size"],
        )
        output.write_json(
            os.path.join("benchmarks", "results", "wall_decision_cost.manifest.json"),
            {
                **git_state,
                "software": output.software(),
                "hardware": output.hardware(),
                "note": "minimum of 5 repetitions; shared machine, other jobs may run",
            },
        )
        return 0
    return evaluate(args, cfg, load_tuned())


if __name__ == "__main__":
    sys.exit(main())
