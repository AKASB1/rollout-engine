"""S9: the realized makespan of every planning model over all of its optimal plans.

    python scripts/s9_tie_ranges.py [--workers N] [--out benchmarks/results/s9_ties]

For each S9 instance (the S7 tuning and evaluation instances of configs/experiments.json) and
each planning model (``det``, ``cvar<lam>`` for the configured lams), the plan the solver
returns is replayed in the simulator as in the benchmark, and the minimum and maximum realized
makespan over all plans that are optimal for the model come from two more MILPs
(``rollout_engine.opt.ties``). Writes ``s9_ties.csv`` (one row per instance), ``summary.csv``
(per set and model: mean ratios to the clairvoyant optimum of the returned plan, of the best and
of the worst optimal plan, and how many instances have more than one realized value), and
``manifest.json`` (with the SHA-256 of this script and of the ``opt`` modules it runs, and whether
they are committed and nothing under ``src/`` is changed). Exits 1 if a replay differs from the cost formula, a solve is not optimal,
or an extremal plan fails its checks (``*_extremes_ok``: covers every sample once, optimal for
the model, replayed where the formula says).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import multiprocessing as mp
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rollout_engine.bench import output  # noqa: E402
from rollout_engine.bench.experiments import load_config  # noqa: E402
from rollout_engine.opt.ties import TIME_LIMIT_S, TOL_MS, tie_row  # noqa: E402


def _row(job):
    seed, lams, which = job
    return {"set": which, **tie_row(seed, lams)}


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT = ["ties.py", "milp.py", "scenario.py", "risk.py", "s7.py"]
CODE = ["scripts/s9_tie_ranges.py"] + [f"src/rollout_engine/opt/{f}" for f in OPT]


def code_state() -> dict:
    """SHA-256 of the code that makes these results, and whether it is tracked and nothing under
    ``src/`` or this script differs from the commit."""
    sha = {p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest() for p in CODE}
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", *CODE], cwd=ROOT, capture_output=True, timeout=10
        )
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--", CODE[0], "src"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        committed = tracked.returncode == 0 and not changed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        committed = None
    return {"code_sha256": sha, "code_committed": committed}


def complete(r: dict, m: str) -> bool:
    """The optimum, the returned plan, and both ends of the range are known and checked."""
    return (
        r["optimum_status"] == "optimal"
        and r[f"{m}_formula_ok"]
        and r[f"{m}_extremes_ok"]
        and r[f"{m}_status"] == r[f"{m}_min_status"] == r[f"{m}_max_status"] == "optimal"
    )


def summarize(rows: list[dict], models: list[str]) -> list[dict]:
    out = []
    for which in sorted({r["set"] for r in rows}):
        for m in models:
            rs = [r for r in rows if r["set"] == which and complete(r, m)]
            if not rs:
                continue
            out.append(
                {
                    "set": which,
                    "model": m,
                    "instances": len(rs),
                    "ratio_returned": sum(r[f"{m}_ratio"] for r in rs) / len(rs),
                    "ratio_best": sum(r[f"{m}_min_ratio"] for r in rs) / len(rs),
                    "ratio_worst": sum(r[f"{m}_max_ratio"] for r in rs) / len(rs),
                    "instances_with_range": sum(r[f"{m}_min_ms"] < r[f"{m}_max_ms"] for r in rs),
                    "instances_left_out": sum(r["set"] == which for r in rows) - len(rs),
                    "solves_retried": sum(r[f"{m}_retried"] for r in rows if r["set"] == which),
                }
            )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="scripts/s9_tie_ranges.py", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default=os.path.join("benchmarks", "results", "s9_ties"))
    argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    git_state = {**output.git_commit(), **code_state()}  # read before any result file is written
    cfg = load_config()
    lams = [float(x) for x in cfg["S9"]["lams"]]
    sets = {"tuning": cfg["seeds"]["s7_tuning"], "evaluation": cfg["seeds"]["s7_evaluation"]}
    jobs = [
        (s, lams, which)
        for which, c in sets.items()
        for s in range(c["start"], c["start"] + c["count"])
    ]
    t0 = time.perf_counter()
    if a.workers <= 1:
        rows = [_row(j) for j in jobs]
    else:
        with cf.ProcessPoolExecutor(
            max_workers=a.workers, mp_context=mp.get_context("spawn")
        ) as ex:
            rows = list(ex.map(_row, jobs))
    rows.sort(key=lambda r: (r["set"], r["seed"]))
    models = ["det"] + [f"cvar{lam:g}" for lam in lams]
    bad = [
        (r["set"], r["seed"], m)
        for r in rows
        for m in models
        if not complete(r, m) or not (r[f"{m}_min_ms"] <= r[f"{m}_ms"] <= r[f"{m}_max_ms"])
    ]
    summary = summarize(rows, models)  # everything computed before any file is written
    output.write_csv(os.path.join(a.out, "s9_ties.csv"), rows, ["set", "seed"])
    output.write_csv(os.path.join(a.out, "summary.csv"), summary, ["set", "model"])
    output.write_json(
        os.path.join(a.out, "manifest.json"),
        {
            **git_state,
            "command": "python scripts/s9_tie_ranges.py " + " ".join(argv),
            "instances": {
                k: f"{c['start']}-{c['start'] + c['count'] - 1}" for k, c in sets.items()
            },
            "lams": lams,
            "tolerance_ms": TOL_MS,
            "time_limit_s": TIME_LIMIT_S,
            "problems": [list(x) for x in bad],
            "software": output.software(),
            "hardware": output.hardware(),
            "workers": a.workers,
            "wall_total_s": round(time.perf_counter() - t0, 1),
            "label": "simulated, assumed parameters; ranges over the plans that are optimal for each planning model",
        },
    )
    print(f"wrote {a.out}: {len(rows)} instances, {len(bad)} problems", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
