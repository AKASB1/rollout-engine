"""Equal-budget tuning on the tuning seeds (never on evaluation seeds).

Per cell: (1) every policy with its default parameters on the tuning seeds; the reference
policy's mean step time (closed loop) or phase time (single phase) is the normalizer R;
(2) random search with a fixed search seed: every policy with tunable parameters gets the
same number of configurations (all of its grid when the grid is smaller); (3) the term
check over every configuration evaluated on the tuning seeds (defaults and candidates): a
J term that no configuration moves away from the reference by more than two standard errors
of the paired difference gets weight 0 in that cell (identical for every policy); (4) the
configuration with the lowest mean J wins. Stage A tunes S1 and S2 and picks the best S2
policy (lowest mean J in S2 heavy/group among deployable policies); stage B tunes the
closed-loop cells, whose other parts are fixed at that policy.
"""

from __future__ import annotations

import copy
import itertools
import math

from rollout_engine.bench.experiments import PolicySpec, build_cells
from rollout_engine.bench.runner import jobs_for, run_jobs
from rollout_engine.bench.stats import j_terms, weight_vector
from rollout_engine.config import config_hash
from rollout_engine.rng import stream


def _ref_value(rows, kind):
    key = "phase_time_s" if kind == "single" else "step_time_mean_s"
    xs = [r[key] for r in rows]
    return sum(xs) / len(xs)


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def term_check(cell, rows_by_policy, R, base_weights):
    """Keep a J term only if it varies across policies by more than its seed-to-seed noise.

    Comparisons are paired (every policy sees the same seeds), so the noise is that of the
    paired difference against the reference: for each policy, the mean over the tuning seeds
    of (term_policy - term_reference) and its standard error. A term is kept when some
    policy's mean difference exceeds twice its standard error (a term that is identical for
    every policy, such as padding on the continuous engine, is dropped)."""
    w = weight_vector(base_weights, cell.kind)
    ref = {r["seed"]: j_terms(r, cell.kind, R) for r in rows_by_policy[cell.reference]}
    report = {}
    for term in list(w):
        best_t, best_diff, best_se = 0.0, 0.0, 0.0
        for label, rows in sorted(rows_by_policy.items()):
            if label == cell.reference:
                continue
            d = [
                j_terms(r, cell.kind, R)[term] - ref[r["seed"]][term]
                for r in rows
                if r["seed"] in ref
            ]
            if len(d) < 2:
                continue
            mean = sum(d) / len(d)
            se = _std(d) / math.sqrt(len(d))
            t = math.inf if (se == 0 and mean != 0) else (abs(mean) / se if se > 0 else 0.0)
            if t > best_t:
                best_t, best_diff, best_se = t, mean, se
        keep = best_t > 2.0
        report[term] = {
            "max_abs_t": best_t if math.isfinite(best_t) else 1e9,
            "mean_diff": best_diff,
            "se": best_se,
            "kept": keep,
        }
        if not keep:
            w[term] = 0.0
    return w, report


def _grid(spec, space, cap_static=None):
    axes = []
    for name in spec.tunables:
        vals = list(space[name])
        if name == "batch_size" and cap_static:
            vals = [v for v in vals if v <= cap_static]
        axes.append([(name, v) for v in vals])
    return [dict(c) for c in itertools.product(*axes)]


def _mean_j(rows, cell, R, w):
    return sum(sum(w[k] * v for k, v in j_terms(r, cell.kind, R).items()) for r in rows) / len(rows)


def tune_cells(cells, cfg, workers, log, tuned):
    seeds = cfg["seeds"]["tuning"]
    space = cfg["tuning"]["space"]
    budget = int(cfg["tuning"]["budget"])
    # (1) defaults
    rows, _, viols = run_jobs(jobs_for(cells, seeds_of=lambda c: seeds), workers, log)
    if viols:
        raise RuntimeError(f"accounting/staleness violations during tuning: {viols[:3]}")
    by_cell: dict[str, dict[str, list]] = {}
    for r in rows:
        by_cell.setdefault(f"{r['scenario']}/{r['variant']}", {}).setdefault(
            r["policy"], []
        ).append(r)
    # (2) the normalizer R: the reference policy on the tuning seeds
    for cell in cells:
        tuned["R"][cell.key] = _ref_value(by_cell[cell.key][cell.reference], cell.kind)
    # (3) random search
    cand_jobs, cand_meta = [], []
    for cell in cells:
        if cell.scenario == "SENS":
            continue  # inherits the parameters of its base cell
        cap = (
            cell.system["rollout"]["static_batch"]
            if cell.system["rollout"]["engine"] == "static"
            else None
        )
        for spec in cell.policies:
            if not spec.tunables:
                continue
            grid = _grid(spec, space, cap)
            rng = stream(int(cfg["tuning"]["search_seed"]), f"tune.{cell.key}.{spec.label}")
            if len(grid) > budget:
                idx = sorted(rng.choice(len(grid), size=budget, replace=False).tolist())
                grid = [grid[i] for i in idx]
            for k, params in enumerate(grid):
                c2 = copy.deepcopy(spec.cfg)
                c2.setdefault("params", {}).update(params)
                label = f"{spec.label}#cand{k}"
                cand_meta.append((cell, spec.label, label, params))
                cand_jobs.append((cell, label, c2))
    jobs = []
    for cell, label, c2 in cand_jobs:
        spec1 = [PolicySpec(label, c2)]
        jobs += jobs_for([cell], seeds_of=lambda c: seeds, policies_of=lambda c, s=spec1: s)
    rows2, _, viols = run_jobs(jobs, workers, log) if jobs else ([], [], [])
    if viols:
        raise RuntimeError(f"accounting/staleness violations during tuning: {viols[:3]}")
    by_cand: dict[tuple, list] = {}
    for r in rows2:
        by_cand.setdefault((f"{r['scenario']}/{r['variant']}", r["policy"]), []).append(r)
    # (4) the term check over every configuration evaluated on the tuning seeds (defaults and
    # candidates): a term that no configuration moves away from the reference carries no
    # information; one that some configuration moves must stay, or tuning would exploit its
    # absence (found in S1, where the default batch sizes do not change the phase time)
    for cell in cells:
        allrows = dict(by_cell[cell.key])
        for (ck, label), rs in by_cand.items():
            if ck == cell.key:
                allrows[label] = rs
        w, rep = term_check(cell, allrows, tuned["R"][cell.key], cfg["weights"][cell.kind])
        tuned["weights"][cell.key] = w
        tuned["term_check"][cell.key] = rep
    best: dict[tuple, tuple] = {}
    for cell, base_label, label, params in cand_meta:
        J = _mean_j(
            by_cand[(cell.key, label)], cell, tuned["R"][cell.key], tuned["weights"][cell.key]
        )
        key = (cell.key, base_label)
        if key not in best or J < best[key][0]:
            best[key] = (J, params)
    for (ck, label), (J, params) in sorted(best.items()):
        tuned["params"].setdefault(ck, {})[label] = params
        tuned["tuned_J"].setdefault(ck, {})[label] = J
    # mean J of every policy at its tuned parameters is recomputed for the best-S2 choice
    return by_cell, by_cand


def tune(cfg, workers, log) -> dict:
    tuned = {
        "schema_version": 1,
        "R": {},
        "weights": {},
        "term_check": {},
        "params": {},
        "tuned_J": {},
        "seeds": cfg["seeds"]["tuning"],
        "budget": cfg["tuning"]["budget"],
        "search_seed": cfg["tuning"]["search_seed"],
    }
    log("tuning stage A: S1, S2")
    cells_a = build_cells(cfg, None, only=["S1", "S2"])
    by_cell, by_cand = tune_cells(cells_a, cfg, workers, log, tuned)
    # best S2 policy: lowest mean J in S2 heavy/group among deployable (non-oracle) policies
    key = "S2/heavy/group"
    cell = next(c for c in cells_a if c.key == key)
    R, w = tuned["R"][key], tuned["weights"][key]
    scores = {}
    for spec in cell.policies:
        if spec.label.startswith("oracle"):
            continue
        if spec.label in tuned["params"].get(key, {}):
            scores[spec.label] = tuned["tuned_J"][key][spec.label]
        else:
            scores[spec.label] = _mean_j(by_cell[key][spec.label], cell, R, w)
    best_label = min(sorted(scores), key=lambda k: scores[k])
    spec = next(p for p in cell.policies if p.label == best_label)
    cfgb = copy.deepcopy(spec.cfg)
    cfgb.setdefault("params", {}).update(tuned["params"].get(key, {}).get(best_label, {}))
    tuned["best_s2"] = {"label": best_label, "cfg": cfgb, "J": scores[best_label], "scores": scores}
    log(f"best S2 policy: {best_label} (J {scores[best_label]:.4f})")
    log("tuning stage B: closed loop")
    cells_b = build_cells(cfg, tuned, only=["S3", "S4", "S5", "S6", "S8", "S10", "SENS"])
    tune_cells(cells_b, cfg, workers, log, tuned)
    tuned["config_hash"] = config_hash(cfg)
    return tuned
