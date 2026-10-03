"""Parallel runner: generate traces into outputs/traces/, replay them from the files, run
every (cell, policy, seed) job in a spawn process pool, and return rows sorted by
(scenario, variant, policy, seed). Results never depend on the order in which jobs finish.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import time

from rollout_engine.batching import batch_cost_ms
from rollout_engine.config import config_hash, parse_system
from rollout_engine.opt.bounds import continuous_lower_bound_ns
from rollout_engine.opt.dp import dp_partition
from rollout_engine.policies.composed import make_policy
from rollout_engine.sim.checks import accounting_violations, staleness_violations
from rollout_engine.sim.driver import simulate
from rollout_engine.telemetry.metrics import run_metrics
from rollout_engine.trace import schema
from rollout_engine.trace.generator import generate, generator_meta


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def trace_path(gen: dict, seed: int, root: str = "outputs/traces") -> str:
    return os.path.join(root, f"{config_hash(gen)}-s{seed}.csv")


def ensure_trace(gen: dict, seed: int, root: str = "outputs/traces") -> str:
    path = trace_path(gen, seed, root)
    if not os.path.exists(path) or not os.path.exists(schema.manifest_path(path)):
        schema.write(path, generate(gen, seed), generator_meta(gen), seed)
    return path


_TRACE_CACHE: dict[str, object] = {}


def _load(path: str):
    tr = _TRACE_CACHE.get(path)
    if tr is None:
        tr, _ = schema.read(path)
        if len(_TRACE_CACHE) > 64:
            _TRACE_CACHE.clear()
        _TRACE_CACHE[path] = tr
    return tr


def run_job(job: dict) -> tuple[dict, dict, list[str]]:
    """One simulation. ``job``: scenario, variant, policy (label), cfg, seed, system,
    trace_path, kind. Returns (deterministic row, wall row, violations)."""
    system = parse_system(job["system"])
    trace = _load(job["trace_path"])
    pol = make_policy(job["cfg"])
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    sim = simulate(system, trace, pol)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    m = run_metrics(sim.core, trace)
    viol = accounting_violations(sim) + staleness_violations(sim)
    row = {
        "scenario": job["scenario"],
        "variant": job["variant"],
        "policy": job["policy"],
        "seed": job["seed"],
        "policy_name": sim.core.policy_name,
        **m,
        "instants": sim.instants,
        "policy_calls": sim.core.policy_calls,
        "violations": len(viol),
    }
    if system.single_phase:
        row["lb_gap"] = _lb_gap(system, trace, m["gen_time_s"])
    d = sorted(sim.decision_ns)
    wrow = {
        "scenario": job["scenario"],
        "variant": job["variant"],
        "policy": job["policy"],
        "seed": job["seed"],
        "wall_run_s": wall,
        "wall_run_cpu_s": cpu,
        "wall_instants_per_s": sim.instants / wall if wall > 0 else math.nan,
        "wall_decision_mean_us": (sum(d) / len(d) / 1000) if d else 0.0,
        "wall_decision_p95_us": d[max(0, math.ceil(0.95 * len(d)) - 1)] / 1000 if d else 0.0,
        "wall_decision_max_us": d[-1] / 1000 if d else 0.0,
    }
    return row, wrow, viol


def _lb_gap(system, trace, gen_time_s: float) -> float:
    """Gap of the generation time to the lower bound of check 3."""
    p = system.engine
    W = system.n_workers
    if p.engine == "continuous":
        groups = [(g.prompt_tokens, list(g.resp_tokens)) for g in trace.groups]
        lb_ms = float(continuous_lower_bound_ns(p, groups, W)) / 1e6
    else:
        L = sorted(r for g in trace.groups for r in g.resp_tokens)
        P = max(g.prompt_tokens for g in trace.groups)
        dp, _ = dp_partition(L, p.static_batch, lambda b, lm, i, j: batch_cost_ms(p, b, P, lm))
        lb_ms = max(dp / W, batch_cost_ms(p, 1, P, L[-1]))
    return gen_time_s * 1000 / lb_ms - 1 if lb_ms > 0 else math.nan


def _sort_key(r: dict):
    return (r["scenario"], r["variant"], r["policy"], r["seed"])


def run_jobs(jobs: list[dict], workers: int, log=None) -> tuple[list[dict], list[dict], list[str]]:
    rows, walls, viols = [], [], []
    done = 0
    t0 = time.perf_counter()

    def note(res):
        nonlocal done
        r, w, v = res
        rows.append(r)
        walls.append(w)
        viols.extend(f"{_sort_key(r)}: {x}" for x in v)
        done += 1
        if log and (done % 200 == 0 or done == len(jobs)):
            log(f"  {done}/{len(jobs)} runs, {time.perf_counter() - t0:.0f} s")

    if workers <= 1:
        for j in jobs:
            note(run_job(j))
    else:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(run_job, j) for j in jobs]
            for f in cf.as_completed(futs):
                note(f.result())
    rows.sort(key=_sort_key)
    walls.sort(key=_sort_key)
    viols.sort()
    return rows, walls, viols


def jobs_for(
    cells, seeds_of=None, policies_of=None, trace_root: str = "outputs/traces"
) -> list[dict]:
    """Expand cells into jobs; generate the traces first (in this process)."""
    jobs = []
    for cell in cells:
        seeds = seeds_of(cell) if seeds_of else cell.seeds
        pols = policies_of(cell) if policies_of else cell.policies
        for seed in seeds:
            path = ensure_trace(cell.generator, seed, trace_root)
            for p in pols:
                jobs.append(
                    {
                        "scenario": cell.scenario,
                        "variant": cell.variant,
                        "policy": p.label,
                        "cfg": p.cfg,
                        "seed": seed,
                        "system": cell.system,
                        "trace_path": path,
                        "kind": cell.kind,
                    }
                )
    return jobs
