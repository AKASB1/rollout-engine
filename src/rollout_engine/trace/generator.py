"""Synthetic rollout trace generator (assumed distributions; docs/simulator.md section 2).

- Tasks: a categorical mix (``share`` per task).
- Prompt lengths: log-normal per task, clipped to ``[prompt_min, prompt_max]``, or one fixed
  ``prompt_fixed`` length for every group.
- Response lengths: log-normal with a group-level latent. For group g and sample i,
  ``log L = log(len_median) + len_sigma * (sqrt(rho) * z_g + sqrt(1 - rho) * e_gi)``, so the
  within-group correlation of the log lengths is ``rho``; ``L`` is rounded up to whole tokens
  and clipped to ``[1, max_tokens]`` (a value equal to ``max_tokens`` is a truncated response).
- Estimates ``est_tokens`` (one per group): ``none`` = the task mean of the clipped length;
  ``group`` = the latent group mean (the conditional mean of the clipped length given z_g)
  times ``exp(error_sigma * u_g)``; ``error_sigma = 0`` gives the exact group mean.
- Verifier times: log-normal per task (median, sigma), capped at ``max_s``, whole ms.

Each quantity has its own random stream, so changing the estimate model or the verifier
times never changes the response lengths of a seed (common random numbers).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from rollout_engine.rng import stream
from rollout_engine.trace.schema import GroupRow, Trace

GENERATOR_NAME = "rollout-gen"
GENERATOR_VERSION = 1

DEFAULT_TASKS = {
    "code": {
        "share": 0.4,
        "prompt_median": 600,
        "prompt_sigma": 0.5,
        "prompt_min": 32,
        "prompt_max": 4096,
        "len_median": 900,
        "len_sigma": 1.1,
        "len_rho": 0.6,
        "verify_median_s": 2.0,
        "verify_sigma": 1.0,
        "verify_max_s": 60.0,
    },
    "math": {
        "share": 0.6,
        "prompt_median": 300,
        "prompt_sigma": 0.5,
        "prompt_min": 32,
        "prompt_max": 4096,
        "len_median": 700,
        "len_sigma": 1.1,
        "len_rho": 0.6,
        "verify_median_s": 0.2,
        "verify_sigma": 0.3,
        "verify_max_s": 10.0,
    },
}

DEFAULT_GENERATOR = {
    "groups": 4608,
    "n_samples": 8,
    "max_tokens": 8192,
    "prompt_fixed": None,
    "estimate": {"model": "group", "error_sigma": 0.3},
    "tasks": DEFAULT_TASKS,
}


def clipped_lognormal_mean(mu, s, cap):
    """E[min(X, cap)] for X ~ LogNormal(mu, s), vectorized over mu and s."""
    mu, s = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(s, dtype=float))
    lc = math.log(cap)
    safe = np.where(s > 0, s, 1.0)
    val = np.exp(mu + safe * safe / 2) * ndtr((lc - mu - safe * safe) / safe) + cap * (
        1 - ndtr((lc - mu) / safe)
    )
    return np.where(s > 0, val, np.minimum(np.exp(mu), cap))


def verify_mean_s(gen: dict) -> dict:
    """Expected verifier service time per task (what policies may know), uncapped log-normal."""
    out = {}
    for name in sorted(gen["tasks"]):
        t = gen["tasks"][name]
        out[name] = round(t["verify_median_s"] * math.exp(t["verify_sigma"] ** 2 / 2), 3)
    return out


def validate(gen: dict) -> None:
    if gen.get("groups", 0) < 0 or gen.get("n_samples", 0) < 1 or gen.get("max_tokens", 0) < 1:
        raise ValueError("generator: groups >= 0, n_samples >= 1, max_tokens >= 1 required")
    tasks = gen.get("tasks") or {}
    if not tasks:
        raise ValueError("generator: at least one task required")
    total = sum(t["share"] for t in tasks.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"generator: task shares sum to {total}, not 1")
    for name, t in tasks.items():
        if not (0 <= t["len_rho"] <= 1) or t["len_sigma"] < 0 or t["len_median"] <= 0:
            raise ValueError(f"generator: bad length parameters for task {name}")
    if gen["estimate"]["model"] not in ("none", "group"):
        raise ValueError("generator: estimate.model must be none or group")


def generate(gen: dict, seed: int) -> Trace:
    validate(gen)
    n = int(gen["groups"])
    k = int(gen["n_samples"])
    cap = int(gen["max_tokens"])
    names = sorted(gen["tasks"])
    tasks = [gen["tasks"][x] for x in names]
    shares = np.cumsum([t["share"] for t in tasks])
    shares[-1] = 1.0

    u = stream(seed, "trace.tasks").random(n)
    tidx = np.searchsorted(shares, u, side="right")
    tidx = np.minimum(tidx, len(names) - 1)

    zp = stream(seed, "trace.prompts").standard_normal(n)
    zg = stream(seed, "trace.lengths.group").standard_normal(n)
    zw = stream(seed, "trace.lengths.sample").standard_normal((n, k))
    ue = stream(seed, "trace.estimates").standard_normal(n)
    zv = stream(seed, "trace.verify").standard_normal((n, k))

    def per_task(key):
        return np.array([tasks[i][key] for i in range(len(tasks))], dtype=float)[tidx]

    if gen.get("prompt_fixed"):
        prompt = np.full(n, int(gen["prompt_fixed"]), dtype=np.int64)
    else:
        pm, ps = per_task("prompt_median"), per_task("prompt_sigma")
        prompt = np.ceil(pm * np.exp(ps * zp)).astype(np.int64)
        prompt = np.clip(
            prompt, per_task("prompt_min").astype(np.int64), per_task("prompt_max").astype(np.int64)
        )

    mu = np.log(per_task("len_median"))
    sig = per_task("len_sigma")
    rho = per_task("len_rho")
    sg = sig * np.sqrt(rho)
    sw = sig * np.sqrt(1 - rho)
    logl = (mu + sg * zg)[:, None] + sw[:, None] * zw
    shift = gen.get("shift") or {}
    at = int(shift.get("at_group", n)) if shift else n
    if at < n:
        # distribution shift (Tier 2): from group `at` on, the log-scale spread is multiplied
        # by len_sigma_factor; the estimates keep the pre-shift calibration
        f = float(shift.get("len_sigma_factor", 1.0))
        logl[at:] = (mu[at:] + f * sg[at:] * zg[at:])[:, None] + (f * sw[at:])[:, None] * zw[at:]
    resp = np.clip(np.ceil(np.exp(logl)), 1, cap).astype(np.int64)

    est_cfg = gen["estimate"]
    if est_cfg["model"] == "none":
        est = clipped_lognormal_mean(mu, sig, cap)
    else:
        exact = clipped_lognormal_mean(mu + sg * zg, sw, cap)
        err = np.full(n, float(est_cfg.get("error_sigma", 0.0)))
        if at < n and shift.get("error_sigma") is not None:
            err[at:] = float(shift["error_sigma"])  # the estimates degrade from group `at` on
        est = exact * np.exp(err * ue)
    est = np.maximum(np.round(est, 3), 0.001)

    vmed, vsig, vmax = (
        per_task("verify_median_s"),
        per_task("verify_sigma"),
        per_task("verify_max_s"),
    )
    vs = np.minimum(vmed[:, None] * np.exp(vsig[:, None] * zv), vmax[:, None])
    vms = np.floor(vs * 1000 + 0.5).astype(np.int64)

    groups = []
    for i in range(n):
        groups.append(
            GroupRow(
                group_id=f"g{i:06d}",
                task=names[int(tidx[i])],
                prompt_tokens=int(prompt[i]),
                max_tokens=cap,
                est_tokens=float(est[i]),
                resp_tokens=tuple(int(x) for x in resp[i]),
                verify_ms=tuple(int(x) for x in vms[i]),
            )
        )
    return Trace(tuple(groups))


def generator_meta(gen: dict) -> dict:
    return {"name": GENERATOR_NAME, "version": GENERATOR_VERSION, "params": gen}
