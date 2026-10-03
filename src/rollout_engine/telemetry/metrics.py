"""Run metrics (docs/contracts.md section 4). Pure functions of the core's records.

Closed loop: the measurement window drops the first ``warmup_steps`` steps and the last step
from every step-level metric; GPU and throughput metrics use the window interval
``[t_sel(warmup), t_sel(T-1))``. Single phase: the whole run. Percentiles are nearest rank.
Times in the result are seconds (floats with millisecond resolution).
"""

from __future__ import annotations

import math

from rollout_engine.scheduler.core import CONSUMED, DROPPED, UNFINISHED, Core, nearest_rank

GPU_STATES = ("gen", "overhead", "train", "idle")


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan


def integrate(tl: list[tuple[int, str]], a: int, b: int) -> dict[str, int]:
    """Milliseconds spent in each state of a step-function timeline within [a, b)."""
    out: dict[str, int] = {}
    if b <= a:
        return out
    for k, (t, st) in enumerate(tl):
        t_next = tl[k + 1][0] if k + 1 < len(tl) else b
        lo, hi = max(t, a), min(t_next, b)
        if hi > lo:
            out[st] = out.get(st, 0) + hi - lo
    return out


def _intervals(tl: list[tuple[int, str]], state: str, end: int) -> list[tuple[int, int]]:
    out = []
    for k, (t, st) in enumerate(tl):
        if st == state:
            t_next = tl[k + 1][0] if k + 1 < len(tl) else end
            if t_next > t:
                out.append((t, t_next))
    return out


def gpu_seconds(core: Core, a: int, b: int) -> tuple[dict[str, int], dict[str, int]]:
    """GPU-milliseconds per state within [a, b): (all GPUs, rollout GPUs)."""
    sys = core.sys
    tp = sys.engine.tp
    roll: dict[str, int] = dict.fromkeys(GPU_STATES, 0)
    train_iv = _intervals(core.trainer_tl, "train", max(b, core.t_end))
    for tl in core.worker_tl:
        d = integrate(tl, a, b)
        for st, ms in d.items():
            if st == "parked":
                # colocated: parked GPUs train while the trainer trains, else they idle
                tr = 0
                for lo, hi in _intervals(tl, "parked", max(b, core.t_end)):
                    for tlo, thi in train_iv:
                        tr += max(0, min(hi, thi, b) - max(lo, tlo, a))
                roll["train"] += tr * tp
                roll["idle"] += (ms - tr) * tp
            else:
                roll[st] += ms * tp
    tot = dict(roll)
    if not sys.colocated and not core.single:
        d = integrate(core.trainer_tl, a, b)
        g = sys.train_gpus
        tot["train"] += d.get("train", 0) * g
        tot["overhead"] += d.get("sync", 0) * g
        tot["idle"] += (d.get("wait", 0) + d.get("switch", 0)) * g
    return tot, roll


def _straggler(core: Core, gidxs: list[int]) -> int:
    rt = sorted(core.groups[g].ready_at for g in gidxs)
    if not rt:
        return 0
    k90 = max(1, math.ceil(0.9 * len(rt)))
    return rt[-1] - rt[k90 - 1]


def run_metrics(core: Core, trace=None) -> dict:
    """Every metric of docs/contracts.md section 4 (closed loop or single phase)."""
    out: dict = {}
    L = None
    if trace is not None:
        L = [r for g in trace.groups for r in g.resp_tokens]
    # whole-run accounting
    tok_cons = sum(g.tokens for g in core.groups if g.state == CONSUMED)
    tok_drop = core.wasted_tokens
    out["groups_consumed"] = sum(1 for g in core.groups if g.state == CONSUMED)
    out["groups_dropped"] = sum(1 for g in core.groups if g.state == DROPPED)
    out["groups_unfinished"] = sum(1 for g in core.groups if g.state == UNFINISHED)
    reasons: dict[str, int] = {}
    for g in core.groups:
        if g.state == DROPPED:
            reasons[g.drop_reason] = reasons.get(g.drop_reason, 0) + 1
    for r in ("policy", "stale", "colocated_switch"):
        out[f"drops_{r}"] = reasons.get(r, 0)
    out["tokens_consumed"] = tok_cons
    out["tokens_dropped"] = tok_drop
    out["waste_frac"] = tok_drop / (tok_drop + tok_cons) if tok_drop + tok_cons else 0.0
    if L is not None:
        cons = [L[s] for g in core.groups if g.state == CONSUMED for s in g.sids]
        drop = [L[s] for g in core.groups if g.state == DROPPED for s in g.sids]
        if drop and cons:
            out["length_bias"] = _mean(cons) / _mean(cons + drop) - 1
        else:
            out["length_bias"] = 0.0
    out["run_s"] = core.t_end / 1000
    if core.single:
        _single_phase(core, out)
    else:
        _closed_loop(core, out)
    return out


def _single_phase(core: Core, out: dict) -> None:
    gen_t = max((s.gen_t for s in core.samples), default=0)
    out["phase_time_s"] = core.t_end / 1000
    out["gen_time_s"] = gen_t / 1000
    out["straggler_s"] = _straggler(core, [g.gidx for g in core.groups]) / 1000
    out["mean_ready_s"] = _mean(g.ready_at for g in core.groups) / 1000
    tot, roll = gpu_seconds(core, 0, gen_t)
    total = sum(roll.values())
    out["gpu_idle_frac"] = roll["idle"] / total if total else 0.0
    for st in ("gen", "overhead"):
        out[f"{st}_frac"] = roll[st] / total if total else 0.0
    if core.batches:
        num = sum(core.samples[s].tokens for b in core.batches for s in b)
        den = sum(len(b) * max(core.samples[s].tokens for s in b) for b in core.batches)
        out["padding_frac"] = 1 - num / den if den else 0.0
    else:
        out["padding_frac"] = 0.0
    vd = [s.ver_end - s.gen_t for s in core.samples if s.ver_end >= 0]
    out["verify_delay_mean_s"] = _mean(vd) / 1000 if vd else 0.0
    out["verify_delay_p95_s"] = nearest_rank(vd, 0.95) / 1000 if vd else 0.0


def _closed_loop(core: Core, out: dict) -> None:
    sys = core.sys
    T, warm = core.T, sys.warmup_steps
    st = core.steps
    win = list(range(warm, T - 1))
    a, b = st[warm].t_sel, st[T - 1].t_sel
    span = b - a
    out["window_s"] = span / 1000
    step_ms = [st[s + 1].t_sel - st[s].t_sel for s in win]
    out["step_time_mean_s"] = span / len(win) / 1000
    out["step_time_p95_s"] = nearest_rank(step_ms, 0.95) / 1000
    n_samples = sum(core.groups[g].spec.n_samples for s in win for g in st[s].groups)
    out["samples_per_s"] = n_samples / (span / 1000)
    tok = sum(x.tokens for x in core.samples if x.end_t >= a and x.end_t < b)
    out["completed_tokens_per_s"] = tok / (span / 1000)
    tot, roll = gpu_seconds(core, a, b)
    tg, rg = sum(tot.values()), sum(roll.values())
    out["gpu_idle_frac"] = tot["idle"] / tg if tg else 0.0
    out["rollout_idle_frac"] = roll["idle"] / rg if rg else 0.0
    for s_ in ("gen", "overhead", "train"):
        out[f"gpu_{s_}_frac"] = tot[s_] / tg if tg else 0.0
    # trainer
    tw = integrate(core.trainer_tl, a, b)
    out["trainer_wait_frac"] = tw.get("wait", 0) / span if span else 0.0
    per_step_wait = [
        integrate(core.trainer_tl, st[s].t_sel, st[s + 1].t_sel).get("wait", 0) for s in win
    ]
    out["trainer_wait_mean_s"] = _mean(per_step_wait) / 1000
    strag = [_straggler(core, st[s].groups) for s in win]
    out["straggler_mean_s"] = _mean(strag) / 1000
    out["straggler_p95_s"] = nearest_rank(strag, 0.95) / 1000
    stale = [x for s in win for x in st[s].staleness]
    out["staleness_mean"] = _mean(stale)
    out["staleness_max"] = max(stale) if stale else 0
    fracs, means = [], []
    for s in win:
        num = den = 0
        lens = []
        for g in st[s].groups:
            for sid in core.groups[g].sids:
                smp = core.samples[sid]
                den += smp.tokens
                lens.append(smp.tokens)
                num += sum(n for v, n in smp.segs if v < s)
        fracs.append(num / den if den else 0.0)
        means.append(_mean(lens))
    out["stale_token_frac"] = _mean(fracs)
    mu = _mean(means)
    sd = math.sqrt(_mean((m - mu) ** 2 for m in means)) if len(means) > 1 else 0.0
    out["batch_len_cv"] = sd / mu if mu else 0.0
    vd = [
        core.samples[sid].ver_end - core.samples[sid].gen_t
        for s in win
        for g in st[s].groups
        for sid in core.groups[g].sids
    ]
    out["verify_delay_mean_s"] = _mean(vd) / 1000
    out["verify_delay_p95_s"] = nearest_rank(vd, 0.95) / 1000
