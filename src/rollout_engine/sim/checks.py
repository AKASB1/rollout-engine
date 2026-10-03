"""Accounting identities (check 4) and staleness/version invariants (check 5) of one run.

Each function returns a list of violation messages (empty = the run is consistent). The
benchmark runs ``accounting_violations`` on every run of the quick benchmark.
"""

from __future__ import annotations

from rollout_engine.scheduler.core import CONSUMED, DROPPED, UNFINISHED, UNLAUNCHED
from rollout_engine.telemetry.metrics import gpu_seconds, integrate


def accounting_violations(sim) -> list[str]:
    core = sim.core
    sys = core.sys
    out: list[str] = []
    T = core.t_end
    # per GPU, the states add up to the run duration
    for w, tl in enumerate(core.worker_tl):
        tot = sum(integrate(tl, 0, T).values())
        if tot != T:
            out.append(f"worker {w}: states sum to {tot} ms, run is {T} ms")
    tr = sum(integrate(core.trainer_tl, 0, T).values())
    if tr != T:
        out.append(f"trainer: states sum to {tr} ms, run is {T} ms")
    alls, _ = gpu_seconds(core, 0, T)
    gpus = sys.total_gpus if not core.single else sys.n_workers * sys.engine.tp
    if sum(alls.values()) != gpus * T:
        out.append(f"GPU-ms {sum(alls.values())} != {gpus} GPUs x {T} ms")
    # every generated token belongs to a consumed, dropped, or unfinished group
    by_state = {CONSUMED: 0, DROPPED: 0, UNFINISHED: 0}
    for g in core.groups:
        if g.state in by_state:
            by_state[g.state] += g.tokens
        elif g.state == UNLAUNCHED:
            if g.tokens or g.launched_at >= 0:
                out.append(f"unlaunched group {g.spec.gid} has tokens or a launch time")
        else:
            out.append(f"group {g.spec.gid} ended in state {g.state}")
    gen = sim.generated_tokens()
    if gen != sum(by_state.values()):
        out.append(
            f"generated tokens {gen} != consumed+dropped+unfinished {sum(by_state.values())}"
        )
    if by_state[DROPPED] != core.wasted_tokens:
        out.append(f"dropped tokens {by_state[DROPPED]} != wasted counter {core.wasted_tokens}")
    # every launched group ends in exactly one terminal state
    for g in core.groups:
        if g.launched_at >= 0 and g.state not in (CONSUMED, DROPPED, UNFINISHED):
            out.append(f"launched group {g.spec.gid} is {g.state}")
    # a sample is generated at most once (the core raises on a second finish); finished
    # samples of consumed groups carry their full length in their version segments
    for g in core.groups:
        if g.state == CONSUMED:
            for sid in g.sids:
                s = core.samples[sid]
                if sum(n for _, n in s.segs) != s.tokens:
                    out.append(f"sample {sid}: version segments do not add up to its tokens")
    # the trainer's time: train intervals equal the training durations
    if not core.single:
        train = integrate(core.trainer_tl, 0, T).get("train", 0)
        exp = sum(sys.train_ms(st.tokens) for st in core.steps if st.t_train_end >= 0)
        if train != exp:
            out.append(f"trainer train time {train} != sum of step training times {exp}")
    # the tokens of the selected batches add up to the tokens of the consumed groups
    cons = sum(
        g.spec.prompt_tokens * g.spec.n_samples + g.tokens
        for g in core.groups
        if g.state == CONSUMED
    )
    sel = sum(st.tokens for st in core.steps) if not core.single else cons
    if sel != cons:
        out.append(f"selected batch tokens {sel} != consumed group tokens {cons}")
    return out


def staleness_violations(sim) -> list[str]:
    core = sim.core
    sys = core.sys
    out: list[str] = []
    if core.single:
        return out
    seen: set[int] = set()
    for st in core.steps:
        if len(st.groups) != core.B:
            out.append(f"step {st.s} consumed {len(st.groups)} groups, B = {core.B}")
        for gi, x in zip(st.groups, st.staleness, strict=True):
            if gi in seen:
                out.append(f"group {gi} consumed twice")
            seen.add(gi)
            if x > sys.eta or x < 0:
                out.append(f"step {st.s}: staleness {x} outside [0, eta={sys.eta}]")
            if sys.eta == 0 and x != 0:
                out.append(f"step {st.s}: eta=0 but staleness {x}")
        if sys.eta == 0:
            for gi in st.groups:
                for sid in core.groups[gi].sids:
                    if any(v < st.s and n > 0 for v, n in core.samples[sid].segs):
                        out.append(f"eta=0: step {st.s} consumed tokens of an older version")
    pubs = [st.t_pub for st in core.steps if st.t_pub >= 0]
    if pubs != sorted(pubs) or core.v_pub != len(pubs):
        out.append(f"publications out of order or repeated: {pubs}, v_pub={core.v_pub}")
    if sys.colocated:
        for st in core.steps:
            for g in core.groups:
                if 0 <= g.launched_at < st.t_sel:
                    ok = (g.state == CONSUMED and g.step <= st.s) or (
                        g.state == DROPPED and g.drop_t <= st.t_sel
                    )
                    if not ok:
                        out.append(f"colocated: group {g.spec.gid} survived step {st.s}")
    return out
