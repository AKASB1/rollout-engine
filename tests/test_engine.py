"""Check 1: closed-form engines against the step-by-step reference, plus hand-computed cases."""

from dataclasses import replace

import numpy as np
import pytest

from rollout_engine.config import EngineParams
from rollout_engine.workers.engine import (
    ContinuousEngine,
    EngineError,
    ESample,
    StaticEngine,
    first_k_exceeding,
    s_of,
    static_batch_ms,
)
from rollout_engine.workers.harness import run_single
from rollout_engine.workers.reference import StepContinuousEngine, StepStaticEngine

MS = 1_000_000


def params(**kw):
    base = dict(
        engine="continuous",
        tp=1,
        a0_ns=MS,
        a1_ns=0,
        a2_ns=0,
        prefill_c0_ns=0,
        prefill_c1_ns=0,
        max_seqs=8,
        kv_tokens=10**9,
        static_batch=8,
    )
    base.update(kw)
    return EngineParams(**base)


_sid = iter(range(10**9))


def smp(gidx, sidx, P, M, L):
    return ESample(next(_sid), gidx, sidx, P, M, L)


def times(log, kind):
    return [(t, e[1]) for t, e in log if e[0] == kind]


# ------------------------------------------------------------------ S(k) inversion helpers
@pytest.mark.parametrize("A,D", [(MS, 0), (1_500_000, 0), (MS, 7), (2_345_678, 1234), (MS, 10**6)])
def test_first_k_exceeding_boundaries(A, D):
    assert first_k_exceeding(-1, A, D) == 0
    assert first_k_exceeding(0, A, D) == 1
    for k in [1, 2, 3, 10, 1000, 123457]:
        assert first_k_exceeding(s_of(k, A, D) - 1, A, D) == k
        assert first_k_exceeding(s_of(k, A, D), A, D) == k + 1


# ----------------------------------------------------------------------- hand-computed
def test_one_sample_alone_and_prefill():
    s = smp(0, 0, P=10, M=100, L=5)
    log, tl = run_single(ContinuousEngine(0, params()), [(0, ("place", [s]))])
    assert times(log, "admit") == [(0, s.sid)] and times(log, "finish") == [(5, s.sid)]
    s = smp(0, 0, P=10, M=100, L=5)
    log, tl = run_single(
        ContinuousEngine(0, params(prefill_c0_ns=2_500_000)), [(0, ("place", [s]))]
    )
    assert times(log, "admit") == [(0, s.sid)]
    assert times(log, "start") == [(3, s.sid)] and times(log, "finish") == [(8, s.sid)]
    assert tl == [(0, "overhead"), (3, "gen"), (8, "idle")]


def test_two_samples_of_different_length():
    a, b = smp(0, 0, 1, 100, 3), smp(0, 1, 1, 100, 5)
    log, _ = run_single(ContinuousEngine(0, params(a1_ns=MS // 2)), [(0, ("place", [a, b]))])
    # b=2: 2 ms per iteration -> a done after 3 iterations at 6 ms; then 1.5 ms per iteration
    # for 2 more iterations -> ceil(3.0) -> 9 ms
    assert times(log, "finish") == [(6, a.sid), (9, b.sid)]


def test_placement_in_the_middle_of_a_phase():
    a, b = smp(0, 0, 1, 100, 10), smp(1, 0, 1, 100, 4)
    eng = ContinuousEngine(0, params(a0_ns=1_500_000))
    log, _ = run_single(eng, [(0, ("place", [a])), (4, ("place", [b]))])
    # boundaries of a's phase at ceil(1.5 j): 2, 3, 5, 6, ... -> first at or after 4 is 5 (j=3)
    assert times(log, "admit") == [(0, a.sid), (5, b.sid)]
    # new phase at 5 (a has 3 tokens): a needs 7 more -> ceil(10.5) = 11 -> 16; b needs 4 -> 5+6=11
    assert times(log, "finish") == [(11, b.sid), (16, a.sid)]


def test_kv_limited_admission_with_prefix_sharing():
    p = params(kv_tokens=25)
    ss = [smp(0, i, P=5, M=10, L=3 + i) for i in range(3)]
    log, _ = run_single(ContinuousEngine(0, p), [(0, ("place", ss))])
    # 5 (prompt once) + 10 + 10 = 25 fits two samples; the third waits for a free reservation
    assert times(log, "admit") == [(0, ss[0].sid), (0, ss[1].sid), (3, ss[2].sid)]
    assert times(log, "finish") == [(3, ss[0].sid), (4, ss[1].sid), (8, ss[2].sid)]


def test_head_of_line_is_never_skipped():
    p = params(kv_tokens=30)
    big = smp(0, 0, P=5, M=25, L=4)
    big2 = smp(1, 0, P=5, M=20, L=1)
    small = smp(2, 0, P=1, M=2, L=1)
    log, _ = run_single(ContinuousEngine(0, p), [(0, ("place", [big, big2, small]))])
    # big2 does not fit beside big; small would fit but must wait behind big2
    assert times(log, "admit") == [(0, big.sid), (4, big2.sid), (4, small.sid)]


def test_draining_worker():
    a, b = smp(0, 0, 1, 100, 10), smp(1, 0, 1, 100, 3)
    eng = ContinuousEngine(0, params(), inflight="drain", swap_ms=2)
    log, tl = run_single(eng, [(0, ("place", [a])), (4, ("publish", 1)), (5, ("place", [b]))])
    assert times(log, "finish") == [(10, a.sid), (15, b.sid)]
    starts = [(t, e[1], e[2]) for t, e in log if e[0] == "start"]
    assert starts == [(0, a.sid, 0), (12, b.sid, 1)]
    fin = {e[1]: e[3] for _, e in log if e[0] == "finish"}
    assert fin[a.sid] == [[0, 10]] and fin[b.sid] == [[1, 3]]
    assert (10, "overhead") in tl and (12, "gen") in tl


def test_swap_and_interrupt():
    a = smp(0, 0, 10, 100, 10)
    log, _ = run_single(
        ContinuousEngine(0, params(), inflight="swap", swap_ms=3),
        [(0, ("place", [a])), (4, ("publish", 1))],
    )
    assert times(log, "finish") == [(13, a.sid)]
    assert [e[3] for _, e in log if e[0] == "finish"] == [[[0, 4], [1, 6]]]
    a = smp(0, 0, 10, 100, 10)
    log, _ = run_single(
        ContinuousEngine(0, params(prefill_c1_ns=100_000), inflight="interrupt", swap_ms=3),
        [(0, ("place", [a])), (4, ("publish", 1))],
    )
    # prefill of the prompt at 0: 10 * 0.1 ms -> 1 ms; tokens start at 1; boundary 5 has 4
    # tokens; pause = 3 ms + re-prefill of 14 tokens (1.4 ms) -> ceil 4.4 -> 5 ms -> resume 10
    assert times(log, "start") == [(1, a.sid)]
    assert times(log, "finish") == [(16, a.sid)]


def test_drop_of_a_running_group_takes_effect_at_the_next_boundary():
    a, b = smp(0, 0, 1, 100, 10), smp(1, 0, 1, 100, 10)
    eng = ContinuousEngine(0, params(a0_ns=1_500_000, max_seqs=1))
    log, _ = run_single(eng, [(0, ("place", [a, b])), (4, ("drop", 0))])
    assert [(t, e[0], e[1], e[2]) for t, e in log if e[0] == "remove"] == [(5, "remove", a.sid, 3)]
    assert times(log, "admit")[-1] == (5, b.sid)


def test_static_batch_hand_case():
    p = params(engine="static", prefill_c0_ns=MS)
    x, y = smp(0, 0, 10, 100, 3), smp(0, 1, 10, 100, 5)
    log, tl = run_single(StaticEngine(0, p), [(0, ("place", [x, y]))])
    assert times(log, "finish") == [(6, x.sid), (6, y.sid)]
    assert tl == [(0, "overhead"), (1, "gen"), (6, "idle")]
    assert static_batch_ms(p, 2, 10, 5) == 6
    with pytest.raises(EngineError):
        run_single(StaticEngine(0, p), [(0, ("place", [smp(0, i, 1, 9, 1) for i in range(9)]))])


# ----------------------------------------------------------- check 1: 2000+ random scenarios
def random_scenario(rng, mode, static=False):
    p = params(
        engine="static" if static else "continuous",
        a0_ns=int(rng.integers(MS, 3 * MS)),
        a1_ns=int(rng.integers(0, 300_000)),
        a2_ns=int(rng.integers(0, 3000)),
        prefill_c0_ns=int(rng.integers(0, 4 * MS)),
        prefill_c1_ns=int(rng.integers(0, 40_000)),
        max_seqs=int(rng.integers(1, 9)),
        kv_tokens=10**9,
        static_batch=int(rng.integers(1, 9)),
    )
    groups = []
    for g in range(int(rng.integers(1, 7))):
        P = int(rng.integers(1, 300))
        M = int(rng.integers(1, 200))
        groups.append(
            [smp(g, i, P, M, int(rng.integers(1, M + 1))) for i in range(int(rng.integers(1, 5)))]
        )
    if not static and rng.random() < 0.5:  # KV-limited
        need = max(s.P + s.M for gr in groups for s in gr)
        p = replace(p, kv_tokens=int(rng.integers(need, need + 3 * need)))
    inputs = []
    t = 0
    for gr in groups:
        t += int(rng.integers(0, 60))
        if static:
            batch = gr[: p.static_batch]
            inputs.append((t, ("place", batch)))
            continue
        if rng.random() < 0.5:
            inputs.append((t, ("place", gr)))
        else:
            for s in gr:
                inputs.append((t + int(rng.integers(0, 30)), ("place", [s])))
    if not static and rng.random() < 0.3:
        g = int(rng.integers(0, len(groups)))
        inputs.append((int(rng.integers(0, t + 200)), ("drop", g)))
    inputs.append((int(rng.integers(0, t + 300)), ("publish", 1)))
    if rng.random() < 0.3:
        inputs.append((int(rng.integers(0, t + 600)), ("publish", 2)))
    return p, inputs, int(rng.integers(0, 40))


def clone_inputs(inputs):
    out = []
    for t, item in inputs:
        if item[0] == "place":
            out.append(
                (t, ("place", [ESample(s.sid, s.gidx, s.sidx, s.P, s.M, s.L) for s in item[1]]))
            )
        else:
            out.append((t, item))
    return out


def _fmt(log):
    return [
        (t, e[0], *[(x if not isinstance(x, list) else [list(y) for y in x]) for x in e[1:]])
        for t, e in log
    ]


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["drain", "swap", "interrupt"])
def test_check1_continuous_closed_form_equals_step(mode):
    rng = np.random.default_rng(
        np.random.SeedSequence([1, ["drain", "swap", "interrupt"].index(mode)])
    )
    n_events = 0
    for _ in range(700):
        p, inputs, swap_ms = random_scenario(rng, mode)
        a = ContinuousEngine(0, p, inflight=mode, swap_ms=swap_ms)
        b = StepContinuousEngine(0, p, inflight=mode, swap_ms=swap_ms)
        la, ta = run_single(a, clone_inputs(inputs))
        lb, tb = run_single(b, clone_inputs(inputs))
        assert _fmt(la) == _fmt(lb)
        assert ta == tb
        n_events += len(la)
    assert n_events > 10_000


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["drain", "interrupt"])
def test_check1_static_closed_form_equals_step(mode):
    rng = np.random.default_rng(np.random.SeedSequence([2, ["drain", "interrupt"].index(mode)]))
    for _ in range(300):
        p, inputs, swap_ms = random_scenario(rng, mode, static=True)
        a = StaticEngine(0, p, inflight=mode, swap_ms=swap_ms)
        b = StepStaticEngine(0, p, inflight=mode, swap_ms=swap_ms)
        # the static engine takes one batch at a time: space placements so each finds it idle
        la, ta = run_single(a, clone_inputs(_serialize(inputs, p)))
        lb, tb = run_single(b, clone_inputs(_serialize(inputs, p)))
        assert _fmt(la) == _fmt(lb) and ta == tb


def _serialize(inputs, p):
    """Shift static placements so that each arrives after the previous batch can end."""
    out, t_free = [], 0
    for t, item in sorted(inputs, key=lambda x: x[0]):
        if item[0] == "place":
            b = item[1]
            t = max(t, t_free)
            dur = static_batch_ms(p, len(b), max(s.P for s in b), max(s.L for s in b))
            t_free = t + dur + 200  # leave room for a swap pause
        out.append((t, item))
    return out
