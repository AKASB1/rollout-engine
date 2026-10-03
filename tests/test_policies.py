"""Unit tests of every policy part on hand-made views (a core with crafted worker reports)."""

import math

import pytest

from helpers import tiny_system
from rollout_engine.api import Drop, Place, specs_from_trace
from rollout_engine.policies.composed import ComposedPolicy, PolicyConfigError, make_policy
from rollout_engine.scheduler.core import Core
from rollout_engine.trace.schema import GroupRow, Trace


def trace_of(ests, n=2, lengths=None, prompt=4, max_tokens=50, tasks=None):
    gs = []
    for i, e in enumerate(ests):
        ls = lengths[i] if lengths else [10] * n
        task = tasks[i] if tasks else "math"
        gs.append(
            GroupRow(f"g{i}", task, prompt, max_tokens, float(e), tuple(ls), tuple([0] * len(ls)))
        )
    return Trace(tuple(gs))


def mk(ests, policy_params=None, n=2, workers=2, B=2, engine="continuous", lengths=None, **sk):
    sysc = tiny_system(workers=workers, B=B, engine=engine, **sk)
    tr = trace_of(ests, n=n, lengths=lengths)
    core = Core(sysc, specs_from_trace(tr))
    pol = ComposedPolicy(policy_params or {})
    pol.bind(sysc, tr)
    return core, pol, tr


def snap(**kw):
    s = dict(
        version=0,
        paused=False,
        draining=False,
        parked=False,
        admitted=0,
        waiting=0,
        kv_used=0,
        iters=0,
        busy=False,
    )
    s.update(kw)
    return s


def act(core, pol, t=0):
    acts = pol.act(core.view_for(t))
    for a in acts:
        core.apply(a, t)
    return acts


def places(acts):
    return [(a.kind, a.ref, a.worker) for a in acts if isinstance(a, Place)]


# ---------------------------------------------------------------- estimators
def test_estimators_prior_group_evidence_oracle():
    core, pol, tr = mk([100, 100], {"estimator": "group_evidence", "k0": 2.0}, n=3, B=2)
    act(core, pol)  # reference-like early launch of g0 -> w0, g1 -> w1
    v = core.view_for(0)
    s0, s1, s2 = core.groups[0].sids
    assert pol.est_sample(v, s0) == 100
    # s0 starts, finishes with 40 tokens; s1 started at iteration 0, worker 0 now at iteration 70
    core.on_report(
        0,
        5,
        [("admit", s0), ("admit", s1), ("start", s0, 0, 0), ("start", s1, 0, 0)],
        [],
        snap(admitted=2),
    )
    core.on_report(0, 40, [("finish", s0, 40, [[0, 40]])], [], snap(admitted=1, iters=70))
    v = core.view_for(40)
    m = (2 * 100 + 40) / 3
    assert pol.est_sample(v, s2) == pytest.approx(m)  # unstarted: (k0 est + sum) / (k0 + n)
    assert pol.est_sample(v, s1) == pytest.approx(max(m, 71))  # running: max(m, tokens + 1)
    assert v.sample(s1).tokens_so_far == 70
    assert pol.remaining(v, s1) == pytest.approx(m - 70)
    prior = ComposedPolicy({"estimator": "prior"})
    prior.bind(core.sys, tr)
    assert prior.est_sample(v, s2) == 100
    oracle = ComposedPolicy({"estimator": "oracle"})
    oracle.bind(core.sys, trace_of([100, 100], n=3, lengths=[[7, 8, 9], [1, 2, 3]]))
    assert oracle.est_sample(v, s2) == 9 and oracle.oracle


# ---------------------------------------------------------------- order and window
@pytest.mark.parametrize(
    "order,expected",
    [("fifo", [0, 1]), ("shortest_first", [3, 1]), ("longest_first", [2, 4])],
)
def test_launch_order_within_window(order, expected):
    core, pol, _ = mk([50, 20, 90, 10, 99], {"order": order, "window_groups": 4})
    acts = act(core, pol)
    assert [a.ref for a in acts] == expected  # group 4 is outside the window of 4


def test_window_anti_starvation():
    # longest_first with window 2: the oldest unlaunched group (g0, smallest) launches after
    # it has been the oldest for 2 launches
    core, pol, _ = mk(
        [1, 50, 60, 70, 80], {"order": "longest_first", "window_groups": 2}, B=1, workers=1
    )
    launched = []
    for _ in range(4):
        acts = act(core, pol)
        launched += [a.ref for a in acts]
        g = core.groups[acts[0].ref]
        core._consume(g, 0)  # free the admission cap (cap = B = 1)
    assert launched == [1, 2, 0, 4]


# ---------------------------------------------------------------- dispatch, binding, assign
def test_early_group_round_robin_is_the_even_split():
    core, pol, _ = mk([10] * 6, make_policy({"name": "reference"}).p, B=4)
    acts = act(core, pol)
    assert places(acts) == [("group", 0, 0), ("group", 1, 1), ("group", 2, 0), ("group", 3, 1)]
    assert core.groups[4].state == "unlaunched"  # cap = B = 4


def test_early_sample_round_robin_spreads_samples():
    core, pol, _ = mk([10] * 4, {"dispatch": "sample"}, n=3, B=1)
    acts = act(core, pol)
    s = core.groups[0].sids
    assert places(acts) == [("sample", s[0], 0), ("sample", s[1], 1), ("sample", s[2], 0)]


def test_late_binding_places_only_on_accepting_workers_with_room():
    core, pol, _ = mk(
        [10] * 6, {"binding": "late", "assign": "first_free"}, n=2, B=6, max_seqs=3, workers=3
    )
    # worker 0 is paused, worker 1 has 2 running (1 slot left), worker 2 is empty
    core.on_report(0, 0, [], [], snap(paused=True))
    core.on_report(1, 0, [], [], snap(admitted=2))
    acts = act(core, pol)
    # groups of 2 samples: only worker 2 fits one group (3 slots); nothing waits anywhere
    assert places(acts) == [("group", 0, 2)]


def test_late_sample_first_free_fills_free_slots_in_id_order():
    core, pol, _ = mk(
        [10] * 4,
        {"binding": "late", "assign": "first_free", "dispatch": "sample"},
        n=2,
        B=4,
        max_seqs=3,
    )
    core.on_report(0, 0, [], [], snap(admitted=2))
    acts = act(core, pol)
    ws = [w for _, _, w in places(acts)]
    assert ws == [0, 1, 1, 1]  # 1 slot on worker 0, then 3 on worker 1; then no room


def test_least_loaded_and_lpt():
    core, pol, _ = mk([100, 10, 50, 80], {"assign": "least_loaded", "dispatch": "group"}, n=1, B=4)
    acts = act(core, pol)
    # loads start at 0: g0 (100) -> w0, g1 (10) -> w1, g2 (50) -> w1 (10 < 100), g3 (80) -> w1 (60 < 100)
    assert places(acts) == [("group", 0, 0), ("group", 1, 1), ("group", 2, 1), ("group", 3, 1)]
    core, pol, _ = mk([100, 10, 50, 80], {"assign": "lpt", "dispatch": "group"}, n=1, B=4)
    acts = act(core, pol)
    # decreasing work: g0 (100) -> w0, g3 (80) -> w1, g2 (50) -> w1? no: loads 100 vs 80 -> w1 (130), g1 (10) -> w0 (110)
    assert places(acts) == [("group", 0, 0), ("group", 3, 1), ("group", 2, 1), ("group", 1, 0)]


# ---------------------------------------------------------------- admission / stragglers
@pytest.mark.parametrize(
    "straggler,rho,cap",
    [("wait", 1.5, 2), ("carry", 1.5, 3), ("carry", 2.0, 4), ("abort", 1.25, 3)],
)
def test_admission_cap(straggler, rho, cap):
    core, pol, _ = mk([10] * 8, {"straggler": straggler, "rho": rho}, B=2)
    acts = act(core, pol)
    assert len(acts) == cap == pol.cap(core.view_for(0))
    assert cap == (2 if straggler == "wait" else math.ceil(rho * 2))


def test_abort_drops_every_unselected_group_after_a_selection():
    core, pol, _ = mk([10] * 8, {"straggler": "abort", "rho": 2.0}, B=2, eta=1)
    act(core, pol)  # 4 groups
    core.s_next = 1  # a selection happened (groups 0, 1 consumed)
    for gi in (0, 1):
        core._consume(core.groups[gi], 0)
    acts = pol.act(core.view_for(5))
    assert [a for a in acts if isinstance(a, Drop)] == [Drop(2), Drop(3)]
    assert len(places(acts)) == 4  # refilled to the cap


def test_gate_closed_launches_nothing():
    core, pol, _ = mk([10] * 4, {}, B=2, eta=0)
    core.s_next = 1  # selected step 0, v_pub still 0 -> gate closed
    assert pol.act(core.view_for(1)) == []


# ---------------------------------------------------------------- verifier order
def test_verifier_orders():
    sysc = tiny_system(workers=1, B=2)
    tr = Trace(
        (
            GroupRow("a", "code", 1, 10, 5.0, (1, 1, 1), (0, 0, 0)),
            GroupRow("b", "math", 1, 10, 5.0, (1, 1), (0, 0)),
        )
    )
    core = Core(sysc, specs_from_trace(tr))
    q = [
        type("E", (), dict(sid=s, gidx=g, task=t))()
        for s, g, t in [(0, 0, "code"), (3, 1, "math"), (1, 0, "code")]
    ]
    core.groups[1].n_verified = 1  # group b has 1 unverified sample left, group a has 3
    v = core.view_for(0)
    assert ComposedPolicy({"verifier": "fifo"}).pick_verification(v, q) == 0
    assert ComposedPolicy({"verifier": "group_first"}).pick_verification(v, q) == 1
    assert (
        ComposedPolicy({"verifier": "shortest_first"}).pick_verification(v, q) == 1
    )  # math 0 s < code 1 s


# ---------------------------------------------------------------- static batch formation
def test_static_batch_formation():
    ests = [30, 10, 20, 40]
    for batch, expected in [
        ("fifo_chunk", [(0, 1), (2, 3)]),  # group order, chunks of 2
        ("sorted_chunk", [(3, 0), (2, 1)]),  # by estimate descending
    ]:
        core, pol, _ = mk(
            ests, {"batch": batch, "batch_size": 2}, n=1, B=4, engine="static", workers=2
        )
        acts = act(core, pol)
        assert [a.ref for a in acts] == expected and [a.worker for a in acts] == [0, 1]
    # dp: optimal contiguous partition of the estimate-sorted pool; longest batch first.
    # cost(b, L) = L * (1 + b) ms: one batch costs 200, [1,1,1] + [40] cost 4 + 80
    core, pol, _ = mk(
        [1, 1, 1, 40],
        {"batch": "dp"},
        n=1,
        B=4,
        engine="static",
        workers=2,
        static_batch=4,
        a1_ns=1_000_000,
    )
    acts = act(core, pol)
    assert [a.ref for a in acts] == [(3,), (0, 1, 2)]


def test_static_only_idle_workers_get_batches():
    core, pol, _ = mk(
        [10] * 4, {"batch": "fifo_chunk", "batch_size": 1}, n=1, B=4, engine="static", workers=2
    )
    core.on_report(0, 0, [], [], snap(busy=True))
    acts = act(core, pol)
    assert [(a.ref, a.worker) for a in acts] == [((0,), 1)]


# ---------------------------------------------------------------- composition and factory
def test_named_compositions_and_factory():
    oa = make_policy({"name": "online_adaptive", "params": {"k0": 4}})
    assert (
        oa.p["estimator"] == "group_evidence"
        and oa.p["dispatch"] == "sample"
        and oa.p["assign"] == "lpt"
    )
    assert oa.name == "longest_first+sample+late+lpt+wait+group_evidence(4)+v_fifo"
    ref = make_policy({"name": "reference"})
    assert ref.name == "fifo+group+early+round_robin+wait+prior+v_fifo"
    assert make_policy(
        {"name": "composed", "params": {"straggler": "carry", "rho": 1.5}}
    ).name.endswith("carry(1.5)+prior+v_fifo")
    assert make_policy({"name": "oracle_lpt"}).oracle
    with pytest.raises(PolicyConfigError):
        make_policy({"name": "nope"})
    with pytest.raises(PolicyConfigError):
        make_policy({"name": "composed", "params": {"binding": "early", "assign": "first_free"}})
    with pytest.raises(PolicyConfigError):
        make_policy({"name": "composed", "params": {"colour": "red"}})
