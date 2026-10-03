"""Checks 3 (continuous bounds), 4 (accounting identities) and 5 (staleness and version
invariants, misbehaving policies) on random configurations and every policy part."""

import itertools

import numpy as np
import pytest

from helpers import MS, tiny_system
from rollout_engine.api import Drop, InvalidAction, Place, RunError
from rollout_engine.opt.bounds import continuous_lower_bound_ns
from rollout_engine.policies.composed import ComposedPolicy, make_policy
from rollout_engine.sim.checks import accounting_violations, staleness_violations
from rollout_engine.sim.driver import simulate
from rollout_engine.trace.schema import GroupRow, Trace

CONT_PARTS = [
    dict(order=o, dispatch=d, binding=b, assign=a)
    for o, d, b, a in itertools.product(
        ["fifo", "shortest_first", "longest_first"],
        ["group", "sample"],
        ["early", "late"],
        ["first_free", "round_robin", "least_loaded", "lpt"],
    )
    if not (b == "early" and a == "first_free")
]


def random_trace(rng, n_groups, n_samples, max_tokens=60, prompt=None):
    groups = []
    for i in range(n_groups):
        P = prompt or int(rng.integers(1, 40))
        k = n_samples if n_samples else int(rng.integers(1, 5))
        ls = tuple(int(x) for x in rng.integers(1, max_tokens + 1, size=k))
        vs = tuple(int(x) for x in rng.integers(0, 6, size=k))
        task = "math" if rng.random() < 0.6 else "code"
        groups.append(GroupRow(f"g{i}", task, P, max_tokens, float(rng.integers(5, 50)), ls, vs))
    return Trace(tuple(groups))


def random_system(rng, static=False, single=False):
    T = int(rng.integers(4, 7))
    B = int(rng.integers(2, 6))
    return tiny_system(
        eta=int(rng.integers(0, 4)),
        T=T,
        B=B,
        F=int(rng.integers(1, 30)),
        S=int(rng.integers(0, 5)),
        W=int(rng.integers(0, 5)),
        switch=int(rng.integers(0, 6)),
        partition="colocated" if rng.random() < 0.25 else "disaggregated",
        inflight=str(rng.choice(["drain", "swap", "interrupt"])),
        warm=int(rng.integers(0, 2)),
        servers=int(rng.integers(1, 4)),
        engine="static" if static else "continuous",
        workers=int(rng.integers(1, 4)),
        mode="single_phase" if single else "closed",
        a0_ns=int(rng.integers(MS, 2 * MS)),
        a1_ns=int(rng.integers(0, 200_000)),
        a2_ns=int(rng.integers(0, 2000)),
        prefill_c0_ns=int(rng.integers(0, 2 * MS)),
        prefill_c1_ns=int(rng.integers(0, 20_000)),
        max_seqs=int(rng.integers(2, 9)),
        kv_tokens=int(rng.integers(150, 600)),
        static_batch=int(rng.integers(1, 7)),
    )


@pytest.mark.slow
def test_check4_check5_random_continuous_every_part():
    rng = np.random.default_rng(404)
    runs = 0
    for parts in CONT_PARTS:
        for straggler in ["wait", "carry", "abort"]:
            sysc = random_system(rng)
            tr = random_trace(rng, 3 * sysc.steps * sysc.groups_per_step + 10, 0)
            params = dict(parts, straggler=straggler, rho=float(rng.choice([1.25, 1.5, 2.0])))
            params["estimator"] = str(rng.choice(["prior", "group_evidence"]))
            params["window_groups"] = int(rng.integers(1, 6))
            params["verifier"] = str(rng.choice(["fifo", "group_first", "shortest_first"]))
            sim = simulate(sysc, tr, ComposedPolicy(params))
            assert accounting_violations(sim) == [], params
            assert staleness_violations(sim) == [], params
            runs += 1
    for name in ["online_adaptive", "oracle_lpt", "reference", "online_eb"]:
        for _ in range(10):
            sysc = random_system(rng)
            tr = random_trace(rng, 3 * sysc.steps * sysc.groups_per_step + 10, 0)
            sim = simulate(sysc, tr, make_policy({"name": name}))
            assert accounting_violations(sim) == [] and staleness_violations(sim) == []
            runs += 1
    assert runs >= 150


@pytest.mark.slow
def test_check4_check5_abort_pred_and_eb():
    rng = np.random.default_rng(406)
    for k in range(30):
        sysc = random_system(rng)
        tr = random_trace(rng, 3 * sysc.steps * sysc.groups_per_step + 10, 0)
        pol = make_policy(
            {
                "name": "online_eb",
                "params": {
                    "straggler": ["abort_pred", "carry", "wait"][k % 3],
                    "rho": 1.5,
                    "kappa": [1.5, 2.0, 4.0][k % 3],
                    "deadline": [0.0, 1.0, 2.0][k % 3],
                    "quantile": [0.0, 0.9, 0.5][(k // 3) % 3],
                },
            }
        )
        sim = simulate(sysc, tr, pol)
        assert accounting_violations(sim) == [] and staleness_violations(sim) == []


@pytest.mark.slow
def test_check4_check5_random_static():
    rng = np.random.default_rng(405)
    for k in range(60):
        sysc = random_system(rng, static=True)
        tr = random_trace(rng, 3 * sysc.steps * sysc.groups_per_step + 10, 0)
        batch = ["fifo_chunk", "sorted_chunk", "dp"][k % 3]
        straggler = ["wait", "carry", "abort"][(k // 3) % 3]
        est = ["prior", "group_evidence", "oracle"][(k // 9) % 3]
        pol = ComposedPolicy(
            dict(
                batch=batch, straggler=straggler, estimator=est, batch_size=int(rng.integers(0, 4))
            )
        )
        sim = simulate(sysc, tr, pol)
        assert accounting_violations(sim) == [] and staleness_violations(sim) == []


@pytest.mark.slow
def test_check3_continuous_bounds_every_part():
    rng = np.random.default_rng(303)
    for parts in CONT_PARTS + [
        dict(PRESET=n) for n in ("online_adaptive", "oracle_lpt", "online_eb")
    ]:
        for _ in range(3):
            sysc = random_system(rng, single=True)
            tr = random_trace(rng, int(rng.integers(2, 12)), 0)
            pol = (
                make_policy({"name": parts["PRESET"]})
                if "PRESET" in parts
                else ComposedPolicy(parts)
            )
            sim = simulate(sysc, tr, pol)
            assert accounting_violations(sim) == []
            gen_ms = max(s.gen_t for s in sim.core.samples)
            groups = [(g.prompt_tokens, list(g.resp_tokens)) for g in tr.groups]
            lb = continuous_lower_bound_ns(sysc.engine, groups, sysc.n_workers)
            assert gen_ms * MS >= lb, (parts, gen_ms, float(lb) / MS)


# ------------------------------------------------------------ misbehaving policies (check 5)
class Bad(ComposedPolicy):
    def __init__(self, mode):
        super().__init__({}, label=f"bad_{mode}")
        self.mode = mode

    def act(self, v):
        if self.mode == "gate":
            # keep launching one group after every selection, ignoring the gate
            if v.s_next >= 1 and v.n_unlaunched():
                return [Place("group", v.window(1)[0].gidx, 0)]
            return super().act(v)
        if self.mode == "drop_unknown":
            return [Drop(10**6)]
        if self.mode == "busy_static":
            g = v.window(2)
            return [Place("batch", (g[0].sids[0],), 0), Place("batch", (g[1].sids[0],), 0)]
        if self.mode == "deadlock":
            return []
        raise AssertionError


@pytest.mark.parametrize(
    "mode,kw,exc,fragment",
    [
        ("gate", dict(eta=0), InvalidAction, "gate"),
        ("drop_unknown", {}, InvalidAction, "unknown group"),
        ("busy_static", dict(engine="static"), InvalidAction, "not idle"),
        ("deadlock", {}, RunError, "deadlock"),
    ],
)
def test_misbehaving_policies_abort_with_a_clear_error(mode, kw, exc, fragment):
    rng = np.random.default_rng(1)
    sysc = tiny_system(T=3, **kw)
    tr = random_trace(rng, 20, 1)
    with pytest.raises(exc) as e:
        simulate(sysc, tr, Bad(mode))
    assert f"bad_{mode}" in str(e.value) and fragment in str(e.value)


def test_running_out_of_stream_is_a_run_error():
    rng = np.random.default_rng(2)
    sysc = tiny_system(T=4, B=2)
    with pytest.raises(RunError, match="stream is exhausted"):
        simulate(sysc, random_trace(rng, 5, 1), make_policy({"name": "reference"}))
