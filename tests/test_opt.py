"""Check 2 (exact references) and the static part of check 3 (bounds, Graham)."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import subprocess_env
from rollout_engine.batching import batch_cost_ms
from rollout_engine.config import EngineParams
from rollout_engine.opt.brute import best_assignment, best_makespan, best_total, set_partitions
from rollout_engine.opt.dp import dp_partition
from rollout_engine.opt.lpt import graham_list_bound, graham_lpt_factor, list_schedule, lpt
from rollout_engine.opt.milp import plan_makespan, solve_makespan
from rollout_engine.opt.s7 import PlanPolicy, cost_fn, evaluate, gen_time_ms, instance
from rollout_engine.policies.composed import ComposedPolicy, make_policy
from rollout_engine.sim.driver import simulate

P = EngineParams("static", 1, 7_480_000, 50_730, 24, 7_480_000, 30_729, 32, 10**6, 16)


def real_cost(b, L):
    return batch_cost_ms(P, b, 512, L)


SYNTH = {
    "real": real_cost,
    "b_times_L": lambda b, L: b * L,
    "L2_plus_b": lambda b, L: L * L + b,
    "step_sqrt": lambda b, L: (L // 300) * 7 + math.sqrt(b),
}


def test_bell_numbers():
    assert sum(1 for _ in set_partitions(8, 8)) == 4140
    assert sum(1 for _ in set_partitions(5, 2)) == 26  # partitions of 5 into blocks of size <= 2


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SYNTH))
def test_check2_dp_equals_brute_force(name):
    cost = SYNTH[name]
    rng = np.random.default_rng(len(name))
    for _ in range(40):
        n = int(rng.integers(1, 9))
        cap = int(rng.integers(1, n + 1))
        L = sorted(int(x) for x in rng.integers(1, 3000, size=n))
        dp, segs = dp_partition(L, cap, lambda b, lm, i, j: cost(b, lm))
        assert dp == pytest.approx(best_total(L, cap, cost), rel=1e-12)
        assert sum(j - i + 1 for i, j in segs) == n and all(j - i + 1 <= cap for i, j in segs)


@pytest.mark.slow
def test_check2_milp_equals_brute_force_and_dp():
    rng = np.random.default_rng(7)
    for _ in range(36):
        n = int(rng.integers(2, 8))
        W = int(rng.integers(1, 4))
        cap = int(rng.integers(1, n + 1))
        L = sorted(int(x) for x in rng.integers(1, 3000, size=n))
        res = solve_makespan(L, W, cap, real_cost)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(best_makespan(L, W, cap, real_cost), rel=1e-6)
        assert plan_makespan(res.plan, L, real_cost) == pytest.approx(res.objective, rel=1e-9)
        lb = max(
            dp_partition(L, cap, lambda b, lm, i, j: real_cost(b, lm))[0] / W, real_cost(1, L[-1])
        )
        assert res.objective >= lb - 1e-9
        if W == 1:
            assert res.objective == pytest.approx(
                dp_partition(L, cap, lambda b, lm, i, j: real_cost(b, lm))[0]
            )


def _solve_in_fresh_process(L, W, cap):
    code = (
        "import json,sys;from rollout_engine.opt.milp import solve_makespan;"
        "from rollout_engine.batching import batch_cost_ms;from rollout_engine.config import EngineParams;"
        "P=EngineParams('static',1,7_480_000,50_730,24,7_480_000,30_729,32,10**6,16);"
        f"r=solve_makespan({L},{W},{cap},lambda b,L: batch_cost_ms(P,b,512,L));"
        "print(json.dumps([r.status,r.objective,r.plan]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=subprocess_env(),
    )
    return json.loads(out.stdout)


@pytest.mark.slow
def test_check2_milp_repeatable_in_fresh_processes():
    L = sorted([40, 3000, 120, 800, 800, 2500, 90, 1500, 4000, 33, 700, 610])
    a = _solve_in_fresh_process(L, 3, 4)
    b = _solve_in_fresh_process(L, 3, 4)
    assert a == b and a[0] == "optimal"


@pytest.mark.slow
def test_check2_engine_replays_milp_plan_and_no_policy_beats_it():
    for seed in range(5000, 5012):
        row = evaluate(seed)
        assert row["milp_status"] == "optimal"
        assert (
            row["milp_replay_ms"] == row["milp_ms"]
        )  # the static engine reproduces the MILP objective
        assert row["milp_ms"] >= row["lower_bound_ms"]
        for name in ("fifo_chunk", "sorted_chunk", "sorted_chunk_oracle", "dp", "oracle_dp"):
            assert row[f"{name}_ms"] >= row["milp_ms"], (seed, name)


# ------------------------------------------------------------------ check 3, static part
@pytest.mark.slow
def test_check3_static_bounds_every_policy():
    for seed in range(7000, 7015):
        system, trace, (n, W, cap) = instance(seed)
        cost = cost_fn(system)
        L = sorted(g.resp_tokens[0] for g in trace.groups)
        dp, _ = dp_partition(L, cap, lambda b, lm, i, j, c=cost: c(b, lm))
        for cfg in (
            {"name": "composed", "params": {"batch": "fifo_chunk"}},
            {"name": "composed", "params": {"batch": "sorted_chunk", "batch_size": 2}},
            {"name": "dp"},
            {"name": "oracle_dp"},
            {"name": "composed", "params": {"batch": "dp", "estimator": "group_evidence"}},
        ):
            g = gen_time_ms(simulate(system, trace, make_policy(cfg)))
            assert g * W >= dp and g >= cost(1, L[-1]), (seed, cfg)


@pytest.mark.slow
def test_check3_graham_list_and_lpt():
    rng = np.random.default_rng(11)
    for _ in range(300):
        W = int(rng.integers(1, 5))
        costs = [int(x) for x in rng.integers(1, 1000, size=int(rng.integers(1, 9)))]
        order = list(rng.permutation(len(costs)))
        ms, _ = list_schedule(costs, W, order)
        assert ms <= graham_list_bound(costs, W) + 1e-9
        opt = best_assignment(costs, W)
        assert lpt(costs, W)[0] <= graham_lpt_factor(W) * opt + 1e-9
        assert ms >= opt


def test_plan_policy_replays_batches_in_order():
    system, trace, (n, W, cap) = instance(5001)
    plan = [[[k] for k in range(n) if k % W == w] for w in range(W)]
    sim = simulate(system, trace, PlanPolicy(plan))
    cost = cost_fn(system)
    L = [g.resp_tokens[0] for g in trace.groups]
    expected = max(sum(cost(1, L[k]) for k in range(n) if k % W == w) for w in range(W))
    assert gen_time_ms(sim) == expected
    assert isinstance(ComposedPolicy({"batch": "dp"}), ComposedPolicy)


def test_scenario_cvar_milp_reduces_to_the_makespan_milp():
    # one scenario equal to the true lengths: expected makespan = CVaR = the makespan, so the
    # objective is (1 + lam) times the makespan optimum
    import numpy as np

    from rollout_engine.opt.scenario import solve_cvar

    rng = np.random.default_rng(3)
    for _ in range(8):
        n, W, cap = int(rng.integers(3, 8)), int(rng.integers(1, 4)), int(rng.integers(1, 5))
        L = sorted(int(x) for x in rng.integers(1, 3000, size=n))
        ref = solve_makespan(L, W, cap, real_cost)
        for lam in (0.0, 1.0):
            r = solve_cvar(np.array([L]), W, cap, real_cost, alpha=0.9, lam=lam)
            assert r.status == "optimal"
            assert r.objective == pytest.approx((1 + lam) * ref.objective, rel=1e-6)


@pytest.mark.slow
def test_s9_risk_rows_are_consistent():
    from rollout_engine.opt.risk import evaluate as s9

    row = s9(5003, lams=(0.0, 1.0))
    assert row["optimum_status"] == "optimal"
    for k in ("det", "cvar0", "cvar1", "dp_policy"):
        assert row[f"{k}_ms"] >= row["optimum_ms"]  # nothing beats the clairvoyant optimum
