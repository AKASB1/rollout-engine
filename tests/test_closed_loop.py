"""Check 6: hand-computed closed-loop cases (2 workers, B = 2, every time computed by hand).

System: a0 = 1 ms (a1 = a2 = 0, no prefill), so a sample of L tokens takes L ms on any
worker; one sample per group; verification takes 0 ms; training takes F ms (ns_per_token =
0), sync S ms. The policy is the reference (fifo + group + early + round_robin + wait) unless
stated, so groups alternate between workers 0 and 1 in launch order.
"""

import pytest

from helpers import tiny_system, tiny_trace
from rollout_engine.policies.composed import make_policy
from rollout_engine.sim.checks import accounting_violations, staleness_violations
from rollout_engine.sim.driver import simulate
from rollout_engine.telemetry.metrics import run_metrics


def run(system, trace, policy="reference", **params):
    sim = simulate(system, trace, make_policy({"name": policy, "params": params}), record_log=True)
    assert accounting_violations(sim) == []
    assert staleness_violations(sim) == []
    return sim, run_metrics(sim.core, trace)


def tsel(sim):
    return [st.t_sel for st in sim.core.steps]


def test_synchronous_eta0():
    sim, m = run(tiny_system(eta=0, T=2, F=10, S=1), tiny_trace([3, 5, 2, 4]))
    c = sim.core
    # step 0 at 5 (g1 ready); train 5-15, sync 15-16, v1 at 16; gate closed until then
    assert tsel(sim) == [5, 20] and c.t_end == 30
    assert [g.launched_at for g in c.groups] == [0, 0, 16, 16]
    assert [st.staleness for st in c.steps] == [[0, 0], [0, 0]]
    assert c.trainer_tl == [(0, "wait"), (5, "train"), (15, "sync"), (16, "wait"), (20, "train")]
    assert m["step_time_mean_s"] == pytest.approx(0.015)
    assert m["trainer_wait_mean_s"] == pytest.approx(0.004)
    # window [5, 20): idle GPU-ms = w0 13 + w1 11 + trainer 4 = 28 of 45
    assert m["gpu_idle_frac"] == pytest.approx(28 / 45)
    assert m["rollout_idle_frac"] == pytest.approx(24 / 30)
    assert m["straggler_mean_s"] == 0 and m["waste_frac"] == 0 and m["length_bias"] == 0


def test_one_step_off_policy_hides_training():
    sim, m = run(tiny_system(eta=1, T=3, F=4, S=1), tiny_trace([3, 5, 4, 6, 2, 2]))
    c = sim.core
    # step 0 at 5; g2, g3 launch at 5 under v0 while step 0 trains (5-9, sync to 10);
    # g3 ready at 11 -> step 1 at 11 with staleness 1; g4, g5 launch at 11 under v1, ready
    # at 13, but the trainer is busy until v2 at 16 -> step 2 at 16
    assert tsel(sim) == [5, 11, 16] and c.t_end == 20
    assert [g.launched_at for g in c.groups] == [0, 0, 5, 5, 11, 11]
    assert [st.staleness for st in c.steps] == [[0, 0], [1, 1], [1, 1]]
    assert [c.samples[s].version for s in range(6)] == [0, 0, 0, 0, 1, 1]
    assert m["step_time_mean_s"] == pytest.approx(0.0055)
    assert m["staleness_mean"] == pytest.approx(0.5) and m["staleness_max"] == 1
    assert m["trainer_wait_mean_s"] == pytest.approx(0.0005)  # wait 10-11 in step 0's interval
    # step 0 consumes v0 tokens at step 0 (fresh), step 1 consumes v0 tokens at step 1 (stale)
    assert m["stale_token_frac"] == pytest.approx(0.5)


def test_dead_drop_eta0_with_carry():
    sim, m = run(
        tiny_system(eta=0, T=2, F=4, S=1),
        tiny_trace([2, 3, 10, 10, 1, 1, 5, 5]),
        "reference",
        straggler="carry",
        rho=2.0,
    )
    c = sim.core
    # 4 groups at 0; step 0 at 3 = {g0, g1}; g2, g3 (v0) are dead for step 1 -> dropped at 3
    # with 3 tokens each; v1 at 8; g4..g7 launch at 8; step 1 at 9; g6, g7 dropped with 1 token
    assert tsel(sim) == [3, 9] and c.t_end == 13
    assert [g.state for g in c.groups] == ["consumed"] * 2 + ["dropped"] * 2 + ["consumed"] * 2 + [
        "dropped"
    ] * 2
    assert [c.groups[i].tokens for i in (2, 3, 6, 7)] == [3, 3, 1, 1]
    assert m["drops_stale"] == 4 and m["tokens_dropped"] == 8 and m["tokens_consumed"] == 7
    assert m["waste_frac"] == pytest.approx(8 / 15)
    assert m["length_bias"] == pytest.approx(1.75 / (37 / 8) - 1)


def test_abort_drops_unselected_and_refills():
    sim, m = run(
        tiny_system(eta=1, T=2, F=4, S=1),
        tiny_trace([2, 3, 10, 10, 1, 1, 1, 1]),
        "reference",
        straggler="abort",
        rho=2.0,
    )
    c = sim.core
    # step 0 at 3; the policy drops g2, g3 (3 tokens each) and launches g4..g7 at 3 (v0);
    # they are ready at 4; the trainer is free at 8 -> step 1 = {g4, g5} (staleness 1);
    # g6, g7 (v0) are then dead for step 2 -> dropped by the engine
    assert tsel(sim) == [3, 8] and c.t_end == 12
    assert [g.drop_reason for g in c.groups if g.state == "dropped"] == [
        "policy",
        "policy",
        "stale",
        "stale",
    ]
    assert m["tokens_dropped"] == 8 and m["tokens_consumed"] == 7
    assert c.steps[1].staleness == [1, 1]
    assert [x[:2] for x in c.log[4:6]] == [(3, "drop"), (3, "drop")]


def test_carry_keeps_unselected_groups():
    sim, m = run(
        tiny_system(eta=1, T=2, F=4, S=1),
        tiny_trace([2, 3, 10, 10, 1, 1, 1, 1]),
        "reference",
        straggler="carry",
        rho=2.0,
    )
    c = sim.core
    # step 0 at 3; g2, g3 carry on; g4, g5 launch at 3 (cap 4), ready at 4; step 1 at 8 =
    # {g4, g5}; then g2, g3 (v0) are dead for step 2 and dropped at 8 with 8 tokens each
    assert tsel(sim) == [3, 8] and c.t_end == 12
    assert [g.launched_at for g in c.groups[:6]] == [0, 0, 0, 0, 3, 3]
    assert [c.groups[i].state for i in (2, 3)] == ["dropped", "dropped"]
    assert [c.groups[i].tokens for i in (2, 3)] == [8, 8]
    assert m["drops_stale"] == 2 and m["drops_policy"] == 0


def test_colocated_step_with_switch_cost():
    sim, m = run(
        tiny_system(eta=1, T=2, F=3, partition="colocated", switch=2), tiny_trace([2, 3, 1, 1])
    )
    c = sim.core
    # step 0 at 3; switch-out 3-5; train 5-8; v1 at 8; switch-in 8-10; g2, g3 (placed at 3,
    # waiting) start at 10 under v1, ready at 11; step 1 at 11; switch 11-13; train 13-16
    assert tsel(sim) == [3, 11] and c.t_end == 16
    assert c.trainer_tl == [
        (0, "wait"),
        (3, "switch"),
        (5, "train"),
        (8, "switch"),
        (10, "wait"),
        (11, "switch"),
        (13, "train"),
    ]
    assert [c.samples[s].version for s in range(4)] == [0, 0, 1, 1]
    assert [st.staleness for st in c.steps] == [[0, 0], [0, 0]]
    # window [3, 11): each worker overhead 4, train 3, gen 1
    assert m["gpu_idle_frac"] == 0
    assert m["gpu_train_frac"] == pytest.approx(3 / 8)
    assert m["gpu_overhead_frac"] == pytest.approx(4 / 8)
    assert m["gpu_gen_frac"] == pytest.approx(1 / 8)


def test_window_straggler_and_batch_cv_with_b10():
    # B = 10, warm-up 1, T = 4: the window holds steps 1 and 2. eta = 0 (synchronous).
    lengths = list(range(1, 11)) + [3, 1, 4, 1, 5, 9, 2, 6, 5, 3] + [2] * 10 + [1] * 10
    sim, m = run(tiny_system(eta=0, T=4, B=10, F=5, S=1, warm=1), tiny_trace(lengths))
    c = sim.core
    # step 0 at 10; train 10-15, sync to 16; step-1 groups 16 + L -> 25; train 25-30, v2 at 31;
    # step-2 groups (L = 2) ready at 33; v3 at 39; step-3 groups ready at 40; train to 45
    assert tsel(sim) == [10, 25, 33, 40] and c.t_end == 45
    assert m["window_s"] == pytest.approx(0.015)
    assert m["step_time_mean_s"] == pytest.approx(0.0075) and m["step_time_p95_s"] == pytest.approx(
        0.008
    )
    # straggler = ready(10th) - ready(9th): step 1: 16+9 - (16+6) = 3 ms; step 2: 0
    assert m["straggler_mean_s"] == pytest.approx(0.0015) and m["straggler_p95_s"] == pytest.approx(
        0.003
    )
    # batch mean lengths 3.9 and 2.0 -> population sd 0.95 over mean 2.95
    assert m["batch_len_cv"] == pytest.approx(0.95 / 2.95)
    assert m["trainer_wait_mean_s"] == pytest.approx(0.0015)  # waits 31-33 and 39-40
    assert m["samples_per_s"] == pytest.approx(20 / 0.015)
    # window [25, 40): each worker decodes 3 ms (31-33, 39-40), trainer idles 3 ms -> 27 of 45
    assert m["gpu_idle_frac"] == pytest.approx(27 / 45)
    assert m["stale_token_frac"] == 0 and m["staleness_max"] == 0
