"""The scaffold's original test, kept in intent (DECISIONS.md): requests are served first in,
first out, and a worker returns one result per requested sample. The scaffold's in-memory
FIFO queue and echo worker became the group stream and the engine-backed mock worker."""

from helpers import tiny_system, tiny_trace
from rollout_engine.policies.composed import make_policy
from rollout_engine.sim.driver import simulate


def test_fifo_and_samples():
    tr = tiny_trace([[3, 4], [2, 2]])  # two groups ("requests") of two samples each
    sim = simulate(
        tiny_system(mode="single_phase", workers=1, T=1, B=2),
        tr,
        make_policy({"name": "reference"}),
    )
    core = sim.core
    assert [g.launched_at for g in core.groups] == [0, 0]
    assert [g.spec.gid for g in core.groups if g.state == "consumed"] == ["g0", "g1"]
    assert all(g.n_generated == g.spec.n_samples == 2 for g in core.groups)
