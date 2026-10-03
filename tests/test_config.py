import copy

import pytest

from rollout_engine.config import ConfigError, deep_merge, derive_system, parse_system


def test_derived_default_system_is_valid_and_documented():
    raw = derive_system()
    sc = parse_system(raw)
    assert sc.n_workers == 8 and sc.rollout_gpus == 8 and sc.train_gpus == 8
    e = sc.engine
    # derived values (docs/simulator.md): weights 15.2 GB at 0.7 x 3.35 TB/s + 1 ms overhead
    assert 7_400_000 <= e.a0_ns <= 7_600_000
    assert 50_000 <= e.a1_ns <= 51_500
    assert e.a2_ns == 24
    assert e.kv_tokens > e.max_seqs * 8192
    # 6 * 7.6e9 / (8 * 989e12 * 0.35) s per token
    assert 16_000 <= sc.ns_per_token <= 16_900
    col = parse_system(derive_system(partition="colocated"))
    assert col.train_gpus == 16 and col.n_workers == 16 and col.switch_ms == 3000
    assert col.ns_per_token * 2 == pytest.approx(sc.ns_per_token, rel=1e-3)


@pytest.mark.parametrize(
    "patch,fragment",
    [
        ({"gpus": {"rollout": 9}}, "!= gpus.total"),
        ({"rollout": {"tp": 3}}, "whole number of workers"),
        ({"loop": {"eta": -1}}, "loop.eta"),
        ({"gpus": {"partition": "mixed"}}, "partition"),
        ({"rollout": {"engine": "paged"}}, "engine"),
        ({"sync": {"inflight": "pause"}}, "inflight"),
        ({"rollout": {"a0_ns": 0}}, "a0_ns"),
        ({"rollout": {"max_seqs": 1.5}}, "integer"),
        ({"loop": {"steps": 3}}, "warmup_steps"),
        ({"schema_version": 2}, "schema_version"),
        ({"verifier": {"tasks": []}}, "verifier.tasks"),
    ],
)
def test_invalid_system_configurations(patch, fragment):
    raw = deep_merge(derive_system(), patch)
    with pytest.raises(ConfigError, match=fragment.replace(".", r"\.")):
        parse_system(raw)


def test_missing_section():
    raw = copy.deepcopy(derive_system())
    del raw["trainer"]
    with pytest.raises(ConfigError, match="trainer"):
        parse_system(raw)
