import subprocess
import sys

import numpy as np
import pytest

from conftest import subprocess_env
from rollout_engine.clock import VirtualClock, ceil_ms, fmt_s, parse_s_to_ms
from rollout_engine.rng import fnv1a64, splitmix64, stream, stream_seeds


def test_fnv1a64_known_answers():
    # Published FNV-1a 64-bit test vectors.
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert fnv1a64("foobar") == 0x85944171F73967E8


def test_splitmix64_known_answers():
    # First outputs of the reference SplitMix64 generator seeded with 0 / 1234567.
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert splitmix64(0x9E3779B97F4A7C15) == 0x6E789E6AA1B965F4
    assert splitmix64(1234567) == 6457827717110365317


def test_stream_seed_derivation_matches_go_formula():
    c = fnv1a64("trace.lengths")
    s1 = splitmix64(7 ^ c)
    s2 = splitmix64(s1 ^ 0x9E3779B97F4A7C15 ^ c)
    assert stream_seeds(7, "trace.lengths") == (s1, s2)


def test_stream_is_reproducible_and_independent():
    a = stream(42, "x").integers(0, 1 << 62, size=8)
    b = stream(42, "x").integers(0, 1 << 62, size=8)
    assert np.array_equal(a, b)
    c = stream(42, "y").integers(0, 1 << 62, size=8)
    d = stream(43, "x").integers(0, 1 << 62, size=8)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)
    # Independence (weak check): two named streams are uncorrelated.
    u = stream(1, "trace.tasks").standard_normal(20000)
    v = stream(1, "trace.lengths").standard_normal(20000)
    assert abs(float(np.corrcoef(u, v)[0, 1])) < 0.03


def test_stream_identical_in_fresh_process():
    code = (
        "from rollout_engine.rng import stream;"
        "print(list(stream(5,'verify').integers(0,1000,size=5)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=subprocess_env(),
    )
    assert out.stdout.strip() == str(list(stream(5, "verify").integers(0, 1000, size=5)))


def test_ceil_ms_and_formatting():
    assert ceil_ms(0) == 0
    assert ceil_ms(1) == 1
    assert ceil_ms(1_000_000) == 1
    assert ceil_ms(1_000_001) == 2
    assert fmt_s(0) == "0.000" and fmt_s(1250) == "1.250" and fmt_s(61001) == "61.001"
    assert parse_s_to_ms("1.25") == 1250 and parse_s_to_ms("0") == 0 and parse_s_to_ms(".5") == 500
    assert parse_s_to_ms("2.000") == 2000 and parse_s_to_ms("3.1000") == 3100
    for bad in ["", "-1", "1.2345", "abc", "1.", "1e3"]:
        with pytest.raises(ValueError):
            parse_s_to_ms(bad)


def test_virtual_clock_monotone():
    c = VirtualClock()
    c.set(5)
    assert c.now_ms() == 5
    with pytest.raises(ValueError):
        c.set(4)
