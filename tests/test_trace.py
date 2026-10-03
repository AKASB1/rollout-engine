"""Trace schema v1 loader/writer/manifest tests and check 10 (generator statistics)."""

import copy
import math

import numpy as np
import pytest

from rollout_engine.trace import schema
from rollout_engine.trace.generator import DEFAULT_GENERATOR, generate, verify_mean_s
from rollout_engine.trace.schema import HEADER, TraceError, dumps, loads

ROW = "g1,0,math,10,100,50.5,40,0.250"


def _csv(*rows, header=HEADER, eol="\n"):
    return (eol.join([header, *rows]) + eol).encode()


def test_round_trip_and_crlf():
    tr = generate({**DEFAULT_GENERATOR, "groups": 30}, 7)
    data = dumps(tr)
    assert loads(data) == tr
    assert loads(data.replace(b"\n", b"\r\n")) == tr
    assert dumps(loads(data)) == data


def test_empty_trace_valid_zero_byte_invalid():
    assert loads((HEADER + "\n").encode()).n_groups == 0
    assert loads(HEADER.encode()).n_groups == 0
    with pytest.raises(TraceError, match="line 1"):
        loads(b"")


@pytest.mark.parametrize(
    "data,line,fragment",
    [
        (_csv(ROW, header="group_id,sample_idx"), 1, "header"),
        (_csv("g1,0,math,10,100,50.5,40"), 2, "fields"),
        (_csv(ROW, "g1,1,math,10,100,50.5,x,0.1"), 3, "resp_tokens"),
        (_csv("g1,0,math,0,100,50.5,40,0.1"), 2, "prompt_tokens"),
        (_csv("g1,0,math,10,100,0,40,0.1"), 2, "est_tokens"),
        (_csv("g1,0,math,10,100,abc,40,0.1"), 2, "est_tokens"),
        (_csv("g1,0,math,10,100,5,101,0.1"), 2, "exceeds max_tokens"),
        (_csv("g1,0,math,10,100,5,0,0.1"), 2, "resp_tokens"),
        (_csv("g1,0,math,10,100,5,10,0.1234"), 2, "three decimals"),
        (_csv("g1,0,math,10,100,5,10,-1"), 2, "verify_s"),
        (_csv(",0,math,10,100,5,10,0.1"), 2, "group_id"),
        (_csv("g1,0,,10,100,5,10,0.1"), 2, "task"),
        (_csv(ROW, "g1,2,math,10,100,50.5,40,0.1"), 3, "out of sequence"),
        (_csv("g1,1,math,10,100,50.5,40,0.1"), 2, "out of sequence"),
        (_csv(ROW, "g1,1,code,10,100,50.5,40,0.1"), 3, "differ"),
        (_csv(ROW, "g1,1,math,11,100,50.5,40,0.1"), 3, "differ"),
        (
            _csv(ROW, "g2,0,math,10,100,50.5,40,0.1", "g1,0,math,10,100,50.5,40,0.1"),
            4,
            "contiguous",
        ),
    ],
)
def test_loader_rejects_with_line_number(data, line, fragment):
    with pytest.raises(TraceError) as e:
        loads(data)
    assert f"line {line}" in str(e.value) and fragment in str(e.value)


def test_manifest_hash_checked(tmp_path):
    tr = generate({**DEFAULT_GENERATOR, "groups": 5}, 1)
    path = str(tmp_path / "t.csv")
    m = schema.write(path, tr, {"name": "x"}, 1)
    assert m["groups"] == 5 and m["samples"] == 40
    tr2, m2 = schema.read(path)
    assert tr2 == tr and m2 == m
    with open(path, "ab") as f:
        f.write(b"g999,0,math,1,1,1,1,0\n")
    with pytest.raises(TraceError, match="content_sha256"):
        schema.read(path)


# ---------------------------------------------------------------- check 10: statistics
N_GROUPS = 12000


def _gen(**over):
    g = copy.deepcopy(DEFAULT_GENERATOR)
    g["groups"] = N_GROUPS
    for k, v in over.items():
        g[k] = v
    return g


def _lengths(tr, task=None):
    return np.array(
        [g.resp_tokens for g in tr.groups if task is None or g.task == task], dtype=float
    )


def _tasks(sigma, median=None):
    t = copy.deepcopy(DEFAULT_GENERATOR["tasks"])
    for v in t.values():
        v["len_sigma"] = sigma
        if median:
            v["len_median"] = median
    return t


@pytest.mark.slow
def test_check10_heavy_and_light_tail():
    heavy = generate(_gen(tasks=_tasks(1.1)), 11)
    for task in ("math", "code"):
        cfg = DEFAULT_GENERATOR["tasks"][task]
        L = _lengths(heavy, task).ravel()
        assert abs(np.median(L) / cfg["len_median"] - 1) < 0.05
        p50, p99 = (
            np.percentile(L, 50, method="inverted_cdf"),
            np.percentile(L, 99, method="inverted_cdf"),
        )
        assert p99 / p50 > 8, (task, p99 / p50)
        # truncation share against the model: P(exp(log L) > max_tokens - 1)
        z = (math.log(8191) - math.log(cfg["len_median"])) / 1.1
        expected = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        assert abs((L == 8192).mean() / expected - 1) < 0.2, ((L == 8192).mean(), expected)
    light = generate(_gen(tasks=_tasks(0.4)), 11)
    for task in ("math", "code"):
        L = _lengths(light, task).ravel()
        p50, p99 = (
            np.percentile(L, 50, method="inverted_cdf"),
            np.percentile(L, 99, method="inverted_cdf"),
        )
        assert p99 / p50 < 3, (task, p99 / p50)
        assert (L == 8192).mean() < 1e-3


@pytest.mark.slow
def test_check10_within_group_correlation():
    tr = generate(_gen(tasks=_tasks(0.6)), 12)  # light enough that clipping is negligible
    for task in ("math", "code"):
        X = np.log(_lengths(tr, task))
        k = X.shape[1]
        msb = k * X.mean(axis=1).var(ddof=1)
        msw = X.var(axis=1, ddof=1).mean()
        icc = (msb - msw) / (msb + (k - 1) * msw)
        assert abs(icc - 0.6) < 0.03, (task, icc)


@pytest.mark.slow
def test_check10_estimate_models():
    exact = generate(_gen(estimate={"model": "group", "error_sigma": 0.0}), 13)
    noisy = generate(_gen(estimate={"model": "group", "error_sigma": 0.5}), 13)
    none = generate(_gen(estimate={"model": "none"}), 13)
    # lengths do not depend on the estimate model (common random numbers)
    assert all(
        a.resp_tokens == b.resp_tokens == c.resp_tokens
        for a, b, c in zip(exact.groups, noisy.groups, none.groups)
    )
    est0 = np.array([g.est_tokens for g in exact.groups])
    est1 = np.array([g.est_tokens for g in noisy.groups])
    means = _lengths(exact).mean(axis=1)
    # level 0: the latent group mean is unbiased for the group's realized mean (within 1%)
    assert abs(means.sum() / est0.sum() - 1) < 0.01
    # the error is log-normal with the configured level, median-unbiased
    r = np.log(est1 / est0)
    assert abs(r.std() - 0.5) < 0.02 and abs(r.mean()) < 0.02
    # none: one value per task, equal to the task mean of the clipped length (within 2%)
    for task in ("math", "code"):
        e = {g.est_tokens for g in none.groups if g.task == task}
        assert len(e) == 1
        L = _lengths(none, task)
        assert abs(L.mean() / e.pop() - 1) < 0.02
    # the exact group mean explains much more of the group means than the task mean
    err_exact = np.mean(np.abs(np.log(means / est0)))
    err_none = np.mean(np.abs(np.log(means / np.array([g.est_tokens for g in none.groups]))))
    assert err_exact < 0.6 * err_none


@pytest.mark.slow
def test_check10_verifier_times_and_task_mix():
    tr = generate(_gen(), 14)
    tasks = [g.task for g in tr.groups]
    assert abs(tasks.count("code") / len(tasks) - 0.4) < 0.02
    for task in ("math", "code"):
        cfg = DEFAULT_GENERATOR["tasks"][task]
        V = np.array([g.verify_ms for g in tr.groups if g.task == task], dtype=float).ravel() / 1000
        assert abs(np.median(V) / cfg["verify_median_s"] - 1) < 0.03
        assert abs(np.log(V[V > 0]).std() / cfg["verify_sigma"] - 1) < 0.05
        assert V.max() <= cfg["verify_max_s"]
        assert abs(V.mean() / verify_mean_s(DEFAULT_GENERATOR)[task] - 1) < 0.05


def test_check10_byte_identical_same_seed():
    g = _gen()
    g["groups"] = 200
    assert dumps(generate(g, 5)) == dumps(generate(g, 5))
    assert dumps(generate(g, 5)) != dumps(generate(g, 6))
