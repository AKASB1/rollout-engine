"""Check 7: the verifier pool against queueing theory.

The pool is driven alone by Poisson arrivals (interarrival times rounded to whole ms) with the
same instant order as the simulator: arrivals are queued, completions free servers, free
servers take the queue head (FIFO). rho ~ 0.7, 10 seeds x 20 000 arrivals.

- One server, exponential / deterministic / log-normal service: the pooled mean wait matches
  Pollaczek-Khinchine lambda E[S^2] / (2 (1 - rho)) with lambda, E[S], E[S^2] taken from the
  realized times, within 6 %; and the waits equal the Lindley recursion exactly.
- 4 and 16 servers, exponential service: the pooled mean wait matches Erlang C within 8 %.
"""

import heapq
import math

import numpy as np
import pytest

from rollout_engine.rng import stream
from rollout_engine.verifier.pool import VerifierPool

N = 20_000
SEEDS = range(10)
MEAN_S = 1000.0  # ms


def run_pool(inter_ms, serv_ms, servers):
    pool = VerifierPool(servers)
    arr = np.cumsum(inter_ms)
    starts = [0] * len(arr)
    comp: list[tuple[int, int]] = []
    i = 0
    n = len(arr)
    while i < n or comp:
        t = min(arr[i] if i < n else math.inf, comp[0][0] if comp else math.inf)
        while i < n and arr[i] == t:
            pool.enqueue(i, i, "x", int(t))
            i += 1
        done = []
        while comp and comp[0][0] == t:
            done.append(heapq.heappop(comp)[1])
        for sv in sorted(done):
            pool.complete(sv)
        for sv, sid in pool.assign(int(t), lambda q: 0):
            starts[sid] = int(t)
            heapq.heappush(comp, (int(t) + int(serv_ms[sid]), sv))
    return np.array(starts) - arr, arr


def draws(seed, dist, servers):
    lam = 0.7 * servers / MEAN_S
    rng = stream(seed, f"check7.{dist}.{servers}")
    inter = np.maximum(1, np.rint(rng.exponential(1 / lam, N))).astype(np.int64)
    if dist == "exp":
        s = rng.exponential(MEAN_S, N)
    elif dist == "det":
        s = np.full(N, MEAN_S)
    else:
        sig = 0.5
        s = rng.lognormal(math.log(MEAN_S) - sig * sig / 2, sig, N)
    return inter, np.maximum(0, np.rint(s)).astype(np.int64)


def erlang_c_wait(lam, mu, c):
    a = lam / mu
    rho = a / c
    s = sum(a**k / math.factorial(k) for k in range(c))
    top = a**c / math.factorial(c) / (1 - rho)
    pc = top / (s + top)
    return pc / (c * mu - lam)


@pytest.mark.slow
@pytest.mark.parametrize("dist", ["exp", "det", "logn"])
def test_mg1_pollaczek_khinchine_and_lindley(dist):
    waits, theory = [], []
    for seed in SEEDS:
        inter, serv = draws(seed, dist, 1)
        w, arr = run_pool(inter, serv, 1)
        # Lindley on the realized, millisecond-rounded times: exact
        lw = np.zeros(N, dtype=np.int64)
        for k in range(N - 1):
            lw[k + 1] = max(0, lw[k] + serv[k] - inter[k + 1])
        assert np.array_equal(w, lw)
        lam = 1 / inter.mean()
        es, es2 = serv.mean(), (serv.astype(float) ** 2).mean()
        theory.append(lam * es2 / (2 * (1 - lam * es)))
        waits.append(w.mean())
    assert abs(np.mean(waits) / np.mean(theory) - 1) < 0.06, (np.mean(waits), np.mean(theory))


@pytest.mark.slow
@pytest.mark.parametrize("c", [4, 16])
def test_mmc_erlang_c(c):
    waits, theory = [], []
    for seed in SEEDS:
        inter, serv = draws(seed, "exp", c)
        w, _ = run_pool(inter, serv, c)
        waits.append(w.mean())
        theory.append(erlang_c_wait(1 / inter.mean(), 1 / serv.mean(), c))
    assert abs(np.mean(waits) / np.mean(theory) - 1) < 0.08, (np.mean(waits), np.mean(theory))


def test_order_hook_and_group_removal():
    pool = VerifierPool(1)
    for sid, (g, task) in enumerate([(0, "code"), (1, "math"), (0, "code")]):
        pool.enqueue(sid, g, task, 0)
    assert pool.assign(0, lambda q: len(q) - 1) == [(0, 2)]
    pool.remove_group(0)
    assert [e.sid for e in pool.queue] == [1]
    assert pool.complete(0) == 2
    with pytest.raises(IndexError):
        pool.assign(1, lambda q: 5)
