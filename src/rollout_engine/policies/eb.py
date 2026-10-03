"""Empirical-Bayes length estimator with right-censoring (Tier 2, research extension).

Model, per task, in log space relative to the prior estimate:

    log L_gi = log est_g + u_g + e_gi,   u_g ~ N(b, su2),   e_gi ~ N(0, se2)

``u_g`` is the group's deviation from its prior estimate (estimate error plus the group
latent), ``e_gi`` the within-group scatter; ``su2 / (su2 + se2)`` plays the role of the
within-group correlation. The hyperparameters (b, su2, se2) are learned online by moments
from the groups whose samples have all finished, shrunk towards a weak prior with ``n0``
pseudo-groups. For one group, the posterior of ``u_g`` uses its finished samples as exact
observations and its running samples as right-censored ones (``L > tokens so far``): the
finished lengths of a group are biased low, because the unfinished samples are the long
ones, so each censored observation is imputed by the mean of the predictive normal truncated
at its threshold, and the posterior is recomputed (three fixed-point passes).

Predictions: an unstarted sample ``E[L] = est * exp(m + (se2 + v)/2)``; a running sample with
``c`` tokens ``E[L | L > c]`` of the same log-normal; both capped at ``max_tokens``. The learned
hyperparameters come from completed groups, which early in a run over-represent short groups;
the prior (b = 0) dominates until enough groups completed.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_ND = NormalDist()
SQRT2 = math.sqrt(2.0)


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _sf(x: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(x / SQRT2)


def trunc_mean(mu: float, s: float, a: float) -> float:
    """E[X | X > a] for X ~ N(mu, s^2)."""
    z = (a - mu) / s
    tail = _sf(z)
    if tail < 1e-300:
        return a + s / max(z, 1e-9)  # asymptotic
    return mu + s * _phi(z) / tail


class EBModel:
    def __init__(self, n0: float = 20.0, b0: float = 0.0, su2_0: float = 0.25, se2_0: float = 0.5):
        self.n0, self.b0, self.su2_0, self.se2_0 = n0, b0, su2_0, se2_0
        self.seen: set[int] = set()
        self.stats: dict[
            str, list[float]
        ] = {}  # task -> [groups, sum mean, sum mean^2, within SS, within dof]
        self._t = -1
        self._post: dict[int, tuple[float, float]] = {}
        self._hyper: dict[str, tuple[float, float, float]] = {}

    # ------------------------------------------------------------------ learning
    def refresh(self, v) -> None:
        if v.t == self._t:
            return
        self._t = v.t
        self._post = {}
        changed = False
        for g in v.outstanding():
            if g.gidx in self.seen or g.finished_n < g.n_samples:
                continue
            self.seen.add(g.gidx)
            le = math.log(g.est_tokens)
            rs = [math.log(v.sample(s).length) - le for s in g.sids]
            k = len(rs)
            mean = sum(rs) / k
            st = self.stats.setdefault(g.task, [0.0, 0.0, 0.0, 0.0, 0.0])
            st[0] += 1
            st[1] += mean
            st[2] += mean * mean
            st[3] += sum((r - mean) ** 2 for r in rs)
            st[4] += k - 1
            changed = True
        if changed:
            self._hyper = {}

    def hyper(self, task: str, k: int) -> tuple[float, float, float]:
        h = self._hyper.get(task)
        if h is not None:
            return h
        n0 = self.n0
        st = self.stats.get(task, [0.0, 0.0, 0.0, 0.0, 0.0])
        G, s1, s2, ss, dof = st
        se2 = (n0 * max(k - 1, 1) * self.se2_0 + ss) / (n0 * max(k - 1, 1) + dof)
        b = (n0 * self.b0 + s1) / (n0 + G)
        if G > 1:
            var_means = max(0.0, (s2 - s1 * s1 / G) / (G - 1))
            su2_hat = max(0.01, var_means - se2 / max(k, 1))
        else:
            su2_hat = self.su2_0
        su2 = (n0 * self.su2_0 + G * su2_hat) / (n0 + G)
        h = (b, su2, se2)
        self._hyper[task] = h
        return h

    # ------------------------------------------------------------------ one group
    def posterior(self, v, g) -> tuple[float, float, float]:
        """(posterior mean of u_g, its variance, se2)."""
        hit = self._post.get(g.gidx)
        if hit is not None:
            return hit
        b, su2, se2 = self.hyper(g.task, g.n_samples)
        le = math.log(g.est_tokens)
        obs, cens = [], []
        for sid in g.sids:
            s = v.sample(sid)
            if s.length is not None:
                obs.append(math.log(s.length) - le)
            elif s.started and s.tokens_so_far > 0:
                cens.append(math.log(s.tokens_so_far + 1) - le)
        prec0 = 1.0 / su2
        n = len(obs) + len(cens)
        post_prec = prec0 + n / se2
        var = 1.0 / post_prec
        if not n:
            res = (b, su2, se2)
        else:
            imputed = list(cens)
            m = b
            for _ in range(3):
                m = (b * prec0 + (sum(obs) + sum(imputed)) / se2) / post_prec
                sp = math.sqrt(se2 + var)
                imputed = [max(c, trunc_mean(m, sp, c)) for c in cens]
            res = (m, var, se2)
        self._post[g.gidx] = res
        return res

    def predict(self, v, sid: int, quantile: float = 0.0) -> float:
        """The predictive mean of the sample's length, or its ``quantile`` when > 0."""
        s = v.sample(sid)
        if s.length is not None:
            return float(s.length)
        g = v.group(s.gidx)
        m, var, se2 = self.posterior(v, g)
        s2 = se2 + var
        mu = math.log(g.est_tokens) + m
        cap = float(g.max_tokens)
        if quantile > 0:
            sd = math.sqrt(s2)
            lo = 0.0
            c = 1.0
            if s.started and s.tokens_so_far > 0:
                c = s.tokens_so_far + 1.0
                lo = 1.0 - _sf((math.log(c) - mu) / sd)
            p = min(1 - 1e-12, lo + quantile * (1.0 - lo))
            return min(cap, max(c, math.exp(mu + sd * _ND.inv_cdf(p))))
        if s.started and s.tokens_so_far > 0:
            c = s.tokens_so_far + 1.0
            a = math.log(c)
            sd = math.sqrt(s2)
            den = _sf((a - mu) / sd)
            if den < 1e-12:
                return min(cap, max(c, c * 1.5))
            est = math.exp(mu + s2 / 2) * _sf((a - mu - s2) / sd) / den
            return min(cap, max(c, est))
        return min(cap, math.exp(mu + s2 / 2))
