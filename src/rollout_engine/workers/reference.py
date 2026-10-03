"""Step-by-step reference engines for check 1 (closed form against step by step).

They keep every rule of ``engine.py`` (admission, pauses, inputs at boundaries, versions) and
replace all phase arithmetic with explicit loops that advance one iteration at a time in
integer nanoseconds with the same rounding rule: the boundary after iteration j of a phase
that started at t0 is at t0 + ceil_ms(sum of the iteration times so far).
"""

from __future__ import annotations

from rollout_engine.clock import ceil_ms
from rollout_engine.workers.engine import ContinuousEngine, ESample, StaticEngine


class StepContinuousEngine(ContinuousEngine):
    def _started(self) -> list[ESample]:
        return [s for s in self.admitted.values() if s.started]

    def _acc_ns(self, j: int) -> int:
        """Sum of the first j iteration times of the current phase, one iteration at a time."""
        p = self.p
        run = self._started()
        b = len(run)
        ctx = sum(s.P + s.g0 + self.iter0 - s.s_iter for s in run)
        acc = 0
        for _ in range(j):
            acc += p.a0_ns + p.a1_ns * b + p.a2_ns * ctx
            ctx += b
        return acc

    def _boundary_time(self, j: int) -> int:
        return self.t0 + ceil_ms(self._acc_ns(j))

    def _first_j_at_or_after(self, t: int) -> int:
        p = self.p
        run = self._started()
        b = len(run)
        ctx = sum(s.P + s.g0 + self.iter0 - s.s_iter for s in run)
        j, acc = 0, 0
        while self.t0 + ceil_ms(acc) < t:
            acc += p.a0_ns + p.a1_ns * b + p.a2_ns * ctx
            ctx += b
            j += 1
        return j

    def _finish_j(self) -> int | None:
        run = self._started()
        if not run:
            return None
        j = 0
        while True:
            j += 1
            for s in run:
                if s.g0 + self.iter0 + j - s.s_iter >= s.L:
                    return j

    def _finishing(self, at_iter: int) -> list[ESample]:
        out = [s for s in self._started() if s.g0 + at_iter - s.s_iter == s.L]
        out.sort(key=ESample.key)
        return out  # _retire marks them removed, so the base-class heap skips them


class StepStaticEngine(StaticEngine):
    def _start_batch(self, t: int, batch: list[ESample]) -> None:
        p = self.p
        b = len(batch)
        pmax = max(s.P for s in batch)
        lmax = max(s.L for s in batch)
        pre = p.prefill_c0_ns + p.prefill_c1_ns * b * pmax
        dec = 0
        for j in range(lmax):
            dec += p.a0_ns + p.a1_ns * b + p.a2_ns * b * (pmax + j)
        super()._start_batch(t, batch)
        # overwrite the closed-form end with the step-by-step one
        self.batch_end = t + ceil_ms(pre) + ceil_ms(dec)
