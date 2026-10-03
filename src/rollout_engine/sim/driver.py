"""The discrete-event driver of the core state machine (docs/simulator.md section 1).

It owns the worker engines (which know the hidden lengths), the verifier completion times
(hidden ``verify_s``), and the training durations, and calls the core in the fixed order of
one instant. One run is single-threaded and deterministic; the only wall-clock numbers are
the policy decision times (``wall_`` columns).
"""

from __future__ import annotations

import heapq
import time

from rollout_engine.api import RunError, specs_from_trace
from rollout_engine.config import SystemConfig
from rollout_engine.scheduler.core import Core
from rollout_engine.trace.schema import Trace
from rollout_engine.workers.engine import ESample, make_engine


class Simulation:
    def __init__(
        self,
        system: SystemConfig,
        trace: Trace,
        policy,
        *,
        record_log: bool = False,
        engine_factory=None,
        max_instants: int = 50_000_000,
    ):
        self.sys = system
        self.trace = trace
        self.policy = policy
        self.core = Core(system, specs_from_trace(trace), getattr(policy, "name", "?"), record_log)
        if hasattr(policy, "bind"):
            policy.bind(system, trace)
        ep = system.engine
        if ep.engine == "continuous":
            for g in trace.groups:
                if g.max_tokens + g.prompt_tokens > ep.kv_tokens:
                    raise RunError(
                        f"group {g.group_id}: max_tokens + prompt_tokens exceeds kv_tokens "
                        "(it could never be admitted)"
                    )
        mk = engine_factory or make_engine
        self.engines = [
            mk(
                w,
                ep,
                inflight=system.inflight,
                swap_ms=system.swap_ms,
                switch_ms=system.switch_ms,
            )
            for w in range(system.n_workers)
        ]
        self.L: list[int] = []
        self.V: list[int] = []
        self.gP: list[int] = []
        self.gM: list[int] = []
        self.sg: list[tuple[int, int]] = []
        for gi, g in enumerate(trace.groups):
            self.gP.append(g.prompt_tokens)
            self.gM.append(g.max_tokens)
            for i, (r, v) in enumerate(zip(g.resp_tokens, g.verify_ms, strict=True)):
                self.L.append(r)
                self.V.append(v)
                self.sg.append((gi, i))
        self.wheap: list[tuple[int, int, int]] = []  # (t, worker, stamp)
        self.wstamp = [0] * len(self.engines)
        self.vheap: list[tuple[int, int]] = []  # (t, server)
        self.train_end: tuple[int, int] | None = None  # (t, step)
        self.decision_ns: list[int] = []
        self.instants = 0
        self.max_instants = max_instants
        self.t = 0

    # ------------------------------------------------------------------ helpers
    def _resched(self, w: int) -> None:
        self.wstamp[w] += 1
        nt = self.engines[w].next_time()
        if nt is not None:
            heapq.heappush(self.wheap, (nt, w, self.wstamp[w]))

    def _next_worker_time(self) -> int | None:
        h = self.wheap
        while h and h[0][2] != self.wstamp[h[0][1]]:
            heapq.heappop(h)
        return h[0][0] if h else None

    def _report(self, w: int, t: int) -> None:
        eng = self.engines[w]
        ev, tl = eng.report()
        self.core.on_report(w, t, ev, tl, eng.snapshot())

    def _flush_inputs(self, t: int, touched: set[int]) -> None:
        core = self.core
        if not core.out_worker:
            return
        out, core.out_worker = core.out_worker, []
        for w, item in out:
            if item[0] == "place":
                es = []
                for sid in item[1]:
                    gi, i = self.sg[sid]
                    es.append(ESample(sid, gi, i, self.gP[gi], self.gM[gi], self.L[sid]))
                item = ("place", es)
            self.engines[w].push(t, item)
            touched.add(w)

    def _flush_train(self, t: int) -> None:
        core = self.core
        while core.out_train:
            step, tokens = core.out_train.pop(0)
            self.train_end = (t + self.sys.train_ms(tokens), step)

    def next_time(self) -> int | None:
        cands = [self._next_worker_time()]
        if self.vheap:
            cands.append(self.vheap[0][0])
        if self.train_end is not None:
            cands.append(self.train_end[0])
        cands.append(self.core.next_internal_time())
        cands = [c for c in cands if c is not None]
        return min(cands) if cands else None

    # ------------------------------------------------------------------ one instant
    def instant(self, t: int, policy_only: bool = False) -> None:
        core = self.core
        self.t = t
        core.t = t
        touched: set[int] = set()
        if not policy_only:
            # 1. worker boundaries, by worker id
            due = []
            h = self.wheap
            while h and h[0][0] == t:
                _, w, stamp = heapq.heappop(h)
                if stamp == self.wstamp[w]:
                    due.append(w)
            for w in sorted(due):
                self.engines[w].boundary(t)
                self._report(w, t)
                touched.add(w)
            # 2-3. verifier completions (by server), assignments; zero-time verifications loop
            while True:
                comps = []
                while self.vheap and self.vheap[0][0] == t:
                    comps.append(heapq.heappop(self.vheap)[1])
                for sv in sorted(comps):
                    core.on_verified(sv, t)
                zero = False
                for sv, sid in core.assign_verifiers(t, self.policy):
                    d = self.V[sid]
                    heapq.heappush(self.vheap, (t + d, sv))
                    zero = zero or d == 0
                if not zero:
                    break
            # 4. trainer events
            if self.train_end is not None and self.train_end[0] == t:
                step = self.train_end[1]
                self.train_end = None
                core.on_train_done(step, t)
                if core.ended:
                    return
            core.trainer_events(t)
            self._flush_train(t)
            # 5-6. selection and the staleness rules
            core.try_select(t)
            self._flush_train(t)
            if core.ended:
                return
            self._flush_inputs(t, touched)
        # 7. the policy, once
        t0 = time.perf_counter_ns()
        core.invoke(self.policy, t)
        self.decision_ns.append(time.perf_counter_ns() - t0)
        self._flush_inputs(t, touched)
        # 8. commits, by worker id
        for w in sorted(touched):
            eng = self.engines[w]
            if eng.wants_commit(t):
                eng.commit(t)
                self._report(w, t)
        core.after_commit(t)
        self._flush_inputs(t, touched)
        self._flush_train(t)
        for w in touched:
            self._resched(w)

    def run(self) -> Simulation:
        core = self.core
        self.instant(0)
        while not core.ended:
            self.instants += 1
            if self.instants > self.max_instants:
                raise RunError(f"policy {core.policy_name!r}: too many instants")
            t = self.next_time()
            if t is None:
                # deadlock rule: call the policy again at the same instant
                self.instant(self.t, policy_only=True)
                if self.next_time() is None:
                    why = "the stream is exhausted" if not core.unlaunched else "nothing can move"
                    raise RunError(
                        f"policy {core.policy_name!r}: deadlock at t={self.t} ms ({why}; "
                        f"{len(core.outstanding)} outstanding, {len(core.ready)} ready, "
                        f"trainer {core.tstate})"
                    )
                continue
            self.instant(t)
        end_tokens: dict[int, int] = {}
        for eng in self.engines:
            end_tokens.update(eng.flush(core.t_end))
        core.finalize(end_tokens)
        return self

    def generated_tokens(self) -> int:
        return sum(e.gen_tokens for e in self.engines)


def simulate(system: SystemConfig, trace: Trace, policy, **kw) -> Simulation:
    return Simulation(system, trace, policy, **kw).run()
