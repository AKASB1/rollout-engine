"""Rollout worker engines (docs/simulator.md sections 3 and 4).

An engine models one rollout worker. It has no clock and no event loop of its own: a driver
(the discrete-event simulator, or a mock worker of the live service) calls

    next_time()       the next instant at which the worker has an event (or None)
    boundary(t)       pre-policy, at that instant: finishes and pause ends
    push(t, item)     an input from the core: ("place", [samples]), ("drop", gidx),
                      ("publish", version), ("switch_out",), ("switch_in",)
    wants_commit(t)   whether the worker is at a boundary at t with something to apply
    commit(t)         post-policy, at that instant: apply inputs, admit, start a phase or pause
    report()          drain the events and state changes since the last report

Inputs take effect at the first boundary at or after their time; an input that arrives after
the worker committed at instant t takes effect at its next boundary after t. Every iteration
takes at least one millisecond (``a0_ns >= 10^6`` is checked), so iteration boundaries of a
phase fall on distinct milliseconds.

Continuous engine (iteration-level batching): between two changes the worker is in a phase
with fixed ``b`` running samples and starting context ``C0``; ``k`` iterations take
``S(k) = k*A + D*k*(k-1)/2`` ns with ``A = a0 + a1*b + a2*C0`` and ``D = a2*b``; the boundary
after iteration ``j`` is at ``t0 + ceil_ms(S(j))``. Everything is integer arithmetic; the
finish iteration of every running sample sits in a heap, so an event costs O(log b).
"""

from __future__ import annotations

import heapq
import math
from collections import deque

from rollout_engine.batching import batch_cost_ms as static_batch_ms  # noqa: F401 (re-export)
from rollout_engine.batching import batch_cost_ns as static_batch_ns
from rollout_engine.clock import NS_PER_MS, ceil_ms
from rollout_engine.config import EngineParams

GEN, OVERHEAD, IDLE, PARKED = "gen", "overhead", "idle", "parked"


class EngineError(RuntimeError):
    pass


class ESample:
    """Engine-side sample. ``L`` is the hidden true length (known only to the worker)."""

    __slots__ = (
        "sid",
        "gidx",
        "sidx",
        "P",
        "M",
        "L",
        "g0",
        "s_iter",
        "seg_iter",
        "seg_version",
        "segs",
        "removed",
        "started",
    )

    def __init__(self, sid: int, gidx: int, sidx: int, P: int, M: int, L: int):
        self.sid = sid
        self.gidx = gidx
        self.sidx = sidx
        self.P = P
        self.M = M
        self.L = L
        self.g0 = 0
        self.s_iter = 0
        self.seg_iter = 0
        self.seg_version = 0
        self.segs: list[list[int]] = []
        self.removed = False
        self.started = False

    def key(self) -> tuple[int, int]:
        return (self.gidx, self.sidx)


def s_of(k: int, A: int, D: int) -> int:
    """S(k) = k*A + D*k*(k-1)/2 in ns (exact integer)."""
    return k * A + D * k * (k - 1) // 2


def first_k_exceeding(X: int, A: int, D: int) -> int:
    """The smallest k >= 0 with S(k) > X (X may be negative; then 0)."""
    if X < 0:
        return 0
    if D == 0:
        k = X // A + 1
    else:
        # D k^2 + (2A - D) k - 2X > 0  -> positive root via isqrt, then fix by neighbours
        B = 2 * A - D
        disc = B * B + 8 * D * X
        k = max(0, (math.isqrt(disc) - B) // (2 * D))
    while k > 0 and s_of(k - 1, A, D) > X:
        k -= 1
    while s_of(k, A, D) <= X:
        k += 1
    return k


class _Base:
    kind = "base"

    def __init__(
        self,
        wid: int,
        p: EngineParams,
        inflight: str = "drain",
        swap_ms: int = 0,
        switch_ms: int = 0,
        version: int = 0,
    ):
        if p.a0_ns < NS_PER_MS:
            raise EngineError("a0_ns must be >= 1 ms so that iteration boundaries are distinct")
        self.wid = wid
        self.p = p
        self.inflight = inflight
        self.swap_ms = swap_ms
        self.switch_ms = switch_ms
        self.version = version
        self.target = version
        self.inbox: list[tuple[int, int, tuple]] = []  # (effective time, seq, item)
        self._seq = 0
        self.last_commit = -1
        self.at_bt = -1  # the instant of the boundary the worker is at
        self.paused_until: int | None = None
        self.pause_kind = ""
        self.parked = False
        self.draining = False
        self.swap_pending = False
        self.switch_in_pending = False
        self.switch_out_pending = False
        self.events: list[tuple] = []
        self.tl: list[tuple[int, str]] = []
        self._state = IDLE
        self.iters = 0

    # -- inputs ---------------------------------------------------------------------------
    def push(self, t: int, item: tuple) -> None:
        # an input after this instant's commit waits for the next boundary, except on an
        # idle worker, which has no iteration in progress and commits again at once
        te = t if (t > self.last_commit or self._idle_now()) else self.last_commit + 1
        self._seq += 1
        self.inbox.append((te, self._seq, item))

    def _take_inputs(self, t: int) -> list[tuple]:
        due = [x for x in self.inbox if x[0] <= t]
        if due:
            self.inbox = [x for x in self.inbox if x[0] > t]
            due.sort(key=lambda x: (x[0], x[1]))
        return [x[2] for x in due]

    def _min_input_time(self) -> int | None:
        return min(x[0] for x in self.inbox) if self.inbox else None

    def _has_control_input(self) -> bool:
        return any(x[2][0] in ("publish", "switch_in", "switch_out") for x in self.inbox)

    def _inbox_placed(self) -> int:
        return sum(len(x[2][1]) for x in self.inbox if x[2][0] == "place")

    # -- state / reports ----------------------------------------------------------------
    def _set_state(self, t: int, state: str) -> None:
        if state != self._state:
            self._state = state
            if self.tl and self.tl[-1][0] == t:
                self.tl[-1] = (t, state)
            else:
                self.tl.append((t, state))

    def _settle_state(self, t: int) -> None:
        if self.paused_until is None:
            self._set_state(t, PARKED if self.parked else (GEN if self._busy() else IDLE))

    def gpu_state(self) -> str:
        return self._state

    def snapshot(self) -> dict:
        return {
            "version": self.version,
            "paused": self.paused_until is not None,
            "draining": self.draining,
            "parked": self.parked,
            "admitted": self._n_admitted(),
            "waiting": self._n_waiting(),
            "kv_used": self._kv_used(),
            "iters": self.iters,
            "busy": self._busy_flag(),
        }

    def report(self) -> tuple[list[tuple], list[tuple[int, str]]]:
        ev, tl = self.events, self.tl
        self.events, self.tl = [], []
        return ev, tl

    def _pause(self, t: int, ms: int, kind: str) -> bool:
        if ms <= 0:
            return False
        self.paused_until = t + ms
        self.pause_kind = kind
        self._set_state(t, OVERHEAD)
        return True


class ContinuousEngine(_Base):
    kind = "continuous"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.waiting: deque[ESample] = deque()
        self.admitted: dict[int, ESample] = {}  # sid -> sample (started or pending)
        self.pending: list[ESample] = []  # admitted, prefilled or in prefill, not started
        self.heap: list[tuple[int, int, int, int, ESample]] = []
        self.kv_used = 0
        self.resident: dict[int, int] = {}  # gidx -> admitted samples of the group
        self.t0 = 0
        self.iter0 = 0
        self.b = 0
        self.sum_const = 0  # sum over started samples of (P + g0 - s_iter)
        self.cur_iter = 0
        self.need_restart = False
        self.gen_tokens = 0  # tokens generated so far (all samples), for accounting checks
        self._acc_iter = 0

    # -- phase arithmetic (overridden by the step-by-step reference) ----------------------
    def _AD(self) -> tuple[int, int]:
        p = self.p
        c0 = self.sum_const + self.b * self.iter0
        return p.a0_ns + p.a1_ns * self.b + p.a2_ns * c0, p.a2_ns * self.b

    def _boundary_time(self, j: int) -> int:
        A, D = self._AD()
        return self.t0 + ceil_ms(s_of(j, A, D))

    def _first_j_at_or_after(self, t: int) -> int:
        """Smallest j >= 0 with t0 + ceil_ms(S(j)) >= t."""
        d = t - self.t0
        if d <= 0:
            return 0
        A, D = self._AD()
        return first_k_exceeding((d - 1) * NS_PER_MS, A, D)

    def _top(self) -> ESample | None:
        h = self.heap
        while h and h[0][4].removed:
            heapq.heappop(h)
        return h[0][4] if h else None

    def _finish_j(self) -> int | None:
        s = self._top()
        if s is None:
            return None
        return s.s_iter + (s.L - s.g0) - self.iter0

    def _finishing(self, at_iter: int) -> list[ESample]:
        """The started samples whose last token is generated by iteration ``at_iter``,
        in (group, sample_idx) order."""
        out = []
        h = self.heap
        while h and (h[0][4].removed or h[0][0] == at_iter):
            s = heapq.heappop(h)[4]
            if not s.removed:
                out.append(s)
        return out

    # -- protocol -------------------------------------------------------------------------
    def next_time(self) -> int | None:
        if self.paused_until is not None:
            return self.paused_until
        te = self._min_input_time()
        if self.b > 0:
            best = self._boundary_time(self._finish_j())
            if te is not None:
                best = min(best, self._boundary_time(self._first_j_at_or_after(te)))
            return best
        return te

    def boundary(self, t: int) -> None:
        if self.paused_until is not None:
            if self.paused_until != t:
                raise EngineError(f"worker {self.wid}: boundary({t}) during a pause")
            self.paused_until = None
            kind = self.pause_kind
            self.pause_kind = ""
            if kind == "switch_out":
                self.parked = True
                self._set_state(t, PARKED)
            self.need_restart = True
            self.at_bt = t
            return
        if self.b > 0:
            fj = self._finish_j()
            if self._boundary_time(fj) == t:
                self._at_iter(self.iter0 + fj)
                for s in self._finishing(self.cur_iter):
                    self._retire(s, self.cur_iter)
                    self.events.append(("finish", s.sid, s.L, s.segs))
                self.need_restart = True
            else:
                j = self._first_j_at_or_after(t)
                if self._boundary_time(j) != t:
                    raise EngineError(f"worker {self.wid}: no boundary at {t}")
                self._at_iter(self.iter0 + j)
        self.at_bt = t

    def wants_commit(self, t: int) -> bool:
        if self.at_bt == t and self.last_commit != t:
            return True
        te = self._min_input_time()
        if te is None or te > t or self.paused_until is not None:
            return False
        if self.b == 0:
            return True
        j = self._first_j_at_or_after(t)
        return self._boundary_time(j) == t

    def _at_iter(self, n: int) -> None:
        self.gen_tokens += self.b * (n - self._acc_iter)
        self._acc_iter = n
        self.cur_iter = n
        self.iters = n

    def flush(self, t: int) -> dict[int, int]:
        """At the end of a run: tokens of the admitted samples at time t (iterations whose
        boundary is at or before t), and the generated-token counter brought up to t."""
        if self.b > 0 and self.paused_until is None:
            self._at_iter(self.iter0 + self._first_j_at_or_after(t + 1) - 1)
        return {s.sid: self.tokens_now(s) for s in self.admitted.values()}

    def _retire(self, s: ESample, at_iter: int) -> int:
        """Remove an admitted sample (finish or drop); return its tokens."""
        if s.started:
            tokens = s.g0 + at_iter - s.s_iter
            s.segs.append([s.seg_version, at_iter - s.seg_iter])
            self.b -= 1
            self.sum_const -= s.P + s.g0 - s.s_iter
        else:
            tokens = 0
            self.pending.remove(s)
        s.removed = True
        del self.admitted[s.sid]
        self.kv_used -= s.M
        n = self.resident[s.gidx] - 1
        if n:
            self.resident[s.gidx] = n
        else:
            del self.resident[s.gidx]
            self.kv_used -= s.P
        return tokens

    def _busy(self) -> bool:
        return self.b > 0

    def _n_admitted(self) -> int:
        return len(self.admitted)

    def _n_waiting(self) -> int:
        return len(self.waiting) + self._inbox_placed()

    def _kv_used(self) -> int:
        return self.kv_used

    def _idle_now(self) -> bool:
        return self.b == 0 and not self.admitted and self.paused_until is None

    def _busy_flag(self) -> bool:
        return False

    def commit(self, t: int) -> None:
        self._commit(t)
        self._settle_state(t)

    def _commit(self, t: int) -> None:
        if self.at_bt != t:
            # an input arrived at an iteration boundary that had no event, or at an idle worker
            if self.b > 0:
                j = self._first_j_at_or_after(t)
                if self._boundary_time(j) != t:
                    raise EngineError(f"worker {self.wid}: commit({t}) between boundaries")
                self._at_iter(self.iter0 + j)
            self.at_bt = t
        if self.paused_until is not None:
            raise EngineError(f"worker {self.wid}: commit({t}) during a pause")
        self.last_commit = t
        changed = self.need_restart
        self.need_restart = False
        p = self.p
        for item in self._take_inputs(t):
            op = item[0]
            if op == "place":
                self.waiting.extend(item[1])
            elif op == "drop":
                g = item[1]
                if any(s.gidx == g for s in self.waiting):
                    self.waiting = deque(s for s in self.waiting if s.gidx != g)
                victims = sorted(
                    (s for s in self.admitted.values() if s.gidx == g), key=ESample.key
                )
                for s in victims:
                    tokens = self._retire(s, self.cur_iter)
                    self.events.append(("remove", s.sid, tokens, s.segs))
                    changed = True
            elif op == "publish":
                if item[1] > self.target:
                    self.target = item[1]
                if self.target > self.version:
                    if self.inflight == "drain":
                        self.draining = True
                    else:
                        self.swap_pending = True
            elif op == "switch_out":
                self.switch_out_pending = True
            elif op == "switch_in":
                self.switch_in_pending = True
            else:
                raise EngineError(f"unknown input {op!r}")
        if self.switch_out_pending:
            self.switch_out_pending = False
            if self.admitted:
                raise EngineError(f"worker {self.wid}: switch_out with admitted samples")
            self._close_phase(t)
            if not self._pause(t, self.switch_ms, "switch_out"):
                self.parked = True
                self._set_state(t, PARKED)
            return
        if self.switch_in_pending:
            self.switch_in_pending = False
            self.parked = False
            self.version = self.target
            self.swap_pending = self.draining = False
            if self._pause(t, self.switch_ms, "switch_in"):
                return
            self._set_state(t, IDLE)
        if self.parked:
            return
        if self.swap_pending:
            self.swap_pending = False
            if self.target > self.version:
                dur = self.swap_ms * NS_PER_MS
                if self.inflight == "interrupt":
                    for s in self.admitted.values():
                        ctx = s.P + (s.g0 + self.cur_iter - s.s_iter if s.started else 0)
                        dur += p.prefill_c0_ns + p.prefill_c1_ns * ctx
                self._close_phase(t)
                self._set_version(self.target)
                if self._pause(t, ceil_ms(dur), "swap"):
                    self.need_restart = True
                    return
                changed = True
        if self.draining:
            if not self.admitted:
                self.draining = False
                self._close_phase(t)
                self._set_version(self.target)
                if self._pause(t, self.swap_ms, "drain"):
                    return
                changed = True
            else:
                if changed:
                    self._restart(t)
                return
        # admission (FIFO, the head is never skipped)
        prefill = 0
        while self.waiting and len(self.admitted) < p.max_seqs:
            s = self.waiting[0]
            need = s.M + (0 if s.gidx in self.resident else s.P)
            if self.kv_used + need > p.kv_tokens:
                break
            self.waiting.popleft()
            if s.gidx in self.resident:
                self.resident[s.gidx] += 1
            else:
                self.resident[s.gidx] = 1
                prefill += p.prefill_c0_ns + p.prefill_c1_ns * s.P
            self.kv_used += need
            self.admitted[s.sid] = s
            self.pending.append(s)
            self.events.append(("admit", s.sid))
            changed = True
        if prefill:
            self._close_phase(t)
            if self._pause(t, ceil_ms(prefill), "prefill"):
                self.need_restart = True
                return
        if changed or self.pending:
            self._restart(t)

    def _set_version(self, v: int) -> None:
        if v == self.version:
            return
        for s in self.admitted.values():
            if s.started:
                s.segs.append([s.seg_version, self.cur_iter - s.seg_iter])
                s.seg_iter = self.cur_iter
                s.seg_version = v
        self.version = v

    def _close_phase(self, t: int) -> None:
        self.t0 = t
        self.iter0 = self.cur_iter

    def _restart(self, t: int) -> None:
        self.t0 = t
        self.iter0 = self.cur_iter
        if self.pending:
            for s in self.pending:
                s.started = True
                s.g0 = 0
                s.s_iter = self.cur_iter
                s.seg_iter = self.cur_iter
                s.seg_version = self.version
                self.b += 1
                self.sum_const += s.P - self.cur_iter
                heapq.heappush(self.heap, (self.cur_iter + s.L, s.gidx, s.sidx, s.sid, s))
                self.events.append(("start", s.sid, self.version, self.cur_iter))
            self.pending = []
        self._set_state(t, GEN if self.b > 0 else IDLE)

    # -- read-only accessors used by the core mirror and tests -------------------------------
    def tokens_now(self, s: ESample) -> int:
        return s.g0 + self.cur_iter - s.s_iter if s.started else 0


class StaticEngine(_Base):
    """Static batching: an idle worker runs one batch to completion (prefill, then Lmax
    iterations with all b samples carried); all samples are generated at the batch end.
    Inputs (drops, publications) take effect at the batch end; swap and interrupt behave as
    drain because no sample is in flight at a batch boundary."""

    kind = "static"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.batch: list[ESample] = []
        self.pending_batch: list[ESample] | None = None
        self.prefill_end = -1
        self.batch_end: int | None = None
        self.gen_tokens = 0

    def next_time(self) -> int | None:
        if self.paused_until is not None:
            return self.paused_until
        if self.batch_end is not None:
            return self.batch_end
        return self._min_input_time()

    def boundary(self, t: int) -> None:
        if self.paused_until is not None:
            if self.paused_until != t:
                raise EngineError(f"worker {self.wid}: boundary({t}) during a pause")
            self.paused_until = None
            if self.pause_kind == "switch_out":
                self.parked = True
                self._set_state(t, PARKED)
            self.pause_kind = ""
        elif self.batch_end == t:
            for s in sorted(self.batch, key=ESample.key):
                s.segs.append([s.seg_version, s.L])
                self.events.append(("finish", s.sid, s.L, s.segs))
                self.gen_tokens += s.L
            self.iters += max(s.L for s in self.batch)
            self.batch = []
            self.batch_end = None
            self._set_state(t, IDLE)
        elif self.batch_end is not None:
            raise EngineError(f"worker {self.wid}: no boundary at {t}")
        # else: an idle worker woken by an input
        self.at_bt = t

    def wants_commit(self, t: int) -> bool:
        if self.at_bt == t and self.last_commit != t:
            return True
        te = self._min_input_time()
        return te is not None and te <= t and self.paused_until is None and self.batch_end is None

    def flush(self, t: int) -> dict[int, int]:
        """At the end of a run: tokens of the samples of the batch in flight at time t."""
        if not self.batch:
            return {}
        out = {}
        if t > self.prefill_end:
            p = self.p
            b = len(self.batch)
            pmax = max(s.P for s in self.batch)
            A = p.a0_ns + p.a1_ns * b + p.a2_ns * b * pmax
            D = p.a2_ns * b
            j = first_k_exceeding((t - self.prefill_end) * NS_PER_MS, A, D) - 1
        else:
            j = 0
        for s in self.batch:
            out[s.sid] = min(s.L, j)
            self.gen_tokens += out[s.sid]
        return out

    def idle(self) -> bool:
        return (
            self.batch_end is None
            and self.paused_until is None
            and self.pending_batch is None
            and not self.parked
        )

    def _busy(self) -> bool:
        return self.batch_end is not None

    def _n_admitted(self) -> int:
        return len(self.batch)

    def _n_waiting(self) -> int:
        return len(self.pending_batch or ())

    def _kv_used(self) -> int:
        return 0

    def _busy_flag(self) -> bool:
        return (
            self.batch_end is not None or self.pending_batch is not None or self._inbox_placed() > 0
        )

    def _idle_now(self) -> bool:
        return self.batch_end is None and self.paused_until is None

    def commit(self, t: int) -> None:
        self._commit(t)
        if self.batch_end is None:
            self._settle_state(t)

    def _commit(self, t: int) -> None:
        if self.paused_until is not None or self.batch_end is not None:
            raise EngineError(f"worker {self.wid}: commit({t}) while busy")
        self.at_bt = t
        self.last_commit = t
        for item in self._take_inputs(t):
            op = item[0]
            if op == "place":
                if self.pending_batch is not None:
                    raise EngineError(f"worker {self.wid}: a second batch placed while one waits")
                if len(item[1]) > self.p.static_batch or not item[1]:
                    raise EngineError(f"worker {self.wid}: batch size {len(item[1])} invalid")
                self.pending_batch = list(item[1])
            elif op == "drop":
                g = item[1]
                if self.pending_batch is not None:
                    keep = []
                    for s in self.pending_batch:
                        if s.gidx == g:
                            self.events.append(("remove", s.sid, 0, s.segs))
                        else:
                            keep.append(s)
                    self.pending_batch = keep or None
            elif op == "publish":
                self.target = max(self.target, item[1])
            elif op == "switch_out":
                self.switch_out_pending = True
            elif op == "switch_in":
                self.switch_in_pending = True
            else:
                raise EngineError(f"unknown input {op!r}")
        if self.switch_out_pending:
            self.switch_out_pending = False
            if self.pending_batch:
                raise EngineError(f"worker {self.wid}: switch_out with a waiting batch")
            if not self._pause(t, self.switch_ms, "switch_out"):
                self.parked = True
                self._set_state(t, PARKED)
            return
        if self.switch_in_pending:
            self.switch_in_pending = False
            self.parked = False
            self.version = self.target
            if self._pause(t, self.switch_ms, "switch_in"):
                return
            self._set_state(t, IDLE)
        if self.parked:
            return
        if self.target > self.version:
            self.version = self.target
            if self._pause(t, self.swap_ms, "swap"):
                return
        if self.pending_batch:
            self._start_batch(t, self.pending_batch)
            self.pending_batch = None

    def _start_batch(self, t: int, batch: list[ESample]) -> None:
        b = len(batch)
        pmax = max(s.P for s in batch)
        lmax = max(s.L for s in batch)
        pre, dec = static_batch_ns(self.p, b, pmax, lmax)
        pre_ms, dec_ms = ceil_ms(pre), ceil_ms(dec)
        self.batch = batch
        self.prefill_end = t + pre_ms
        self.batch_end = t + pre_ms + dec_ms
        for s in sorted(batch, key=ESample.key):
            s.started = True
            s.seg_version = self.version
            self.events.append(("admit", s.sid))
            self.events.append(("start", s.sid, self.version, self.iters))
        if pre_ms:
            self._set_state(t, OVERHEAD)
        self._set_state(t + pre_ms, GEN)
        self.at_bt = t


def make_engine(wid: int, p: EngineParams, **kw) -> _Base:
    if p.engine == "continuous":
        return ContinuousEngine(wid, p, **kw)
    return StaticEngine(wid, p, **kw)
