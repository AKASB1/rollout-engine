"""The core state machine shared by the simulator and the live controller.

The core holds the scheduling state only: groups and samples, a mirror of each worker built
from the worker's reports, the verifier queue, the trainer, versions and the staleness rules.
It never sees a hidden length or verifier time; the worker engines (in the simulator driver
or in the mock workers of the live service) and the verifier servers report them.

Order at one instant (docs/simulator.md section 1): worker boundaries (by id) ->
verifier completions (by server) and assignments -> readiness -> trainer events ->
selection -> staleness rules -> the policy (once) -> worker commits (by id) -> the dead-group
check for samples that started in those commits. A driver calls the methods in that order.
"""

from __future__ import annotations

import bisect
import math

from rollout_engine.api import Drop, GroupSpec, InvalidAction, Place, RunError
from rollout_engine.config import SystemConfig
from rollout_engine.verifier.pool import VerifierPool

# group states
UNLAUNCHED, OUTSTANDING, CONSUMED, DROPPED, UNFINISHED = (
    "unlaunched",
    "outstanding",
    "consumed",
    "dropped",
    "unfinished",
)
# sample states
UNPLACED, WAITING, RUNNING, GENERATED, VERIFYING, VERIFIED = (
    "unplaced",
    "waiting",
    "running",
    "generated",
    "verifying",
    "verified",
)


class CGroup:
    __slots__ = (
        "gidx",
        "spec",
        "state",
        "sids",
        "n_unplaced",
        "n_generated",
        "n_verified",
        "fin_sum",
        "fin_n",
        "version",
        "launched_at",
        "ready_at",
        "step",
        "drop_reason",
        "drop_t",
        "tokens",
        "workers",
    )

    def __init__(self, gidx: int, spec: GroupSpec, first_sid: int):
        self.gidx = gidx
        self.spec = spec
        self.state = UNLAUNCHED
        self.sids = list(range(first_sid, first_sid + spec.n_samples))
        self.n_unplaced = spec.n_samples
        self.n_generated = 0
        self.n_verified = 0
        self.fin_sum = 0
        self.fin_n = 0
        self.version: int | None = None
        self.launched_at = -1
        self.ready_at = -1
        self.step = -1
        self.drop_reason = ""
        self.drop_t = -1
        self.tokens = 0  # tokens of finished or removed samples
        self.workers: dict[int, int] = {}  # worker -> samples placed there (not yet removed)


class CSample:
    __slots__ = (
        "sid",
        "gidx",
        "sidx",
        "state",
        "worker",
        "admitted",
        "started",
        "version",
        "start_iter",
        "tokens",
        "segs",
        "gen_t",
        "ver_start",
        "ver_end",
        "start_t",
        "removed",
        "batch",
        "end_t",
        "attempts",
    )

    def __init__(self, sid: int, gidx: int, sidx: int):
        self.sid = sid
        self.gidx = gidx
        self.sidx = sidx
        self.state = UNPLACED
        self.worker = -1
        self.admitted = False
        self.started = False
        self.version = -1
        self.start_iter = 0
        self.tokens = 0
        self.segs: list = []
        self.gen_t = -1
        self.ver_start = -1
        self.ver_end = -1
        self.start_t = -1
        self.removed = False
        self.batch = -1
        self.end_t = -1  # time of the finish or removal report
        self.attempts = 0  # placements lost with a worker (live service)


class CWorker:
    __slots__ = (
        "wid",
        "version",
        "paused",
        "draining",
        "parked",
        "admitted",
        "waiting",
        "kv_used",
        "iters",
        "busy",
        "report_t",
        "gpu",
        "sids",
        "resident",
        "alive",
    )

    def __init__(self, wid: int):
        self.wid = wid
        self.version = 0
        self.paused = False
        self.draining = False
        self.parked = False
        self.admitted = 0
        self.waiting = 0
        self.kv_used = 0
        self.iters = 0
        self.busy = False
        self.report_t = 0
        self.gpu = "idle"
        self.sids: dict[int, None] = {}  # samples placed here and not finished/removed (ordered)
        self.resident: dict[int, int] = {}  # gidx -> admitted samples (prefix residency)
        self.alive = True  # False once the live service declared the worker lost


class StepRec:
    __slots__ = (
        "s",
        "t_sel",
        "groups",
        "tokens",
        "staleness",
        "stale_tokens",
        "t_train_start",
        "t_train_end",
        "t_pub",
    )

    def __init__(self, s: int, t_sel: int, groups: list[int]):
        self.s = s
        self.t_sel = t_sel
        self.groups = groups
        self.tokens = 0
        self.staleness: list[int] = []
        self.stale_tokens = 0
        self.t_train_start = -1
        self.t_train_end = -1
        self.t_pub = -1


class Core:
    def __init__(
        self,
        system: SystemConfig,
        specs: list[GroupSpec],
        policy_name: str = "?",
        record_log: bool = False,
    ):
        self.sys = system
        self.policy_name = policy_name
        self.groups: list[CGroup] = []
        self.samples: list[CSample] = []
        self.unlaunched: list[int] = []
        self.outstanding: dict[int, CGroup] = {}
        self.ready: list[tuple[int, int]] = []  # (ready_at, gidx), sorted
        self.workers = [CWorker(w) for w in range(system.n_workers)]
        self.vpool = VerifierPool(system.verifier_servers)
        self.single = system.single_phase
        self.T = 1 if self.single else system.steps
        self.eta = system.eta
        self.s_next = 0
        self.v_pub = 0
        self.tstate = "wait"
        self.t_until = -1  # end of sync or switch-in
        self.training: int = -1  # step in training
        self.await_park = False
        self.steps: list[StepRec] = []
        self.trainer_tl: list[tuple[int, str]] = [(0, "wait")]
        self.worker_tl: list[list[tuple[int, str]]] = [[(0, "idle")] for _ in self.workers]
        self.log: list[tuple] | None = [] if record_log else None
        self.out_worker: list[tuple[int, tuple]] = []
        self.out_train: list[tuple[int, int]] = []
        self.ended = False
        self.t_end = -1
        self.t = 0
        self.n_consumed = 0
        self.n_dropped = 0
        self.wasted_tokens = 0
        self.batches: list[list[int]] = []  # static batches (sids), for padding_frac
        self._new_versions: list[int] = []  # groups whose version was set or lowered at commit
        self.add_groups(specs)
        self.B = len(self.groups) if self.single else system.groups_per_step
        self.policy_calls = 0

    # ------------------------------------------------------------------ stream
    def add_groups(self, specs: list[GroupSpec]) -> None:
        for spec in specs:
            gidx = len(self.groups)
            g = CGroup(gidx, spec, len(self.samples))
            self.groups.append(g)
            for i in range(spec.n_samples):
                self.samples.append(CSample(len(self.samples), gidx, i))
            self.unlaunched.append(gidx)
        if getattr(self, "_view", None) is not None:
            self._view._sync()  # the live queue grew

    # ------------------------------------------------------------------ helpers
    def gate_open(self) -> bool:
        return self.single or self.s_next - self.v_pub <= self.eta

    def _set_trainer(self, t: int, state: str) -> None:
        if self.trainer_tl[-1][1] != state:
            if self.trainer_tl[-1][0] == t:
                self.trainer_tl[-1] = (t, state)
            else:
                self.trainer_tl.append((t, state))
        self.tstate = state

    def alive_workers(self) -> list[CWorker]:
        return [w for w in self.workers if w.alive]

    def _push(self, wid: int, item: tuple) -> None:
        self.out_worker.append((wid, item))

    def accepting(self, w: CWorker) -> bool:
        """Whether a unit placed on ``w`` now can start being admitted at this instant."""
        if not w.alive or w.paused or w.draining or w.parked or w.version < self.v_pub:
            return False
        if self.sys.colocated and self.tstate in ("train", "switch"):
            return False
        if self.sys.engine.engine == "static":
            return not w.busy
        if w.waiting:
            return False
        return w.report_t == self.t or w.admitted == 0

    # ------------------------------------------------------------------ worker reports
    def on_report(self, wid: int, t: int, events, timeline, snap: dict) -> None:
        self.t = t
        w = self.workers[wid]
        for ev in events:
            kind = ev[0]
            sid = ev[1]
            s = self.samples[sid]
            g = self.groups[s.gidx]
            if kind == "admit":
                s.admitted = True
                if s.state == WAITING:
                    s.state = RUNNING
                w.resident[s.gidx] = w.resident.get(s.gidx, 0) + 1
            elif kind == "start":
                s.started = True
                s.version = ev[2]
                s.start_iter = ev[3]
                s.start_t = t
                if g.state == OUTSTANDING and (g.version is None or ev[2] < g.version):
                    g.version = ev[2]
                    self._new_versions.append(g.gidx)
            elif kind in ("finish", "remove"):
                s.tokens = ev[2]
                s.segs = [list(x) for x in ev[3]]
                s.removed = kind == "remove"
                s.end_t = t
                if s.admitted:
                    n = w.resident[s.gidx] - 1
                    if n:
                        w.resident[s.gidx] = n
                    else:
                        del w.resident[s.gidx]
                w.sids.pop(sid, None)
                c = g.workers.get(wid, 0) - 1
                if c > 0:
                    g.workers[wid] = c
                else:
                    g.workers.pop(wid, None)
                g.tokens += s.tokens
                if g.state == DROPPED:
                    self.wasted_tokens += s.tokens
                if kind == "finish":
                    if s.gen_t >= 0:
                        raise RunError(f"sample {sid} generated twice")
                    s.gen_t = t
                    s.state = GENERATED
                    g.n_generated += 1
                    g.fin_sum += s.tokens
                    g.fin_n += 1
                    if g.state == OUTSTANDING:
                        self.vpool.enqueue(sid, g.gidx, g.spec.task, t)
            else:
                raise RunError(f"unknown worker event {kind!r}")
        if timeline:
            tl = self.worker_tl[wid]
            for tt, st in timeline:
                if tl[-1][1] == st:
                    continue
                if tl[-1][0] == tt:
                    tl[-1] = (tt, st)
                    if len(tl) > 1 and tl[-2][1] == st:
                        tl.pop()
                else:
                    tl.append((tt, st))
        w.version = snap["version"]
        w.paused = snap["paused"]
        w.draining = snap["draining"]
        w.parked = snap["parked"]
        w.admitted = snap["admitted"]
        w.waiting = snap["waiting"]
        w.kv_used = snap["kv_used"]
        w.iters = snap["iters"]
        w.busy = snap["busy"]
        w.report_t = t
        if self.await_park and w.parked and all(x.parked for x in self.alive_workers()):
            self.await_park = False
            self._start_training(t)

    # ------------------------------------------------------------------ verifier
    def on_verified(self, server: int, t: int) -> None:
        sid = self.vpool.complete(server)
        s = self.samples[sid]
        g = self.groups[s.gidx]
        s.ver_end = t
        if g.state != OUTSTANDING:
            return  # a verification of a dropped group runs to its end; result discarded
        s.state = VERIFIED
        g.n_verified += 1
        if g.n_verified == g.spec.n_samples:
            g.ready_at = t
            bisect.insort(self.ready, (t, g.gidx))

    def assign_verifiers(self, t: int, policy) -> list[tuple[int, int]]:
        if not self.vpool.queue or self.vpool.busy() == self.vpool.n:
            return []
        out = self.vpool.assign(t, lambda q: policy.pick_verification(self.view_for(t), q))
        for _, sid in out:
            s = self.samples[sid]
            s.state = VERIFYING
            s.ver_start = t
        return out

    # ------------------------------------------------------------------ trainer
    def next_internal_time(self) -> int | None:
        if self.tstate in ("sync", "switch") and self.t_until >= 0:
            return self.t_until
        return None

    def trainer_events(self, t: int) -> None:
        """Timers due at t: end of sync (publication) or end of the switch back to rollout;
        and, colocated, the start of training once every worker is parked."""
        self.t = t
        if self.tstate == "sync" and 0 <= self.t_until <= t:
            self._publish(t)
            self._set_trainer(t, "wait")
            self.t_until = -1
        elif self.tstate == "switch" and 0 <= self.t_until <= t and self.training < 0:
            self._set_trainer(t, "wait")
            self.t_until = -1
        if self.await_park and all(w.parked for w in self.alive_workers()):
            self.await_park = False
            self._start_training(t)

    def _publish(self, t: int) -> None:
        self.v_pub += 1
        self.steps[self.v_pub - 1].t_pub = t
        for w in self.alive_workers():
            self._push(w.wid, ("publish", self.v_pub))

    def _start_training(self, t: int) -> None:
        st = self.steps[self.s_next - 1]
        st.t_train_start = t
        self.training = st.s
        self._set_trainer(t, "train")
        self.out_train.append((st.s, st.tokens))

    def on_train_done(self, step: int, t: int) -> None:
        self.t = t
        if step != self.training:
            raise RunError(f"train_done for step {step}, but step {self.training} is training")
        self.steps[step].t_train_end = t
        self.training = -1
        if step == self.T - 1:
            self._end(t)
            return
        if self.sys.colocated:
            self._publish(t)
            for w in self.alive_workers():
                self._push(w.wid, ("switch_in",))
            self._set_trainer(t, "switch")
            self.t_until = t + self.sys.switch_ms
            if self.sys.switch_ms == 0:
                self._set_trainer(t, "wait")
                self.t_until = -1
        else:
            self._set_trainer(t, "sync")
            self.t_until = t + self.sys.sync_ms
            if self.sys.sync_ms == 0:
                self._publish(t)
                self._set_trainer(t, "wait")
                self.t_until = -1

    def try_select(self, t: int) -> bool:
        """Select a batch when the trainer is free and B eligible groups are ready."""
        self.t = t
        if self.ended or self.tstate != "wait":
            return False
        if self.single:
            if len(self.ready) == len(self.groups):
                for _, gi in self.ready:
                    self._consume(self.groups[gi], 0)
                self.steps.append(StepRec(0, t, [gi for _, gi in self.ready]))
                self.ready = []
                self._end(t)
                return True
            return False
        elig = []
        for ready_at, gi in self.ready:
            g = self.groups[gi]
            if self.s_next - g.version <= self.eta:
                elig.append((ready_at, gi))
                if len(elig) == self.B:
                    break
        if len(elig) < self.B:
            return False
        s = self.s_next
        st = StepRec(s, t, [gi for _, gi in elig])
        chosen = set(st.groups)
        self.ready = [x for x in self.ready if x[1] not in chosen]
        for gi in st.groups:
            g = self.groups[gi]
            self._consume(g, s)
            st.staleness.append(s - g.version)
            for sid in g.sids:
                smp = self.samples[sid]
                st.tokens += g.spec.prompt_tokens + smp.tokens
                for v, n in smp.segs:
                    if v < s:
                        st.stale_tokens += n
        self.steps.append(st)
        self.s_next = s + 1
        if self.sys.colocated:
            for gi in sorted(self.outstanding):
                self._drop(self.groups[gi], t, "colocated_switch")
            for w in self.alive_workers():
                self._push(w.wid, ("switch_out",))
            self._set_trainer(t, "switch")
            self.t_until = -1
            self.await_park = True
        else:
            self._start_training(t)
            self._dead_check(t, sorted(self.outstanding))
        return True

    def _consume(self, g: CGroup, s: int) -> None:
        g.state = CONSUMED
        g.step = s
        del self.outstanding[g.gidx]
        self.n_consumed += 1

    def _end(self, t: int) -> None:
        self.ended = True
        self.t_end = t
        self._set_trainer(t, self.tstate)

    # ------------------------------------------------------------------ staleness
    def _dead_check(self, t: int, gidxs) -> None:
        for gi in gidxs:
            g = self.groups[gi]
            if (
                g.state == OUTSTANDING
                and g.version is not None
                and self.s_next - g.version > self.eta
            ):
                self._drop(g, t, "stale")

    def after_commit(self, t: int) -> None:
        """Rule (1) for samples that started in this instant's commits (a group's version
        was set or lowered); the drops reach the workers at their next boundary."""
        if self._new_versions and not self.single:
            gs = sorted(set(self._new_versions))
            self._new_versions = []
            self._dead_check(t, gs)
        else:
            self._new_versions = []

    def _drop(self, g: CGroup, t: int, reason: str) -> None:
        if g.state != OUTSTANDING:
            return
        g.state = DROPPED
        g.drop_reason = reason
        g.drop_t = t
        del self.outstanding[g.gidx]
        self.n_dropped += 1
        self.wasted_tokens += g.tokens
        if g.ready_at >= 0:
            self.ready = [x for x in self.ready if x[1] != g.gidx]
        self.vpool.remove_group(g.gidx)
        for wid in sorted(g.workers):
            self._push(wid, ("drop", g.gidx))
        if self.log is not None:
            self.log.append((t, "drop", g.spec.gid, reason))

    # ------------------------------------------------------------------ policy actions
    def invoke(self, policy, t: int) -> None:
        self.t = t
        self.policy_calls += 1
        actions = policy.act(self.view_for(t))
        for a in actions or ():
            self.apply(a, t)

    def _bad(self, a, reason: str):
        raise InvalidAction(self.policy_name, a, reason)

    def apply(self, a, t: int) -> None:
        if isinstance(a, Drop):
            if not (0 <= a.gidx < len(self.groups)):
                self._bad(a, "unknown group")
            g = self.groups[a.gidx]
            if g.state != OUTSTANDING:
                self._bad(a, f"group is {g.state}, not outstanding")
            self._drop(g, t, "policy")
            return
        if not isinstance(a, Place):
            self._bad(a, "not an action")
        if not (0 <= a.worker < len(self.workers)):
            self._bad(a, "unknown worker")
        w = self.workers[a.worker]
        if not w.alive:
            self._bad(a, f"worker {w.wid} is lost")
        static = self.sys.engine.engine == "static"
        if a.kind == "group":
            if static:
                self._bad(a, "the static engine takes batches")
            if not (isinstance(a.ref, int) and 0 <= a.ref < len(self.groups)):
                self._bad(a, "unknown group")
            g = self.groups[a.ref]
            sids = [sid for sid in g.sids if self.samples[sid].state == UNPLACED]
            if not sids:
                self._bad(a, "group has no unplaced samples")
        elif a.kind == "sample":
            if static:
                self._bad(a, "the static engine takes batches")
            if not (isinstance(a.ref, int) and 0 <= a.ref < len(self.samples)):
                self._bad(a, "unknown sample")
            sids = [a.ref]
        elif a.kind == "batch":
            if not static:
                self._bad(a, "batches are for the static engine")
            sids = list(a.ref)
            if not sids or len(set(sids)) != len(sids):
                self._bad(a, "empty batch or repeated sample")
            if len(sids) > self.sys.engine.static_batch:
                self._bad(a, f"batch of {len(sids)} exceeds static_batch")
            if w.busy or w.paused or w.parked or not self.accepting(w):
                self._bad(a, f"static worker {w.wid} is not idle")
            if any(not (isinstance(x, int) and 0 <= x < len(self.samples)) for x in sids):
                self._bad(a, "unknown sample")
        else:
            self._bad(a, "unknown unit kind")
        launching = []
        for sid in sids:
            s = self.samples[sid]
            g = self.groups[s.gidx]
            if s.state != UNPLACED:
                self._bad(a, f"sample {sid} is already placed")
            if g.version is None and not self.gate_open():
                # rule (2): a sample of a group that has not started yet
                self._bad(a, "staleness gate closed (s_next - v_pub > eta)")
            if g.state == UNLAUNCHED:
                if g.gidx not in launching:
                    launching.append(g.gidx)
            elif g.state != OUTSTANDING:
                self._bad(a, f"group {g.spec.gid} is {g.state}")
        for gi in launching:
            g = self.groups[gi]
            g.state = OUTSTANDING
            g.launched_at = t
            self.unlaunched.remove(gi)
            self.outstanding[gi] = g
        for sid in sids:
            s = self.samples[sid]
            g = self.groups[s.gidx]
            s.state = WAITING
            s.worker = w.wid
            g.n_unplaced -= 1
            g.workers[w.wid] = g.workers.get(w.wid, 0) + 1
            w.sids[sid] = None
        if static:
            w.busy = True
            self.batches.append(sids)
            for sid in sids:
                self.samples[sid].batch = len(self.batches) - 1
        self._push(w.wid, ("place", sids))
        if self.log is not None:
            self.log.append(
                (
                    t,
                    "place",
                    tuple(
                        self.groups[self.samples[x].gidx].spec.gid + f"/{self.samples[x].sidx}"
                        for x in sids
                    ),
                    w.wid,
                )
            )

    # ------------------------------------------------------------------ live service
    def worker_lost(self, wid: int, t: int, max_attempts: int) -> list[int]:
        """A worker went silent: its waiting and running samples go back to the pool with
        an attempt counter; a group whose sample exceeded ``max_attempts`` is dropped with
        reason ``retries_exhausted``. Returns the samples put back."""
        self.t = t
        w = self.workers[wid]
        w.alive = False
        back = []
        for sid in list(w.sids):
            s = self.samples[sid]
            g = self.groups[s.gidx]
            s.state = UNPLACED
            s.worker = -1
            s.admitted = s.started = s.removed = False
            s.tokens = 0
            s.attempts += 1
            g.n_unplaced += 1
            c = g.workers.get(wid, 0) - 1
            if c > 0:
                g.workers[wid] = c
            else:
                g.workers.pop(wid, None)
            back.append(sid)
            if self.log is not None:
                self.log.append((t, "lost", g.spec.gid + f"/{s.sidx}", wid))
        w.sids.clear()
        w.resident.clear()
        w.admitted = w.waiting = w.kv_used = 0
        w.busy = w.paused = w.draining = w.parked = False
        for sid in back:
            g = self.groups[self.samples[sid].gidx]
            if g.state == OUTSTANDING and self.samples[sid].attempts > max_attempts:
                self._drop(g, t, "retries_exhausted")
        return back

    # ------------------------------------------------------------------ view
    def view_for(self, t: int):
        from rollout_engine.scheduler.view import View

        v = getattr(self, "_view", None)
        if v is None:
            v = self._view = View(self)
        v.t = t
        return v

    # ------------------------------------------------------------------ end of run
    def finalize(self, end_tokens: dict[int, int] | None = None) -> None:
        """Mark outstanding groups unfinished and add the tokens of samples in flight."""
        end_tokens = end_tokens or {}
        for g in list(self.outstanding.values()):
            g.state = UNFINISHED
            for sid in g.sids:
                s = self.samples[sid]
                if s.state in (WAITING, RUNNING) and not s.removed:
                    s.tokens = end_tokens.get(sid, 0)
                    g.tokens += s.tokens
        for g in self.groups:
            if g.state == DROPPED:
                for sid in g.sids:
                    s = self.samples[sid]
                    if s.state in (WAITING, RUNNING) and not s.removed and sid in end_tokens:
                        s.tokens = end_tokens[sid]
                        g.tokens += s.tokens
                        self.wasted_tokens += s.tokens
        self.outstanding = {}


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def nearest_rank(values: list, p: float):
    if not values:
        return math.nan
    xs = sorted(values)
    k = max(1, math.ceil(p * len(xs)))
    return xs[k - 1]
