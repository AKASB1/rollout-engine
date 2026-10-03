"""The read-only view a policy sees at one decision instant (docs/contracts.md section 5).

Wrappers are created once per group, sample, and worker and only expose read-only
properties; nothing here reveals the true length of an unfinished sample or a verifier time
before the verification ends. Policies import nothing from the simulator.
"""

from __future__ import annotations


class GroupView:
    __slots__ = ("_g", "_core")

    def __init__(self, g, core):
        self._g = g
        self._core = core

    gidx = property(lambda self: self._g.gidx)
    gid = property(lambda self: self._g.spec.gid)
    task = property(lambda self: self._g.spec.task)
    prompt_tokens = property(lambda self: self._g.spec.prompt_tokens)
    max_tokens = property(lambda self: self._g.spec.max_tokens)
    n_samples = property(lambda self: self._g.spec.n_samples)
    est_tokens = property(lambda self: self._g.spec.est_tokens)
    state = property(lambda self: self._g.state)
    version = property(lambda self: self._g.version)
    n_unplaced = property(lambda self: self._g.n_unplaced)
    n_generated = property(lambda self: self._g.n_generated)
    n_verified = property(lambda self: self._g.n_verified)
    finished_sum = property(lambda self: self._g.fin_sum)  # observed lengths of finished samples
    finished_n = property(lambda self: self._g.fin_n)
    launched_at = property(lambda self: self._g.launched_at)
    ready_at = property(lambda self: self._g.ready_at)

    @property
    def sids(self) -> tuple[int, ...]:
        return tuple(self._g.sids)

    def unplaced(self) -> list[int]:
        smp = self._core.samples
        return [s for s in self._g.sids if smp[s].state == "unplaced"]

    def workers(self) -> tuple[int, ...]:
        return tuple(sorted(self._g.workers))


class SampleView:
    __slots__ = ("_s", "_core")

    def __init__(self, s, core):
        self._s = s
        self._core = core

    sid = property(lambda self: self._s.sid)
    gidx = property(lambda self: self._s.gidx)
    sample_idx = property(lambda self: self._s.sidx)
    attempts = property(lambda self: self._s.attempts)
    state = property(lambda self: self._s.state)
    worker = property(lambda self: self._s.worker)
    admitted = property(lambda self: self._s.admitted)
    started = property(lambda self: self._s.started)
    version = property(lambda self: self._s.version if self._s.started else None)
    started_at = property(lambda self: self._s.start_t if self._s.started else None)
    generated_at = property(lambda self: self._s.gen_t if self._s.gen_t >= 0 else None)

    @property
    def tokens_so_far(self) -> int:
        """Tokens generated so far (as of the worker's latest report)."""
        s = self._s
        if s.state in ("generated", "verifying", "verified") or s.removed:
            return s.tokens
        if s.started and self._core.sys.engine.engine == "continuous":
            return self._core.workers[s.worker].iters - s.start_iter
        return 0

    @property
    def length(self) -> int | None:
        """The observed length once the sample finished, else None."""
        s = self._s
        return s.tokens if s.state in ("generated", "verifying", "verified") else None


class WorkerView:
    __slots__ = ("_w", "_core")

    def __init__(self, w, core):
        self._w = w
        self._core = core

    wid = property(lambda self: self._w.wid)
    version = property(lambda self: self._w.version)
    running = property(lambda self: self._w.admitted)
    waiting = property(lambda self: self._w.waiting)
    kv_used = property(lambda self: self._w.kv_used)
    busy = property(lambda self: self._w.busy)
    alive = property(lambda self: self._w.alive)

    @property
    def state(self) -> str:
        w = self._w
        if w.paused or w.parked:
            return "paused"
        if w.draining:
            return "draining"
        if w.admitted or w.busy:
            return "running"
        return "idle"

    @property
    def free_slots(self) -> int:
        return max(0, self._core.sys.engine.max_seqs - self._w.admitted - self._w.waiting)

    @property
    def free_kv(self) -> int:
        return self._core.sys.engine.kv_tokens - self._w.kv_used

    @property
    def accepting(self) -> bool:
        return self._core.accepting(self._w)

    def resident(self, gidx: int) -> bool:
        return gidx in self._w.resident

    def sids(self) -> tuple[int, ...]:
        """Samples placed on this worker and not finished or removed (placement order)."""
        return tuple(self._w.sids)


class View:
    def __init__(self, core):
        self._c = core
        self.t = 0
        self._g = [GroupView(g, core) for g in core.groups]
        self._s = [SampleView(s, core) for s in core.samples]
        self._w = [WorkerView(w, core) for w in core.workers]

    def _sync(self) -> None:
        c = self._c
        if len(self._g) != len(c.groups):  # the live queue grew
            self._g += [GroupView(g, c) for g in c.groups[len(self._g) :]]
            self._s += [SampleView(s, c) for s in c.samples[len(self._s) :]]

    # trainer and loop
    s_next = property(lambda self: self._c.s_next)
    v_pub = property(lambda self: self._c.v_pub)
    B = property(lambda self: self._c.B)
    eta = property(lambda self: self._c.eta)
    single_phase = property(lambda self: self._c.single)
    trainer_state = property(lambda self: self._c.tstate)
    gate_open = property(lambda self: self._c.gate_open())
    engine = property(lambda self: self._c.sys.engine.engine)
    max_seqs = property(lambda self: self._c.sys.engine.max_seqs)
    kv_tokens = property(lambda self: self._c.sys.engine.kv_tokens)
    static_batch = property(lambda self: self._c.sys.engine.static_batch)
    n_workers = property(lambda self: len(self._c.workers))
    consumed = property(lambda self: self._c.n_consumed)
    dropped = property(lambda self: self._c.n_dropped)
    wasted_tokens = property(lambda self: self._c.wasted_tokens)
    colocated = property(lambda self: self._c.sys.colocated)

    @property
    def trainer_busy(self) -> bool:
        return self._c.tstate != "wait"

    def verify_mean_s(self, task: str) -> float:
        return self._c.sys.verify_mean_s.get(task, 0.0)

    # entities
    def workers(self) -> list[WorkerView]:
        return self._w

    def group(self, gidx: int) -> GroupView:
        return self._g[gidx]

    def sample(self, sid: int) -> SampleView:
        return self._s[sid]

    def window(self, k: int) -> list[GroupView]:
        """The next k unlaunched groups in stream order."""
        self._sync()
        return [self._g[i] for i in self._c.unlaunched[:k]]

    def n_unlaunched(self) -> int:
        return len(self._c.unlaunched)

    def outstanding(self) -> list[GroupView]:
        """Launched groups not yet consumed or dropped, in launch order."""
        self._sync()
        return [self._g[i] for i in self._c.outstanding]

    def n_outstanding(self) -> int:
        return len(self._c.outstanding)

    def selection_times(self) -> list[int]:
        """Selection times of the steps so far (ms)."""
        return [st.t_sel for st in self._c.steps]

    def ready(self) -> list[GroupView]:
        return [self._g[i] for _, i in self._c.ready]

    def verifier_queue(self) -> list[tuple[int, str, int]]:
        """(sample id, task, wait so far in ms) in arrival order."""
        return [(e.sid, e.task, self.t - e.arrived) for e in self._c.vpool.queue]
