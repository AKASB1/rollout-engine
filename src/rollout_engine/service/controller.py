"""The live rollout controller: the core state machine under an asyncio driver.

The controller runs ``scheduler.core.Core`` exactly as the simulator does; only the live layer
differs: a durable queue and store (sqlite3), the HTTP API (``handle``), worker registration,
liveness and retry of lost work, backpressure, verifier workers, and Prometheus metrics.

Instants. Every message is buffered; the driver waits until the loop has delivered all
messages of the current instant (``clock.settle``) and then applies them in the simulator's
order: worker boundary reports (by worker id) -> verifier results (by server) and assignments
-> trainer events -> selection and staleness rules -> the policy (once) -> inputs to the
workers -> worker commit reports (by worker id) -> the dead-group check. On a virtual-time
loop this reproduces the simulator exactly (check 9); on the wall clock it is the same logic
with real timing.
"""

from __future__ import annotations

import asyncio
import json
import math

from rollout_engine.api import GroupSpec, InvalidAction, RunError
from rollout_engine.config import SystemConfig
from rollout_engine.policies.composed import make_policy
from rollout_engine.scheduler.core import CONSUMED, DROPPED, OUTSTANDING, UNLAUNCHED, Core, StepRec
from rollout_engine.storage import Store
from rollout_engine.telemetry.prometheus import render

GROUP_FIELDS = ("group_id", "task", "prompt_tokens", "max_tokens", "n_samples", "est_tokens")


class ApiError(Exception):
    def __init__(self, status: int, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.headers = headers or {}


class _Worker:
    def __init__(self, wid: int, name: str):
        self.wid = wid
        self.name = name
        self.inbox: list[dict] = []
        self.commit_now = False
        self.poll: asyncio.Future | None = None
        self.last_seen = 0
        self.lost = False


class Controller:
    def __init__(
        self,
        system: SystemConfig,
        policy_cfg: dict,
        *,
        clock,
        store_path: str,
        max_queue_groups: int = 100_000,
        heartbeat_timeout_s: float = 10.0,
        max_attempts: int = 3,
        record_log: bool = True,
    ):
        self.sys = system
        self.clock = clock
        self.policy_cfg = policy_cfg
        self.policy = make_policy(policy_cfg)
        self.policy.bind(system, None)
        self.core = Core(system, [], self.policy.name, record_log=record_log)
        self.max_queue = max_queue_groups
        self.hb_timeout_ms = int(heartbeat_timeout_s * 1000)
        self.max_attempts = max_attempts
        self.store = Store(store_path)
        self.gid_index: dict[str, int] = {}
        self.workers: dict[str, _Worker] = {}
        self.by_wid: dict[int, _Worker] = {}
        self.boundary: dict[int, tuple] = {}  # wid -> (events, timeline, snapshot, future)
        self.commits: dict[int, tuple] = {}
        self.results: dict[int, dict] = {}  # sid -> posted result, until its report
        self.vnames: list[str] = []
        self.vjobs: dict[int, dict | None] = {}  # server -> job waiting to be fetched
        self.vpolls: dict[int, asyncio.Future] = {}
        self.vresults: list[int] = []  # servers whose verification ended
        self.train_ready: list[dict] = []
        self.train_polls: list[asyncio.Future] = []
        self.train_done: list[int] = []
        self.wake = asyncio.Event()
        self.error: BaseException | None = None
        self.stopped = False
        self.counters = {"retries": 0, "backpressure": 0, "tokens": 0, "trainer_wait_ms": 0}
        self._pending_lost: list[int] = []
        self._tw_mark = 0
        for w in self.core.workers:  # a worker slot takes work only once a worker registered
            w.alive = False
        if self.store.has_data():
            self._recover()
        else:
            self.store.set_meta("system", system.raw)
            self.store.set_meta("policy", policy_cfg)
            self.store.set_meta("clock_offset_ms", 0)

    # ================================================================ recovery
    def _recover(self) -> None:
        """Restart on an existing store: the queue comes back in order; consumed and dropped
        groups stay terminal; samples that were waiting or running go back to the pool as
        lost (attempt + 1); finished samples keep their results."""
        data = self.store.load()
        core = self.core
        specs = [
            GroupSpec(
                g["gid"],
                g["task"],
                g["prompt_tokens"],
                g["max_tokens"],
                g["n_samples"],
                g["est_tokens"],
            )
            for g in data["groups"]
        ]
        core.add_groups(specs)
        for g in data["groups"]:
            self.gid_index[g["gid"]] = g["gidx"]
        srows = {s["sid"]: s for s in data["samples"]}
        for grow in data["groups"]:
            g = core.groups[grow["gidx"]]
            st = grow["state"]
            if st == UNLAUNCHED:
                continue
            core.unlaunched.remove(g.gidx)
            g.state = st
            g.version = grow["version"]
            g.step = grow["step"] if grow["step"] is not None else -1
            g.drop_reason = grow["drop_reason"] or ""
            g.launched_at = 0
            for sid in g.sids:
                r = srows.get(sid)
                s = core.samples[sid]
                if r is None:
                    continue
                s.attempts = r["attempts"]
                if (
                    r["gen_ms"] is not None
                    and r["tokens"] is not None
                    and r["state"] not in ("waiting", "running")
                ):
                    s.state = r["state"]
                    s.tokens = r["tokens"]
                    s.gen_t = r["gen_ms"]
                    s.started = True
                    s.version = r["version"] if r["version"] is not None else -1
                    s.segs = [[s.version, s.tokens]]
                    g.n_unplaced -= 1
                    g.n_generated += 1
                    g.fin_sum += s.tokens
                    g.fin_n += 1
                    g.tokens += s.tokens
                    if r["ver_end_ms"] is not None:
                        s.ver_start, s.ver_end = r["ver_start_ms"], r["ver_end_ms"]
                        s.state = "verified"
                        g.n_verified += 1
                elif st == OUTSTANDING and r["state"] in ("waiting", "running"):
                    s.attempts += 1
                    self.counters["retries"] += 1
            if st == OUTSTANDING:
                core.outstanding[g.gidx] = g
                for sid in g.sids:
                    s = core.samples[sid]
                    if s.state == "generated":
                        core.vpool.enqueue(sid, g.gidx, g.spec.task, data["last_ms"])
                if g.n_verified == g.spec.n_samples:
                    g.ready_at = data["last_ms"]
                    core.ready.append((g.ready_at, g.gidx))
            elif st == CONSUMED:
                core.n_consumed += 1
            elif st == DROPPED:
                core.n_dropped += 1
        for srow in data["steps"]:
            st = StepRec(srow["step"], srow["t_sel_ms"], json.loads(srow["groups"]))
            st.staleness = json.loads(srow["staleness"])
            st.tokens = srow["tokens"]
            st.t_train_end = srow["t_train_end_ms"] if srow["t_train_end_ms"] is not None else -1
            st.t_pub = srow["t_pub_ms"] if srow["t_pub_ms"] is not None else -1
            core.steps.append(st)
        core.s_next = len(core.steps)
        core.v_pub = sum(1 for st in core.steps if st.t_pub >= 0)
        if core.steps and core.steps[-1].t_train_end < 0:
            last = core.steps[-1]
            core.training = last.s
            core._set_trainer(data["last_ms"], "train")
            self.train_ready.append(self._batch_payload(last))
        elif core.steps and core.steps[-1].t_pub < 0:
            core._set_trainer(data["last_ms"], "sync")
            core.t_until = data["last_ms"]
        for w in data["workers"]:
            self._add_worker(w["name"], w["wid"])
            self.by_wid[w["wid"]].lost = True
            core.workers[w["wid"]].alive = False
        self.recovered_at = data["last_ms"]

    # ================================================================ helpers
    def now(self) -> int:
        return self.clock.now_ms()

    def _add_worker(self, name: str, wid: int) -> _Worker:
        w = _Worker(wid, name)
        self.workers[name] = w
        self.by_wid[wid] = w
        return w

    def _input_payload(self, item: tuple) -> dict:
        op = item[0]
        if op == "place":
            out = []
            for sid in item[1]:
                s = self.core.samples[sid]
                g = self.core.groups[s.gidx]
                out.append(
                    {
                        "sid": sid,
                        "gidx": s.gidx,
                        "group_id": g.spec.gid,
                        "sample_idx": s.sidx,
                        "prompt_tokens": g.spec.prompt_tokens,
                        "max_tokens": g.spec.max_tokens,
                    }
                )
            return {"op": "place", "samples": out}
        if op == "drop":
            return {"op": "drop", "gidx": item[1], "group_id": self.core.groups[item[1]].spec.gid}
        if op == "publish":
            return {"op": "publish", "version": item[1]}
        return {"op": op}

    def _batch_payload(self, st) -> dict:
        core = self.core
        groups = []
        for gi, x in zip(st.groups, st.staleness, strict=True):
            g = core.groups[gi]
            groups.append(
                {
                    "group_id": g.spec.gid,
                    "version": g.version,
                    "staleness": x,
                    "samples": [
                        {"sample_idx": core.samples[s].sidx, "tokens": core.samples[s].tokens}
                        for s in g.sids
                    ],
                }
            )
        return {"step": st.s, "version": core.v_pub, "tokens": st.tokens, "groups": groups}

    def _ingest(self, wid: int, t: int, rep: tuple) -> None:
        events, timeline, snap, _ = rep
        core = self.core
        full = []
        for ev in events:
            kind = ev[0]
            if kind == "finish":
                r = self.results.pop(ev[1], None)
                if r is None:
                    raise RunError(
                        f"worker {wid}: finish of sample {ev[1]} without a posted result"
                    )
                full.append(("finish", ev[1], r["tokens"], r["segments"]))
                self.counters["tokens"] += r["tokens"]
            elif kind == "remove":
                full.append(("remove", ev[1], ev[2], ev[3]))
                self.counters["tokens"] += ev[2]
            elif kind == "start":
                full.append(("start", ev[1], ev[2], ev[3]))
            else:
                full.append(tuple(ev))
        core.on_report(wid, t, full, [tuple(x) for x in timeline], snap)
        for ev in full:
            s = core.samples[ev[1]]
            fr = None
            if ev[0] == "finish":
                fr = "length" if ev[2] >= core.groups[s.gidx].spec.max_tokens else "stop"
                self.store.attempt_end(s.sid, s.attempts, "finished", t)
            elif ev[0] == "remove":
                self.store.attempt_end(s.sid, s.attempts, "dropped", t)
            self.store.sample(s, fr)
            self.store.event(
                t,
                ev[0],
                {
                    "sid": ev[1],
                    "worker": wid,
                    **({"tokens": ev[2]} if ev[0] in ("finish", "remove") else {}),
                },
            )

    def _deliver(self, t: int, commit_now: bool, responding: dict[int, asyncio.Future]) -> None:
        """Route the core's worker inputs: to the response of a worker that reported at this
        instant, else to the worker's inbox (its long poll)."""
        out, self.core.out_worker = self.core.out_worker, []
        per: dict[int, list[dict]] = {}
        for wid, item in out:
            per.setdefault(wid, []).append(self._input_payload(item))
            if item[0] == "place":
                for sid in item[1]:
                    s = self.core.samples[sid]
                    self.store.attempt(sid, s.attempts, wid, t)
                    self.store.sample(s)
                    self.store.group(self.core.groups[s.gidx])
            if item[0] == "drop":
                self.store.group(self.core.groups[item[1]])
        for wid, fut in sorted(responding.items()):
            if not fut.done():
                fut.set_result({"inputs": per.pop(wid, []), "commit_now": commit_now})
        for wid, items in sorted(per.items()):
            w = self.by_wid.get(wid)
            if w is None or w.lost:
                raise RunError(f"inputs for worker {wid}, which is not registered or lost")
            w.inbox += items
            w.commit_now = commit_now
            if w.poll is not None and not w.poll.done():
                w.poll.set_result(None)

    def _flush_train(self, t: int) -> None:
        core = self.core
        while core.out_train:
            step, _tokens = core.out_train.pop(0)
            st = core.steps[step]
            self.store.step(st)
            self.store.event(t, "train_start", {"step": step})
            self.train_ready.append(self._batch_payload(st))
        while self.train_ready and self.train_polls:
            f = self.train_polls.pop(0)
            if not f.done():
                f.set_result(self.train_ready.pop(0))

    def _log_actions(self, mark: int, t: int) -> int:
        """Copy the core's placement/drop/lost log entries since ``mark`` into the event log."""
        log = self.core.log
        if log is None:
            return 0
        for e in log[mark:]:
            self.store.event(
                e[0], e[1], {"detail": [list(x) if isinstance(x, tuple) else x for x in e[2:]]}
            )
        return len(log)

    # ================================================================ the driver
    def _next_timer(self) -> int | None:
        cands = []
        nt = self.core.next_internal_time()
        if nt is not None:
            cands.append(nt)
        if self.hb_timeout_ms > 0:
            for w in self.workers.values():
                if not w.lost:
                    cands.append(w.last_seen + self.hb_timeout_ms + 1)
        return min(cands) if cands else None

    async def run(self) -> None:
        """Process instants until the last training step ends (or ``stop``)."""
        try:
            while not self.core.ended and not self.stopped:
                nt = self._next_timer()
                waiter = asyncio.ensure_future(self.wake.wait())
                timer = (
                    asyncio.ensure_future(self.clock.sleep_until(nt)) if nt is not None else None
                )
                await asyncio.wait(
                    [x for x in (waiter, timer) if x is not None],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for x in (waiter, timer):
                    if x is not None and not x.done():
                        x.cancel()
                self.wake.clear()
                await self.clock.settle()
                t = self.now()
                lost = self._check_liveness(t)
                internal = self.core.next_internal_time()
                if not (
                    self.boundary
                    or self.commits
                    or self.vresults
                    or self.train_done
                    or lost
                    or (internal is not None and internal <= t)
                    or self._kick
                ):
                    continue
                self._kick = False
                await self._instant(t)
        except BaseException as e:  # a policy bug or an invalid action stops the service
            self.error = e
            raise
        finally:
            self._finish()

    _kick = True  # process one instant at start-up (the policy launches work)

    def _check_liveness(self, t: int) -> bool:
        if self.hb_timeout_ms <= 0:
            return False
        any_lost = False
        for w in sorted(self.workers.values(), key=lambda x: x.wid):
            if not w.lost and t - w.last_seen > self.hb_timeout_ms:
                w.lost = True
                back = self.core.worker_lost(w.wid, t, self.max_attempts)
                self.counters["retries"] += len(back)
                self.store.event(t, "worker_lost", {"worker": w.name, "samples": back})
                self.store.worker(w.wid, w.name, {}, w.last_seen, "lost")
                for sid in back:
                    s = self.core.samples[sid]
                    self.store.attempt_end(sid, s.attempts - 1, "lost", t)
                    self.store.sample(s)
                any_lost = True
        return any_lost

    async def _instant(self, t: int) -> None:
        core = self.core
        core.t = t
        mark = len(core.log) if core.log is not None else 0
        # 1. worker boundary reports, by worker id
        reps = sorted(self.boundary.items())
        self.boundary = {}
        for wid, rep in reps:
            self._ingest(wid, t, rep)
        # 2-3. verifier results (by server) and assignments; zero-time verifications loop
        while True:
            done, self.vresults = sorted(self.vresults), []
            for sv in done:
                sid = core.vpool.serving[sv]
                core.on_verified(sv, t)
                s = core.samples[sid]
                self.store.sample(s)
                self.store.event(t, "verified", {"sid": sid, "server": sv})
                g = core.groups[s.gidx]
                if g.ready_at == t:
                    self.store.event(t, "ready", {"group": g.spec.gid})
            assigned = core.assign_verifiers(t, self.policy)
            for sv, sid in assigned:
                s = core.samples[sid]
                g = core.groups[s.gidx]
                self.vjobs[sv] = {
                    "sample_id": sid,
                    "group_id": g.spec.gid,
                    "sample_idx": s.sidx,
                    "task": g.spec.task,
                }
                f = self.vpolls.pop(sv, None)
                if f is not None and not f.done():
                    f.set_result(None)
            if not assigned:
                break
            await self.clock.settle()
            if not self.vresults:
                break
        # 4. trainer events
        for step in sorted(self.train_done):
            core.on_train_done(step, t)
            self.store.step(core.steps[step])
            self.store.event(t, "train_done", {"step": step})
        self.train_done = []
        if not core.ended:
            core.trainer_events(t)
            # 5-6. selection and the staleness rules
            if core.try_select(t):
                st = core.steps[-1]
                self.store.step(st)
                self.store.event(
                    t,
                    "select",
                    {"step": st.s, "groups": [core.groups[g].spec.gid for g in st.groups]},
                )
                for gi in st.groups:
                    self.store.group(core.groups[gi])
            for st in core.steps:
                if st.t_pub == t:
                    self.store.step(st)
                    self.store.event(t, "publish", {"version": st.s + 1})
            self._flush_train(t)
        if core.ended:
            self._log_actions(mark, t)
            self._release_all(reps)
            self.store.flush()
            return
        # 7. the policy, once
        try:
            core.invoke(self.policy, t)
        except InvalidAction:
            self._release_all(reps)
            raise
        mark = self._log_actions(mark, t)
        self._deliver(t, True, {wid: rep[3] for wid, rep in reps})
        await self.clock.settle()
        # 8. commit reports, by worker id; the dead-group check
        coms = sorted(self.commits.items())
        self.commits = {}
        for wid, rep in coms:
            self._ingest(wid, t, rep)
        core.after_commit(t)
        self._log_actions(mark, t)
        self._flush_train(t)
        self._deliver(t, False, {wid: rep[3] for wid, rep in coms})
        for gi in list(core.outstanding)[-8:]:
            self.store.group(core.groups[gi])
        self.store.flush()

    def _release_all(self, reps) -> None:
        for _, rep in reps:
            if not rep[3].done():
                rep[3].set_result({"inputs": [], "commit_now": False})

    def _finish(self) -> None:
        for f in self.train_polls:
            if not f.done():
                f.set_result({"done": True})
        for w in self.workers.values():
            if w.poll is not None and not w.poll.done():
                w.poll.set_result(None)
        for f in self.vpolls.values():
            if not f.done():
                f.set_result(None)
        for rep in list(self.boundary.values()) + list(self.commits.values()):
            if not rep[3].done():
                rep[3].set_result({"inputs": [], "commit_now": False, "done": True})
        try:
            self.store.flush()
        except Exception:  # noqa: BLE001 - shutting down
            pass

    def stop(self) -> None:
        self.stopped = True
        self.wake.set()

    # ================================================================ HTTP API
    async def handle(self, method: str, path: str, body: dict | None) -> tuple[int, object, dict]:
        """Route one API call; returns (status, JSON body or text, extra headers)."""
        try:
            parts = [p for p in path.split("?")[0].split("/") if p]
            if method == "GET" and parts == ["metrics"]:
                return 200, self.metrics_text(), {"Content-Type": "text/plain; version=0.0.4"}
            if method == "GET" and parts == ["healthz"]:
                return 200, {"ok": self.error is None, "ended": self.core.ended}, {}
            if method == "GET" and parts == ["v1", "status"]:
                return 200, self.status(), {}
            if method != "POST":
                raise ApiError(405, "method not allowed")
            body = body or {}
            if parts == ["v1", "groups"]:
                return await self.post_groups(body)
            if parts == ["v1", "workers"]:
                return await self.post_worker(body)
            if len(parts) == 4 and parts[:2] == ["v1", "workers"] and parts[3] == "heartbeat":
                return await self.post_heartbeat(parts[2], body)
            if len(parts) == 4 and parts[:2] == ["v1", "samples"] and parts[3] == "result":
                return await self.post_result(parts[2], body)
            if parts == ["v1", "verifiers"]:
                return await self.post_verifier(body)
            if len(parts) == 4 and parts[:2] == ["v1", "verifiers"] and parts[3] == "poll":
                return await self.post_verifier_poll(parts[2], body)
            if len(parts) == 4 and parts[:2] == ["v1", "verifications"] and parts[3] == "result":
                return await self.post_verification(parts[2], body)
            if parts == ["v1", "trainer", "next"]:
                return await self.post_trainer_next(body)
            if parts == ["v1", "trainer", "done"]:
                return await self.post_trainer_done(body)
            raise ApiError(404, f"no route {method} {path}")
        except ApiError as e:
            return e.status, {"error": e.message}, e.headers

    async def post_groups(self, body: dict):
        groups = body.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ApiError(400, "field 'groups' must be a non-empty list")
        specs, seen = [], set()
        for k, g in enumerate(groups):
            if not isinstance(g, dict):
                raise ApiError(400, f"groups[{k}] must be an object")
            for f in GROUP_FIELDS:
                if f not in g:
                    raise ApiError(400, f"groups[{k}].{f} is required")
            gid = g["group_id"]
            if not isinstance(gid, str) or not gid or "," in gid:
                raise ApiError(
                    400, f"groups[{k}].group_id must be a non-empty string without commas"
                )
            if not isinstance(g["task"], str) or not g["task"]:
                raise ApiError(400, f"groups[{k}].task must be a non-empty string")
            for f in ("prompt_tokens", "max_tokens", "n_samples"):
                if not isinstance(g[f], int) or isinstance(g[f], bool) or g[f] < 1:
                    raise ApiError(400, f"groups[{k}].{f} must be an integer >= 1")
            est = g["est_tokens"]
            if (
                not isinstance(est, (int, float))
                or isinstance(est, bool)
                or not (est > 0 and math.isfinite(est))
            ):
                raise ApiError(400, f"groups[{k}].est_tokens must be a number > 0")
            if gid in self.gid_index or gid in seen:
                raise ApiError(409, f"duplicate group_id {gid!r}")
            seen.add(gid)
            specs.append(
                GroupSpec(
                    gid, g["task"], g["prompt_tokens"], g["max_tokens"], g["n_samples"], float(est)
                )
            )
        depth = len(self.core.unlaunched)
        if depth + len(specs) > self.max_queue:
            self.counters["backpressure"] += 1
            raise ApiError(
                429, f"queue holds {depth} groups (max {self.max_queue})", {"Retry-After": "1"}
            )
        t = self.now()
        base = len(self.core.groups)
        self.core.add_groups(specs)
        for i, sp in enumerate(specs):
            self.gid_index[sp.gid] = base + i
            self.store.group(self.core.groups[base + i], submitted_ms=t)
            for sid in self.core.groups[base + i].sids:
                self.store.sample(self.core.samples[sid])
        self.store.event(t, "submit", {"groups": [s.gid for s in specs]})
        self.store.flush()
        self._kick = True
        self.wake.set()
        return 202, {"accepted": len(specs), "queue_depth": len(self.core.unlaunched)}, {}

    async def post_worker(self, body: dict):
        name = body.get("worker_id")
        if not isinstance(name, str) or not name:
            raise ApiError(400, "field 'worker_id' is required")
        for f in ("max_seqs", "kv_tokens", "tp"):
            if not isinstance(body.get(f), int) or body[f] < 1:
                raise ApiError(400, f"field '{f}' must be an integer >= 1")
        t = self.now()
        if name in self.workers:
            w = self.workers[name]
            if not w.lost:
                raise ApiError(409, f"worker {name!r} is already registered")
            w.lost = False
            self.core.workers[w.wid].alive = True
            w.inbox = [{"op": "publish", "version": self.core.v_pub}] if self.core.v_pub else []
        else:
            if len(self.workers) >= len(self.core.workers):
                raise ApiError(409, f"all {len(self.core.workers)} worker slots are taken")
            w = self._add_worker(name, len(self.workers))
            self.core.workers[w.wid].alive = True
            w.inbox = [{"op": "publish", "version": self.core.v_pub}] if self.core.v_pub else []
        w.last_seen = t
        w.commit_now = True
        self.store.worker(w.wid, name, body, t, "registered")
        self.store.event(t, "register", {"worker": name, "wid": w.wid})
        self.store.flush()
        self._kick = True
        self.wake.set()
        p = self.sys.engine
        return (
            200,
            {
                "wid": w.wid,
                "engine": p.engine,
                "max_seqs": p.max_seqs,
                "kv_tokens": p.kv_tokens,
                "version": self.core.v_pub,
            },
            {},
        )

    def _worker(self, name: str) -> _Worker:
        w = self.workers.get(name)
        if w is None:
            raise ApiError(404, f"unknown worker {name!r}")
        if w.lost:
            raise ApiError(410, f"worker {name!r} was declared lost; register again")
        return w

    async def post_heartbeat(self, name: str, body: dict):
        """phase=poll: liveness plus a long poll for inputs; phase=boundary|commit: a report
        (events, timeline, snapshot) answered with the inputs of that instant."""
        w = self._worker(name)
        t = self.now()
        w.last_seen = t
        phase = body.get("phase", "poll")
        if phase == "poll":
            if not w.inbox:
                w.poll = asyncio.get_running_loop().create_future()
                wait_s = float(body.get("wait_s", 0))
                try:
                    if wait_s > 0:
                        await asyncio.wait_for(asyncio.shield(w.poll), timeout=wait_s)
                except TimeoutError:
                    pass
                w.poll = None
                if not w.lost:
                    w.last_seen = max(w.last_seen, self.now())
            items, w.inbox = w.inbox, []
            return (
                200,
                {
                    "inputs": items,
                    "commit_now": w.commit_now if items else False,
                    "ended": self.core.ended,
                },
                {},
            )
        if phase not in ("boundary", "commit"):
            raise ApiError(400, "field 'phase' must be poll, boundary, or commit")
        for f in ("events", "snapshot"):
            if f not in body:
                raise ApiError(400, f"field '{f}' is required")
        fut = asyncio.get_running_loop().create_future()
        rep = ([tuple(e) for e in body["events"]], body.get("timeline", []), body["snapshot"], fut)
        if self.core.ended:
            return 200, {"inputs": [], "commit_now": False, "ended": True}, {}
        (self.boundary if phase == "boundary" else self.commits)[w.wid] = rep
        self.wake.set()
        res = await fut
        return 200, {**res, "ended": self.core.ended}, {}

    async def post_result(self, sid_s: str, body: dict):
        try:
            sid = int(sid_s)
        except ValueError:
            raise ApiError(400, "sample id must be an integer") from None
        if not (0 <= sid < len(self.core.samples)):
            raise ApiError(404, f"unknown sample {sid}")
        w = self._worker(body.get("worker_id", ""))
        tokens = body.get("tokens")
        if not isinstance(tokens, int) or tokens < 1:
            raise ApiError(400, "field 'tokens' must be an integer >= 1")
        if body.get("finish_reason") not in ("stop", "length"):
            raise ApiError(400, "field 'finish_reason' must be stop or length")
        w.last_seen = self.now()
        self.results[sid] = {
            "tokens": tokens,
            "segments": body.get("segments") or [[0, tokens]],
            "finish_reason": body["finish_reason"],
        }
        return 200, {"ok": True}, {}

    async def post_verifier(self, body: dict):
        name = body.get("verifier_id")
        if not isinstance(name, str) or not name:
            raise ApiError(400, "field 'verifier_id' is required")
        if name in self.vnames:
            return 200, {"server": self.vnames.index(name)}, {}
        if len(self.vnames) >= self.core.vpool.n:
            raise ApiError(409, f"all {self.core.vpool.n} verifier slots are taken")
        self.vnames.append(name)
        return 200, {"server": len(self.vnames) - 1}, {}

    async def post_verifier_poll(self, sv_s: str, body: dict):
        sv = int(sv_s)
        if not (0 <= sv < len(self.vnames)):
            raise ApiError(404, f"unknown verifier {sv}")
        job = self.vjobs.pop(sv, None)
        if job is None:
            f = asyncio.get_running_loop().create_future()
            self.vpolls[sv] = f
            try:
                await asyncio.wait_for(
                    asyncio.shield(f), timeout=float(body.get("wait_s", 0)) or None
                )
            except TimeoutError:
                pass
            self.vpolls.pop(sv, None)
            job = self.vjobs.pop(sv, None)
        if job is None:
            return 204, {"ended": self.core.ended}, {}
        return 200, job, {}

    async def post_verification(self, sid_s: str, body: dict):
        sid = int(sid_s)
        sv = body.get("server")
        if (
            not isinstance(sv, int)
            or not (0 <= sv < self.core.vpool.n)
            or self.core.vpool.serving[sv] != sid
        ):
            raise ApiError(409, f"sample {sid} is not in service on server {sv}")
        self.vresults.append(sv)
        self.wake.set()
        return 200, {"ok": True}, {}

    async def post_trainer_next(self, body: dict):
        if self.core.ended:
            return 200, {"done": True}, {}
        if not self.train_ready:
            f = asyncio.get_running_loop().create_future()
            self.train_polls.append(f)
            try:
                res = await asyncio.wait_for(
                    asyncio.shield(f), timeout=float(body.get("wait_s", 0)) or None
                )
            except TimeoutError:
                if f in self.train_polls:
                    self.train_polls.remove(f)
                return 204, {}, {}
            return 200, res, {}
        return 200, self.train_ready.pop(0), {}

    async def post_trainer_done(self, body: dict):
        step = body.get("step")
        if not isinstance(step, int) or step != self.core.training:
            raise ApiError(409, f"step {step} is not training (training: {self.core.training})")
        self.train_done.append(step)
        self.wake.set()
        return 200, {"ok": True}, {}

    # ================================================================ observability
    def status(self) -> dict:
        core = self.core
        return {
            "t_ms": self.now(),
            "policy": self.policy.name,
            "queue_depth": len(core.unlaunched),
            "outstanding": len(core.outstanding),
            "ready": len(core.ready),
            "consumed": core.n_consumed,
            "dropped": core.n_dropped,
            "s_next": core.s_next,
            "v_pub": core.v_pub,
            "trainer": core.tstate,
            "ended": core.ended,
            "workers": {w.name: {"wid": w.wid, "lost": w.lost} for w in self.workers.values()},
            "error": repr(self.error) if self.error else None,
        }

    def metrics_text(self) -> str:
        core = self.core
        drops: dict[str, int] = {}
        for g in core.groups:
            if g.state == DROPPED:
                drops[g.drop_reason] = drops.get(g.drop_reason, 0) + 1
        steps = core.steps
        last_step = (steps[-1].t_sel - steps[-2].t_sel) / 1000 if len(steps) >= 2 else 0.0
        stal = steps[-1].staleness if steps else []
        from rollout_engine.telemetry.metrics import integrate

        tw = integrate(core.trainer_tl, 0, max(self.now(), core.trainer_tl[-1][0])).get("wait", 0)
        m = [
            (
                "rollout_queue_depth",
                "gauge",
                "Groups queued and not launched.",
                [({}, len(core.unlaunched))],
            ),
            (
                "rollout_outstanding_groups",
                "gauge",
                "Launched groups not consumed or dropped.",
                [({}, len(core.outstanding))],
            ),
            (
                "rollout_running_samples",
                "gauge",
                "Samples admitted on workers.",
                [({}, sum(w.admitted for w in core.workers))],
            ),
            (
                "rollout_healthy_workers",
                "gauge",
                "Registered workers not declared lost.",
                [({}, sum(1 for w in self.workers.values() if not w.lost))],
            ),
            (
                "rollout_tokens_generated_total",
                "counter",
                "Tokens of finished or removed samples.",
                [({}, self.counters["tokens"])],
            ),
            (
                "rollout_trainer_wait_seconds_total",
                "counter",
                "Time the trainer waited for groups.",
                [({}, tw / 1000)],
            ),
            ("rollout_steps_total", "counter", "Training steps selected.", [({}, len(steps))]),
            (
                "rollout_step_seconds",
                "gauge",
                "Duration of the last completed step interval.",
                [({}, last_step)],
            ),
            (
                "rollout_staleness",
                "gauge",
                "Mean staleness of the last selected batch.",
                [({}, sum(stal) / len(stal) if stal else 0.0)],
            ),
            (
                "rollout_verifier_queue_depth",
                "gauge",
                "Samples waiting for a verifier.",
                [({}, len(core.vpool.queue))],
            ),
            (
                "rollout_drops_total",
                "counter",
                "Groups dropped, by reason.",
                [
                    ({"reason": r}, drops.get(r, 0))
                    for r in ("policy", "stale", "colocated_switch", "retries_exhausted")
                ],
            ),
            (
                "rollout_retries_total",
                "counter",
                "Samples returned to the pool by a lost worker.",
                [({}, self.counters["retries"])],
            ),
            (
                "rollout_backpressure_rejections_total",
                "counter",
                "Submissions rejected with 429.",
                [({}, self.counters["backpressure"])],
            ),
        ]
        return render(m)
