"""Mock workers, verifiers, and trainer for the live service.

They stand in for a model, a sandbox, and a trainer: a mock worker runs the rollout engine of
docs/simulator.md and reads the hidden response lengths from the trace; a mock verifier reads
the hidden verifier times; the mock trainer trains for the simulated training time. They make
the same API calls a real component would; the API never carries the hidden values.
"""

from __future__ import annotations

import asyncio

from rollout_engine.workers.engine import ESample, make_engine


class InProcessClient:
    """Calls the controller's API handler directly (no sockets)."""

    def __init__(self, controller):
        self.c = controller

    async def post(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        status, obj, _ = await self.c.handle("POST", path, body or {})
        return status, obj

    async def get(self, path: str):
        status, obj, _ = await self.c.handle("GET", path, None)
        return status, obj


def _decode(item: dict, lengths: dict) -> tuple:
    op = item["op"]
    if op == "place":
        es = [
            ESample(
                s["sid"],
                s["gidx"],
                s["sample_idx"],
                s["prompt_tokens"],
                s["max_tokens"],
                lengths[(s["group_id"], s["sample_idx"])],
            )
            for s in item["samples"]
        ]
        return ("place", es)
    if op == "drop":
        return ("drop", item["gidx"])
    if op == "publish":
        return ("publish", item["version"])
    return (op,)


class MockWorker:
    def __init__(self, name: str, client, system, trace, clock, poll_s: float = 5.0):
        self.name = name
        self.api = client
        self.sys = system
        self.clock = clock
        self.poll_s = poll_s
        self.lengths = {
            (g.group_id, i): r for g in trace.groups for i, r in enumerate(g.resp_tokens)
        }
        self.engine = None
        self.wid = None
        self._mt: dict[int, int] = {}
        self._last_t = 0

    def _max_tokens(self, sid: int) -> int:
        return self._mt.get(sid, 1 << 62)

    def _snap(self):
        return self.engine.snapshot()

    async def _report(self, phase: str, t: int) -> dict:
        ev, tl = self.engine.report()
        wire = []
        for e in ev:
            if e[0] == "finish":
                L = e[2]
                status, _ = await self.api.post(
                    f"/v1/samples/{e[1]}/result",
                    {
                        "worker_id": self.name,
                        "tokens": L,
                        "finish_reason": "length" if L >= self._max_tokens(e[1]) else "stop",
                        "segments": [list(x) for x in e[3]],
                    },
                )
                if status != 200:
                    raise RuntimeError(f"result rejected: {status}")
                wire.append(["finish", e[1]])
            elif e[0] in ("remove",):
                wire.append(["remove", e[1], e[2], [list(x) for x in e[3]]])
            else:
                wire.append(list(e))
        status, res = await self.api.post(
            f"/v1/workers/{self.name}/heartbeat",
            {
                "phase": phase,
                "t": t,
                "events": wire,
                "timeline": [list(x) for x in tl],
                "snapshot": self._snap(),
            },
        )
        if status != 200:
            raise RuntimeError(f"heartbeat rejected: {status} {res}")
        return res

    async def _apply(self, res: dict, t: int) -> None:
        for item in res.get("inputs", []):
            if item["op"] == "place":
                for smp in item["samples"]:
                    self._mt[smp["sid"]] = smp["max_tokens"]
            self.engine.push(t, _decode(item, self.lengths))

    async def _commit_if_due(self, t: int) -> None:
        if self.engine.wants_commit(t):
            self.engine.commit(t)
            res = await self._report("commit", t)
            await self._apply(res, t)

    async def run(self) -> None:
        p = self.sys.engine
        status, reg = await self.api.post(
            "/v1/workers",
            {"worker_id": self.name, "max_seqs": p.max_seqs, "kv_tokens": p.kv_tokens, "tp": p.tp},
        )
        if status != 200:
            raise RuntimeError(f"registration failed: {status} {reg}")
        self.wid = reg["wid"]
        self.engine = make_engine(
            self.wid,
            p,
            inflight=self.sys.inflight,
            swap_ms=self.sys.swap_ms,
            switch_ms=self.sys.switch_ms,
        )
        poll = None
        try:
            while True:
                if poll is None:
                    poll = asyncio.ensure_future(
                        self.api.post(
                            f"/v1/workers/{self.name}/heartbeat",
                            {"phase": "poll", "wait_s": self.poll_s},
                        )
                    )
                nt = self.engine.next_time()
                sleeper = (
                    asyncio.ensure_future(self.clock.sleep_until(nt)) if nt is not None else None
                )
                done, _ = await asyncio.wait(
                    [x for x in (poll, sleeper) if x is not None],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if sleeper is not None and sleeper not in done:
                    sleeper.cancel()
                # the engine acts at its own scheduled instants (on a wall clock "now" may
                # already be later); inputs are stamped with the current time
                t = max(self.clock.now_ms(), self._last_t)
                if poll in done:
                    status, res = poll.result()
                    poll = None
                    nt2 = self.engine.next_time()
                    if nt2 is not None and nt2 < t:
                        t = nt2  # an overdue boundary is processed first, at its own time
                        self.engine.boundary(t)
                        res0 = await self._report("boundary", t)
                        await self._apply(res0, t)
                        await self._commit_if_due(t)
                    if status != 200 or res.get("ended"):
                        if status == 410 or res.get("ended"):
                            return
                    await self._apply(res, t)
                    if res.get("commit_now"):
                        await self._commit_if_due(t)
                    continue
                t = nt
                if self.engine.next_time() == t:
                    self._last_t = t
                    self.engine.boundary(t)
                    res = await self._report("boundary", t)
                    if res.get("ended"):
                        return
                    await self._apply(res, t)
                    await self._commit_if_due(t)
        finally:
            if poll is not None and not poll.done():
                poll.cancel()


class MockVerifier:
    def __init__(self, name: str, client, trace, clock, poll_s: float = 30.0):
        self.name = name
        self.api = client
        self.clock = clock
        self.poll_s = poll_s
        self.vms = {(g.group_id, i): v for g in trace.groups for i, v in enumerate(g.verify_ms)}

    async def run(self) -> None:
        status, reg = await self.api.post("/v1/verifiers", {"verifier_id": self.name})
        if status != 200:
            raise RuntimeError(f"verifier registration failed: {status} {reg}")
        sv = reg["server"]
        while True:
            status, job = await self.api.post(f"/v1/verifiers/{sv}/poll", {"wait_s": self.poll_s})
            if status == 204:
                if job.get("ended"):
                    return
                continue
            d = self.vms[(job["group_id"], job["sample_idx"])]
            await self.clock.sleep_until(self.clock.now_ms() + d)
            await self.api.post(f"/v1/verifications/{job['sample_id']}/result", {"server": sv})


class MockTrainer:
    """Polls for batches and trains for the simulated time (fixed_ms + ns_per_token * tokens)."""

    def __init__(self, client, system, clock, poll_s: float = 30.0):
        self.api = client
        self.sys = system
        self.clock = clock
        self.poll_s = poll_s
        self.batches: list[dict] = []

    async def run(self) -> None:
        while True:
            status, b = await self.api.post("/v1/trainer/next", {"wait_s": self.poll_s})
            if status == 204:
                continue
            if b.get("done"):
                return
            self.batches.append(b)
            await self.clock.sleep_until(self.clock.now_ms() + self.sys.train_ms(b["tokens"]))
            await self.api.post("/v1/trainer/done", {"step": b["step"]})


def submit_payload(trace) -> dict:
    return {
        "groups": [
            {
                "group_id": g.group_id,
                "task": g.task,
                "prompt_tokens": g.prompt_tokens,
                "max_tokens": g.max_tokens,
                "n_samples": g.n_samples,
                "est_tokens": g.est_tokens,
            }
            for g in trace.groups
        ]
    }


async def run_session(
    controller,
    client,
    system,
    trace,
    clock,
    *,
    submit: bool = True,
    crash: dict | None = None,
    until_step: int | None = None,
):
    """Run a live session: submit the trace, start the mocks, run until the last training step
    ends (or ``until_step`` is selected). ``crash`` = {"worker": name, "at_ms": t} stops a worker."""
    if submit:
        status, res = await client.post("/v1/groups", submit_payload(trace))
        if status != 202:
            raise RuntimeError(f"submission failed: {status} {res}")
    workers = [MockWorker(f"w{i}", client, system, trace, clock) for i in range(system.n_workers)]
    verifiers = [
        MockVerifier(f"v{i}", client, trace, clock) for i in range(system.verifier_servers)
    ]
    trainer = MockTrainer(client, system, clock)
    tasks = [asyncio.ensure_future(m.run()) for m in workers + verifiers]
    tasks.append(asyncio.ensure_future(trainer.run()))
    ctl = asyncio.ensure_future(controller.run())
    extra = []
    if crash:
        victim = next(w for w, m in zip(tasks, workers, strict=False) if m.name == crash["worker"])

        async def kill():
            await clock.sleep_until(crash["at_ms"])
            victim.cancel()

        extra.append(asyncio.ensure_future(kill()))
    if until_step is not None:

        async def watch():
            while len(controller.core.steps) <= until_step and not controller.core.ended:
                await clock.sleep_until(clock.now_ms() + 50)
            controller.stop()

        extra.append(asyncio.ensure_future(watch()))
    try:
        await ctl
    finally:
        for x in tasks + extra:
            x.cancel()
        await asyncio.gather(*tasks, *extra, return_exceptions=True)
    return workers, verifiers, trainer
