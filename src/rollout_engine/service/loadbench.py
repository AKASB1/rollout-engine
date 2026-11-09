"""Wall-clock load benchmark of the live service.

    python -m rollout_engine.service.loadbench [--workers 2,4,8] [--steps 4] [--speed 200]
                                               [--out benchmarks/results/load] [--load-note TEXT]

For each worker count, start the controller on 127.0.0.1 (a free port), attach mock workers,
verifiers, and a trainer over real HTTP (one connection per request, as in the service), submit
the committed sample trace, and run until the last training step. Every request is timed on the
client side. The mocks run the simulated engine on a clock accelerated ``--speed`` times, so the
offered load is set by the worker count, the speed, and the trace. Writes ``wall_load.csv`` (one
row per worker count and endpoint, plus a session row) and ``manifest.json``.

These are wall-clock measurements: they depend on the machine and its load, are not simulated
results, and are not deterministic. Long-poll endpoints (marked in the CSV) include the time the
server holds the request open.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import tempfile
import time

from rollout_engine.bench import output
from rollout_engine.config import deep_merge, derive_system, parse_system
from rollout_engine.service.controller import Controller
from rollout_engine.service.http import HttpClient, serve
from rollout_engine.service.mock import MockTrainer, MockVerifier, MockWorker, submit_payload
from rollout_engine.service.vloop import ScaledClock
from rollout_engine.trace import schema

SAMPLE = os.path.join("configs", "traces", "sample.csv")
GROUPS_PER_STEP = 8
TEARDOWN_S = 5.0
# requests the controller holds open until something happens or wait_s passes; a worker
# heartbeat is a long poll only in phase "poll" (boundary and commit reports wait for the
# controller's scheduling round trip, which is what they measure)
LONG_POLL = {
    "POST /v1/trainer/next",
    "POST /v1/verifiers/{id}/poll",
    "POST /v1/workers/{id}/heartbeat [poll]",
}


def endpoint(method: str, path: str) -> str:
    """Path template of a request: ids become {id}."""
    parts = [p for p in path.split("?")[0].split("/") if p]
    if (
        len(parts) == 4
        and parts[0] == "v1"
        and parts[1] in ("workers", "samples", "verifiers", "verifications")
    ):
        parts[2] = "{id}"
    return f"{method} /" + "/".join(parts)


class TimedClient(HttpClient):
    """HttpClient that records (endpoint, status, seconds) for every request."""

    def __init__(self, host: str, port: int):
        super().__init__(host, port)
        self.records: list[tuple[str, int, float]] = []

    async def _req(self, method: str, path: str, body: dict | None):
        ep = endpoint(method, path)
        if ep.endswith("/heartbeat") and body:
            ep += f" [{body.get('phase', '-')}]"
        t0 = time.perf_counter()
        status, obj, headers = await super()._req(method, path, body)
        self.records.append((ep, status, time.perf_counter() - t0))
        return status, obj, headers


def load_system(workers: int, steps: int):
    raw = derive_system(
        total_gpus=2 * workers,
        rollout_gpus=workers,
        steps=steps,
        groups_per_step=GROUPS_PER_STEP,
        eta=1,
        warmup_steps=0,
        verifier_servers=max(4, workers),
    )
    return parse_system(deep_merge(raw, {"loop": {"warmup_steps": 0}}))


def quantile(xs: list[float], q: float) -> float:
    """Nearest-rank quantile of a sorted list."""
    return xs[max(0, math.ceil(q * len(xs)) - 1)]


async def session(
    workers: int, steps: int, speed: float, trace, max_seconds: float, store_dir: str
) -> tuple[list[dict], dict]:
    sysc = load_system(workers, steps)
    clock = ScaledClock(speed)
    store = os.path.join(
        tempfile.mkdtemp(prefix=f"session-{workers}-", dir=store_dir), "service.db"
    )
    ctl = Controller(
        sysc, {"name": "reference"}, clock=clock, store_path=store, heartbeat_timeout_s=30.0 * speed
    )
    server = None
    client = None
    ctl_task = None
    tasks: list[asyncio.Future] = []
    trainer = None
    failure = None
    t0 = time.perf_counter()
    try:
        server = await serve(ctl, "127.0.0.1", 0)
        client = TimedClient("127.0.0.1", server.sockets[0].getsockname()[1])
        ctl_task = asyncio.ensure_future(ctl.run())
        status, res = await client.post("/v1/groups", submit_payload(trace))
        if status != 202:
            raise RuntimeError(f"submission failed: {status} {res}")
        poll = min(max(5.0 * speed / 1000, 0.5), 10.0)  # well inside the 30 s heartbeat timeout
        mocks = [
            MockWorker(f"w{i}", client, sysc, trace, clock, poll_s=poll)
            for i in range(sysc.n_workers)
        ]
        mocks += [
            MockVerifier(f"v{i}", client, trace, clock, poll_s=1.0)
            for i in range(sysc.verifier_servers)
        ]
        trainer = MockTrainer(client, sysc, clock, poll_s=1.0)
        tasks = [asyncio.ensure_future(m.run()) for m in mocks + [trainer]]
        await asyncio.wait_for(asyncio.shield(ctl_task), timeout=max_seconds)
    except TimeoutError:
        failure = f"timeout after {max_seconds:g} s"
    except Exception as e:  # noqa: BLE001 - recorded in the manifest, the run goes on
        failure = f"{type(e).__name__}: {e}"
    finally:
        ctl.stop()
        for x in tasks:
            x.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if ctl_task is not None:
            try:  # the controller ends on stop(); cancel it if it does not
                await asyncio.wait_for(asyncio.shield(ctl_task), timeout=TEARDOWN_S)
            except BaseException:  # noqa: BLE001 - any outcome is recorded below
                ctl_task.cancel()
                await asyncio.gather(ctl_task, return_exceptions=True)
        if server is not None:
            server.close()
            try:  # handlers waiting on a future the controller never resolved must not block us
                await asyncio.wait_for(server.wait_closed(), timeout=TEARDOWN_S)
            except TimeoutError:
                failure = failure or "server did not close within the teardown time"
        ctl.store.close()
    records = client.records if client is not None else []
    batches = trainer.batches if trainer is not None else []
    wall = time.perf_counter() - t0
    ok = failure is None and ctl.core.ended and len(batches) == steps and ctl.error is None
    n_samples = {g.group_id: g.n_samples for g in trace.groups}
    groups = sum(len(b["groups"]) for b in batches)
    samples = sum(n_samples[g["group_id"]] for b in batches for g in b["groups"])
    key = {"workers": sysc.n_workers, "verifiers": sysc.verifier_servers}
    rows = []
    by: dict[str, list[tuple[int, float]]] = {}
    for ep, st, s in records:
        by.setdefault(ep, []).append((st, s))
    for ep in sorted(by):
        lat = sorted(s * 1000 for _, s in by[ep])
        rows.append(
            {
                **key,
                "endpoint": ep,
                "long_poll": ep in LONG_POLL,
                "wall_requests": len(lat),
                "wall_status_429": sum(st == 429 for st, _ in by[ep]),
                "wall_status_5xx": sum(st >= 500 for st, _ in by[ep]),
                "wall_p50_ms": quantile(lat, 0.5),
                "wall_p95_ms": quantile(lat, 0.95),
                "wall_p99_ms": quantile(lat, 0.99),
                "wall_max_ms": lat[-1],
            }
        )
    rows.append(
        {
            **key,
            "endpoint": "session",
            "long_poll": False,
            "wall_requests": len(records),
            "wall_status_429": sum(st == 429 for _, st, _ in records),
            "wall_status_5xx": sum(st >= 500 for _, st, _ in records),
            "wall_seconds": wall,
            "wall_requests_per_s": len(records) / wall,
            "wall_groups_per_s": groups / wall,
            "wall_samples_per_s": samples / wall,
        }
    )
    info = {
        **key,
        "steps": len(batches),
        "groups": groups,
        "samples": samples,
        "completed": ok,
        "failure": failure or (str(ctl.error) if ctl.error else None),
    }
    return rows, info


async def main_async(args, trace, argv: list[str]) -> int:
    git_state = output.git_commit()  # read before any result file is (re)written
    rows, sessions = [], []
    with tempfile.TemporaryDirectory(prefix="rollout-load-") as tmp:
        for w in [int(x) for x in args.workers.split(",")]:
            r, info = await session(w, args.steps, args.speed, trace, args.max_seconds, tmp)
            rows += r
            sessions.append(info)
            print(
                f"workers {info['workers']}: {info['steps']} steps, {info['groups']} groups, "
                f"completed={info['completed']}"
                + (f" ({info['failure']})" if info["failure"] else ""),
                flush=True,
            )
    output.write_csv(
        os.path.join(args.out, "wall_load.csv"),
        rows,
        ["workers", "verifiers", "endpoint", "long_poll"],
    )
    output.write_json(
        os.path.join(args.out, "manifest.json"),
        {
            **git_state,
            "command": "python -m rollout_engine.service.loadbench " + " ".join(argv),
            "trace": args.trace,
            "speed": args.speed,
            "steps": args.steps,
            "sessions": sessions,
            "software": output.software(),
            "hardware": output.hardware(),
            "load_note": args.load_note,
            "label": "wall-clock measurements of the live service with mock workers on an accelerated simulated clock; not simulated results, not deterministic",
            "notes": "the client, the mocks, and the controller share one event loop, so the timings include queueing behind the mocks' work; requests still in flight when a session ends are cancelled and not counted",
        },
    )
    print(f"wrote {args.out}", flush=True)
    return 0 if all(s["completed"] for s in sessions) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rollout_engine.service.loadbench", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument(
        "--workers", default="2,4,8", help="comma-separated rollout worker counts, one session each"
    )
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument(
        "--speed", type=float, default=200.0, help="clock acceleration (simulated ms per real ms)"
    )
    ap.add_argument("--trace", default=SAMPLE)
    ap.add_argument("--out", default=os.path.join("benchmarks", "results", "load"))
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--load-note", default="", help="machine load during the run, for the manifest")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    if args.steps < 2:
        print(
            "--steps must be at least 2 (one warm-up step is not allowed in the system config)",
            file=sys.stderr,
        )
        return 2
    trace, _ = schema.read(args.trace)
    if args.steps * GROUPS_PER_STEP > len(trace.groups):
        print(
            f"{args.trace} has {len(trace.groups)} groups; {args.steps} steps need "
            f"{args.steps * GROUPS_PER_STEP}",
            file=sys.stderr,
        )
        return 2
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(main_async(args, trace, argv))


if __name__ == "__main__":
    sys.exit(main())
