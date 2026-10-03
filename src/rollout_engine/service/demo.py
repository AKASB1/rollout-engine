"""Demo: start the live rollout service on 127.0.0.1, attach mock workers, verifiers, and a
mock trainer over HTTP, submit the committed sample trace, and consume training batches.

    python -m rollout_engine.service.demo [--port 18300] [--speed 200] [--steps 3]

The mock components run the simulated engine on an accelerated wall clock (``--speed`` times
real time), so a few minutes of simulated rollout take seconds. With ``--serve`` only the
controller runs (for external workers) until ``--max-seconds`` pass. Everything is simulated.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile

from rollout_engine.config import deep_merge, derive_system, parse_system
from rollout_engine.service.controller import Controller
from rollout_engine.service.http import HttpClient, serve
from rollout_engine.service.mock import MockTrainer, MockVerifier, MockWorker, submit_payload
from rollout_engine.service.vloop import ScaledClock
from rollout_engine.trace import schema

SAMPLE = os.path.join("configs", "traces", "sample.csv")


def demo_system(steps: int) -> object:
    raw = derive_system(
        total_gpus=4,
        rollout_gpus=2,
        steps=steps,
        groups_per_step=4,
        eta=1,
        warmup_steps=0,
        verifier_servers=4,
    )
    return parse_system(deep_merge(raw, {"loop": {"warmup_steps": 0}}))


async def main_async(args) -> int:
    trace, _ = schema.read(args.trace)
    sysc = demo_system(args.steps)
    clock = ScaledClock(args.speed)
    store = args.store or os.path.join(tempfile.mkdtemp(prefix="rollout-demo-"), "service.db")
    ctl = Controller(
        sysc,
        {"name": args.policy},
        clock=clock,
        store_path=store,
        heartbeat_timeout_s=args.heartbeat_timeout * args.speed,
    )
    server = await serve(ctl, "127.0.0.1", args.port)
    port = server.sockets[0].getsockname()[1]
    print(
        f"service on http://127.0.0.1:{port} (store {store}); clock x{args.speed:g}; simulated, assumed parameters",
        flush=True,
    )
    ctl_task = asyncio.ensure_future(ctl.run())
    if args.serve:
        try:
            await asyncio.wait_for(ctl_task, timeout=args.max_seconds)
        except TimeoutError:
            ctl.stop()
        server.close()
        return 0
    client = HttpClient("127.0.0.1", port)
    status, res = await client.post("/v1/groups", submit_payload(trace))
    print(f"POST /v1/groups -> {status} {res}", flush=True)
    poll = 5.0 * args.speed / 1000  # mock poll intervals in real seconds
    workers = [
        MockWorker(f"w{i}", client, sysc, trace, clock, poll_s=max(poll, 0.5))
        for i in range(sysc.n_workers)
    ]
    verifiers = [
        MockVerifier(f"v{i}", client, trace, clock, poll_s=1.0)
        for i in range(sysc.verifier_servers)
    ]
    trainer = MockTrainer(client, sysc, clock, poll_s=1.0)
    tasks = [asyncio.ensure_future(m.run()) for m in workers + verifiers + [trainer]]
    try:
        await asyncio.wait_for(ctl_task, timeout=args.max_seconds)
    finally:
        for x in tasks:
            x.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    for b in trainer.batches:
        stal = [g["staleness"] for g in b["groups"]]
        print(
            f"  trainer got step {b['step']}: {len(b['groups'])} groups, {b['tokens']} tokens, staleness {stal}",
            flush=True,
        )
    status, text = await client.get("/metrics")
    keep = [
        ln
        for ln in text.splitlines()
        if ln.startswith(
            (
                "rollout_steps_total",
                "rollout_tokens_generated_total",
                "rollout_healthy_workers",
                "rollout_trainer_wait_seconds_total",
            )
        )
    ]
    print("metrics: " + "; ".join(keep), flush=True)
    server.close()
    await server.wait_closed()
    ok = ctl.core.ended and len(trainer.batches) == args.steps and ctl.error is None
    print("demo " + ("finished" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rollout_engine.service.demo", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--port", type=int, default=int(os.environ.get("ROLLOUT_PORT", "18300")))
    ap.add_argument(
        "--speed", type=float, default=200.0, help="clock acceleration (simulated ms per real ms)"
    )
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--policy", default="reference")
    ap.add_argument("--trace", default=SAMPLE)
    ap.add_argument("--store", default="")
    ap.add_argument("--heartbeat-timeout", type=float, default=30.0, help="real seconds")
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--serve", action="store_true", help="run only the controller (no mocks)")
    args = ap.parse_args(argv)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
