"""A single-worker driver used by check 1 and by the engine microbenchmarks.

It feeds timed inputs to one engine with the same protocol as the simulator (boundary
before inputs, commit after) and returns the event log and the GPU-state timeline.
"""

from __future__ import annotations


def run_single(engine, inputs: list[tuple[int, tuple]], max_steps: int = 1_000_000):
    pending = sorted(inputs, key=lambda x: x[0])
    k = 0
    log: list[tuple[int, tuple]] = []
    timeline: list[tuple[int, str]] = []
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("run_single: too many steps")
        tn = engine.next_time()
        ti = pending[k][0] if k < len(pending) else None
        cands = [x for x in (tn, ti) if x is not None]
        if not cands:
            break
        t = min(cands)
        if tn == t:
            engine.boundary(t)
        while k < len(pending) and pending[k][0] == t:
            engine.push(t, pending[k][1])
            k += 1
        if engine.wants_commit(t):
            engine.commit(t)
        ev, tl = engine.report()
        log.extend((t, e) for e in ev)
        timeline.extend(tl)
    return log, timeline
