"""The live service: check 9 (sim-live conformance), lost-worker retry, restart recovery,
backpressure, Prometheus metrics, the HTTP transport, and trace export from the store."""

import asyncio
import copy
import math
import os
import re

import pytest

from rollout_engine.config import deep_merge, derive_system, parse_system
from rollout_engine.policies.composed import make_policy
from rollout_engine.service.controller import Controller
from rollout_engine.service.http import HttpClient, serve
from rollout_engine.service.mock import InProcessClient, run_session, submit_payload
from rollout_engine.service.vloop import LoopClock, run_virtual
from rollout_engine.sim.driver import simulate
from rollout_engine.storage import Store, export_trace
from rollout_engine.trace.generator import DEFAULT_GENERATOR, generate


def small(
    engine="continuous",
    eta=1,
    inflight="drain",
    partition="disaggregated",
    steps=5,
    seed=3,
    workers=3,
):
    raw = derive_system(
        total_gpus=workers + 2 if partition == "disaggregated" else workers,
        rollout_gpus=workers,
        partition=partition,
        steps=steps,
        groups_per_step=4,
        eta=eta,
        inflight=inflight,
        engine=engine,
        verifier_servers=3,
    )
    raw = deep_merge(
        raw,
        {
            "rollout": {"max_seqs": 8, "static_batch": 6},
            "trainer": {"fixed_ms": 500},
            "loop": {"warmup_steps": 1},
        },
    )
    sysc = parse_system(raw)
    gen = copy.deepcopy(DEFAULT_GENERATOR)
    gen.update({"groups": 4 * steps * 3, "n_samples": 4, "max_tokens": 800})
    for t in gen["tasks"].values():
        t["len_median"] = 150
    return sysc, generate(gen, seed)


def live(sysc, tr, pcfg, tmp, **kw):
    session_kw = {k: kw.pop(k) for k in ("crash", "until_step", "submit") if k in kw}

    async def main(loop):
        clock = LoopClock(loop, kw.pop("offset", 0))
        ctl = Controller(
            sysc,
            pcfg,
            clock=clock,
            store_path=os.path.join(tmp, "store.db"),
            **{"heartbeat_timeout_s": 0, **kw},
        )
        client = InProcessClient(ctl)
        mocks = await run_session(ctl, client, sysc, tr, clock, **session_kw)
        ctl.store.flush()
        return ctl, mocks

    return run_virtual(main)


def steps_of(core):
    return [
        (s.s, s.t_sel, s.groups, s.staleness, s.tokens, s.t_train_end, s.t_pub) for s in core.steps
    ]


CASES = [
    ("continuous", 0, "drain", "reference"),
    ("continuous", 0, "interrupt", "online_adaptive"),
    ("continuous", 1, "drain", "lpt"),
    ("continuous", 1, "interrupt", "reference"),
    (
        "continuous",
        1,
        "drain",
        {
            "name": "composed",
            "params": {
                "dispatch": "sample",
                "binding": "late",
                "assign": "first_free",
                "straggler": "carry",
                "rho": 1.5,
            },
        },
    ),
    (
        "continuous",
        1,
        "interrupt",
        {
            "name": "composed",
            "params": {
                "order": "longest_first",
                "straggler": "abort",
                "rho": 2.0,
                "assign": "least_loaded",
            },
        },
    ),
    ("static", 0, "drain", {"name": "composed", "params": {"batch": "fifo_chunk"}}),
    ("static", 0, "interrupt", "dp"),
    ("static", 1, "drain", "dp"),
    (
        "static",
        1,
        "interrupt",
        {"name": "composed", "params": {"batch": "sorted_chunk", "straggler": "carry"}},
    ),
]


@pytest.mark.parametrize("engine,eta,inflight,pol", CASES)
def test_check9_sim_live_conformance(engine, eta, inflight, pol, tmp_path):
    sysc, tr = small(engine, eta, inflight)
    pcfg = pol if isinstance(pol, dict) else {"name": pol}
    sim = simulate(sysc, tr, make_policy(copy.deepcopy(pcfg)), record_log=True)
    ctl, _ = live(sysc, tr, copy.deepcopy(pcfg), str(tmp_path))
    assert ctl.error is None and ctl.core.ended
    assert ctl.core.log == sim.core.log  # placements and drops: time, action, ids, worker
    assert steps_of(ctl.core) == steps_of(sim.core)
    assert ctl.core.trainer_tl == sim.core.trainer_tl
    assert len(sim.core.log) > 10


def test_check9_colocated(tmp_path):
    sysc, tr = small(partition="colocated", eta=1)
    sim = simulate(sysc, tr, make_policy({"name": "reference"}), record_log=True)
    ctl, _ = live(sysc, tr, {"name": "reference"}, str(tmp_path))
    assert ctl.core.log == sim.core.log and steps_of(ctl.core) == steps_of(sim.core)


def test_lost_worker_samples_are_retried(tmp_path):
    sysc, tr = small(eta=1, steps=5)
    ctl, (workers, _, trainer) = live(
        sysc,
        tr,
        {"name": "reference"},
        str(tmp_path),
        heartbeat_timeout_s=20.0,
        crash={"worker": "w1", "at_ms": 30_000},
    )
    c = ctl.core
    assert c.ended and ctl.error is None and len(trainer.batches) == sysc.steps
    lost = [e for e in c.log if e[1] == "lost"]
    assert lost and all(e[3] == 1 for e in lost)
    assert ctl.counters["retries"] == len(lost)
    # every lost sample was placed again on a live worker and the group finished normally
    for e in lost:
        gid, sidx = e[2].split("/")
        s = c.samples[c.groups[ctl.gid_index[gid]].sids[int(sidx)]]
        assert s.attempts == 1
        assert c.groups[s.gidx].state in ("consumed", "outstanding", "unfinished")
    assert not c.workers[1].alive and "rollout_healthy_workers 2" in ctl.metrics_text()
    # the lost worker's attempt ended as lost in the store
    rows = ctl.store.db.execute("SELECT COUNT(*) FROM attempts WHERE outcome = 'lost'").fetchone()[
        0
    ]
    assert rows == len(lost)


def test_retries_exhausted_drops_the_group(tmp_path):
    sysc, tr = small(eta=1, steps=5)
    ctl, _ = live(
        sysc,
        tr,
        {"name": "reference"},
        str(tmp_path),
        heartbeat_timeout_s=20.0,
        max_attempts=0,
        crash={"worker": "w1", "at_ms": 30_000},
    )
    reasons = [g.drop_reason for g in ctl.core.groups if g.state == "dropped"]
    assert reasons and set(reasons) == {"retries_exhausted"}
    assert (
        'rollout_drops_total{reason="retries_exhausted"} ' + str(len(reasons)) in ctl.metrics_text()
    )


def test_restart_recovers_queue_and_returns_outstanding_samples(tmp_path):
    sysc, tr = small(eta=1, steps=6)
    ctl, _ = live(sysc, tr, {"name": "reference"}, str(tmp_path), until_step=2)
    c1 = ctl.core
    n_steps = len(c1.steps)
    unl = [c1.groups[g].spec.gid for g in c1.unlaunched]
    inflight = {
        s.sid
        for s in c1.samples
        if s.state in ("waiting", "running") and c1.groups[s.gidx].state == "outstanding"
    }
    ctl.store.close()
    assert 2 <= n_steps < sysc.steps and inflight
    offset = Store(str(tmp_path / "store.db")).last_ms()
    ctl2, (_, _, trainer) = live(
        sysc, tr, {"name": "reference"}, str(tmp_path), submit=False, offset=offset
    )
    c2 = ctl2.core
    assert c2.ended and ctl2.error is None
    assert [st.groups for st in c2.steps[:n_steps]] == [st.groups for st in c1.steps[:n_steps]]
    assert len(c2.steps) == sysc.steps
    # the queue came back in order, and the in-flight samples went back to the pool as lost
    launched_after = [c2.groups[ctl2.gid_index[g]].launched_at for g in unl]
    assert all(x >= offset for x in launched_after if x >= 0)
    assert all(c2.samples[sid].attempts == 1 for sid in inflight)
    assert ctl2.counters["retries"] == len(inflight)


def test_backpressure_duplicates_and_validation(tmp_path):
    sysc, tr = small()

    async def main(loop):
        ctl = Controller(
            sysc,
            {"name": "reference"},
            clock=LoopClock(loop),
            store_path=str(tmp_path / "s.db"),
            max_queue_groups=10,
        )
        api = InProcessClient(ctl)
        p = submit_payload(tr)
        out = [await api.post("/v1/groups", {"groups": p["groups"][:8]})]
        out.append(await ctl.handle("POST", "/v1/groups", {"groups": p["groups"][8:13]}))
        out.append(await api.post("/v1/groups", {"groups": p["groups"][:1]}))
        bad = dict(p["groups"][20])
        del bad["est_tokens"]
        out.append(await api.post("/v1/groups", {"groups": [bad]}))
        out.append(
            await api.post("/v1/groups", {"groups": [dict(p["groups"][20], prompt_tokens=0)]})
        )
        return out, ctl.metrics_text()

    (ok, full, dup, missing, rng), metrics = run_virtual(main)
    assert ok == (202, {"accepted": 8, "queue_depth": 8})
    assert full[0] == 429 and full[2] == {"Retry-After": "1"}
    assert dup[0] == 409 and "duplicate" in dup[1]["error"]
    assert missing[0] == 400 and "est_tokens" in missing[1]["error"]
    assert rng[0] == 400 and "prompt_tokens" in rng[1]["error"]
    assert "rollout_backpressure_rejections_total 1" in metrics


LINE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*"(,[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*")*)?\})? (\S+)$'
)


def parse_prometheus(text: str) -> dict:
    """A small parser of the text exposition format: every sample line must parse and belong
    to a metric with HELP and TYPE."""
    helps, types, samples = set(), {}, {}
    for line in text.splitlines():
        if line.startswith("# HELP "):
            helps.add(line.split()[2])
        elif line.startswith("# TYPE "):
            _, _, name, typ = line.split()
            assert typ in ("counter", "gauge", "histogram", "summary", "untyped")
            types[name] = typ
        elif line.strip():
            m = LINE.match(line)
            assert m, f"unparsable line {line!r}"
            name = m.group(1)
            assert name in helps and name in types, f"{name} lacks HELP/TYPE"
            v = float(m.group(5))
            assert not math.isnan(v)
            samples.setdefault(name, []).append((m.group(3) or "", v))
    return samples


def test_metrics_parse_as_prometheus_text(tmp_path):
    sysc, tr = small()
    ctl, _ = live(sysc, tr, {"name": "reference"}, str(tmp_path))
    m = parse_prometheus(ctl.metrics_text())
    for name in (
        "rollout_queue_depth",
        "rollout_outstanding_groups",
        "rollout_running_samples",
        "rollout_healthy_workers",
        "rollout_tokens_generated_total",
        "rollout_trainer_wait_seconds_total",
        "rollout_step_seconds",
        "rollout_staleness",
        "rollout_verifier_queue_depth",
        "rollout_drops_total",
        "rollout_retries_total",
        "rollout_backpressure_rejections_total",
    ):
        assert name in m
    assert m["rollout_steps_total"][0][1] == sysc.steps
    assert m["rollout_tokens_generated_total"][0][1] > 0


def test_http_transport_on_a_random_port(tmp_path):
    sysc, tr = small()

    async def main():
        from rollout_engine.service.vloop import ScaledClock

        ctl = Controller(
            sysc, {"name": "reference"}, clock=ScaledClock(), store_path=str(tmp_path / "s.db")
        )
        server = await serve(ctl, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        cl = HttpClient("127.0.0.1", port)
        try:
            health = await asyncio.wait_for(cl.get("/healthz"), 10)
            bad = await asyncio.wait_for(cl.post("/v1/groups", {"groups": "x"}), 10)
            ok = await asyncio.wait_for(cl.post("/v1/groups", submit_payload(tr)), 10)
            st, text = await asyncio.wait_for(cl.get("/metrics"), 10)
            nf = await asyncio.wait_for(cl.post("/v1/nope", {}), 10)
        finally:
            server.close()
            await server.wait_closed()
        return health, bad, ok, (st, text), nf

    health, bad, ok, (st, text), nf = asyncio.run(main())
    assert health == (200, {"ended": False, "ok": True})
    assert bad[0] == 400 and ok[0] == 202 and nf[0] == 404
    assert st == 200 and "rollout_queue_depth " in text


def test_store_exports_a_replayable_trace(tmp_path):
    sysc, tr = small()
    ctl, _ = live(sysc, tr, {"name": "reference"}, str(tmp_path))
    exp = export_trace(ctl.store)
    orig = {g.group_id: g for g in tr.groups}
    assert len(exp.groups) >= sysc.steps * sysc.groups_per_step
    for g in exp.groups:
        o = orig[g.group_id]
        assert g.resp_tokens == o.resp_tokens and g.verify_ms == o.verify_ms
        assert (g.task, g.prompt_tokens, g.max_tokens, g.est_tokens) == (
            o.task,
            o.prompt_tokens,
            o.max_tokens,
            o.est_tokens,
        )
    kinds = {k for _, k, _ in ctl.store.events()}
    assert {
        "submit",
        "register",
        "place",
        "finish",
        "verified",
        "select",
        "train_start",
        "train_done",
        "publish",
    } <= kinds
