# Architecture

![framework](figures/framework.png)

## Modules (`src/rollout_engine/`)

| Module | Role |
|---|---|
| `clock.py`, `rng.py` | time conventions (`ceil_ms`, decimal seconds) and per-component random streams |
| `config.py` | system configuration v1: loader, validation, and `derive_system` (engine and trainer parameters from hardware and model numbers) |
| `api/` | records shared by every layer: `GroupSpec`, the actions `Place` and `Drop`, `InvalidAction`, `RunError` |
| `trace/` | rollout trace schema v1 (loader, writer, manifest) and the generator |
| `workers/` | the rollout engines: continuous (closed form) and static, the step-by-step references, a single-worker harness |
| `batching/` | the static batch cost model shared by the engine, the policies, and the exact references |
| `verifier/` | the verifier pool (queue, servers, order hook) |
| `scheduler/` | **the core state machine** (`core.py`) and the read-only view policies see (`view.py`) |
| `policies/` | policy parts, named combinations, oracles, and the factory `make_policy` |
| `sim/` | the discrete-event driver and the accounting / staleness checkers |
| `telemetry/` | run metrics (window, GPU states) and the Prometheus text renderer |
| `opt/` | exact references: DP, MILP, LPT, brute force, lower bounds, the S7 experiment |
| `bench/` | experiments, parallel runner, tuning, statistics, decision-cost microbenchmarks, the CLI |
| `service/` | the live controller, HTTP transport, virtual-time loop, mock components, demo |
| `storage/` | the sqlite3 store and trace export |

Policy, metric, and trace code imports neither the simulator nor `asyncio` nor `http`, and never reads the wall clock.

## The core state machine and its two drivers

`scheduler.core.Core` holds the scheduling state only: groups and samples, a mirror of every worker built from the worker's reports (version, pause and drain flags, admitted and waiting counts, KV in use, prefix residency, iteration count), the verifier queue, the trainer (`s_next`, `v_pub`, state, timers), the staleness rules, and the action validation. It never sees a hidden length or verifier time. Its interface:

- inputs from workers: `on_report(worker, t, events, timeline, snapshot)` (admit, start, finish, remove);
- inputs from verifiers and the trainer: `on_verified(server, t)`, `on_train_done(step, t)`;
- internal timers: `next_internal_time()`, `trainer_events(t)` (end of sync = publication, end of a switch);
- decisions: `assign_verifiers(t, policy)`, `try_select(t)`, `invoke(policy, t)` (applies the actions), `after_commit(t)` (dead-group rule);
- outputs: `out_worker` (inputs for the workers: place, drop, publish, switch_out, switch_in) and `out_train` (a training step starts with these tokens).

Two drivers call it in the same order (`docs/simulator.md` section 1):

1. **Simulator** (`sim/driver.py`): owns one engine per worker, the verifier completion times, and the training durations, all derived from the trace; keeps a heap of next event times; runs one instant at a time. A default 24-step closed-loop run of the reference policy takes about half a second on one core of the development machine (decision and engine costs in `benchmarks/README.md`).
2. **Live controller** (`service/controller.py`): the same core under asyncio; the engines run in the mock workers, which talk to it through the HTTP API; a durable queue, a store, retries, and backpressure surround it.

Because both drivers run the same core and the same engines, the live service on a virtual-time loop reproduces the simulator's placement log and step records exactly (check 9); any difference would be a bug in the live layer.

## Flow of a benchmark run

`python -m rollout_engine.bench` → `bench/experiments.py` expands `configs/experiments.json` and the frozen `configs/tuned/tuned.json` into cells (scenario × variant: a system configuration, a generator configuration, policies, seeds) → `bench/runner.py` generates every needed trace once into `outputs/traces/` (CSV + manifest) and replays it from the file in each job → jobs run in a `spawn` process pool (default: half the logical processors) → each job: `simulate` → `run_metrics` → accounting and staleness checks → rows sorted by (scenario, variant, policy, seed) → `J` with the frozen normalizers and weights → `aggregate.csv` (mean, 95 % Student-t interval), `paired.csv` (paired differences, win/tie/loss) → S7 (MILP, DP, policies on small static instances, solve-time sweep) → `manifest.json`. `scripts/plot_results.py` turns the committed results into the figures and tables of the README.

Tuning (`python -m rollout_engine.bench tune`) runs only on tuning seeds and writes `configs/tuned/tuned.json`, which is committed before the evaluation.

## The live service

See `docs/service.md`: API, the instant protocol between controller and workers, the store, lost-worker handling, restart recovery, backpressure, metrics, and the demo.
