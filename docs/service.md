# Live rollout service

`src/rollout_engine/service/`: the controller (`controller.py`), the HTTP transport (`http.py`), the virtual-time loop and clocks (`vloop.py`), mock workers, verifiers, and trainer (`mock.py`), and the demo (`demo.py`). Store: `src/rollout_engine/storage/`. Contract: `docs/contracts.md` section 7. Everything the mock components do is **simulated**: they stand in for a model, a sandbox, and a trainer.

## Design

- The controller runs `scheduler.core.Core`, the same state machine as the simulator, with the same policy factory. Messages (worker reports, sample results, verifier results, trainer calls) are buffered; once every message of the current instant has arrived (`clock.settle()`), the controller applies them in the simulator's order (`docs/simulator.md` section 1): worker boundary reports by worker id, verifier results by server, trainer events, selection and staleness rules, the policy once, the inputs to the workers, worker commit reports by worker id, the dead-group check.
- **Workers** run the engine of `docs/simulator.md` (mock workers read the hidden lengths from the trace by `group_id` and `sample_idx`; the API never carries them). A worker reports at every boundary of its engine (`phase: boundary`), receives the inputs of that instant in the response, commits, reports again (`phase: commit`), and receives the inputs that arrive after its commit. Between boundaries it long-polls (`phase: poll`) for inputs addressed to it in the middle of a phase. Each request also counts as a heartbeat.
- **Verifier workers** long-poll for the next sample; the verifier order of the policy decides which queued sample a freed server takes.
- **The trainer** long-polls for a batch; the batch is handed out when training starts (at the selection when disaggregated, after every worker has switched out when colocated) and carries every sample's length, the versions, and the staleness. When it reports `done`, the controller runs the sync (`sync_ms`) and publishes the next version to every worker.
- **Durability.** One sqlite3 file in WAL mode; writes are committed once per instant. Tables `groups`, `samples`, `attempts`, `workers`, `steps`, `meta`, and the append-only `events` log (`submit`, `register`, `place`, `admit`, `start`, `finish`, `remove`, `verified`, `ready`, `select`, `train_start`, `train_done`, `publish`, `drop`, `lost`, `worker_lost`), each with its timestamp. `storage.export_trace` rebuilds a rollout trace from the observed lengths and verification durations, so a live run can be replayed through the simulator. Restarting the controller on the same file recovers the queue in order; consumed and dropped groups stay terminal; samples that were waiting or running go back to the pool as lost (attempt + 1); a step selected but not yet trained is handed to the trainer again.
- **Lost workers.** A worker silent for `heartbeat_timeout_s` is declared lost: its waiting and running samples return to the pool with an attempt counter (`rollout_retries_total`), and a group whose sample has been lost more than `max_attempts` times is dropped with reason `retries_exhausted`. Its slot takes no more work until a worker registers again under the same name.
- **Backpressure.** The queue (groups submitted but not launched) is bounded by `max_queue_groups`; a submission that would exceed it is rejected whole with `429` and `Retry-After`.
- **Clocks.** In tests the controller and the mocks share a `VirtualTimeLoop` (an asyncio loop with a virtual clock that never sleeps, opens no socket, and advances only when nothing can run at the current instant). There the live run produces the same placement and drop log and the same step records as the simulator (check 9, `tests/test_service.py`). On the wall clock (`ScaledClock`, optionally accelerated) the same code runs with real timing; conformance is not claimed there.

## Run it

```bash
python -m rollout_engine.service.demo --port 18300
```

starts the service on `127.0.0.1:18300`, attaches two mock workers, four mock verifiers, and a mock trainer over HTTP, submits the committed sample trace (`configs/traces/sample.csv`), consumes three training batches, prints them with a few metrics, and exits (`--port 0` picks a free port; `--speed` sets the clock acceleration, default 200; `--serve --max-seconds N` runs only the controller for external workers).

## API, one example per call

All bodies are JSON (UTF-8). The examples were produced by the code (`InProcessClient`) on a small configuration.

`POST /v1/groups` queues groups; the queue is the stream (first in, first in the launch window).

```text
request:  {"groups": [{"group_id": "q-0001", "task": "math", "prompt_tokens": 312, "max_tokens": 8192, "n_samples": 2, "est_tokens": 640.5}]}
response: 202 {"accepted": 1, "queue_depth": 1}
again:    409 {"error": "duplicate group_id 'q-0001'"}
full:     429 {"error": "queue holds 1 groups (max 2)"}   with header Retry-After: 1
invalid:  400 {"error": "groups[0].est_tokens is required"}
```

`POST /v1/workers` registers a worker; the response names its slot and the engine settings of the configuration.

```text
request:  {"worker_id": "gpu-0", "max_seqs": 32, "kv_tokens": 955636, "tp": 1}
response: 200 {"wid": 0, "engine": "continuous", "max_seqs": 32, "kv_tokens": 955636, "version": 0}
```

`POST /v1/workers/{id}/heartbeat` with `phase: poll` is a heartbeat and a long poll for inputs:

```text
request:  {"phase": "poll", "wait_s": 5}
response: 200 {"inputs": [{"op": "place", "samples": [{"sid": 0, "gidx": 0, "group_id": "q-0001", "sample_idx": 0, "prompt_tokens": 312, "max_tokens": 8192}, {"sid": 1, "gidx": 0, "group_id": "q-0001", "sample_idx": 1, "prompt_tokens": 312, "max_tokens": 8192}]}], "commit_now": true, "ended": false}
```

Other inputs: `{"op": "drop", "gidx": 7, "group_id": "q-0008"}`, `{"op": "publish", "version": 3}`, `{"op": "switch_out"}`, `{"op": "switch_in"}`. With `phase: boundary` or `phase: commit` the body is the worker's report: the engine events since its last report (`["admit", sid]`, `["start", sid, version, iteration]`, `["finish", sid]` after the result was posted, `["remove", sid, tokens, segments]`), its GPU-state changes, and a snapshot; the response carries the inputs of that instant:

```text
request:  {"phase": "boundary", "t": 31337, "events": [["finish", 0]], "timeline": [[31337, "gen"]],
           "snapshot": {"version": 0, "paused": false, "draining": false, "parked": false, "admitted": 1,
                        "waiting": 0, "kv_used": 8504, "iters": 3210, "busy": false}}
response: 200 {"inputs": [], "commit_now": true, "ended": false}
```

`POST /v1/samples/{id}/result` reports a finished sample (sent just before the boundary report that lists it):

```text
request:  {"worker_id": "gpu-0", "tokens": 3210, "finish_reason": "stop", "segments": [[0, 3210]]}
response: 200 {"ok": true}
```

`POST /v1/verifiers`, `POST /v1/verifiers/{server}/poll`, `POST /v1/verifications/{id}/result`:

```text
request:  {"verifier_id": "sandbox-0"}            -> 200 {"server": 0}
request:  {"wait_s": 30}                           -> 200 {"sample_id": 0, "group_id": "q-0001", "sample_idx": 0, "task": "math"}
                                                      (204 when nothing arrives in time)
request:  {"server": 0}                            -> 200 {"ok": true}
```

`POST /v1/trainer/next` (long poll) and `POST /v1/trainer/done`:

```text
request:  {"wait_s": 30}
response: 200 {"step": 0, "version": 0, "tokens": 9046, "groups": [{"group_id": "q-0001", "version": 0, "staleness": 0,
               "samples": [{"sample_idx": 0, "tokens": 3210}, {"sample_idx": 1, "tokens": 5212}]}]}
          (204 when no batch starts in time; {"done": true} after the last step)
request:  {"step": 0}  -> 200 {"ok": true}   (409 when that step is not training)
```

`GET /healthz` → `200 {"ended": false, "ok": true}`. `GET /v1/status` → queue depth, outstanding and ready groups, consumed and dropped counts, `s_next`, `v_pub`, trainer state, workers, and the last error. `GET /metrics` (Prometheus text format 0.0.4):

```text
# HELP rollout_queue_depth Groups queued and not launched.
# TYPE rollout_queue_depth gauge
rollout_queue_depth 0
# HELP rollout_outstanding_groups Launched groups not consumed or dropped.
# TYPE rollout_outstanding_groups gauge
rollout_outstanding_groups 1
...
```

Metrics: `rollout_queue_depth`, `rollout_outstanding_groups`, `rollout_running_samples`, `rollout_healthy_workers`, `rollout_tokens_generated_total`, `rollout_trainer_wait_seconds_total`, `rollout_steps_total`, `rollout_step_seconds`, `rollout_staleness`, `rollout_verifier_queue_depth`, `rollout_drops_total{reason}` (`policy`, `stale`, `colocated_switch`, `retries_exhausted`), `rollout_retries_total`, `rollout_backpressure_rejections_total`.

## Tests

`tests/test_service.py`: check 9 on 11 cases (continuous and static engines, `eta` 0 and 1, `drain` and `interrupt`, carry and abort, colocated); lost-worker retry; retries exhausted; restart recovery on the same store; backpressure, duplicates, and field validation; `/metrics` parsed by a small Prometheus text parser; the HTTP transport on a random port; trace export from the store.
