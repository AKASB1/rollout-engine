# Contracts, version 1

This document is the binding, language-neutral definition of the formats, metric definitions, policy interface, and service API of `rollout-engine`. Other projects follow this document; they do not import the Python package. Changes are deliberate, versioned, and logged.

Version: **1**.

The general conventions (time, randomness, units, determinism, CSV and manifest rules) follow version 1 of the conventions of the sibling project `llm-serving-control` (its `docs/contracts.md`, sections 1 and 2); the evaluation protocol follows `gpu-cluster-scheduler`. They are restated here so that this document stands alone. The rollout trace schema, the system configuration, the metrics, the policy interface, and the service API below are this project's own.

## 1. Conventions

- **Time.** Policies never read a clock and never sleep; they read time from the view. Inside the simulator, time is an integer number of milliseconds since the start of the run, and every engine parameter is an integer number of nanoseconds. A duration becomes an event time by rounding up to the next millisecond, `ceil_ms(ns) = ceil(ns / 10^6)` in integer arithmetic, so event order never depends on float rounding. Files and reports show decimal seconds with at most three decimals. The live service reads a clock object (virtual in tests, monotonic wall clock otherwise).
- **Randomness.** One stream per component, derived from the run seed and the component name exactly as in `llm-serving-control`: `c = fnv1a64(name)` (64-bit FNV-1a over the UTF-8 bytes), `s1 = splitmix64(seed XOR c)`, `s2 = splitmix64(s1 XOR 0x9e3779b97f4a7c15 XOR c)`, where `splitmix64(x)` adds `0x9e3779b97f4a7c15` and applies the SplitMix64 output function (xor-shift-multiply by `0xbf58476d1ce4e5b9` and `0x94d049bb133111eb`). The stream is NumPy's `Generator` over `PCG64DXSM` with its 128-bit state set to `(s1 << 64) | s2` and the fixed odd increment `((6364136223846793005 << 64) | 1442695040888963407) | 1`. The streams are **not bit-identical** to the Go streams of the sibling projects (different generator); only the derivation of the seeds is shared. Component names used by the trace generator: `trace.tasks`, `trace.prompts`, `trace.lengths.group`, `trace.lengths.sample`, `trace.estimates`, `trace.verify`; policies use `policy.<name>`. The trace a seed produces depends only on the generator configuration and the seed, never on which policy runs (common random numbers): response lengths, verifier times, and length estimates are all part of the trace. Python's `random` module and `hash()` never influence a result.
- **Units.** Tokens and GPUs are integers; time in files is seconds; bytes and FLOPs appear only in the derivation of the system parameters; there is no currency.
- **Determinism.** The same system configuration, trace, policy configuration, and seed give identical outputs. Ties break by an explicit total order (groups: position in the trace; samples: group, then `sample_idx`; workers and verifier servers: integer id), never by dict or set iteration order. The only non-deterministic outputs are wall-clock measurements, in columns whose names start with `wall_` (kept in separate files); byte-comparison tests exclude them. Reproducibility is claimed for one machine and one software stack, which every manifest records, not across platforms (`exp` and `log` may differ in the last bit between math libraries).

## 2. Rollout trace schema v1

CSV, UTF-8, exactly this header (8 columns), one row per sample. Groups appear in the order the dataset serves prompts (the stream order); the rows of one group are contiguous.

```text
group_id,sample_idx,task,prompt_tokens,max_tokens,est_tokens,resp_tokens,verify_s
```

| Column | Type | Rule |
|---|---|---|
| `group_id` | string | non-empty, no comma; unique per group (not per row) |
| `sample_idx` | integer | 0, 1, ... within the group, ascending, no gaps; the group size `n_samples` is the row count |
| `task` | string | non-empty (for example `math`, `code`); one value per group; selects the verifier kind |
| `prompt_tokens` | integer | >= 1; one value per group |
| `max_tokens` | integer | >= 1; the generation cap; one value per group |
| `est_tokens` | decimal | > 0, finite; the prior estimate of the mean response length of the group's samples, the only length information a policy has before a sample of the group finishes; one value per group |
| `resp_tokens` | integer | 1 to `max_tokens`; the length the sample will generate (equal to `max_tokens`: a truncated response); hidden from policies until the sample finishes |
| `verify_s` | decimal | >= 0 with at most three decimals (whole milliseconds; trailing zeros beyond three decimals are accepted); the verifier service time of the sample; hidden until its verification ends |

A loader rejects, naming the 1-based line number: a missing or different header (line 1), a wrong field count, an unparsable or out-of-range value, an empty required field, a group whose rows are not contiguous or whose group-level fields (`task`, `prompt_tokens`, `max_tokens`, `est_tokens`) differ, a `sample_idx` out of sequence, and a duplicate `group_id`. Line endings LF or CRLF. A header with no rows is a valid empty trace; a zero-byte file is invalid. Writers use LF, integers without sign or leading `+`, `est_tokens` with three decimals, and `verify_s` with exactly three decimals.

**Manifest.** Beside `<name>.csv` lies `<name>.manifest.json`:

```json
{
  "schema_version": 1,
  "generator": {"name": "rollout-gen", "version": 1, "params": {"...": "..."}},
  "seed": 7,
  "groups": 4608,
  "samples": 36864,
  "content_sha256": "<hex SHA-256 of the CSV bytes>"
}
```

A replayer checks `schema_version` and `content_sha256` before use.

## 3. System configuration v1

JSON with `schema_version: 1` and the sections below. Every numeric parameter is documented in `docs/simulator.md` as assumed or derived with its source; every result is labelled "simulated, assumed parameters". `rollout_engine.config.derive_system` builds a configuration from hardware and model numbers (the derivations are code). An optional `derivation` object records those inputs; it is ignored by the loader.

| Section | Field | Rule |
|---|---|---|
| `gpus` | `total` | integer >= 1 |
| | `partition` | `disaggregated` or `colocated` |
| | `rollout`, `train` | `disaggregated` only: integers >= 1 with `rollout + train = total` |
| | `switch_ms` | `colocated` only: integer >= 0, the cost of one change between rollout and training |
| `rollout` | `engine` | `continuous` or `static` |
| | `tp` | GPUs per worker, integer >= 1; workers = rollout GPUs / `tp` (all GPUs when colocated), which must be whole |
| | `a0_ns` (>= 10^6), `a1_ns`, `a2_ns` | iteration time `a0 + a1 * b + a2 * C` ns (`b` running samples, `C` context tokens); a model restriction: every iteration takes at least 1 ms |
| | `prefill_c0_ns`, `prefill_c1_ns` | prefill time `c0 + c1 * tokens` ns |
| | `max_seqs` (>= 1), `kv_tokens` (>= 1) | continuous engine: concurrency cap and KV capacity in tokens |
| | `static_batch` (>= 1) | static engine: batch-size cap |
| `verifier` | `servers` | integer >= 1 |
| | `tasks.<task>.verify_mean_s` | decimal >= 0: the expected service time policies may know |
| `trainer` | `fixed_ms` (>= 1), `ns_per_token` (>= 0) | a step trains for `fixed_ms + ceil_ms(ns_per_token * tokens)` (at least 1 ms) |
| `sync` | `sync_ms`, `swap_ms` | integers >= 0 |
| | `inflight` | `drain`, `swap`, or `interrupt` |
| `loop` | `mode` | `closed` (default) or `single_phase` |
| | `steps` (T), `groups_per_step` (B) | integers >= 1; `closed` needs `steps >= warmup_steps + 2` |
| | `eta` | staleness bound, integer >= 0 |
| | `warmup_steps` | integer >= 0, default 2 |

Integers must be JSON integers (not `1.0`). The experiment configuration sets how many groups the generator makes (default `3 * T * B`); running out of stream is a run error.

## 4. Metrics

**Measurement window.** Step-level metrics drop the first `warmup_steps` steps and the last step (those steps still run); the window interval is `[t_sel(warmup), t_sel(T-1))`. A single-phase run (no trainer; one step that takes every group of the trace; S1, S2, S7) uses the whole run. Percentiles are nearest rank (the `ceil(p * n)`-th smallest). Times in result files are seconds.

Per step `s` (its interval runs from the selection time `t_sel(s)` to `t_sel(s + 1)`):

- step time `= t_sel(s + 1) - t_sel(s)`; trainer wait = time in the interval with the trainer in state `wait`;
- straggler delay = ready time of the `B`-th consumed group minus that of the `ceil(0.9 * B)`-th, in ready order (`t100 - t90`);
- staleness of a consumed group `= s - version`; `stale_token_frac` = response tokens of the batch generated under a version older than `s`, over all response tokens of the batch.

Per run (closed loop; column names of `runs.csv`):

| Column | Definition |
|---|---|
| `step_time_mean_s`, `step_time_p95_s` | window length / window steps; nearest-rank P95 of the window step times |
| `samples_per_s` | samples of the window steps' batches / window length |
| `completed_tokens_per_s` | tokens of samples whose finish (or drop removal) report lies in the window / window length (tokens are credited when their sample completes, not when they are generated) |
| `gpu_idle_frac`, `gpu_gen_frac`, `gpu_overhead_frac`, `gpu_train_frac` | GPU-seconds per state (`docs/simulator.md` section 6) over all GPU-seconds of the window, rollout and training GPUs together |
| `rollout_idle_frac` | idle GPU-seconds of the rollout GPUs over their GPU-seconds |
| `trainer_wait_frac`, `trainer_wait_mean_s` | trainer wait over the window length; mean trainer wait per window step |
| `straggler_mean_s`, `straggler_p95_s` | over window steps |
| `staleness_mean`, `staleness_max`, `stale_token_frac` | over consumed groups / window steps |
| `waste_frac` | tokens generated by groups that ended `dropped` / (those + tokens of `consumed` groups), whole run |
| `length_bias` | mean true response length of the samples of consumed groups / the same mean over consumed and dropped groups, minus 1; whole run; 0 when nothing is dropped |
| `batch_len_cv` | coefficient of variation (population) across window steps of the mean response length of the batch |
| `verify_delay_mean_s`, `verify_delay_p95_s` | `verified - generated` over the samples of the window steps' batches |
| `groups_consumed`, `groups_dropped`, `groups_unfinished`, `drops_policy`, `drops_stale`, `drops_colocated_switch`, `tokens_consumed`, `tokens_dropped`, `run_s`, `window_s` | counts and totals |

Single phase: `phase_time_s` (start to the ready time of the last group), `gen_time_s` (the last `generated` stamp), `straggler_s` (`t100 - t90` over all groups), `gpu_idle_frac` over `[0, gen_time]` (rollout GPUs), `gen_frac`, `overhead_frac`, `mean_ready_s`, `padding_frac` (static engine: `1 - sum of response lengths / sum over batches of b * Lmax`), `lb_gap` (`gen_time` over the lower bound of check 3, minus 1), and the verifier delays.

Non-deterministic columns start with `wall_` and live in separate files (`wall_runs.csv`, `wall_s7.csv`, `wall_s7_sweep.csv`, `wall_decision_cost.csv`): run wall and CPU time, instants per second, the decision time per policy invocation (mean, P95, max), solver wall and process CPU time.

**Objective** (tuning and summary score, column `J`). Closed loop:

```text
J = alpha * step_time_mean / R_step + beta * gpu_idle_frac + gamma * P95(straggler) / R_step + delta * waste_frac + epsilon * |length_bias|
```

Single phase: `J = alpha * phase_time / R_phase + beta * gpu_idle_frac + gamma * padding_frac`. `R_step` and `R_phase` are the means of the reference policy over the tuning seeds of the same cell, computed once by the tuning command and recorded in `configs/tuned/tuned.json`, identical for every policy. Weights come from `configs/experiments.json` (1, 0.5, 0.25, 1, 1 and 1, 0.5, 1); a term gets weight 0 in a cell when no configuration evaluated on the tuning seeds (every policy's defaults and every search candidate) moves it away from the reference by more than two standard errors of the paired difference (recorded in `tuned.json` as `term_check`, identical for every policy).

## 5. Policy interface

A policy is `{"name": ..., "params": {...}}` built by one factory (`policies.composed.make_policy`) for the simulator, the benchmark, and the live service. It reads a read-only view, keeps private state, is deterministic, and never imports the simulator, `asyncio`, or `http`, or reads a clock.

**View at one decision instant:** the time; the trainer (`s_next`, `v_pub`, `B`, `eta`, `gate_open = s_next - v_pub <= eta`, state); each worker (id, version, state `running` / `draining` / `paused` / `idle`, running and waiting counts, free slots, free KV tokens, prefix residency per group, whether it can admit now, placed samples, whether it is alive); the launch window (the next `window_groups` unlaunched groups in stream order with `task`, `prompt_tokens`, `max_tokens`, `n_samples`, `est_tokens`); every outstanding group with the state of each sample (unplaced, waiting, running with tokens so far as of its worker's latest report, generated or verified with the observed length) and its version; the ready pool; the verifier queue (sample, task, wait so far); counters (consumed, dropped, tokens wasted). Never the true length of an unfinished sample, and never `verify_s` before a verification ends.

**Actions,** applied in order: `Place(kind, ref, worker)` with `kind` `group` (its unplaced samples), `sample`, or `batch` (static engine; a tuple of sample ids), and `Drop(group)`. The engine checks every action (the unit exists and is unplaced; the worker exists and is alive; for the static engine the worker is idle and the batch within `static_batch`; the staleness gate for a group not yet launched; a drop names an outstanding group). An invalid action aborts the run with an error that names the policy and the action. A second hook, `pick_verification(view, queue)`, returns the index of the queued sample a freed verifier serves next.

**Parts** (all named in the policy's display name, e.g. `longest_first+sample+late+lpt+carry(1.5)+group_evidence(2)+v_fifo`): estimator `prior` | `group_evidence(k0)` | `oracle`; order `fifo` | `shortest_first` | `longest_first` with the anti-starvation window; dispatch `group` | `sample`; binding `early` | `late`; assignment `first_free` (late only) | `round_robin` | `least_loaded` | `lpt`; batch formation (static) `fifo_chunk` | `sorted_chunk` | `dp`; straggler/admission `wait` | `carry(rho)` | `abort(rho)`; verifier order `fifo` | `group_first` | `shortest_first`. Named combinations: `reference` (`fifo+group+early+round_robin+wait+prior+v_fifo`; static engine: `fifo_chunk` with full batches), `online_adaptive` (`longest_first+sample+late+lpt` with `group_evidence`; static: `dp` with `group_evidence`), `oracle_lpt`, `oracle_dp` (oracles: they read the trace, run only in the benchmark, and are never presented as deployable), `dp`, `lpt`. Semantics of each part: `IMPLEMENTATION.md` and the docstrings of `src/rollout_engine/policies/composed.py`. Tuned parameters: `window_groups`, `rho` (or `cap_groups`), `k0`, `batch_size`.

**Tier 2 parts:** estimator `eb(n0)` (empirical Bayes in log space relative to the estimate, hyperparameters learned from completed groups, running samples as right-censored observations; `policies/eb.py`) and its `quantile` option (plan on a quantile of the predictive length instead of its mean); straggler part `abort_pred(rho, kappa)` (after a selection, drop only the unselected groups whose predicted remaining time exceeds `kappa` times the median of the outstanding groups); `isolate(x)` (late binding: a unit predicted longer than `x` times the median of the units placed in the same call goes to the candidate worker with the most free slots); `deadline(x)` admission (a group beyond the first `B` outstanding is not launched when its predicted ready time, `x` times its estimated length times the observed decoding time per token, falls after the extrapolated selection of the last step it can serve). Presets: `online_eb` (`longest_first+sample+late+lpt+isolate(2)` with `eb`), `quantile_lpt` (`eb` with `quantile 0.9`). Additional tuned parameters: `kappa`, `deadline`.

## 6. Result files and manifests

`benchmarks/results/<full|quick>/`: `runs.csv` (one row per run; first columns `scenario,variant,policy,seed,policy_name,J`), `wall_runs.csv`, `aggregate.csv` (`scenario,variant,policy,metric,mean,ci95,n`: mean and the half-width of a 95 % Student-t interval across seeds), `paired.csv` (`diff_mean,diff_ci,n,wins,ties,losses` of the paired differences policy minus reference on the common seeds, for `J` and the step or phase time; a seed is a tie when `|diff| <= tie_band * |reference|`, lower is better), `s7.csv`, `wall_s7*.csv`, and `manifest.json` (commit and dirty flag, hashes of the experiment and tuned configurations and of the expanded cells, seeds, normalizers and weights per cell, Python, NumPy, SciPy, and HiGHS versions, CPU model, logical CPUs, RAM, OS, workers, a load note, violations). CSV: UTF-8, LF, header row, floats with 6 significant digits; rows sorted by (scenario, variant, policy, seed), never by completion order.

## 7. Live service API (v1)

HTTP/1.1, JSON, UTF-8, `127.0.0.1`, default port 18300 (configurable; tests use port 0); the controller is one asyncio process. Every call with an example: `docs/service.md`.

| Call | Purpose and responses |
|---|---|
| `POST /v1/groups` | queue groups durably (`group_id`, `task`, `prompt_tokens`, `max_tokens`, `n_samples`, `est_tokens`); the queue is the stream. `202` with `accepted` and `queue_depth`; `409` duplicate `group_id`; `400` names the field; `429` with `Retry-After` when the queue would exceed `max_queue_groups` |
| `POST /v1/workers` | register `{worker_id, max_seqs, kv_tokens, tp}`; `200` with the worker slot `wid`; a worker declared lost registers again under its name |
| `POST /v1/workers/{id}/heartbeat` | `phase: poll` (liveness and a long poll for inputs, `wait_s`), `boundary` or `commit` (the worker's report: events, GPU-state timeline, snapshot), answered with the inputs of that instant (`place`, `drop`, `publish`, `switch_out`, `switch_in`; never a hidden length) |
| `POST /v1/samples/{id}/result` | a finished sample: `worker_id`, `tokens`, `finish_reason` (`stop` or `length`), version segments |
| `POST /v1/verifiers`, `POST /v1/verifiers/{server}/poll`, `POST /v1/verifications/{id}/result` | verifier workers: register, long-poll for the next sample (the verifier order decides), report the end of a verification |
| `POST /v1/trainer/next` | long poll (`wait_s`): `200` with `step`, `version`, `tokens`, and the selected groups with their versions, staleness, and sample lengths when training starts; `204` when nothing is ready in time; `{"done": true}` after the last step |
| `POST /v1/trainer/done` | `{step}`: training ended; the controller runs the sync and publishes the next version |
| `GET /metrics`, `GET /healthz`, `GET /v1/status` | Prometheus text (0.0.4), liveness, a JSON status |

A worker silent for `heartbeat_timeout_s` is lost: its waiting and running samples go back to the pool with an attempt counter, and after `max_attempts` the group is dropped with reason `retries_exhausted`. Store: one sqlite3 file (WAL) with groups, samples, attempts, workers, steps, and an append-only event log with timestamps, enough to rebuild the observed trace (`storage.export_trace`) and replay it through the simulator; restarting the controller on the same file recovers the queue and puts the outstanding samples back in the pool as lost.

## 8. Versioning

This is version 1. A change to a format, a definition, or the API is deliberate, gets a new version number here, and is logged with its reason.
