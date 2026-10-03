# Implementation

## Main boundary

The rollout engine does not own the training algorithm. It accepts groups of prompts, schedules their samples on rollout workers, verifies them, and hands batches with their versions and staleness to a trainer, which stays outside the repository. Everything here runs in simulation or on mock workers; no GPU backend is attached.

## Core records

- **Group** (`api.GroupSpec`, trace schema v1): `group_id`, `task`, `prompt_tokens`, `max_tokens`, `n_samples`, `est_tokens`; hidden per sample: `resp_tokens`, `verify_s`.
- **Sample state** (`scheduler.core.CSample`): unplaced, waiting, running, generated, verifying, verified; its worker, version, tokens per version, attempts.
- **Group state**: unlaunched, outstanding, then exactly one of consumed, dropped (by a policy, the staleness rules, a colocated switch, or exhausted retries), unfinished.
- **Step** (`StepRec`): selection time, the batch, staleness per group, tokens, training end, publication.
- **Batch handed to the trainer** (service): step, version, tokens, and per group its version, staleness, and sample lengths.

## Scheduler

The policy is composed of independent parts (`docs/contracts.md` section 5): estimator (`prior`, `group_evidence(k0)`, `oracle`), launch order within a window (`fifo`, `shortest_first`, `longest_first`), dispatch unit (`group`, `sample`), binding (`early`: assigned at launch and queued on the worker; `late`: placed only on a worker that can admit now), assignment (`first_free`, `round_robin`, `least_loaded`, `lpt`), batch formation for the static engine (`fifo_chunk`, `sorted_chunk`, `dp`), straggler handling and admission (`wait`, `carry(rho)`, `abort(rho)`), and verifier order (`fifo`, `group_first`, `shortest_first`). `reference` = `fifo+group+early+round_robin+wait+prior+v_fifo`, the even static split of a batch across workers; `online_adaptive` = `longest_first+sample+late+lpt` with `group_evidence`. The exact references for static batching (DP, MILP, LPT) are in `docs/optimization.md`.

## Delivery order

Revised: the rollout simulator and the benchmark came before any external queue, cluster framework, or inference adapter.

- [x] 1. local mock worker (the engine-backed mock worker of the live service)
- [x] 2. request/result schema (rollout trace schema v1, system configuration v1, service API v1)
- [x] 3. rollout simulator with a generation-length model, and metrics
- [x] 4. baseline schedulers (FIFO, shortest/longest estimated first, even split, late binding, LPT) and the benchmark tables
- [x] 5. dynamic batching (iteration-level continuous engine; static engine with batch formation)
- [x] 6. verifier service (pool with order policies; verifier workers in the live service)
- [x] 7. optimization-based (DP, MILP, LPT) and online (`online_adaptive`) schedulers on the same traces
- [ ] 8. Redis queue (the durable queue is sqlite3; a Redis adapter is Tier 3, not started)
- [ ] 9. Ray worker pool (Tier 3, not started)
- [ ] 10. vLLM adapter and scheduling experiments on real workers (Tier 3, not started)

## Decision problem

- **Decisions:** which groups to launch from the stream and when (admission under the staleness bound), in which unit (group or sample) and on which worker (early or late binding, assignment), how to form static batches, which outstanding groups to drop, and which queued sample a freed verifier serves.
- **Uncertainty:** the response length of every sample (only a group-level estimate before a sample of the group finishes; the finished samples of a group are evidence about the others), and the verifier time.
- **Objective actually used** (`docs/contracts.md` section 4): closed loop `J = alpha * step_time / R_step + beta * gpu_idle_frac + gamma * P95(straggler) / R_step + delta * waste_frac + epsilon * |length_bias|`; single phase `J = alpha * phase_time / R_phase + beta * gpu_idle_frac + gamma * padding_frac`, with `R` the reference policy's value on the tuning seeds of the same cell and the weights of `configs/experiments.json` (a term that does not vary across policies more than its seed noise is weighted 0 in that cell, the same for every policy). The waste and bias terms stop a policy from winning by throwing data away.
- The link to stochastic optimization is the length distribution: a policy plans on the prior estimate, adapts online from the finished samples of the same group, or (as a labelled oracle) knows the lengths.

## Evaluation and acceptance

`benchmarks/README.md` has the protocol, scenarios S1–S7 and SENS, how to reproduce, and the results; `docs/simulator.md` the model and its assumptions. Checks 1–10 of the task are tests (`tests/`). Every number in the README comes from `benchmarks/results/`, produced by one command with fixed seeds, frozen tuned configurations, and a manifest.

## Cross-project contracts

The conventions (time, randomness, units, determinism, CSV and manifest rules) are those of `llm-serving-control`, contracts version 1; the evaluation protocol (disjoint tuning and evaluation seeds, an objective normalized by a reference policy, paired differences, win/tie/loss, a sensitivity check) is that of `gpu-cluster-scheduler`. Nothing is imported from either project. This project defines its own rollout trace schema, system configuration, metrics, policy interface, and service API in `docs/contracts.md`: a rollout group is not a cluster job (it has samples, versions, and a verifier), so the job trace schema and policy interface of `gpu-cluster-scheduler` are not reused. Shared: the random-stream derivation (the seeds of a stream, not the generator), the CSV and manifest rules, nearest-rank percentiles, `wall_` columns kept apart from deterministic results.

## Scaffold checkpoint

Replaced. The scaffold's records, FIFO queue, echo worker, verifier boundary, and result store were superseded by the trace schema and core records, the group stream, the engine-backed mock worker, the verifier pool, and the sqlite3 store; its single test was rewritten with the same intent (`tests/test_rollout.py`). Current state: Tier 1 implemented and tested (see the README status); Tier 2 and Tier 3 as listed in the README `## TODO`.
