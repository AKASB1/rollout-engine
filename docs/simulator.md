# Simulator model and assumptions

Every result produced with this simulator is **simulated, with assumed parameters**. It models the rollout side of RL post-training (a trainer, rollout workers, a verifier pool) at the level of iterations, versions, and queues. It is not a measurement of any GPU, inference engine, or training stack, and it has **no model of learning**: staleness and length bias are reported as measured quantities; whether they matter for training is out of scope.

Code: `workers/engine.py` (engines), `workers/reference.py` (step-by-step references), `scheduler/core.py` (the core state machine), `scheduler/view.py` (what policies see), `verifier/pool.py`, `sim/driver.py` (the discrete-event driver), `telemetry/metrics.py`, `sim/checks.py`, `opt/bounds.py`, `trace/generator.py`, `config.py` (system configuration and its derivation). Formats and metric definitions: `docs/contracts.md`.

## 1. Time and the order of one instant

- Discrete event, virtual clock, no sleeps; one run is single-threaded and deterministic. Time is an integer number of milliseconds; engine parameters are integer nanoseconds; a duration becomes an event time by `ceil_ms` (integer arithmetic).
- Events at the same instant are applied in this order (`Simulation.instant`):
  1. worker boundaries, by worker id (generation completions, ends of pauses; within a worker, samples by group, then `sample_idx`);
  2. verifier completions, by server id, then the free servers take queued samples through the verifier-order hook; a 0 ms verification completes within the same instant (the loop repeats);
  3. group readiness (a group is ready when its last sample is verified);
  4. trainer events (end of training, end of sync = publication, end of a colocated switch);
  5. the trainer's selection of a batch;
  6. the staleness rules (section 5);
  7. the policy, invoked **once**; its actions are applied in order;
  8. worker commits, by worker id: each worker at a boundary applies its inputs, admits, and starts a phase or a pause;
  9. the dead-group check for samples that started in those commits.
- Inputs produced after a worker's commit at instant `t` (the drops of step 9) reach it at its next boundary after `t`. An input for a worker that did not commit at `t` (or an idle one) reaches it at its first boundary at or after `t`; when that boundary is `t` itself, the driver processes a further instant at the same time `t` (the worker reports, the policy is called, the worker commits). Training lasts at least 1 ms (`fixed_ms >= 1`), so a training end never falls into the instant of its selection.
- Deadlock rule: when no future event exists and the run has not ended, the policy is called once more at the same instant; if it still creates no event, the run aborts with an error that names the policy (a policy bug, not a result). Running out of the trace stream is reported the same way.
- Policy decisions take no virtual time; their wall-clock time is measured apart (`wall_` columns).

## 2. Continuous engine (iteration-level batching)

A worker has a local FIFO waiting list and a running set. Placing a unit appends its samples to the waiting list.

- **Admission** happens only at a boundary of the worker: while the running count is below `max_seqs` and the reservation fits in `kv_tokens`, the head of the waiting list is admitted; the head is never skipped. A sample reserves its `max_tokens` (nobody knows the true length); the prompt of a group is reserved once, `prompt_tokens`, while at least one sample of the group is admitted on the worker (prefix sharing).
- **Prefill.** Admitting a sample whose group prompt is not resident pauses the whole worker for `prefill_c0_ns + prefill_c1_ns * prompt_tokens` (summed over the groups that become resident at that boundary, then rounded up to whole ms); no decoding during a prefill; one prefill per group per residency. Samples start decoding when the pause ends (if that boundary admits more, the worker pauses again first).
- **Iterations.** An iteration with `b` running samples and `C` context tokens (prompt plus tokens generated so far, per sample) takes `a0_ns + a1_ns * b + a2_ns * C` and generates one token per running sample. Between two changes the worker is in a *phase* with fixed `b` and starting context `C0`: `k` iterations take `S(k) = k * A + D * k * (k - 1) / 2` ns with `A = a0 + a1 * b + a2 * C0`, `D = a2 * b`; the boundary after iteration `j` is at `t0 + ceil_ms(S(j))`.
- **Closed form.** Each running sample's finish iteration sits in a heap; the next finish boundary and "the first boundary at or after time `t`" are computed by inverting `S` with `math.isqrt` and correcting by testing the neighbours (`first_k_exceeding`), so an event costs `O(log b)`, not one step per token. Every iteration is required to take at least 1 ms (`a0_ns >= 10^6`, checked), so iteration boundaries of a phase fall on distinct milliseconds.
- **Phases.** A sample finishes at the boundary of the iteration that generates its `resp_tokens`-th token: its slot and reservation are freed at that instant and it is stamped `generated`. A phase is restarted (new `t0`, `C0`) only when the running set changes or the worker pauses; a boundary at which nothing changes continues the phase (restarting it would change the rounding).
- **Inputs at boundaries.** A placement, a publication, or a drop that arrives in the middle of a phase takes effect at the first boundary at or after its time. A drop removes the group's waiting samples and its running samples (their tokens so far are waste) and frees their reservations.
- **Validation.** A group whose `max_tokens + prompt_tokens` exceeds `kv_tokens` could never be admitted; the run is refused at the start.

**Step-by-step reference** (`workers/reference.py`): the same rules with every phase computation replaced by a loop over single iterations in integer ns with the same rounding. Check 1 compares both on 2100 random worker scenarios and on hand-computed cases.

## 3. Static engine

An idle worker takes a batch of at most `static_batch` samples chosen by the policy and runs it to completion: one prefill of the padded prompts, `ceil_ms(prefill_c0_ns + prefill_c1_ns * b * Pmax)`, then `Lmax` iterations in which all `b` samples are carried until the longest finishes (the padding cost); iteration `j = 0 .. Lmax-1` takes `a0_ns + a1_ns * b + a2_ns * b * (Pmax + j)`, and the decode part is rounded up once, `ceil_ms(sum)`. All samples of the batch are `generated` at the end (a static engine returns a batch at once). The batch time depends on the batch only through `b`, `Pmax`, `Lmax` and is nondecreasing in `Lmax` and `b` (`batching.batch_cost_ms`; also the cost model of `docs/optimization.md`). A batch is atomic: drops and publications take effect at its end, so `swap` and `interrupt` behave as `drain` on this engine.

## 4. Verifier

A `generated` sample enters one queue; one of `servers` identical servers serves it for `ceil_ms(verify_s)` (the trace stores whole milliseconds). The queue order is the verifier-order part of the policy (`pick_verification`), consulted whenever a server frees up and the queue is not empty; free servers are filled in id order. A sample of a dropped group leaves the queue; a verification in service runs to its end and its result is discarded. Check 7 compares the pool with Pollaczek-Khinchine, Erlang C, and the Lindley recursion.

## 5. Trainer, versions, staleness, partitions

- The trainer holds version `v_pub` (the weights after `v_pub` steps; 0 at the start) and the index `s_next` of its next step. Step `s` is selected at the first instant at which the trainer is free (state `wait`) and at least `B` ready groups are eligible; the batch is the `B` earliest ready groups (ties by trace order). Training lasts `fixed_ms + ceil_ms(ns_per_token * tokens)` with `tokens = sum over the batch's samples of prompt_tokens + resp_tokens`; the trainer is then busy for `sync_ms` (state `sync`) and version `s + 1` is published at the end of the sync. The run ends when the training of step `T - 1` ends.
- **Versions.** The version of a sample is the version in force on its worker when its first token is generated (the version of the phase it starts in); the version of a group is the oldest version among its started samples. Each sample records how many of its tokens were generated under each version.
- **Staleness rules** (enforced by the engine, not the policies): (1) dead groups: after every selection, and whenever a group's version is set or lowered by a sample that starts, every outstanding group (any state, ready included) with `s_next - version > eta` is dropped at once (reason `stale`) and frees its slots (at each worker's next boundary); (2) the gate: placing a sample of a group that has not started yet (no sample of it has generated a token) when `s_next - v_pub > eta` is an invalid action; (3) with `eta = 0` nothing generated under an older version is ever consumed (a consequence of (1); checked).
- **Publication and in-flight handling.** At publication every rollout worker adopts the new version by `inflight`: `drain` (stop admitting; when the running set is empty, pause `swap_ms` and resume admitting under the new version), `swap` (at the next boundary pause `swap_ms` and continue the running samples with their KV cache under the new weights), `interrupt` (as `swap`, plus a re-prefill of the context of every admitted sample, `prefill_c0_ns + prefill_c1_ns * context` summed, before decoding resumes).
- **Partition.** `disaggregated`: rollout workers and the trainer run concurrently on separate GPUs, and policies may place groups while the trainer trains (the gate decides). `colocated`: the same GPUs alternate. When a batch is selected, every other outstanding group is dropped (reason `colocated_switch`, its work is waste), every worker pauses `switch_ms` at its next boundary (switch-out) and then parks; training starts when every worker is parked and uses all GPUs; the new version is published when training ends, and the workers pause `switch_ms` again (switch-in) before they admit anything; `sync_ms` is not charged and staleness is 0. Groups placed while continuous workers are parked wait in their local lists; a static worker takes no batch while switching or parked (it is not idle). A switch-out waits for each worker's next boundary: for the static engine, the end of its current batch.
- **Termination.** Groups outstanding when the last training step ends are `unfinished` and excluded from every metric except where stated. Worker failures are not simulated (the live service handles lost workers).

## 6. GPU states

Every GPU is in exactly one state at any instant: `gen` (a worker decoding), `overhead` (prefill, swap, re-prefill, switch, the trainer's sync), `train`, or `idle`. A rollout worker of `tp` GPUs contributes its state `tp` times. Disaggregated training GPUs are `train` while training, `overhead` during sync, `idle` otherwise. Colocated GPUs are `train` while parked during training and `idle` while parked otherwise (waiting for the other workers to finish their switch-out).

## 7. Lower bounds (check 3)

For a single phase with every worker idle at time 0 and `W` workers (`opt/bounds.py`, exact rational arithmetic in ns):

- **Work bound (continuous).** A sample of prompt `P` and length `L` needs `L` iterations; each iteration costs at least `a0 / max_seqs` of its fixed part (at most `max_seqs` samples share it), `a1`, and `a2` times its own context `P + g` for `g = 0 .. L-1`. Every group is prefilled at least once. Summed over samples and divided over `W` workers: `(sum over samples of [L * (a0 / max_seqs + a1) + a2 * (L * P + L * (L - 1) / 2)] + sum over groups of (prefill_c0 + prefill_c1 * P)) / W`. Rounding up to ms and pauses only add time.
- **Longest-sample bound (continuous).** Every iteration a sample takes part in costs at least `a0 + a1 + a2 * (its own context)` (`b >= 1`, `C >=` its own context), and its first iteration comes after its group's prefill, so the generation time is at least the largest alone-time `prefill_c0 + prefill_c1 * P + L * (a0 + a1 + a2 * P) + a2 * L * (L - 1) / 2`.
- **Static engine** (one prompt length): at least the DP optimum (the minimum total batch time) divided by `W`, and at least the cost of a batch of one sample with the longest response (`docs/optimization.md`).

The benchmark reports `lb_gap = gen_time / max(bounds) - 1` for every single-phase run.

## 8. Trace generator (assumed distributions)

`trace/generator.py`; each quantity draws from its own stream, so changing the estimate model or the verifier times never changes the lengths of a seed.

| Quantity | Model | Default (assumed) |
|---|---|---|
| task mix | categorical | `math` 0.6, `code` 0.4 (S5: `code` 0.7); S1 and S2 use `math` only |
| prompt length | log-normal per task, rounded up, clipped to `[32, 4096]` | median 300 (`math`), 600 (`code`), log-sd 0.5; S1 and S7: one length, 512 |
| group size | fixed | 8 samples |
| response length | `log L = log(median) + sigma * (sqrt(rho) * z_group + sqrt(1 - rho) * e_sample)`, rounded up, clipped to `[1, max_tokens]` | median 700 (`math`), 900 (`code`); heavy tail `sigma = 1.1` (P99/P50 > 8), light tail `sigma = 0.4` (< 3); within-group correlation of log lengths `rho = 0.6`; `max_tokens = 8192` |
| estimate `est_tokens` | `none`: the task mean of the clipped length; `group`: the latent group mean (conditional mean of the clipped length given `z_group`) times `exp(error_sigma * u)` | `group` with `error_sigma = 0.3` (S1: 0.5 and 0; SENS: 1.0 and `none`) |
| verifier time | log-normal per task, capped, whole ms | `math`: median 0.2 s, log-sd 0.3, cap 10 s; `code`: median 2 s, log-sd 1.0, cap 60 s (a sandbox running tests) |

Distribution shift (Tier 2, S8): with `shift = {at_group, len_sigma_factor, error_sigma}`, the groups from position `at_group` on draw their log lengths with the spread multiplied by `len_sigma_factor` (same latent draws) and/or their estimate error with level `error_sigma`; their estimates keep the pre-shift calibration. Without `shift` the traces are byte-identical to before.

`verify_mean_s` in the system configuration is the uncapped log-normal mean `median * exp(sigma^2 / 2)` (0.209 s and 3.297 s), the only verifier information policies get. Check 10 tests every row of this table.

## 9. Default system (derived from hardware and model numbers)

`config.derive_system`; values printed by `tests/test_config.py`.

| Input | Value | Kind and source |
|---|---|---|
| GPU memory, bandwidth, dense BF16 | 80 GB, 3.35 TB/s, 989 TFLOP/s | datasheet of an 80 GB data-center GPU (H100 SXM class), transcribed |
| model | 7.6 B parameters, 2 bytes each | a public dense 7B-class model card |
| KV cache per token | 28 layers × 4 KV heads × 128 × 2 (K, V) × 2 B = 57 344 B | architecture of a public 7B grouped-query-attention model |
| `eff_mem_bw`, `eff_flops` | 0.7, 0.5 | assumed |
| iteration overhead, per-sequence overhead | 1 ms, 20 µs | assumed |
| `mem_util`, activation reserve | 0.9, 2 GB per GPU | assumed |
| `max_seqs`, `static_batch` | 32, 16 | assumed engine settings |
| training MFU, `fixed_ms` | 0.35, 2000 ms | assumed |
| `sync_ms`, `swap_ms`, `switch_ms` | 1000, 500, 3000 ms | assumed |
| GPUs | 16: 8 rollout (`tp = 1`) + 8 training; colocated: 16 | assumed (S6 varies the split) |
| loop | `T = 24` (12 in the quick run), `B = 64` groups of 8, `eta = 1`, `warmup_steps = 2`, `inflight = drain` | assumed |
| verifier servers | 32 | assumed (S5 varies 4–32) |

Derived (`a*` and `prefill*` rounded to whole ns):

```text
a0 = weights / (bandwidth * eff_mem_bw) + iteration overhead = 15.2 GB / 2.345 TB/s + 1 ms = 7 481 876 ns
a1 = 2 * params / (peak * eff_flops) + per-sequence overhead  = 30 738 ns + 20 000 ns     = 50 738 ns
a2 = KV bytes per token / (bandwidth * eff_mem_bw)            = 57 344 / 2.345e12 s       = 24 ns
prefill_c0 = a0 (one weight pass), prefill_c1 = 2 * params / (peak * eff_flops)          = 30 738 ns
kv_tokens = (memory * mem_util - weights - reserve) / KV bytes per token                  = 955 636
ns_per_token (trainer) = 6 * params / (training GPUs * peak * MFU)                        = 16 467 ns (8 GPUs)
```

KV capacity does not bind in the default system (`32 * (8192 + 4096) < 955 636`); `max_seqs` does. KV-limited admission is exercised by the tests. Two properties the benchmark needs hold under the reference policy (tuning seeds, window steps): generation of a batch takes 5.0 to 5.9 times as long as its training (89–104 s against 18 s), and a worker gets 64 samples per step, twice `max_seqs`.

The linear iteration model sums memory and compute time instead of taking a roofline maximum; it has not been fitted to any measurement (calibration is a Tier 2 item and was not done).

## 10. What the model leaves out

No preemption or swapping of KV cache (reservations are worst-case `max_tokens`); no chunked prefill (a prefill pauses the worker); no speculative decoding; one model per worker; no network or weight-transfer model beyond the fixed `sync_ms`, `swap_ms`, `switch_ms`; no worker failures in the simulator; no learning. The trainer's cost is linear in tokens (the 6ND rule) with a fixed per-step overhead.
