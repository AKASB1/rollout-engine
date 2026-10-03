# Exact references for static batching

Code: `src/rollout_engine/batching/__init__.py` (cost model), `src/rollout_engine/opt/` (`dp.py`, `milp.py`, `lpt.py`, `brute.py`, `bounds.py`, `s7.py`). Tests: `tests/test_opt.py` (check 2 and the static part of check 3).

Everything here is **simulated, with assumed parameters**, and the plans are **clairvoyant**: they use the true response lengths of a single phase. They are references that bound what a batch-formation policy could achieve on the static engine of `docs/simulator.md`; they are not deployable policies (the `dp` policy applies the same dynamic program to *estimated* lengths, and `oracle_dp` to the true ones, labelled as an oracle).

## 1. Batch cost model

A static batch of `b` samples with longest prompt `Pmax` and longest response `Lmax` takes, in whole milliseconds,

```text
cost(b, Pmax, Lmax) = ceil_ms(c0 + c1 * b * Pmax)                                   (prefill of the padded prompts)
                    + ceil_ms(sum_{j=0}^{Lmax-1} (a0 + a1 * b + a2 * b * (Pmax + j)))  (Lmax iterations, all b carried)
```

exactly the duration the static engine produces (`static_batch_ms`). It depends on the batch only through `b`, `Pmax`, and `Lmax`, and it is nondecreasing in each of them (every term is, and `ceil` preserves order). With **one prompt length** (S1, S7) the cost is a function `cost(b, L)` of the size and the longest response. Costs are integers in milliseconds, so a makespan is of order 10^3 to 10^5: the model is solved at that scale (section 5).

## 2. Contiguity and the dynamic program

**Claim.** Let the cost of a batch depend only on its size and its longest response and be nondecreasing in the longest response. For any partition of `n` samples into batches of at most `cap` samples there is a partition with the same batch sizes in which every batch is a contiguous run of the samples sorted by length, and no batch costs more than its counterpart.

**Proof (exchange argument in its direct form).** Take any partition with batch sizes `b_1, ..., b_m`, ordered so that their longest responses satisfy `M_1 <= M_2 <= ... <= M_m`. Build the contiguous partition with the same sizes: batch `1'` gets the `b_1` shortest samples, batch `2'` the next `b_2`, and so on. Let `S_k = b_1 + ... + b_k`. The original batches `1..k` hold `S_k` samples, all at most `M_k`, so the `S_k`-th shortest sample, which is the longest member of batch `k'`, is at most `M_k`. Hence `cost(b_k, max(k')) <= cost(b_k, M_k)`: no batch costs more than its counterpart. (Equivalently: swapping a longer member of a batch with a smaller maximum against a shorter member of a batch with a larger maximum never raises either cost; the construction applies all such swaps at once.) Batch `k'` inherits the worker of batch `k`, so the argument holds for **any objective that is monotone in the batch costs**: the total cost (the sum), and the makespan of the batches on `W` workers.

**Dynamic program** (total cost, `opt/dp.py`): over the ascending lengths,

```text
F[0] = 0,   F[j] = min over i in [max(1, j - cap + 1), j] of  F[i-1] + cost(j - i + 1, L_j)
```

`F[n]` is the minimum total batch time (`DP` below), found with `O(n * cap)` cost evaluations; ties keep the largest batch ending at `j`. Check 2 compares it with brute force over **all** set partitions for `n <= 8` under the real cost and three synthetic costs nondecreasing in `Lmax` (`b * L`, `L^2 + b`, and a step function of `L` plus `sqrt(b)`).

With several prompt lengths the cost also depends on `Pmax` and the contiguity claim no longer holds; the `dp` policy then still evaluates each segment with its own `Pmax` (a heuristic, not an optimum).

## 3. MILP for the makespan on W workers

By section 2 an optimal makespan schedule exists whose batches are contiguous segments `[i, j]` of the sorted lengths. With the cost table `C[i][j] = cost(j - i + 1, L_j)` for `j - i + 1 <= cap`:

```text
minimize    T
subject to  sum over segments [i, j] containing k, over workers w, of y[i, j, w] = 1     for every sample k
            sum over segments [i, j] of C[i][j] * y[i, j, w] <= T                           for every worker w
            sum over [i, j] of C[i][j] * y[i, j, w] >= sum over [i, j] of C[i][j] * y[i, j, w+1]   (w < W-1, symmetry breaking)
            y[i, j, w] in {0, 1},  T >= 0
```

The order of batches on one worker does not change its load, so the load constraint is exact; the symmetry-breaking rows only order identical workers by load. The model is exact under the exchange argument. Its lower bound `max(DP / W, cost(1, Lmax))` holds because the total work of any plan is at least `DP`, and the batch holding the longest sample costs at least a batch of that sample alone.

What the model leaves out: one prompt length (no `Pmax` term); estimates are not used (the plan is clairvoyant); all samples are available at time 0 (a single phase, no release times, no closed loop); the order of batches on a worker and idle time between batches (neither changes the makespan here).

## 4. LPT planning and Graham's bounds

`opt/lpt.py`: given a batch plan (for example the DP partition), list scheduling assigns each batch in the given order to the worker that becomes free first; **LPT** first sorts the batches by decreasing cost. For identical workers (Graham, "Bounds on Multiprocessing Timing Anomalies"):

- list scheduling of any order: `makespan <= total / W + (1 - 1/W) * max batch <= (2 - 1/W) * max(total / W, max batch)`;
- LPT: `makespan <= (4/3 - 1/(3W)) * OPT`, where `OPT` is the optimal assignment of the same batches (brute force on small instances in check 3).

## 5. Solver settings

`scipy.optimize.milp` (HiGHS; SciPy 1.13.1 vendors HiGHS 1.2.0, whose banner reports `HiGHS 1.2.0`), with deterministic limits: `mip_rel_gap = 0` (prove optimality), a node limit (`node_limit = 200000`), and a wall-clock `time_limit` only as a safety net. A solve that stops on the time limit is reported with status `solver_capped` and left out of byte-identical comparisons; one that stops on the node limit reports status `node_limit` with its gap. Every solve reports status, objective, dual bound, gap, wall time, and process CPU time. Costs enter in milliseconds: in a prototype of this model, costs in nanoseconds (about 10^9) made HiGHS return a plan 0.8 % above the optimum (its tolerances are absolute), while milliseconds returned the exact one; check 2 verifies the MILP against the DP and brute force at the scale used.

## 6. Experiment S7

`opt/s7.py`, run by the benchmark: 50 tuning and 50 evaluation instances (disjoint seeds), `n` in 8..16 samples of one prompt length, `W` in {2, 3}, `cap` in 4..6, lengths log-normal (heavy tail) with group estimates. Per instance: the MILP optimum, `DP`, the lower bound, and the generation makespan of `fifo_chunk`, `sorted_chunk`, and `dp` (with true and with estimated lengths) in the simulator's single-phase run on the static engine. These policies re-plan the remaining pool whenever a worker becomes idle (the `dp` policy re-runs the DP on what is left and hands the costliest batch first), so they are online list schedules, not the replay of one plan. The gap is `makespan / MILP - 1`, computed only when the MILP was solved to optimality (otherwise the bound is reported). The tuning set (50 instances, seeds 5000-5049) is run by the tuning command (`configs/tuned/s7_tuning.csv`; S7 has no tuned parameter) and the evaluation set (seeds 6000-6049) by the evaluation. A second sweep measures the MILP solve time against `n` until the node limit or the safety cap stops it.

## 7. Risk-aware static batching (Tier 2, experiment S9)

`opt/scenario.py`, `opt/risk.py`. The plans of sections 2–4 are clairvoyant. A deployable planner knows only estimates; on the static engine it must fix batches before the lengths are known. The **scenario MILP** draws `S = 16` length scenarios from the planner's belief (the estimate times `exp(0.5 z)`, the error level of the S7 estimates; fixed seed) and chooses one plan for all of them:

```text
minimize    (1/S) * sum_s T_s  +  lam * (eta + 1 / ((1 - alpha) * S) * sum_s z_s)
subject to  every sample covered once
            sum over segments of C_s[seg] * y[seg, w] <= T_s          for every worker w and scenario s
            z_s >= T_s - eta,  z_s >= 0                               (Rockafellar-Uryasev: eta + mean excess = CVaR_alpha)
            expected loads non-increasing in the worker index        (symmetry)
            y[seg, w] in {0, 1}
```

with `alpha = 0.9` and `C_s[seg] = cost(size, longest length of the segment in scenario s)`. The batches are segments of the **estimate**-sorted order; the scenarios order the samples differently, so contiguity is a restriction here (the plan is optimal within that family, not over all partitions). `lam` is chosen on the 50 tuning instances (`lam` in {0, 0.5, 1, 2}, by the mean realized makespan relative to the optimum) and frozen in `configs/tuned/tuned.json`; the evaluation instances are the S7 evaluation set. Compared, each replayed on the true lengths in the simulator: the deterministic makespan MILP on the estimates (`det`), the scenario plans (`cvar(lam)`), the online `dp` policy (re-planning at every idle worker), and the clairvoyant optimum. Same deterministic limits as section 5.
