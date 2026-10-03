"""Aggregation: objective J, mean with a 95 % Student-t interval, paired differences against
the reference policy, and win/tie/loss counts with a relative tie band."""

from __future__ import annotations

import math

from scipy.stats import t as student_t

CLOSED_TERMS = ("step", "idle", "straggler", "waste", "bias")
SINGLE_TERMS = ("phase", "idle", "padding")


def j_terms(row: dict, kind: str, R: float) -> dict:
    if kind == "single":
        return {
            "phase": row["phase_time_s"] / R,
            "idle": row["gpu_idle_frac"],
            "padding": row.get("padding_frac", 0.0),
        }
    return {
        "step": row["step_time_mean_s"] / R,
        "idle": row["gpu_idle_frac"],
        "straggler": row["straggler_p95_s"] / R,
        "waste": row["waste_frac"],
        "bias": abs(row.get("length_bias", 0.0)),
    }


def weight_vector(weights: dict, kind: str) -> dict:
    if kind == "single":
        return {"phase": weights["alpha"], "idle": weights["beta"], "padding": weights["gamma"]}
    return {
        "step": weights["alpha"],
        "idle": weights["beta"],
        "straggler": weights["gamma"],
        "waste": weights["delta"],
        "bias": weights["epsilon"],
    }


def objective(row: dict, kind: str, R: float, w: dict) -> float:
    terms = j_terms(row, kind, R)
    return sum(w[k] * v for k, v in terms.items())


def mean_ci(xs: list[float], level: float = 0.95) -> tuple[float, float, int]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return math.nan, math.nan, 0
    m = sum(xs) / n
    if n < 2:
        return m, math.nan, n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, float(student_t.ppf(0.5 + level / 2, n - 1)) * sd / math.sqrt(n), n


def paired(ref: dict[int, float], pol: dict[int, float], tie: float) -> dict:
    """Paired differences pol - ref over common seeds; lower is better. A seed is a tie when
    |diff| <= tie * |ref|."""
    seeds = sorted(set(ref) & set(pol))
    diffs = [pol[s] - ref[s] for s in seeds]
    m, ci, n = mean_ci(diffs)
    w = sum(1 for s in seeds if pol[s] - ref[s] < -tie * abs(ref[s]))
    lo = sum(1 for s in seeds if pol[s] - ref[s] > tie * abs(ref[s]))
    return {"diff_mean": m, "diff_ci": ci, "n": n, "wins": w, "ties": n - w - lo, "losses": lo}
