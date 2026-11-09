"""Result figures and tables from the committed benchmark results.

    python scripts/plot_results.py [--results benchmarks/results/full] [--out docs/figures]

Reads aggregate.csv, paired.csv, s7.csv, s9.csv, and wall_s7_sweep.csv, and the S9 ranges in
../s9_ties/summary.csv when present; writes PNG figures (each below 300 KB) and ``tables.md``
beside the results. Everything shown is simulated with assumed parameters. Colours: a fixed categorical order (blue, orange, aqua, yellow, magenta,
green, violet, red), never cycled; identity is also carried by markers and labels.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARK = ["o", "s", "D", "^", "v", "P", "X", "*"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


class Results:
    def __init__(self, root):
        self.root = root
        self.agg = {}
        for r in read(os.path.join(root, "aggregate.csv")):
            self.agg[(r["scenario"], r["variant"], r["policy"], r["metric"])] = (
                fnum(r["mean"]),
                fnum(r["ci95"]),
                int(r["n"]),
            )
        self.paired = {}
        for r in read(os.path.join(root, "paired.csv")):
            self.paired[(r["scenario"], r["variant"], r["policy"], r["metric"])] = r
        self.s7 = read(os.path.join(root, "s7.csv"))
        self.s9 = read(os.path.join(root, "s9.csv"))
        self.tuned_lam = None
        tpath = os.path.join("configs", "tuned", "tuned.json")
        if os.path.exists(tpath):
            import json

            with open(tpath, encoding="utf-8") as f:
                self.tuned_lam = json.load(f).get("s9", {}).get("lam")
        self.sweep = read(os.path.join(root, "wall_s7_sweep.csv"))
        # ranges over the plans that are optimal for each S9 planning model (scripts/s9_tie_ranges.py)
        ties = os.path.join(os.path.dirname(os.path.normpath(root)), "s9_ties", "summary.csv")
        self.s9_ties = {r["model"]: r for r in read(ties) if r.get("set") == "evaluation"}
        if self.s9 and not self.s9_ties:
            print(f"note: {ties} not found; the S9 figure and table show the returned plans only")
        for key, tie in list(self.s9_ties.items()):  # the ranges must belong to the same evaluation
            g = [
                fnum(r[f"{key}_ratio"])
                for r in self.s9
                if r.get(f"{key}_ratio") not in (None, "", "nan")
            ]
            if not g or abs(sum(g) / len(g) - fnum(tie["ratio_returned"])) > 5e-5:
                print(f"warning: {ties} does not match s9.csv for {key}; S9 ranges left out")
                self.s9_ties = {}
                break

    def m(self, s, v, p, metric):
        return self.agg.get((s, v, p, metric), (math.nan, math.nan, 0))

    def variants(self, s):
        return sorted({k[1] for k in self.agg if k[0] == s})

    def policies(self, s, v):
        return sorted({k[2] for k in self.agg if k[0] == s and k[1] == v})


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    kb = os.path.getsize(path) / 1024
    if kb > 300:
        raise SystemExit(f"{name} is {kb:.0f} KB (> 300 KB)")
    print(f"wrote {path} ({kb:.0f} KB)")


def note(
    fig, text="simulated, assumed parameters; 95 % Student-t intervals across seeds", top=0.93
):
    fig.tight_layout(rect=(0, 0.06, 1, top))
    fig.text(0.01, 0.01, text, fontsize=7, color=INK2, ha="left", va="bottom")


# ----------------------------------------------------------------------------- S2
def fig_s2(R, out):
    cells = [("light", "none"), ("light", "group"), ("heavy", "none"), ("heavy", "group")]
    cells = [c for c in cells if f"{c[0]}/{c[1]}" in R.variants("S2")]
    if not cells:
        return
    ref_key = f"{cells[-1][0]}/{cells[-1][1]}"
    pols = [p for p in R.policies("S2", ref_key) if p != "reference"]

    def rel(v, p):
        pr = R.paired.get(("S2", v, p, "phase_time_s"))
        ref = R.m("S2", v, "reference", "phase_time_s")[0]
        if not pr or not ref:
            return math.nan, math.nan
        return 100 * fnum(pr["diff_mean"]) / ref, 100 * fnum(pr["diff_ci"]) / ref

    pols.sort(key=lambda p: rel(ref_key, p)[0])
    fig, axes = plt.subplots(
        1, len(cells), figsize=(2.3 * len(cells) + 2.2, 0.24 * len(pols) + 1.4), sharey=True
    )
    axes = list(axes) if len(cells) > 1 else [axes]
    y = list(range(len(pols)))
    for ax, (tail, est) in zip(axes, cells, strict=True):
        v = f"{tail}/{est}"
        xs = [rel(v, p) for p in pols]
        for k, p in enumerate(pols):
            c = CAT[2] if p.startswith("oracle") else (CAT[1] if p == "online_adaptive" else CAT[0])
            ax.errorbar(
                xs[k][0],
                k,
                xerr=xs[k][1] if not math.isnan(xs[k][1]) else None,
                fmt="o",
                ms=4,
                color=c,
                elinewidth=1.2,
                capsize=0,
            )
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(f"{tail} tail, estimates: {est}", fontsize=8.5)
        ax.set_xlabel("phase time vs reference (%)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(pols, fontsize=7)
    fig.suptitle(
        "S2: one phase, continuous engine, 64 groups x 8 on 8 workers (lower is better)",
        fontsize=9.5,
        x=0.02,
        ha="left",
    )
    note(
        fig,
        "simulated, assumed parameters; paired differences against the reference (fifo+group+early+round_robin), 95 % Student-t intervals; aqua = oracle, orange = online_adaptive",
        top=0.9,
    )
    save(fig, out, "s2_policies.png")


# ----------------------------------------------------------------------------- S1
def fig_s1(R, out):
    vs = R.variants("S1")
    if not vs:
        return
    caps = sorted({v.split("/")[0] for v in vs}, key=lambda c: int(c[3:]))
    ests = ["none", "group_moderate", "group_exact"]
    pols = ["reference", "fifo_chunk", "sorted_chunk", "dp", "oracle_dp"]
    fig, axes = plt.subplots(1, len(caps), figsize=(4.2 * len(caps), 3.0), sharey=True)
    axes = list(axes) if len(caps) > 1 else [axes]
    for ax, cap in zip(axes, caps, strict=True):
        for k, p in enumerate(pols):
            xs = [i + (k - 2) * 0.12 for i in range(len(ests))]
            ms = [R.m("S1", f"{cap}/{e}", p, "lb_gap") for e in ests]
            ax.errorbar(
                xs,
                [100 * m[0] for m in ms],
                yerr=[100 * m[1] if not math.isnan(m[1]) else 0 for m in ms],
                fmt=MARK[k],
                ms=5,
                color=CAT[k],
                label=p,
                elinewidth=1.2,
                capsize=0,
                ls="none",
            )
        ax.set_xticks(range(len(ests)))
        ax.set_xticklabels(["no estimate", "group, error 0.5", "exact group mean"])
        ax.set_title(f"static engine, batch cap {cap[3:]}", fontsize=8.5)
        ax.set_ylabel("generation time above the lower bound (%)")
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        "S1: batch formation, one phase, 64 samples on 4 workers (lower is better)",
        fontsize=9.5,
        x=0.02,
        ha="left",
    )
    note(fig, top=0.9)
    save(fig, out, "s1_batching.png")


# ----------------------------------------------------------------------------- S3 / S4
def short(p):
    return (
        p.replace("best_s2+", "best ")
        .replace("online_adaptive+", "online ")
        .replace("reference", "reference")
    )


def fig_s3_s4(R, out):
    v3 = [v for v in R.variants("S3") if "-" not in v]
    v4 = R.variants("S4")
    if not v3 and not v4:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    if v3:
        ax = axes[0]
        etas = sorted(int(v[3:]) for v in v3)
        for k, p in enumerate(["reference", "best_s2+wait", "best_s2+carry"]):
            ms = [R.m("S3", f"eta{e}", p, "step_time_mean_s") for e in etas]
            ax.errorbar(
                etas,
                [m[0] for m in ms],
                yerr=[m[1] if not math.isnan(m[1]) else 0 for m in ms],
                marker=MARK[k],
                ms=5,
                color=CAT[k],
                label=short(p),
                capsize=0,
            )
            ax.annotate(
                short(p),
                (etas[-1], ms[-1][0]),
                textcoords="offset points",
                xytext=(6, 0),
                fontsize=7,
                color=INK2,
                va="center",
            )
        ax.set_xticks(etas)
        ax.set_xlabel("staleness bound eta")
        ax.set_ylabel("mean step time (s)")
        ax.set_title("S3: step time against the staleness bound", fontsize=8.5)
        ax.legend(fontsize=7, loc="upper right")
    if v4:
        ax = axes[1]
        v = v4[0]
        pols = R.policies("S4", v)
        for p in pols:
            fam = 0 if p == "reference" else (1 if p.startswith("best_s2") else 2)
            kind = "o" if p.endswith("wait") or p == "reference" else ("s" if "carry" in p else "^")
            x, y = R.m("S4", v, p, "waste_frac")[0], R.m("S4", v, p, "step_time_mean_s")
            ax.errorbar(
                100 * x,
                y[0],
                yerr=y[1] if not math.isnan(y[1]) else None,
                fmt=kind,
                ms=5,
                color=CAT[fam],
                capsize=0,
            )
            ax.annotate(
                short(p),
                (100 * x, y[0]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=6.3,
                color=INK2,
            )
        ax.set_xlabel("waste: tokens of dropped groups (%)")
        ax.set_ylabel("mean step time (s)")
        ax.set_title(
            f"S4: stragglers at {v.replace('eta', 'eta = ')} (circle wait, square carry, triangle abort)",
            fontsize=8.5,
        )
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D([], [], color=CAT[i], marker="o", ls="none", label=lab)
                for i, lab in enumerate(["reference", "best S2 policy", "online_adaptive"])
            ],
            fontsize=7,
            loc="upper right",
        )
    if not v3:
        axes[0].set_visible(False)
    if not v4:
        axes[1].set_visible(False)
    note(fig, top=1.0)
    save(fig, out, "s3_s4_staleness_stragglers.png")


# ----------------------------------------------------------------------------- S5 / S6
def fig_s5_s6(R, out):
    v5 = R.variants("S5")
    v6 = R.variants("S6")
    if not v5 and not v6:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    if v5:
        ax = axes[0]
        servers = sorted(int(v[7:]) for v in v5)
        pols = R.policies("S5", f"servers{servers[0]}")
        for k, p in enumerate(pols):
            ms = [R.m("S5", f"servers{s}", p, "step_time_mean_s") for s in servers]
            ax.errorbar(
                servers,
                [m[0] for m in ms],
                yerr=[m[1] if not math.isnan(m[1]) else 0 for m in ms],
                marker=MARK[k],
                ms=5,
                color=CAT[k],
                label=short(p),
                capsize=0,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(servers)
        ax.set_xticklabels([str(s) for s in servers])
        ax.set_xlabel("verifier servers (70 % code tasks)")
        ax.set_ylabel("mean step time (s)")
        ax.set_title("S5: verifier capacity and order", fontsize=8.5)
        ax.legend(fontsize=7)
    if v6:
        ax = axes[1]
        splits = sorted(int(v[7:]) for v in v6 if v.startswith("rollout"))
        for k, p in enumerate(["reference", "best_s2+wait"]):
            ms = [R.m("S6", f"rollout{r}", p, "step_time_mean_s") for r in splits]
            ax.errorbar(
                splits,
                [m[0] for m in ms],
                yerr=[m[1] if not math.isnan(m[1]) else 0 for m in ms],
                marker=MARK[k],
                ms=5,
                color=CAT[k],
                label=f"{short(p)}, disaggregated",
                capsize=0,
            )
            c = R.m("S6", "colocated", p, "step_time_mean_s")
            if not math.isnan(c[0]):
                ax.axhline(
                    c[0], color=CAT[k], lw=1.2, ls="--", label=f"{short(p)}, colocated (16 GPUs)"
                )
        ax.set_xticks(splits)
        ax.set_xlabel("rollout GPUs of 16 (the rest train)")
        ax.set_ylabel("mean step time (s)")
        ax.set_title("S6: partition of 16 GPUs", fontsize=8.5)
        ax.legend(fontsize=7)
    if not v5:
        axes[0].set_visible(False)
    if not v6:
        axes[1].set_visible(False)
    note(fig, top=1.0)
    save(fig, out, "s5_s6_verifier_partition.png")


# ----------------------------------------------------------------------------- S7
def fig_s7(R, out):
    if not R.s7:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    names = ["fifo_chunk", "sorted_chunk", "dp", "sorted_chunk_oracle", "oracle_dp"]
    for k, nme in enumerate(names):
        g = [
            100 * fnum(r[f"{nme}_gap"])
            for r in R.s7
            if r.get(f"{nme}_gap") not in (None, "", "nan")
        ]
        if not g:
            continue
        mu = sum(g) / len(g)
        sd = math.sqrt(sum((x - mu) ** 2 for x in g) / max(1, len(g) - 1))
        ax.barh(
            k,
            mu,
            height=0.55,
            color=CAT[k],
            xerr=1.96 * sd / math.sqrt(len(g)),
            error_kw={"elinewidth": 1, "ecolor": INK2},
        )
        ax.annotate(
            f"{mu:.1f} %",
            (mu, k),
            textcoords="offset points",
            xytext=(18, -3),
            fontsize=7,
            color=INK2,
        )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [
            "fifo_chunk",
            "sorted_chunk (estimates)",
            "dp (estimates)",
            "sorted_chunk (true lengths, oracle)",
            "dp (true lengths, oracle)",
        ],
        fontsize=7,
    )
    ax.set_xlabel("makespan above the MILP optimum (%), mean over instances")
    ax.set_title(f"S7: gap to the exact optimum ({len(R.s7)} instances)", fontsize=8.5)
    ax = axes[1]
    if R.sweep:
        by = defaultdict(list)
        for r in R.sweep:
            by[int(r["n"])].append(
                (fnum(r["wall_solve_s"]), fnum(r["wall_solve_cpu_s"]), r["status"])
            )
        ns = sorted(by)
        ax.plot(
            ns,
            [max(x[0] for x in by[n]) for n in ns],
            marker="o",
            ms=5,
            color=CAT[0],
            label="wall, slowest of 3 seeds",
        )
        ax.plot(
            ns,
            [max(x[1] for x in by[n]) for n in ns],
            marker="s",
            ms=5,
            color=CAT[1],
            label="process CPU, slowest of 3",
        )
        for n in ns:
            if any(x[2] != "optimal" for x in by[n]):
                ax.annotate(
                    "not optimal",
                    (n, max(x[0] for x in by[n])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=INK2,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("samples n (W = 3, cap 6)")
        ax.set_ylabel("MILP solve time (s)")
        ax.set_title("S7: MILP solve time (HiGHS via SciPy; shared machine)", fontsize=8.5)
        ax.legend(fontsize=7)
    if not R.sweep:
        axes[1].set_visible(False)
    note(
        fig,
        "simulated, assumed parameters; clairvoyant references; solve times are wall-clock on a shared machine",
        top=1.0,
    )
    save(fig, out, "s7_exact_references.png")


# ----------------------------------------------------------------------------- SENS
def fig_sens(R, out):
    vs = R.variants("SENS")
    if not vs:
        return
    rows = []
    for v in vs:
        for p in R.policies("SENS", v):
            pr = R.paired.get(("SENS", v, p, "J"))
            if pr:
                rows.append((v, p, fnum(pr["diff_mean"]), fnum(pr["diff_ci"])))
    if not rows:
        return
    pols = sorted({r[1] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(vs) * 1.0 + 2.2))
    for k, p in enumerate(pols):
        pts = [
            (vs.index(r[0]) + (k - (len(pols) - 1) / 2) * 0.18, r[2], r[3])
            for r in rows
            if r[1] == p
        ]
        ax.errorbar(
            [x[1] for x in pts],
            [x[0] for x in pts],
            xerr=[x[2] if not math.isnan(x[2]) else 0 for x in pts],
            fmt=MARK[k],
            ms=4.5,
            color=CAT[k],
            label=short(p),
            capsize=0,
            ls="none",
        )
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels(vs, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("J minus the reference's J (paired, lower is better)")
    ax.set_title("SENS: does the ranking survive other assumptions?", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    note(fig, top=1.0)
    save(fig, out, "sensitivity.png")


# ----------------------------------------------------------------------------- S9 (Tier 2)
S9_PLANS = [("det", "MILP on estimates"), ("dp_policy", "dp policy (re-plans)")]


def s9_plans(R):
    if not R.s9:
        return []
    lams = sorted(
        {k[4:-6] for k in R.s9[0] if k.startswith("cvar") and k.endswith("_ratio")}, key=float
    )
    tuned = R.tuned_lam
    plans = list(S9_PLANS)
    for lam in lams:
        tag = " (tuned)" if tuned is not None and float(lam) == tuned else ""
        plans.append((f"cvar{lam}", f"scenario CVaR, lam {lam}{tag}"))
    return plans


def fig_s9(R, out):
    plans = s9_plans(R)
    if not plans:
        return
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(plans) + 1.3))
    for k, (key, lab) in enumerate(plans):
        g = [
            100 * (fnum(r[f"{key}_ratio"]) - 1)
            for r in R.s9
            if r.get(f"{key}_ratio") not in (None, "", "nan")
        ]
        if not g:
            continue
        mu = sum(g) / len(g)
        sd = math.sqrt(sum((x - mu) ** 2 for x in g) / max(1, len(g) - 1))
        tuned = "tuned" in lab
        ax.barh(
            k,
            mu,
            height=0.55,
            color=CAT[1] if tuned else (CAT[0] if key.startswith("cvar") else CAT[6]),
            xerr=1.96 * sd / math.sqrt(len(g)),
            error_kw={"elinewidth": 1, "ecolor": INK2},
        )
        right = mu + 1.96 * sd / math.sqrt(len(g))
        tie = R.s9_ties.get(key)
        if tie:
            lo, hi = 100 * (fnum(tie["ratio_best"]) - 1), 100 * (fnum(tie["ratio_worst"]) - 1)
            ax.plot([lo, hi], [k + 0.36, k + 0.36], color=INK, lw=1.4, solid_capstyle="butt")
            ax.plot(
                [lo, lo, hi, hi],
                [k + 0.28, k + 0.44, k + 0.28, k + 0.44],
                ls="none",
                marker="|",
                ms=5,
                color=INK,
            )
            right = max(right, hi)
        ax.annotate(
            f"{mu:.1f} %"
            + (f"  (optimal plans: {lo:.1f}-{hi:.1f} %)" if tie and hi - lo >= 0.05 else ""),
            (right, k),
            textcoords="offset points",
            xytext=(18, -3),
            fontsize=7,
            color=INK2,
        )
    ax.set_yticks(range(len(plans)))
    ax.set_yticklabels([lab for _, lab in plans], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("realized makespan above the clairvoyant optimum (%), mean over instances")
    ax.set_title(
        f"S9 (Tier 2): plans made on estimates, replayed on true lengths ({len(R.s9)} instances)",
        fontsize=9,
        loc="left",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1.0))
    fig.text(
        0.01,
        0.01,
        "simulated, assumed parameters; static engine; lam chosen on the tuning instances. MILP plans: the bar is the plan HiGHS\n"
        "returned, one of the plans that are optimal for the model (95 % normal interval across instances); the black line runs\n"
        "from the best to the worst of those plans, each averaged over the instances (benchmarks/results/s9_ties).\n"
        "dp policy: online re-planning, no solver choice.",
        fontsize=7,
        color=INK2,
        ha="left",
        va="bottom",
    )
    save(fig, out, "s9_risk.png")


# ----------------------------------------------------------------------------- S10 (Tier 2)
def fig_s10(R, out):
    vs = R.variants("S10")
    if not vs:
        return
    v = vs[0]
    pols = R.policies("S10", v)
    rhos = sorted({p.split("carry(")[1].split(")")[0] for p in pols if "carry(" in p}, key=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, (suffix, lab) in enumerate(
        [("", "greedy carry"), ("+deadline", "deadline-aware carry")]
    ):
        xs, ys, es = [], [], []
        for r in rhos:
            p = f"best_s2+carry({r}){suffix}"
            xs.append(100 * R.m("S10", v, p, "waste_frac")[0])
            m = R.m("S10", v, p, "step_time_mean_s")
            ys.append(m[0])
            es.append(m[1] if not math.isnan(m[1]) else 0)
        ax.errorbar(xs, ys, yerr=es, marker=MARK[k], ms=5, color=CAT[k], label=lab, capsize=0)
        for x, y, r in zip(xs, ys, rhos, strict=True):
            ax.annotate(
                f"rho {r}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=6.5,
                color=INK2,
            )
    for k, p in enumerate(["best_s2+wait", "reference"]):
        m = R.m("S10", v, p, "step_time_mean_s")
        if not math.isnan(m[0]):
            ax.errorbar(
                [100 * R.m("S10", v, p, "waste_frac")[0]],
                [m[0]],
                yerr=[m[1] if not math.isnan(m[1]) else 0],
                fmt=MARK[k + 2],
                ms=6,
                color=CAT[k + 2],
                label=short(p),
                capsize=0,
            )
    ax.set_xlabel("waste: tokens of dropped groups (%)")
    ax.set_ylabel("mean step time (s)")
    ax.set_title(
        f"S10 (Tier 2): admission under the staleness bound, {v.replace('eta', 'eta = ')}",
        fontsize=9,
        loc="left",
    )
    ax.legend(fontsize=7)
    note(fig, top=1.0)
    save(fig, out, "s10_deadline.png")


# ----------------------------------------------------------------------------- S8 (Tier 2)
def fig_s8(R, out):
    vs = R.variants("S8")
    if not vs:
        return
    order = [v for v in vs if v.endswith("/none")] + [v for v in vs if not v.endswith("/none")]
    pols = R.policies("S8", order[0])
    pols.sort(key=lambda p: R.m("S8", order[0], p, "J")[0])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(pols) + 1.4))
    for k, v in enumerate(order):
        ms = [R.m("S8", v, p, "step_time_mean_s") for p in pols]
        ys = [i + (k - 1) * 0.22 for i in range(len(pols))]
        ax.errorbar(
            [m[0] for m in ms],
            ys,
            xerr=[m[1] if not math.isnan(m[1]) else 0 for m in ms],
            fmt=MARK[k],
            ms=5,
            color=CAT[k],
            label=v.split("/")[1].replace("_", " "),
            capsize=0,
            ls="none",
        )
    ax.set_yticks(range(len(pols)))
    ax.set_yticklabels([short(p) for p in pols], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean step time (s); waste and length bias are in the table")
    ax.set_title(
        "S8 (Tier 2): distribution shift half way through the stream, eta = 2",
        fontsize=9,
        loc="left",
    )
    ax.legend(fontsize=7, loc="lower right", title="shift", title_fontsize=7)
    note(fig, top=1.0)
    save(fig, out, "s8_shift.png")


# ----------------------------------------------------------------------------- tables
def fmt(m, digits=2, pct=False):
    mu, ci, _ = m
    if math.isnan(mu):
        return "–"
    if pct:
        return f"{100 * mu:.{max(0, digits - 1)}f} %" + (
            f" ± {100 * ci:.{max(0, digits - 1)}f}" if not math.isnan(ci) else ""
        )
    return f"{mu:.{digits}f}" + (f" ± {ci:.{digits}f}" if not math.isnan(ci) else "")


def wtl(R, s, v, p):
    pr = R.paired.get((s, v, p, "J"))
    return f"{pr['wins']}/{pr['ties']}/{pr['losses']}" if pr else "–"


def tables(R) -> str:
    out = [
        "<!-- generated by scripts/plot_results.py from benchmarks/results; simulated, assumed parameters -->"
    ]
    closed = [
        ("S3", "eta1"),
        ("S3", "eta0"),
        ("S4", "eta2"),
        ("S5", "servers8"),
        ("S6", "colocated"),
    ]
    out.append(
        "\n**Closed loop** (mean ± 95 % CI across seeds; W/T/L = paired wins/ties/losses on J against the reference, tie band 1 %)\n"
    )
    out.append(
        "| Scenario | Policy | J | step time (s) | GPU idle | P95 straggler (s) | waste | length bias | staleness | W/T/L |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for s, v in closed:
        for p in R.policies(s, v):
            out.append(
                f"| {s} {v} | {p} | {fmt(R.m(s, v, p, 'J'), 3)} | {fmt(R.m(s, v, p, 'step_time_mean_s'), 1)} | {fmt(R.m(s, v, p, 'gpu_idle_frac'), 1, True)} | "
                f"{fmt(R.m(s, v, p, 'straggler_p95_s'), 1)} | {fmt(R.m(s, v, p, 'waste_frac'), 1, True)} | {fmt(R.m(s, v, p, 'length_bias'), 1, True)} | "
                f"{fmt(R.m(s, v, p, 'staleness_mean'), 2)} | {wtl(R, s, v, p)} |"
            )
    out.append("\n**One phase, S2 heavy tail with group estimates** (every policy)\n")
    out.append("| Policy | J | phase time (s) | gap to lower bound | GPU idle | W/T/L |")
    out.append("|---|---|---|---|---|---|")
    v = "heavy/group"
    pols = sorted(R.policies("S2", v), key=lambda p: R.m("S2", v, p, "J")[0])
    for p in pols:
        out.append(
            f"| {p} | {fmt(R.m('S2', v, p, 'J'), 3)} | {fmt(R.m('S2', v, p, 'phase_time_s'), 1)} | {fmt(R.m('S2', v, p, 'lb_gap'), 1, True)} | {fmt(R.m('S2', v, p, 'gpu_idle_frac'), 1, True)} | {wtl(R, 'S2', v, p)} |"
        )
    if R.variants("S8"):
        out.append(
            "\n**S8 (Tier 2): distribution shift half way through the stream** "
            "(eta = 2; J is normalized within each shift cell, so compare J within a cell; "
            "step-time change = relative to the same policy without shift)\n"
        )
        out.append(
            "| Shift | Policy | J | step-time change | step time (s) | waste | length bias |"
        )
        out.append("|---|---|---|---|---|---|---|")
        base = next(v for v in R.variants("S8") if v.endswith("/none"))
        for v in [base] + [x for x in R.variants("S8") if x != base]:
            for p in sorted(R.policies("S8", v), key=lambda p, v=v: R.m("S8", v, p, "J")[0]):
                d = (
                    R.m("S8", v, p, "step_time_mean_s")[0]
                    / R.m("S8", base, p, "step_time_mean_s")[0]
                    - 1
                )
                ch = "–" if v == base else f"{100 * d:+.1f} %"
                out.append(
                    f"| {v.split('/')[1]} | {p} | {fmt(R.m('S8', v, p, 'J'), 3)} | {ch} | "
                    f"{fmt(R.m('S8', v, p, 'step_time_mean_s'), 1)} | "
                    f"{fmt(R.m('S8', v, p, 'waste_frac'), 1, True)} | {fmt(R.m('S8', v, p, 'length_bias'), 1, True)} |"
                )
    if R.variants("S6"):
        out.append(
            "\n**S6 partition planning**: the GPU split with the lowest mean step time, per policy (16 GPUs)\n"
        )
        out.append("| Policy | best split | its step time (s) | colocated step time (s) |")
        out.append("|---|---|---|---|")
        splits = [v for v in R.variants("S6") if v.startswith("rollout")]
        for p in sorted(R.policies("S6", splits[0]) if splits else []):
            best = min(splits, key=lambda v, p=p: R.m("S6", v, p, "step_time_mean_s")[0])
            out.append(
                f"| {p} | {best[7:]} rollout + {16 - int(best[7:])} training GPUs | "
                f"{fmt(R.m('S6', best, p, 'step_time_mean_s'), 1)} | {fmt(R.m('S6', 'colocated', p, 'step_time_mean_s'), 1)} |"
            )
    if R.s9:
        out.append(
            f"\n**S9 (Tier 2): static plans made on estimates, replayed on the true lengths** ({len(R.s9)} evaluation instances)\n"
        )
        out.append(
            "Realized makespan / clairvoyant optimum. A planning MILP can have several optimal plans with different "
            "realized makespans, and the solver (HiGHS) returns one of them: for the MILP plans, the first and the last "
            "column are the plan HiGHS returned, one choice among the optimal plans. The range is the mean over the "
            "instances of the best and of the worst plan that is optimal for the model (`benchmarks/results/s9_ties/`); "
            "the `dp` policy re-plans online and involves no solver choice.\n"
        )
        out.append(
            "| Plan | mean, plan returned by the solver | range over the model's optimal plans | instances with a range | worst instance, returned plan |"
        )
        out.append("|---|---|---|---|---|")
        for key, lab in s9_plans(R):
            g = [
                fnum(r[f"{key}_ratio"])
                for r in R.s9
                if r.get(f"{key}_ratio") not in (None, "", "nan")
            ]
            if g:
                tie = R.s9_ties.get(key)
                rng = (
                    f"{fnum(tie['ratio_best']):.3f}–{fnum(tie['ratio_worst']):.3f}"
                    if tie
                    else ("– (re-plans online)" if key == "dp_policy" else "–")
                )
                cnt = f"{tie['instances_with_range']} of {tie['instances']}" if tie else "–"
                out.append(f"| {lab} | {sum(g) / len(g):.3f} | {rng} | {cnt} | {max(g):.3f} |")
    if R.s7:
        out.append(
            f"\n**S7 exact references** ({len(R.s7)} evaluation instances; statuses: {sorted({r['milp_status'] for r in R.s7})})\n"
        )
        out.append("| Plan | mean gap to the MILP optimum | worst gap |")
        out.append("|---|---|---|")
        for nme in ("fifo_chunk", "sorted_chunk", "dp", "sorted_chunk_oracle", "oracle_dp"):
            g = [
                fnum(r[f"{nme}_gap"]) for r in R.s7 if r.get(f"{nme}_gap") not in (None, "", "nan")
            ]
            if g:
                out.append(f"| {nme} | {100 * sum(g) / len(g):.1f} % | {100 * max(g):.1f} % |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- README version
def readme_s2(R, out):
    """S2 in the default cell (heavy tail, group estimates) at README font sizes (880 px wide at
    110 dpi; text 9.5 pt, titles and axis labels 11 pt; no tool metadata): for each launch order,
    dispatch, and binding the best of its assignments, and the other policies. The four-cell figure
    with every policy stays s2_policies.png."""
    v = "heavy/group"
    if v not in R.variants("S2"):
        return
    pols = [p for p in R.policies("S2", v) if p != "reference"]
    ref = R.m("S2", v, "reference", "phase_time_s")[0]

    def rel(p):
        pr = R.paired.get(("S2", v, p, "phase_time_s"))
        if not pr or not ref:
            return math.nan, math.nan
        return 100 * fnum(pr["diff_mean"]) / ref, 100 * fnum(pr["diff_ci"]) / ref

    # one row per launch order, dispatch, and binding (the best of its assignments), plus the rest
    best = {}
    for p in pols:
        key = p.rsplit("+", 1)[0] if p.count("+") == 3 else p
        if key not in best or rel(p)[0] < rel(best[key])[0]:
            best[key] = p
    pols = sorted(best.values(), key=lambda p: rel(p)[0])
    rc = {"font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 11}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8.0, 0.3 * len(pols) + 1.5))
        for k, p in enumerate(pols):
            m, ci = rel(p)
            c = CAT[2] if p.startswith("oracle") else (CAT[1] if p == "online_adaptive" else CAT[0])
            ax.errorbar(
                m, k, xerr=None if math.isnan(ci) else ci, fmt="o", ms=5, color=c, capsize=0
            )
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(list(range(len(pols))))
        ax.set_yticklabels(pols, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_xlabel("phase time against the reference (%, mean and 95 % interval)")
        ax.set_title("S2, heavy tail with group estimates (simulated; lower is better)", loc="left")
        fig.tight_layout()
        path = os.path.join(out, "readme_s2.png")
        fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None})
        plt.close(fig)
        print(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join("benchmarks", "results", "full"))
    ap.add_argument("--out", default=os.path.join("docs", "figures"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    R = Results(a.results)
    if not R.agg:
        raise SystemExit(f"no results in {a.results}")
    for f in (
        fig_s1,
        fig_s2,
        fig_s3_s4,
        fig_s5_s6,
        fig_s7,
        fig_sens,
        fig_s8,
        fig_s9,
        fig_s10,
        readme_s2,
    ):
        f(R, a.out)
    with open(os.path.join(a.results, "tables.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(tables(R))
    print(f"wrote {os.path.join(a.results, 'tables.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
