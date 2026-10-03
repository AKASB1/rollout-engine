"""Timeline figure of closed-loop runs: trainer and worker lanes over two steps.

    python scripts/plot_timeline.py [--out docs/figures]

Runs the simulator (default system, eta = 1, 6 steps, evaluation seed 1) for the reference
policy and for longest_first+sample+late+lpt, and draws each GPU lane's state (decoding,
overhead, training, idle) between the selections of steps 2 and 4. Simulated, assumed
parameters.
"""

from __future__ import annotations

import argparse
import copy
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from rollout_engine.config import derive_system, parse_system  # noqa: E402
from rollout_engine.policies.composed import make_policy  # noqa: E402
from rollout_engine.sim.driver import simulate  # noqa: E402
from rollout_engine.trace.generator import DEFAULT_GENERATOR, generate, verify_mean_s  # noqa: E402

COL = {
    "gen": "#2a78d6",
    "overhead": "#eb6834",
    "train": "#1baf7a",
    "idle": "#e6e5e0",
    "parked": "#e6e5e0",
    "wait": "#e6e5e0",
    "sync": "#eb6834",
    "switch": "#eb6834",
}
INK2, SURF = "#52514e", "#fcfcfb"


def lanes(core, a, b):
    out = [("trainer", core.trainer_tl)]
    out += [(f"worker {w}", tl) for w, tl in enumerate(core.worker_tl)]
    bars = []
    for name, tl in out:
        segs = []
        for k, (t, st) in enumerate(tl):
            t2 = tl[k + 1][0] if k + 1 < len(tl) else core.t_end
            lo, hi = max(t, a), min(t2, b)
            if hi > lo:
                segs.append(((lo - a) / 1000, (hi - lo) / 1000, COL.get(st, "#e6e5e0")))
        bars.append((name, segs))
    return bars


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "figures"))
    a = ap.parse_args(argv)
    gen = copy.deepcopy(DEFAULT_GENERATOR)
    gen["groups"] = 3 * 6 * 64
    sysc = parse_system(derive_system(steps=6, eta=1, verify_mean_s=verify_mean_s(gen)))
    tr = generate(gen, 1)
    runs = [
        ("reference: fifo+group+early+round_robin (even split)", "reference"),
        ("longest_first+sample+late+lpt", "lpt"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), facecolor=SURF)
    for ax, (title, name) in zip(axes, runs, strict=True):
        core = simulate(sysc, tr, make_policy({"name": name})).core
        t0, t1 = core.steps[2].t_sel, core.steps[4].t_sel
        bars = lanes(core, t0, t1)
        for i, (_lane, segs) in enumerate(bars):
            ax.broken_barh(
                [(x, w) for x, w, _ in segs],
                (i - 0.4, 0.8),
                facecolors=[c for _, _, c in segs],
                linewidth=0,
            )
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([n for n, _ in bars], fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline((core.steps[3].t_sel - t0) / 1000, color=INK2, lw=1, ls="--")
        ax.text(
            (core.steps[3].t_sel - t0) / 1000,
            -0.9,
            " step 3 selected",
            fontsize=7,
            color=INK2,
            va="bottom",
        )
        ax.set_xlim(0, (t1 - t0) / 1000)
        ax.set_facecolor(SURF)
        ax.set_title(f"{title}: steps 2-3 take {(t1 - t0) / 1000:.0f} s", fontsize=9, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("seconds after the selection of step 2")
    axes[0].legend(
        handles=[
            Patch(color=COL["gen"], label="decoding"),
            Patch(color=COL["overhead"], label="prefill / swap / sync"),
            Patch(color=COL["train"], label="training"),
            Patch(color=COL["idle"], label="idle / waiting"),
        ],
        fontsize=7.5,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0, 1.08),
        frameon=False,
    )
    fig.text(
        0.01,
        0.005,
        "simulated, assumed parameters: default system (8 rollout + 8 training GPUs, B = 64 groups x 8, eta = 1, drain), seed 1",
        fontsize=7,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "timeline.png")
    fig.savefig(path, dpi=110)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
