"""Framework figure: one core state machine, two drivers.

python scripts/plot_framework.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=SURF, ec=color, lw=1.8
        )
    )
    ax.text(x + 0.08, y + h - 0.12, title, fontsize=8.5, weight="bold", color=INK, va="top")
    ax.text(x + 0.08, y + h - 0.42, body, fontsize=7, color=INK2, va="top", linespacing=1.35)


def arrow(ax, a, b, text="", color=INK2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.1,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.08, text, fontsize=6.5, color=INK2, ha="center"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "figures"))
    a = ap.parse_args(argv)
    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor=SURF)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0.5, 5.4)
    ax.axis("off")
    box(
        ax,
        0.2,
        3.7,
        2.2,
        1.45,
        "Trace generator",
        "groups of samples, heavy-tailed\nlengths with a group latent,\nestimates, verifier times\n(schema v1 CSV + manifest)",
        YELLOW,
    )
    box(
        ax,
        3.9,
        3.55,
        2.7,
        1.65,
        "Core state machine",
        "groups, samples, worker mirrors,\nverifier queue, trainer, versions,\nstaleness gate + dead-group rule,\naction validation, metrics records",
        BLUE,
    )
    box(
        ax,
        7.9,
        3.7,
        2.4,
        1.45,
        "Policy (parts)",
        "estimator, launch order, dispatch,\nbinding, assignment, batching,\nstragglers, verifier order\nreads a read-only view",
        ORANGE,
    )
    arrow(ax, (6.6, 4.55), (7.9, 4.55), "view")
    arrow(ax, (7.9, 4.2), (6.6, 4.2), "place / drop")
    box(
        ax,
        0.2,
        0.75,
        4.6,
        2.1,
        "Simulator driver (discrete events, virtual ms)",
        "worker engines in closed form:\n  continuous (iteration-level, KV, prefill,\n  drain / swap / interrupt) and static batching\nverifier completions and training times\n(the hidden lengths and verifier times)\nfixed order of one instant -> deterministic runs\nbenchmark: tuning, evaluation, S7 MILP / DP",
        AQUA,
    )
    box(
        ax,
        5.6,
        0.75,
        4.7,
        2.1,
        "Live service (asyncio)",
        "HTTP API, durable queue + sqlite3 store\nworker registration, heartbeats,\nretry of lost work, backpressure (429)\nmock workers run the same engines,\nmock verifiers and trainer call the API\nPrometheus /metrics; virtual-time loop in tests\n=> same placement log as the simulator (check 9)",
        AQUA,
    )
    arrow(ax, (2.4, 4.4), (3.9, 4.4), "trace")
    arrow(ax, (4.6, 3.55), (2.8, 2.85), "reports in,\ninputs out", rad=0.1)
    arrow(ax, (5.9, 3.55), (7.6, 2.85), "same core,\nsame policies", rad=-0.1)
    fig.text(
        0.01,
        0.01,
        "rollout-engine: one core state machine and one policy factory under two drivers; everything is simulated with assumed parameters",
        fontsize=7,
        color=INK2,
    )
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "framework.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
