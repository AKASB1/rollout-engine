"""Fit the iteration and prefill model to a small decoder on the local GPU (optional).

    python scripts/calibrate_gpu.py [--out benchmarks/results/calibration] [--memory-fraction 0.5]

Needs PyTorch with CUDA (not a dependency of the package; nothing in the default path, CI, or the
README numbers uses this script). The model has the shape of a 0.5-billion-parameter decoder
(24 layers, hidden 896, 14 query and 2 key/value heads of 64, SwiGLU 4864, vocabulary 151936,
tied embeddings) with random bf16 weights, so no weights are downloaded. Measured:

- decode iterations: batch b in 1..64, per-sample context L in 128..4096 (C = b * L);
- prefills: b prompts of P tokens processed together (b * P tokens);
- static batches: one padded prefill, then lmax iterations carrying all b samples.

The iteration model a0 + a1*b + a2*C (docs/simulator.md) and the prefill model c0 + c1*tokens are
fitted by least squares on alternating grid points and checked on the others (held-out error);
the static batches are compared with the simulator's batch cost under the fitted parameters.
Columns of wall_calibration.csv: kind, b, context (per-sample context; pmax for static rows),
tokens (b * context; lmax for static rows), wall_ns (median of the repetitions), and for static
rows predicted_ns and relative_error.
A small model on a consumer GPU says nothing about a data-center rollout worker; these are
wall-clock measurements, not simulated results, and they are not deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

DECODE_B = [1, 2, 4, 8, 16, 32, 64]
DECODE_L = [128, 256, 512, 1024, 2048, 4096]
PREFILL = [
    (1, 128),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (4, 256),
    (4, 1024),
    (16, 128),
    (16, 512),
]
STATIC = [(4, 256, 256), (8, 512, 256), (16, 256, 512), (32, 128, 256)]  # (b, pmax, lmax)
SHAPE = {
    "layers": 24,
    "hidden": 896,
    "heads": 14,
    "kv_heads": 2,
    "head_dim": 64,
    "ffn": 4864,
    "vocab": 151936,
}


def build(torch, dev):
    """Random bf16 weights of the model shape above; returns (params, n_params)."""
    s, g = SHAPE, torch.Generator(device="cpu").manual_seed(0)

    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(dev, torch.bfloat16)

    layers = []
    for _ in range(s["layers"]):
        layers.append(
            {
                "q": w(s["heads"] * s["head_dim"], s["hidden"]),
                "k": w(s["kv_heads"] * s["head_dim"], s["hidden"]),
                "v": w(s["kv_heads"] * s["head_dim"], s["hidden"]),
                "o": w(s["hidden"], s["heads"] * s["head_dim"]),
                "gate": w(s["ffn"], s["hidden"]),
                "up": w(s["ffn"], s["hidden"]),
                "down": w(s["hidden"], s["ffn"]),
            }
        )
    emb = w(s["vocab"], s["hidden"])
    n = sum(t.numel() for lay in layers for t in lay.values()) + emb.numel()
    return {"layers": layers, "emb": emb}, n


def forward(torch, prm, x, kv, pos):
    """One forward pass over x [b, t, hidden] at positions pos..pos+t-1, writing into kv."""
    s, F = SHAPE, torch.nn.functional
    b, t, _ = x.shape
    rep = s["heads"] // s["kv_heads"]
    for i, lay in enumerate(prm["layers"]):
        h = F.rms_norm(x, (s["hidden"],))
        q = (h @ lay["q"].T).view(b, t, s["heads"], s["head_dim"]).transpose(1, 2)
        k = (h @ lay["k"].T).view(b, t, s["kv_heads"], s["head_dim"]).transpose(1, 2)
        v = (h @ lay["v"].T).view(b, t, s["kv_heads"], s["head_dim"]).transpose(1, 2)
        kv[i][0][:, :, pos : pos + t] = k
        kv[i][1][:, :, pos : pos + t] = v
        kk = kv[i][0][:, :, : pos + t].repeat_interleave(rep, dim=1)
        vv = kv[i][1][:, :, : pos + t].repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, kk, vv, is_causal=t > 1)
        x = x + a.transpose(1, 2).reshape(b, t, -1) @ lay["o"].T
        h = F.rms_norm(x, (s["hidden"],))
        x = x + (F.silu(h @ lay["gate"].T) * (h @ lay["up"].T)) @ lay["down"].T
    return F.rms_norm(x[:, -1:], (s["hidden"],)) @ prm["emb"].T  # logits of the last position


def kv_cache(torch, dev, b, length):
    s = SHAPE
    return [
        [
            torch.zeros(b, s["kv_heads"], length, s["head_dim"], device=dev, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        for _ in range(s["layers"])
    ]


def timed(torch, fn, reps):
    """Median wall time in ns of fn() over reps runs (CUDA events, after one warm-up)."""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        z.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(z) * 1e6)
    out.sort()
    return out[len(out) // 2]


def lstsq(rows, cols):
    """Least squares y = X @ coef for rows of (features..., y); returns coef (pure Python)."""
    n = len(cols)
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    aty = [sum(r[i] * r[-1] for r in rows) for i in range(n)]
    for c in range(n):  # Gauss-Jordan
        p = max(range(c, n), key=lambda r: abs(ata[r][c]))
        ata[c], ata[p], aty[c], aty[p] = ata[p], ata[c], aty[p], aty[c]
        for r in range(n):
            if r != c:
                f = ata[r][c] / ata[c][c]
                ata[r] = [x - f * y for x, y in zip(ata[r], ata[c], strict=True)]
                aty[r] -= f * aty[c]
    return {k: aty[i] / ata[i][i] for i, k in enumerate(cols)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join("benchmarks", "results", "calibration"))
    ap.add_argument(
        "--memory-fraction", type=float, default=0.5, help="cap on this process's GPU memory"
    )
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args(argv)
    try:
        import torch
    except ImportError:
        print(
            "PyTorch is not installed; this optional script needs PyTorch with CUDA",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from rollout_engine.batching import batch_cost_ns
    from rollout_engine.config import EngineParams

    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    prm, n_params = build(torch, dev)
    tok = lambda b, t: prm["emb"][torch.zeros(b, t, dtype=torch.long, device=dev)]  # noqa: E731
    rows = []
    with torch.inference_mode():
        for b in DECODE_B:
            for length in DECODE_L:
                kv = kv_cache(torch, dev, b, length + 1)
                x = tok(b, 1)
                ns = timed(
                    torch,
                    lambda b=b, kv=kv, x=x, length=length: forward(torch, prm, x, kv, length),
                    args.reps,
                )
                rows.append(
                    {
                        "kind": "decode",
                        "b": b,
                        "context": length,
                        "tokens": b * length,
                        "wall_ns": ns,
                    }
                )
                del kv
        for b, p in PREFILL:
            kv = kv_cache(torch, dev, b, p)
            x = tok(b, p)
            ns = timed(torch, lambda kv=kv, x=x: forward(torch, prm, x, kv, 0), args.reps)
            rows.append({"kind": "prefill", "b": b, "context": p, "tokens": b * p, "wall_ns": ns})
            del kv
        for b, pmax, lmax in STATIC:

            def run(b=b, pmax=pmax, lmax=lmax):
                kv = kv_cache(torch, dev, b, pmax + lmax)
                forward(torch, prm, tok(b, pmax), kv, 0)
                x = tok(b, 1)
                for j in range(lmax):
                    forward(torch, prm, x, kv, pmax + j)

            ns = timed(torch, run, 3)
            rows.append({"kind": "static", "b": b, "context": pmax, "tokens": lmax, "wall_ns": ns})
    dec = [r for r in rows if r["kind"] == "decode"]
    pre = [r for r in rows if r["kind"] == "prefill"]
    fit_d = lstsq(
        [(1, r["b"], r["tokens"], r["wall_ns"]) for r in dec[0::2]], ["a0_ns", "a1_ns", "a2_ns"]
    )
    fit_p = lstsq(
        [(1, r["tokens"], r["wall_ns"]) for r in pre[0::2]], ["prefill_c0_ns", "prefill_c1_ns"]
    )

    def rel(pred, meas):
        return abs(pred - meas) / meas

    held_d = [
        rel(fit_d["a0_ns"] + fit_d["a1_ns"] * r["b"] + fit_d["a2_ns"] * r["tokens"], r["wall_ns"])
        for r in dec[1::2]
    ]
    held_p = [
        rel(fit_p["prefill_c0_ns"] + fit_p["prefill_c1_ns"] * r["tokens"], r["wall_ns"])
        for r in pre[1::2]
    ]
    p = EngineParams(
        engine="static",
        tp=1,
        a0_ns=max(1, round(fit_d["a0_ns"])),
        a1_ns=round(fit_d["a1_ns"]),
        a2_ns=round(fit_d["a2_ns"]),
        prefill_c0_ns=round(fit_p["prefill_c0_ns"]),
        prefill_c1_ns=round(fit_p["prefill_c1_ns"]),
        max_seqs=64,
        kv_tokens=1 << 30,
        static_batch=64,
    )
    for r in rows:
        if r["kind"] == "static":
            pre_ns, dec_ns = batch_cost_ns(p, r["b"], r["context"], r["tokens"])
            r["predicted_ns"] = pre_ns + dec_ns
            r["relative_error"] = rel(r["predicted_ns"], r["wall_ns"])
    os.makedirs(args.out, exist_ok=True)
    cols = ["kind", "b", "context", "tokens", "wall_ns", "predicted_ns", "relative_error"]
    with open(
        os.path.join(args.out, "wall_calibration.csv"), "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    ""
                    if r.get(c) is None
                    else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                    for c in cols
                )
                + "\n"
            )
    fit = {
        "model_shape": SHAPE,
        "parameters": n_params,
        "dtype": "bfloat16",
        "weights": "random (seed 0), no download",
        "iteration_fit": {k: round(v, 1) for k, v in fit_d.items()},
        "prefill_fit": {k: round(v, 3) for k, v in fit_p.items()},
        "fit_points": "even grid points; held-out: odd grid points",
        "held_out_relative_error": {
            "decode_median": sorted(held_d)[len(held_d) // 2],
            "decode_max": max(held_d),
            "prefill_median": sorted(held_p)[len(held_p) // 2],
            "prefill_max": max(held_p),
        },
        "static_batch_relative_error_max": max(
            r["relative_error"] for r in rows if r["kind"] == "static"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "reps": args.reps,
        "label": "wall-clock measurements of a 0.5B-parameter decoder shape with random weights on one consumer GPU; "
        "says nothing about a data-center rollout worker; not used by the simulator's defaults",
    }
    with open(os.path.join(args.out, "fit.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(fit, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                k: fit[k]
                for k in (
                    "iteration_fit",
                    "prefill_fit",
                    "held_out_relative_error",
                    "static_batch_relative_error_max",
                )
            },
            indent=1,
        )
    )
    return 0 if math.isfinite(fit_d["a0_ns"]) else 1


if __name__ == "__main__":
    sys.exit(main())
