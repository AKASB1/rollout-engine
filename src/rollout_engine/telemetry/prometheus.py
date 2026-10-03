"""Prometheus text exposition format (version 0.0.4), standard library only."""

from __future__ import annotations


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\n").replace('"', '\\"')


def render(metrics: list[tuple[str, str, str, list[tuple[dict, float]]]]) -> str:
    """``metrics``: (name, type, help, [(labels, value), ...]) in a fixed order."""
    lines = []
    for name, mtype, help_, samples in metrics:
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} {mtype}")
        for labels, value in samples:
            lab = ""
            if labels:
                lab = "{" + ",".join(f'{k}="{_esc(str(labels[k]))}"' for k in sorted(labels)) + "}"
            v = float(value)
            sval = repr(int(v)) if v.is_integer() and abs(v) < 2**53 else repr(v)
            lines.append(f"{name}{lab} {sval}")
    return "\n".join(lines) + "\n"
