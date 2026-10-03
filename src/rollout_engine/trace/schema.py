"""Rollout trace schema v1: CSV loader, writer, and manifest (docs/contracts.md section 2).

One row per sample; the rows of one group are contiguous; groups appear in stream order.
Hidden columns (``resp_tokens``, ``verify_s``) are kept in the trace object for the
simulator driver and the mock workers; policies never receive them (oracles excepted).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from rollout_engine.clock import fmt_s, parse_s_to_ms

SCHEMA_VERSION = 1
HEADER = "group_id,sample_idx,task,prompt_tokens,max_tokens,est_tokens,resp_tokens,verify_s"
N_FIELDS = 8


class TraceError(ValueError):
    """A malformed trace; the message names the 1-based line number."""


@dataclass(frozen=True, slots=True)
class GroupRow:
    group_id: str
    task: str
    prompt_tokens: int
    max_tokens: int
    est_tokens: float
    resp_tokens: tuple[int, ...]  # hidden until a sample finishes
    verify_ms: tuple[int, ...]  # hidden until a verification ends

    @property
    def n_samples(self) -> int:
        return len(self.resp_tokens)


@dataclass(frozen=True, slots=True)
class Trace:
    groups: tuple[GroupRow, ...]

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_samples(self) -> int:
        return sum(g.n_samples for g in self.groups)


def fmt_est(x: float) -> str:
    s = f"{x:.3f}"
    return s


def dumps(trace: Trace) -> bytes:
    lines = [HEADER]
    for g in trace.groups:
        est = fmt_est(g.est_tokens)
        for i, (r, v) in enumerate(zip(g.resp_tokens, g.verify_ms, strict=True)):
            lines.append(
                f"{g.group_id},{i},{g.task},{g.prompt_tokens},{g.max_tokens},{est},{r},{fmt_s(v)}"
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _int(field: str, name: str, line: int, lo: int = 1) -> int:
    try:
        if not field or not (field.isdigit() or (field[0] == "-" and field[1:].isdigit())):
            raise ValueError
        value = int(field)
    except ValueError:
        raise TraceError(f"line {line}: {name} is not an integer: {field!r}") from None
    if value < lo:
        raise TraceError(f"line {line}: {name} must be >= {lo}, got {value}")
    return value


def loads(data: bytes) -> Trace:
    if len(data) == 0:
        raise TraceError("line 1: empty file (a trace needs at least the header)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise TraceError(f"line 1: not UTF-8: {e}") from None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in lines]
    if not lines or lines[0] != HEADER:
        raise TraceError(f"line 1: header must be exactly {HEADER!r}")
    groups: list[GroupRow] = []
    seen: set[str] = set()
    cur: dict | None = None

    def close(c: dict) -> None:
        groups.append(
            GroupRow(
                c["gid"], c["task"], c["p"], c["m"], c["est"], tuple(c["resp"]), tuple(c["ver"])
            )
        )

    for n, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        if len(fields) != N_FIELDS:
            raise TraceError(f"line {n}: expected {N_FIELDS} fields, got {len(fields)}")
        gid, sidx_s, task, p_s, m_s, est_s, r_s, v_s = fields
        if not gid:
            raise TraceError(f"line {n}: group_id is empty")
        if not task:
            raise TraceError(f"line {n}: task is empty")
        sidx = _int(sidx_s, "sample_idx", n, lo=0)
        p = _int(p_s, "prompt_tokens", n)
        m = _int(m_s, "max_tokens", n)
        try:
            est = float(est_s)
        except ValueError:
            raise TraceError(f"line {n}: est_tokens is not a decimal: {est_s!r}") from None
        if not (est > 0 and est < float("inf")):
            raise TraceError(f"line {n}: est_tokens must be > 0 and finite, got {est_s!r}")
        r = _int(r_s, "resp_tokens", n)
        if r > m:
            raise TraceError(f"line {n}: resp_tokens {r} exceeds max_tokens {m}")
        try:
            v = parse_s_to_ms(v_s)
        except ValueError as e:
            raise TraceError(f"line {n}: verify_s: {e}") from None
        if cur is not None and gid == cur["gid"]:
            if (task, p, m, est) != (cur["task"], cur["p"], cur["m"], cur["est"]):
                raise TraceError(f"line {n}: group-level fields differ within group {gid!r}")
            if sidx != len(cur["resp"]):
                raise TraceError(
                    f"line {n}: sample_idx {sidx} out of sequence (expected {len(cur['resp'])})"
                )
            cur["resp"].append(r)
            cur["ver"].append(v)
            continue
        if cur is not None:
            close(cur)
        if gid in seen:
            raise TraceError(f"line {n}: group {gid!r} is not contiguous or is duplicated")
        if sidx != 0:
            raise TraceError(f"line {n}: sample_idx {sidx} out of sequence (expected 0)")
        seen.add(gid)
        cur = {"gid": gid, "task": task, "p": p, "m": m, "est": est, "resp": [r], "ver": [v]}
    if cur is not None:
        close(cur)
    return Trace(tuple(groups))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_path(csv_path: str) -> str:
    base = csv_path[:-4] if csv_path.endswith(".csv") else csv_path
    return base + ".manifest.json"


def write(path: str, trace: Trace, generator: dict, seed: int) -> dict:
    """Write ``<name>.csv`` and ``<name>.manifest.json``; return the manifest."""
    data = dumps(trace)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": generator,
        "seed": seed,
        "groups": trace.n_groups,
        "samples": trace.n_samples,
        "content_sha256": sha256_hex(data),
    }
    with open(manifest_path(path), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def read(path: str, check_manifest: bool = True) -> tuple[Trace, dict | None]:
    """Load a trace; when its manifest exists (or is required), check version and hash."""
    with open(path, "rb") as f:
        data = f.read()
    manifest = None
    mpath = manifest_path(path)
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise TraceError(f"manifest {mpath}: unsupported schema_version")
        if manifest.get("content_sha256") != sha256_hex(data):
            raise TraceError(f"manifest {mpath}: content_sha256 does not match the CSV bytes")
    elif check_manifest:
        raise TraceError(f"missing manifest {mpath}")
    return loads(data), manifest
