"""System configuration v1 (docs/contracts.md section 3) and the derivation helper.

Every numeric parameter is documented in docs/simulator.md as assumed or derived. The
helper ``derive_system`` turns hardware and model numbers into engine parameters, so the
derivations are code, not constants.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass

SCHEMA_VERSION = 1
PARTITIONS = ("disaggregated", "colocated")
ENGINES = ("continuous", "static")
INFLIGHT = ("drain", "swap", "interrupt")
LOOP_MODES = ("closed", "single_phase")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EngineParams:
    engine: str
    tp: int
    a0_ns: int
    a1_ns: int
    a2_ns: int
    prefill_c0_ns: int
    prefill_c1_ns: int
    max_seqs: int
    kv_tokens: int
    static_batch: int


@dataclass(frozen=True, slots=True)
class SystemConfig:
    raw: dict
    partition: str
    total_gpus: int
    rollout_gpus: int  # GPUs used by rollout workers (all GPUs when colocated)
    train_gpus: int  # GPUs used by training (all GPUs when colocated)
    switch_ms: int
    engine: EngineParams
    n_workers: int
    verifier_servers: int
    verify_mean_s: dict  # task -> float (known to policies)
    fixed_ms: int
    ns_per_token: int
    sync_ms: int
    swap_ms: int
    inflight: str
    mode: str
    steps: int
    groups_per_step: int
    eta: int
    warmup_steps: int

    @property
    def colocated(self) -> bool:
        return self.partition == "colocated"

    @property
    def single_phase(self) -> bool:
        return self.mode == "single_phase"

    def train_ms(self, tokens: int) -> int:
        return self.fixed_ms + -(-self.ns_per_token * tokens // 1_000_000)

    def config_hash(self) -> str:
        return config_hash(self.raw)


def config_hash(obj) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise ConfigError(f"{where}.{key} is required")
    return d[key]


def _int(d: dict, key: str, where: str, lo: int = 0, default=None) -> int:
    v = d.get(key, default) if default is not None else _req(d, key, where)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{where}.{key} must be an integer, got {v!r}")
    if v < lo:
        raise ConfigError(f"{where}.{key} must be >= {lo}, got {v}")
    return v


def parse_system(raw: dict) -> SystemConfig:
    """Validate ranges and cross-field rules and return the typed configuration."""
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    for section in ("gpus", "rollout", "verifier", "trainer", "sync", "loop"):
        if not isinstance(raw.get(section), dict):
            raise ConfigError(f"section {section!r} is required")
    g, r, v, t, s, lp = (raw[k] for k in ("gpus", "rollout", "verifier", "trainer", "sync", "loop"))
    total = _int(g, "total", "gpus", lo=1)
    partition = _req(g, "partition", "gpus")
    if partition not in PARTITIONS:
        raise ConfigError(f"gpus.partition must be one of {PARTITIONS}")
    switch_ms = 0
    if partition == "disaggregated":
        rollout = _int(g, "rollout", "gpus", lo=1)
        train = _int(g, "train", "gpus", lo=1)
        if rollout + train != total:
            raise ConfigError(
                f"gpus.rollout + gpus.train = {rollout + train} != gpus.total {total}"
            )
    else:
        rollout = train = total
        switch_ms = _int(g, "switch_ms", "gpus", lo=0)
    engine = _req(r, "engine", "rollout")
    if engine not in ENGINES:
        raise ConfigError(f"rollout.engine must be one of {ENGINES}")
    tp = _int(r, "tp", "rollout", lo=1)
    if rollout % tp:
        raise ConfigError(f"rollout GPUs {rollout} are not a whole number of workers of tp={tp}")
    ep = EngineParams(
        engine=engine,
        tp=tp,
        a0_ns=_int(r, "a0_ns", "rollout", lo=1_000_000),  # every iteration takes >= 1 ms
        a1_ns=_int(r, "a1_ns", "rollout", lo=0),
        a2_ns=_int(r, "a2_ns", "rollout", lo=0),
        prefill_c0_ns=_int(r, "prefill_c0_ns", "rollout", lo=0),
        prefill_c1_ns=_int(r, "prefill_c1_ns", "rollout", lo=0),
        max_seqs=_int(r, "max_seqs", "rollout", lo=1),
        kv_tokens=_int(r, "kv_tokens", "rollout", lo=1),
        static_batch=_int(r, "static_batch", "rollout", lo=1),
    )
    servers = _int(v, "servers", "verifier", lo=1)
    tasks = _req(v, "tasks", "verifier")
    if not isinstance(tasks, dict) or not tasks:
        raise ConfigError("verifier.tasks must be a non-empty object")
    vm = {}
    for name in sorted(tasks):
        mean = _req(tasks[name], "verify_mean_s", f"verifier.tasks.{name}")
        if not isinstance(mean, (int, float)) or isinstance(mean, bool) or not mean >= 0:
            raise ConfigError(f"verifier.tasks.{name}.verify_mean_s must be >= 0")
        vm[name] = float(mean)
    mode = lp.get("mode", "closed")
    if mode not in LOOP_MODES:
        raise ConfigError(f"loop.mode must be one of {LOOP_MODES}")
    steps = _int(lp, "steps", "loop", lo=1)
    b = _int(lp, "groups_per_step", "loop", lo=1)
    eta = _int(lp, "eta", "loop", lo=0)
    warmup = _int(lp, "warmup_steps", "loop", lo=0, default=2)
    inflight = _req(s, "inflight", "sync")
    if inflight not in INFLIGHT:
        raise ConfigError(f"sync.inflight must be one of {INFLIGHT}")
    if mode == "closed" and steps < warmup + 2:
        raise ConfigError(f"loop.steps must be >= warmup_steps + 2 = {warmup + 2}")
    return SystemConfig(
        raw=copy.deepcopy(raw),
        partition=partition,
        total_gpus=total,
        rollout_gpus=rollout,
        train_gpus=train,
        switch_ms=switch_ms,
        engine=ep,
        n_workers=rollout // tp,
        verifier_servers=servers,
        verify_mean_s=vm,
        fixed_ms=_int(t, "fixed_ms", "trainer", lo=1),  # training takes >= 1 ms
        ns_per_token=_int(t, "ns_per_token", "trainer", lo=0),
        sync_ms=_int(s, "sync_ms", "sync", lo=0),
        swap_ms=_int(s, "swap_ms", "sync", lo=0),
        inflight=inflight,
        mode=mode,
        steps=steps,
        groups_per_step=b,
        eta=eta,
        warmup_steps=warmup,
    )


def load_system(path: str) -> SystemConfig:
    with open(path, encoding="utf-8") as f:
        return parse_system(json.load(f))


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------------------
# Derivation helper. Datasheet and model-card figures are transcribed; efficiencies and
# overheads are assumptions (docs/simulator.md, section "Default system").

GPU_H100_SXM = {
    "name": "H100-SXM5-80GB",
    "mem_bytes": 80e9,  # datasheet
    "mem_bw_Bps": 3.35e12,  # datasheet
    "peak_bf16_flops": 989e12,  # datasheet, dense
}
MODEL_7B_GQA = {
    "name": "dense-7.6B-GQA",
    "params": 7.6e9,  # public 7B-class model card
    "kv_bytes_per_token": 28
    * 4
    * 128
    * 2
    * 2,  # layers x KV heads x head dim x (K,V) x 2 B = 57344
    "bytes_per_param": 2,  # BF16
}
ASSUMED = {
    "eff_mem_bw": 0.7,  # achievable fraction of memory bandwidth in decode
    "eff_flops": 0.5,  # achievable fraction of peak FLOP/s in prefill / per-sequence compute
    "tp_efficiency": 0.85,  # for tp > 1
    "iter_overhead_ns": 1_000_000,  # scheduler + sampling per iteration
    "seq_overhead_ns": 20_000,  # per running sequence per iteration
    "mem_util": 0.9,  # fraction of GPU memory the engine may use
    "act_reserve_bytes": 2e9,  # per GPU, activations and buffers
    "train_mfu": 0.35,
    "train_fixed_ms": 2000,  # optimizer step, data movement, logging per step
    "sync_ms": 1000,  # trainer-side weight broadcast after a step
    "swap_ms": 500,  # rollout-side weight load per worker
    "switch_ms": 3000,  # colocated: offload/onload between rollout and training
    "max_seqs": 32,  # engine concurrency cap
    "static_batch": 16,  # static engine batch cap
}


def derive_engine(gpu: dict, model: dict, tp: int, assumed: dict) -> dict:
    eff_tp = 1.0 if tp == 1 else assumed["tp_efficiency"]
    bw = gpu["mem_bw_Bps"] * tp * eff_tp * assumed["eff_mem_bw"]
    flops = gpu["peak_bf16_flops"] * tp * eff_tp * assumed["eff_flops"]
    weights = model["params"] * model["bytes_per_param"]
    a0 = weights / bw * 1e9 + assumed["iter_overhead_ns"]
    per_token_compute = 2 * model["params"] / flops * 1e9
    a1 = per_token_compute + assumed["seq_overhead_ns"]
    a2 = model["kv_bytes_per_token"] / bw * 1e9
    kv = (
        tp * gpu["mem_bytes"] * assumed["mem_util"] - weights - tp * assumed["act_reserve_bytes"]
    ) / model["kv_bytes_per_token"]
    return {
        "a0_ns": round(a0),
        "a1_ns": round(a1),
        "a2_ns": round(a2),
        "prefill_c0_ns": round(a0),  # one weight pass per prefill
        "prefill_c1_ns": round(per_token_compute),
        "kv_tokens": math.floor(kv),
    }


def derive_trainer_ns_per_token(gpu: dict, model: dict, train_gpus: int, mfu: float) -> int:
    """6 * parameters FLOPs per token over the training GPUs at the assumed MFU."""
    return round(6 * model["params"] / (train_gpus * gpu["peak_bf16_flops"] * mfu) * 1e9)


def derive_system(
    *,
    gpu: dict = GPU_H100_SXM,
    model: dict = MODEL_7B_GQA,
    assumed: dict | None = None,
    total_gpus: int = 16,
    partition: str = "disaggregated",
    rollout_gpus: int = 8,
    tp: int = 1,
    engine: str = "continuous",
    verifier_servers: int = 32,
    verify_mean_s: dict | None = None,
    inflight: str = "drain",
    steps: int = 24,
    groups_per_step: int = 64,
    eta: int = 1,
    warmup_steps: int = 2,
    mode: str = "closed",
) -> dict:
    a = dict(ASSUMED)
    if assumed:
        a.update(assumed)
    train_gpus = total_gpus if partition == "colocated" else total_gpus - rollout_gpus
    gpus: dict = {"total": total_gpus, "partition": partition}
    if partition == "disaggregated":
        gpus.update({"rollout": rollout_gpus, "train": train_gpus})
    else:
        gpus["switch_ms"] = a["switch_ms"]
    eng = derive_engine(gpu, model, tp, a)
    raw = {
        "schema_version": SCHEMA_VERSION,
        "gpus": gpus,
        "rollout": {
            "engine": engine,
            "tp": tp,
            **{k: eng[k] for k in ("a0_ns", "a1_ns", "a2_ns", "prefill_c0_ns", "prefill_c1_ns")},
            "max_seqs": a["max_seqs"],
            "kv_tokens": eng["kv_tokens"],
            "static_batch": a["static_batch"],
        },
        "verifier": {
            "servers": verifier_servers,
            "tasks": {
                k: {"verify_mean_s": v}
                for k, v in sorted((verify_mean_s or {"math": 0.2, "code": 3.0}).items())
            },
        },
        "trainer": {
            "fixed_ms": a["train_fixed_ms"],
            "ns_per_token": derive_trainer_ns_per_token(gpu, model, train_gpus, a["train_mfu"]),
        },
        "sync": {"sync_ms": a["sync_ms"], "swap_ms": a["swap_ms"], "inflight": inflight},
        "loop": {
            "mode": mode,
            "steps": steps,
            "groups_per_step": groups_per_step,
            "eta": eta,
            "warmup_steps": warmup_steps,
        },
        "derivation": {"gpu": gpu, "model": model, "assumed": a},
    }
    parse_system(raw)
    return raw
