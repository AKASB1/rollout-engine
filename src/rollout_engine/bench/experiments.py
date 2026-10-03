"""Scenario expansion: the committed experiment configuration -> cells and policies.

A cell is one (scenario, variant): a system configuration, a generator configuration, a list
of policies, and the label of its reference policy. The tuned parameters and the "best S2
policy" come from ``configs/tuned/tuned.json`` (frozen before the evaluation); before tuning
exists, documented defaults stand in.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
from dataclasses import dataclass, field

from rollout_engine.config import config_hash, deep_merge, derive_system
from rollout_engine.trace.generator import DEFAULT_GENERATOR, verify_mean_s

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def repo_path(*parts: str) -> str:
    return os.path.join(os.getcwd(), *parts)


@dataclass
class PolicySpec:
    label: str
    cfg: dict
    tunables: tuple[str, ...] = ()


@dataclass
class Cell:
    scenario: str
    variant: str
    kind: str  # single | closed
    system: dict
    generator: dict
    policies: list[PolicySpec]
    reference: str
    seeds: list[int]
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.scenario}/{self.variant}"


def load_config(path: str | None = None) -> dict:
    with open(path or repo_path("configs", "experiments.json"), encoding="utf-8") as f:
        return json.load(f)


def load_tuned(path: str | None = None) -> dict | None:
    p = path or repo_path("configs", "tuned", "tuned.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _math_only(g: dict) -> dict:
    """Single-phase scenarios use one task with fast verification, so that the phase time
    measures generation; the verifier is studied in S5."""
    g["tasks"] = {"math": dict(g["tasks"]["math"], share=1.0)}
    return g


def _generator(cfg: dict, tail: str = "heavy", estimate: dict | None = None, **over) -> dict:
    g = copy.deepcopy(DEFAULT_GENERATOR)
    gc = cfg["generator"]
    g["n_samples"] = gc["n_samples"]
    g["max_tokens"] = gc["max_tokens"]
    g["estimate"] = copy.deepcopy(estimate or gc["estimate"])
    for t in g["tasks"].values():
        t["len_sigma"] = cfg["tails"][tail]
    g.update(over)
    return g


def _closed_system(cfg: dict, steps: int, gen: dict, **kw) -> dict:
    s = dict(cfg["system"])
    s.update(kw)
    s["steps"] = steps
    s["verify_mean_s"] = verify_mean_s(gen)
    assumed = s.pop("assumed", None)
    return derive_system(assumed=assumed, **s)


def _single_system(
    workers: int, groups: int, gen: dict, engine: str, cap: int | None = None
) -> dict:
    raw = derive_system(
        total_gpus=workers + 1,
        rollout_gpus=workers,
        engine=engine,
        mode="single_phase",
        steps=1,
        groups_per_step=groups,
        verifier_servers=64,
        verify_mean_s=verify_mean_s(gen),
    )
    if cap:
        raw = deep_merge(raw, {"rollout": {"static_batch": cap}})
    return raw


S2_GRID = [
    (o, d, b, a)
    for o, d, b, a in itertools.product(
        ["fifo", "longest_first"],
        ["group", "sample"],
        ["early", "late"],
        ["first_free", "round_robin", "least_loaded", "lpt"],
    )
    if not (b == "early" and a == "first_free")
]


def s2_policies() -> list[PolicySpec]:
    out = [PolicySpec("reference", {"name": "reference"})]
    for o, d, b, a in S2_GRID:
        label = f"{o}+{d}+{b}+{a}"
        if label == "fifo+group+early+round_robin":
            continue  # that is the reference
        params = {"order": o, "dispatch": d, "binding": b, "assign": a}
        out.append(
            PolicySpec(
                label,
                {"name": "composed", "params": params},
                ("window_groups",) if o != "fifo" else (),
            )
        )
    out.append(PolicySpec("online_adaptive", {"name": "online_adaptive"}, ("window_groups", "k0")))
    out.append(PolicySpec("online_eb", {"name": "online_eb"}, ("window_groups",)))  # Tier 2
    out.append(PolicySpec("oracle_lpt", {"name": "oracle_lpt"}, ("window_groups",)))
    return out


DEFAULT_BEST_S2 = {
    "label": "longest_first+sample+late+lpt",
    "cfg": {
        "name": "composed",
        "params": {
            "order": "longest_first",
            "dispatch": "sample",
            "binding": "late",
            "assign": "lpt",
        },
    },
}


def best_s2(tuned: dict | None) -> dict:
    return (
        copy.deepcopy(tuned["best_s2"])
        if tuned and "best_s2" in tuned
        else copy.deepcopy(DEFAULT_BEST_S2)
    )


def tuned_params(tuned: dict | None, cell_key: str, label: str) -> dict:
    if not tuned:
        return {}
    return dict(tuned.get("params", {}).get(cell_key, {}).get(label, {}))


def with_params(spec: PolicySpec, extra: dict) -> PolicySpec:
    cfg = copy.deepcopy(spec.cfg)
    cfg.setdefault("params", {}).update(extra)
    return PolicySpec(spec.label, cfg, spec.tunables)


def _variant(base: dict, label: str, **params) -> PolicySpec:
    cfg = copy.deepcopy(base["cfg"])
    cfg.setdefault("params", {}).update(params)
    return PolicySpec(label, cfg)


def build_cells(
    cfg: dict, tuned: dict | None, quick: bool = False, only: list[str] | None = None
) -> list[Cell]:
    seeds = cfg["quick"]["evaluation"] if quick else cfg["seeds"]["evaluation"]
    sens_seeds = cfg["quick"]["evaluation"] if quick else cfg["seeds"]["sensitivity_evaluation"]
    steps = cfg["quick"]["steps"] if quick else cfg["system"]["steps"]
    scen = set(
        only
        or (
            cfg["quick"]["scenarios"]
            if quick
            else ["S1", "S2", "S3", "S4", "S5", "S6", "S8", "S10", "SENS"]
        )
    )
    cells: list[Cell] = []
    best = best_s2(tuned)
    # ---------------------------------------------------------------- S1
    if "S1" in scen:
        c = cfg["S1"]
        for cap in c["caps"]:
            for est_name, est in c["estimates"].items():
                gen = _math_only(
                    _generator(
                        cfg,
                        "heavy",
                        est,
                        groups=c["samples"] // c["n_samples"],
                        n_samples=c["n_samples"],
                        prompt_fixed=c["prompt"],
                    )
                )
                pols = [
                    PolicySpec(
                        "reference", {"name": "composed", "params": {"batch": "fifo_chunk"}}
                    ),
                    PolicySpec(
                        "fifo_chunk",
                        {"name": "composed", "params": {"batch": "fifo_chunk"}},
                        ("batch_size",),
                    ),
                    PolicySpec(
                        "sorted_chunk",
                        {"name": "composed", "params": {"batch": "sorted_chunk"}},
                        ("batch_size",),
                    ),
                    PolicySpec("dp", {"name": "dp"}),
                    PolicySpec("oracle_dp", {"name": "oracle_dp"}),
                ]
                cells.append(
                    Cell(
                        "S1",
                        f"cap{cap}/{est_name}",
                        "single",
                        _single_system(c["workers"], gen["groups"], gen, "static", cap),
                        gen,
                        pols,
                        "reference",
                        seeds,
                    )
                )
    # ---------------------------------------------------------------- S2
    if "S2" in scen:
        c = cfg["S2"]
        pols_all = s2_policies()
        if quick:
            keep = set(cfg["quick"]["s2_policies"])
            pols_all = [p for p in pols_all if p.label in keep]
        for tail in ("light", "heavy"):
            for est_name, est in c["estimates"].items():
                gen = _math_only(_generator(cfg, tail, est, groups=c["groups"]))
                cells.append(
                    Cell(
                        "S2",
                        f"{tail}/{est_name}",
                        "single",
                        _single_system(c["workers"], c["groups"], gen, "continuous"),
                        gen,
                        list(pols_all),
                        "reference",
                        seeds,
                    )
                )
    # ---------------------------------------------------------------- closed loop
    T = steps
    B = cfg["system"]["groups_per_step"]
    ngroups = cfg["generator"]["groups_factor"] * T * B
    base_gen = _generator(cfg, "heavy", None, groups=ngroups)

    def closed_pols(extra_variants: list[PolicySpec]) -> list[PolicySpec]:
        return [PolicySpec("reference", {"name": "reference"})] + extra_variants

    best_wait = _variant(best, "best_s2+wait", straggler="wait")
    if "S3" in scen:
        c = cfg["S3"]
        for eta in c["etas"]:
            pols = closed_pols(
                [
                    best_wait,
                    PolicySpec(
                        "best_s2+carry", _variant(best, "", straggler="carry").cfg, ("rho",)
                    ),
                ]
            )
            cells.append(
                Cell(
                    "S3",
                    f"eta{eta}",
                    "closed",
                    _closed_system(cfg, T, base_gen, eta=eta),
                    base_gen,
                    pols,
                    "reference",
                    seeds,
                )
            )
        for mode in c["inflight_modes"]:
            pols = closed_pols(
                [
                    best_wait,
                    PolicySpec(
                        "best_s2+carry", _variant(best, "", straggler="carry").cfg, ("rho",)
                    ),
                ]
            )
            cells.append(
                Cell(
                    "S3",
                    f"eta{c['inflight_eta']}-{mode}",
                    "closed",
                    _closed_system(cfg, T, base_gen, eta=c["inflight_eta"], inflight=mode),
                    base_gen,
                    pols,
                    "reference",
                    seeds,
                )
            )
    if "S4" in scen:
        c = cfg["S4"]
        oa = {
            "label": "online_adaptive",
            "cfg": {
                "name": "online_adaptive",
                "params": tuned_params(tuned, "S2/heavy/group", "online_adaptive"),
            },
        }
        pols = [PolicySpec("reference", {"name": "reference"})]
        for base, tag in ((best, "best_s2"), (oa, "online_adaptive")):
            pols.append(_variant(base, f"{tag}+wait", straggler="wait"))
            for r in c["rhos"]:
                pols.append(_variant(base, f"{tag}+carry({r:g})", straggler="carry", rho=r))
                pols.append(_variant(base, f"{tag}+abort({r:g})", straggler="abort", rho=r))
        cells.append(
            Cell(
                "S4",
                f"eta{c['eta']}",
                "closed",
                _closed_system(cfg, T, base_gen, eta=c["eta"]),
                base_gen,
                pols,
                "reference",
                seeds,
            )
        )
    if "S5" in scen:
        c = cfg["S5"]
        gen5 = copy.deepcopy(base_gen)
        gen5["tasks"]["code"]["share"] = c["code_share"]
        gen5["tasks"]["math"]["share"] = round(1 - c["code_share"], 10)
        for servers in c["servers"]:
            pols = [PolicySpec("reference", {"name": "reference"})]
            for order in c["orders"]:
                pols.append(_variant(best, f"best_s2+v_{order}", straggler="wait", verifier=order))
            cells.append(
                Cell(
                    "S5",
                    f"servers{servers}",
                    "closed",
                    _closed_system(cfg, T, gen5, verifier_servers=servers),
                    gen5,
                    pols,
                    "reference",
                    seeds,
                )
            )
    if "S6" in scen:
        c = cfg["S6"]
        for r in c["rollout_gpus"] + ["colocated"]:
            if r == "colocated":
                sysraw = _closed_system(
                    cfg, T, base_gen, total_gpus=c["total_gpus"], partition="colocated"
                )
                var = "colocated"
            else:
                sysraw = _closed_system(
                    cfg, T, base_gen, total_gpus=c["total_gpus"], rollout_gpus=r
                )
                var = f"rollout{r}"
            cells.append(
                Cell(
                    "S6",
                    var,
                    "closed",
                    sysraw,
                    base_gen,
                    [PolicySpec("reference", {"name": "reference"}), best_wait],
                    "reference",
                    seeds,
                )
            )
    if "S8" in scen:  # Tier 2: distribution shift half way through the stream
        c = cfg["S8"]
        rho = c["rho"]
        oa = {
            "label": "online_adaptive",
            "cfg": {
                "name": "online_adaptive",
                "params": tuned_params(tuned, "S2/heavy/group", "online_adaptive"),
            },
        }
        oeb = {
            "label": "online_eb",
            "cfg": {
                "name": "online_eb",
                "params": tuned_params(tuned, "S2/heavy/group", "online_eb"),
            },
        }
        for name, sh in c["shifts"].items():
            g = copy.deepcopy(base_gen)
            if sh:
                g["shift"] = {"at_group": T * B // 2, **sh}
            pols = [
                PolicySpec("reference", {"name": "reference"}),
                _variant(best, "best_s2+wait", straggler="wait"),
                _variant(oa, "online_adaptive+wait", straggler="wait"),
                _variant(oeb, "online_eb+wait", straggler="wait"),
                _variant(best, f"best_s2+abort({rho:g})", straggler="abort", rho=rho),
                _variant(oa, f"online_adaptive+abort({rho:g})", straggler="abort", rho=rho),
                PolicySpec(
                    f"online_eb+abort_pred({rho:g})",
                    _variant(oeb, "", straggler="abort_pred", rho=rho).cfg,
                    ("kappa",),
                ),
            ]
            cells.append(
                Cell(
                    "S8",
                    f"eta{c['eta']}/{name}",
                    "closed",
                    _closed_system(cfg, T, g, eta=c["eta"]),
                    g,
                    pols,
                    "reference",
                    seeds,
                )
            )
    if "S10" in scen:  # Tier 2: deadline-aware admission against greedy carry
        c = cfg["S10"]
        pols = [
            PolicySpec("reference", {"name": "reference"}),
            _variant(best, "best_s2+wait", straggler="wait"),
        ]
        for r in c["rhos"]:
            pols.append(_variant(best, f"best_s2+carry({r:g})", straggler="carry", rho=r))
            dl = _variant(best, "", straggler="carry", rho=r, deadline=1.0).cfg
            pols.append(PolicySpec(f"best_s2+carry({r:g})+deadline", dl, ("deadline",)))
        cells.append(
            Cell(
                "S10",
                f"eta{c['eta']}",
                "closed",
                _closed_system(cfg, T, base_gen, eta=c["eta"]),
                base_gen,
                pols,
                "reference",
                seeds,
            )
        )
    if "SENS" in scen:
        c = cfg["SENS"]
        variants = []
        for f in c["tail_factors"]:
            g = copy.deepcopy(base_gen)
            for t in g["tasks"].values():
                t["len_sigma"] = round(t["len_sigma"] * f, 6)
            variants.append((f"tail{f:g}", g, {}))
        for name, est in c["error_levels"].items():
            g = copy.deepcopy(base_gen)
            g["estimate"] = copy.deepcopy(est)
            variants.append((f"est_{name}", g, {}))
        for mfu in c["mfu"]:
            variants.append((f"mfu{mfu:g}", base_gen, {"assumed": {"train_mfu": mfu}}))
        rho = c["rho"]
        for vname, g, kw in variants:
            s3p = closed_pols(
                [best_wait, PolicySpec("best_s2+carry", _variant(best, "", straggler="carry").cfg)]
            )
            cells.append(
                Cell(
                    "SENS",
                    f"S3eta1/{vname}",
                    "closed",
                    _closed_system(cfg, T, g, eta=1, **kw),
                    g,
                    s3p,
                    "reference",
                    sens_seeds,
                    {"base": "S3/eta1"},
                )
            )
            s4p = closed_pols(
                [
                    best_wait,
                    _variant(best, f"best_s2+carry({rho:g})", straggler="carry", rho=rho),
                    _variant(best, f"best_s2+abort({rho:g})", straggler="abort", rho=rho),
                ]
            )
            cells.append(
                Cell(
                    "SENS",
                    f"S4eta2/{vname}",
                    "closed",
                    _closed_system(cfg, T, g, eta=2, **kw),
                    g,
                    s4p,
                    "reference",
                    sens_seeds,
                    {"base": "S4/eta2"},
                )
            )
    # apply frozen tuned parameters (SENS inherits from its base cell)
    for cell in cells:
        src = cell.extra.get("base", cell.key)
        cell.policies = [with_params(p, tuned_params(tuned, src, p.label)) for p in cell.policies]
    return cells


def cells_hash(cells: list[Cell]) -> str:
    return config_hash(
        [
            {
                "key": c.key,
                "system": c.system,
                "generator": c.generator,
                "policies": [(p.label, p.cfg) for p in c.policies],
                "seeds": c.seeds,
            }
            for c in cells
        ]
    )
