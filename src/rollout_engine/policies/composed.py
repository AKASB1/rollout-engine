"""Rollout scheduling policies composed of independent parts (docs/contracts.md section 5).

A policy is ``{"name": ..., "params": {...}}``; it reads only the view, keeps private state,
and returns actions. It never imports the simulator, asyncio, or http, and never reads a
clock. Parts:

- estimator: ``prior`` | ``group_evidence(k0)`` | ``eb`` (empirical Bayes with right-censoring,
  Tier 2, ``policies/eb.py``) | ``oracle`` (true lengths; benchmark only)
- order (launch within the window): ``fifo`` | ``shortest_first`` | ``longest_first``
- dispatch (continuous engine): ``group`` | ``sample``
- binding: ``early`` | ``late``
- assign: ``first_free`` (late only) | ``round_robin`` | ``least_loaded`` | ``lpt``
- batch (static engine): ``fifo_chunk`` | ``sorted_chunk`` | ``dp``
- straggler / admission: ``wait`` (cap = B) | ``carry`` (cap = ceil(rho B)) | ``abort``
  (cap = ceil(rho B); every outstanding group not selected is dropped after a selection) |
  ``abort_pred`` (Tier 2: after a selection, drop only the unselected groups whose predicted
  remaining time exceeds ``kappa`` times the median of the outstanding groups; carry the rest)
- ``isolate`` (Tier 2, late binding): a unit predicted to be long (more than ``isolate`` times the
  median of the units placed in the same call) goes to the candidate worker with the most free
  slots, where it runs in a small batch
- verifier order: ``fifo`` | ``group_first`` | ``shortest_first``
"""

from __future__ import annotations

import math

from rollout_engine.api import Drop, Place
from rollout_engine.batching import batch_cost_ms
from rollout_engine.opt.dp import dp_partition

ESTIMATORS = ("prior", "group_evidence", "eb", "oracle")
ORDERS = ("fifo", "shortest_first", "longest_first")
DISPATCH = ("group", "sample")
BINDING = ("early", "late")
ASSIGN = ("first_free", "round_robin", "least_loaded", "lpt")
BATCH = ("fifo_chunk", "sorted_chunk", "dp")
STRAGGLER = ("wait", "carry", "abort", "abort_pred")
VERIFIER = ("fifo", "group_first", "shortest_first")

DEFAULTS = {
    "estimator": "prior",
    "k0": 2.0,
    "order": "fifo",
    "window_groups": 16,
    "dispatch": "group",
    "binding": "early",
    "assign": "round_robin",
    "batch": "fifo_chunk",
    "batch_size": 0,  # 0 = static_batch
    "straggler": "wait",
    "rho": 1.5,
    "cap_groups": 0,  # 0 = from the straggler part
    "verifier": "fifo",
    "eb_n0": 20.0,  # eb: pseudo-groups of the hyperparameter prior
    "kappa": 2.0,  # abort_pred: drop threshold relative to the median predicted remaining time
    "isolate": 0.0,  # 0 = off
    "quantile": 0.0,  # eb: plan on this quantile of the predictive length (0 = the mean)
    "deadline": 0.0,  # > 0: deadline-aware admission with this slack on the predicted duration
}


class PolicyConfigError(ValueError):
    pass


def _fmt_num(x) -> str:
    return f"{x:g}"


class ComposedPolicy:
    def __init__(self, params: dict | None = None, label: str | None = None):
        p = dict(DEFAULTS)
        for k, v in (params or {}).items():
            if k not in DEFAULTS:
                raise PolicyConfigError(f"unknown policy parameter {k!r}")
            p[k] = v
        for key, allowed in (
            ("estimator", ESTIMATORS),
            ("order", ORDERS),
            ("dispatch", DISPATCH),
            ("binding", BINDING),
            ("assign", ASSIGN),
            ("batch", BATCH),
            ("straggler", STRAGGLER),
            ("verifier", VERIFIER),
        ):
            if p[key] not in allowed:
                raise PolicyConfigError(f"{key} must be one of {allowed}, got {p[key]!r}")
        if p["binding"] == "early" and p["assign"] == "first_free":
            raise PolicyConfigError("first_free is used only under late binding")
        if int(p["window_groups"]) < 1:
            raise PolicyConfigError("window_groups must be >= 1")
        if float(p["rho"]) < 1:
            raise PolicyConfigError("rho must be >= 1")
        if float(p["k0"]) < 0:
            raise PolicyConfigError("k0 must be >= 0")
        if not 0 <= float(p["quantile"]) < 1:
            raise PolicyConfigError("quantile must be in [0, 1)")
        self.p = p
        self.params = {k: p[k] for k in DEFAULTS if (params or {}).get(k) is not None}
        self.label = label
        self.name = label or self.describe()
        self.oracle = p["estimator"] == "oracle"
        self._L: list[int] | None = None
        self.reset()

    # ------------------------------------------------------------------ identity
    def describe(self, engine: str | None = None) -> str:
        p = self.p
        est = {
            "prior": "prior",
            "oracle": "oracle",
            "group_evidence": f"group_evidence({_fmt_num(float(p['k0']))})",
            "eb": f"eb({_fmt_num(float(p['eb_n0']))})"
            if not float(p["quantile"])
            else f"eb({_fmt_num(float(p['eb_n0']))},q{_fmt_num(float(p['quantile']))})",
        }[p["estimator"]]
        strag = (
            p["straggler"]
            if p["straggler"] == "wait"
            else f"{p['straggler']}({_fmt_num(float(p['rho']))})"
        )
        if p["straggler"] == "abort_pred":
            strag = f"abort_pred({_fmt_num(float(p['rho']))},{_fmt_num(float(p['kappa']))})"
        if engine == "static":
            bs = int(p["batch_size"])
            batch = p["batch"] + (f"({bs})" if bs else "")
            return f"{batch}+{strag}+{est}+v_{p['verifier']}"
        iso = f"+isolate({_fmt_num(float(p['isolate']))})" if float(p["isolate"]) > 0 else ""
        if float(p["deadline"]) > 0:
            strag += f"+deadline({_fmt_num(float(p['deadline']))})"
        return f"{p['order']}+{p['dispatch']}+{p['binding']}+{p['assign']}{iso}+{strag}+{est}+v_{p['verifier']}"

    def reset(self) -> None:
        self.rr = 0
        self.last_s = 0
        self.oldest = -1
        self.oldest_launches = 0
        self.eb = None
        self._rate_t = -1
        self._rate_seen: set[int] = set()
        self._rate_ms = 0
        self._rate_tok = 0
        if self.p["estimator"] == "eb":
            from rollout_engine.policies.eb import EBModel

            self.eb = EBModel(n0=float(self.p["eb_n0"]))

    def bind(self, system, trace) -> None:
        """Called by the simulator before a run. Only oracle policies keep the true lengths."""
        self.reset()
        self.engine = system.engine.engine
        self.name = self.label or self.describe(self.engine)
        self._sys = system
        if self.oracle:
            if trace is None:
                raise PolicyConfigError("oracle policies read the trace; they cannot run live")
            self._L = [r for g in trace.groups for r in g.resp_tokens]

    # ------------------------------------------------------------------ estimates
    def est_sample(self, v, sid: int) -> float:
        """Estimated total length of a sample."""
        s = v.sample(sid)
        if self.oracle:
            return float(self._L[sid])
        if self.eb is not None:
            self.eb.refresh(v)
            return self.eb.predict(v, sid, float(self.p["quantile"]))
        g = v.group(s.gidx)
        if s.length is not None:
            return float(s.length)
        if self.p["estimator"] == "group_evidence":
            k0 = float(self.p["k0"])
            m = (
                (k0 * g.est_tokens + g.finished_sum) / (k0 + g.finished_n)
                if (k0 + g.finished_n) > 0
                else g.est_tokens
            )
        else:
            m = g.est_tokens
        if s.started:
            return max(m, s.tokens_so_far + 1.0)
        return m

    def remaining(self, v, sid: int) -> float:
        s = v.sample(sid)
        return max(1.0, self.est_sample(v, sid) - s.tokens_so_far)

    def group_work(self, v, g) -> float:
        if self.oracle:
            return float(sum(self._L[x] for x in g.sids))
        return g.n_samples * g.est_tokens

    def worker_load(self, v, w) -> float:
        return sum(self.remaining(v, sid) for sid in w.sids())

    # ------------------------------------------------------------------ launch order
    def _next_group(self, v, skip: set[int]):
        """The next unlaunched group to launch (order rule within the window; a group that
        has been the oldest unlaunched one for window_groups launches goes next)."""
        k = int(self.p["window_groups"])
        win = [g for g in v.window(k + len(skip)) if g.gidx not in skip][:k]
        if not win:
            return None
        oldest = win[0]
        if oldest.gidx != self.oldest:
            self.oldest = oldest.gidx
            self.oldest_launches = 0
        if self.oldest_launches >= k or self.p["order"] == "fifo":
            return oldest
        if self.p["order"] == "shortest_first":
            return min(win, key=lambda g: (self.group_work(v, g), g.gidx))
        return min(win, key=lambda g: (-self.group_work(v, g), g.gidx))

    def _note_launch(self, gidx: int) -> None:
        if gidx == self.oldest:
            self.oldest = -1
            self.oldest_launches = 0
        else:
            self.oldest_launches += 1

    # ------------------------------------------------------------------ deadline (Tier 2)
    def _learn_rate(self, v) -> None:
        """Observed decoding time per token, from samples that finished (started -> generated)."""
        if v.t == self._rate_t:
            return
        self._rate_t = v.t
        for g in v.outstanding():
            if g.finished_n == 0:
                continue
            for sid in g.sids:
                if sid in self._rate_seen:
                    continue
                s = v.sample(sid)
                if s.generated_at is not None and s.started_at is not None and s.length:
                    self._rate_seen.add(sid)
                    self._rate_ms += s.generated_at - s.started_at
                    self._rate_tok += s.length

    def _misses_deadline(self, v, g) -> bool:
        """Predict whether a group launched now is ready before the selection of step
        ``v_pub + eta`` (the last step it is eligible for if it starts under ``v_pub``): ready
        ~ now + deadline * (its estimated length) * (observed ms per token); selection times
        extrapolated from the mean of the observed step intervals. Without enough history,
        launch."""
        if v.single_phase:
            return False
        self._learn_rate(v)
        sel = v.selection_times()
        if len(sel) < 2 or self._rate_tok <= 0:
            return False
        mean_step = (sel[-1] - sel[0]) / (len(sel) - 1)
        k = v.v_pub + v.eta  # last step it can serve
        t_dead = sel[-1] + (k - (len(sel) - 1)) * mean_step
        est = self.est_sample(v, g.sids[0]) if g.sids else g.est_tokens
        ready = v.t + float(self.p["deadline"]) * est * self._rate_ms / self._rate_tok
        return ready > t_dead

    def cap(self, v) -> int:
        if int(self.p["cap_groups"]) > 0:
            return int(self.p["cap_groups"])
        if self.p["straggler"] == "wait" or v.single_phase:
            return v.B
        return math.ceil(float(self.p["rho"]) * v.B)

    # ------------------------------------------------------------------ main hook
    def act(self, v) -> list:
        actions: list = []
        dropped: set[int] = set()
        if v.s_next > self.last_s:
            self.last_s = v.s_next
            if self.p["straggler"] == "abort":
                for g in v.outstanding():
                    actions.append(Drop(g.gidx))
                    dropped.add(g.gidx)
            elif self.p["straggler"] == "abort_pred":
                rem = {}
                for g in v.outstanding():
                    left = [sid for sid in g.sids if v.sample(sid).length is None]
                    if left:
                        rem[g.gidx] = max(self.remaining(v, sid) for sid in left)
                if rem:
                    med = sorted(rem.values())[(len(rem) - 1) // 2]
                    for gidx in sorted(rem):
                        if rem[gidx] > float(self.p["kappa"]) * med:
                            actions.append(Drop(gidx))
                            dropped.add(gidx)
        if v.engine == "static":
            actions += self._act_static(v, dropped)
        else:
            actions += self._act_continuous(v, dropped)
        return actions

    def _units(self, v, g, sids=None) -> list[tuple[int, list[int]]]:
        sids = sids if sids is not None else g.unplaced()
        if self.p["dispatch"] == "group":
            return [(g.gidx, sids)] if sids else []
        return [(g.gidx, [s]) for s in sids]

    def _act_continuous(self, v, dropped: set[int]) -> list:
        p = self.p
        workers = v.workers()
        n_out = v.n_outstanding() - len(dropped)
        cap = self.cap(v)
        late = p["binding"] == "late"
        if late:
            cand = [w for w in workers if w.accepting]
            if not cand:
                return []
            slots = {w.wid: v.max_seqs - w.running for w in cand}
            kv = {w.wid: w.free_kv for w in cand}
            res_new: dict[int, set[int]] = {w.wid: set() for w in cand}
            wobj = {w.wid: w for w in cand}
        elif (n_out >= cap or not v.gate_open or v.n_unlaunched() == 0) and not any(
            g.n_unplaced and (g.version is not None or v.gate_open) for g in v.outstanding()
        ):
            return []
        units: list[tuple[int, list[int]]] = []  # in the order they become candidates
        # (a) unplaced samples of launched groups (late binding with sample dispatch; and,
        # under either binding, samples returned to the pool by a lost worker)
        gate = v.gate_open
        for g in v.outstanding():
            if g.gidx in dropped or g.n_unplaced == 0 or (g.version is None and not gate):
                continue  # rule (2): no sample of a group that has not started through a closed gate
            units += self._units(v, g)
        out: list = []
        load: dict[int, float] = {}

        def fits(wid: int, gidx: int, sids: list[int]) -> bool:
            g = v.group(gidx)
            need = len(sids) * g.max_tokens
            if not (wobj[wid].resident(gidx) or gidx in res_new[wid]):
                need += g.prompt_tokens
            return slots[wid] >= len(sids) and kv[wid] >= need

        def take(wid: int, gidx: int, sids: list[int]) -> None:
            g = v.group(gidx)
            need = len(sids) * g.max_tokens
            if not (wobj[wid].resident(gidx) or gidx in res_new[wid]):
                need += g.prompt_tokens
                res_new[wid].add(gidx)
            slots[wid] = max(0, slots[wid] - len(sids))
            kv[wid] = max(0, kv[wid] - need) if kv[wid] >= need else -1

        def work(sids: list[int]) -> float:
            return sum(self.remaining(v, s) for s in sids)

        def too_big(gidx: int, sids: list[int]) -> bool:
            g = v.group(gidx)
            return (
                len(sids) > v.max_seqs or len(sids) * g.max_tokens + g.prompt_tokens > v.kv_tokens
            )

        def choose(gidx: int, sids: list[int]) -> int | None:
            if late:
                ok = [wid for wid in sorted(slots) if fits(wid, gidx, sids)]
                if not ok and too_big(gidx, sids):
                    # a unit larger than an empty worker goes to an empty worker (the rest
                    # of it waits there); otherwise late binding could never place it
                    ok = [
                        wid
                        for wid in sorted(slots)
                        if slots[wid] == v.max_seqs
                        and kv[wid] == v.kv_tokens
                        and not wobj[wid].waiting
                    ]
            else:
                ok = [w.wid for w in workers if w.alive]
            if not ok:
                return None
            a = p["assign"]
            if (
                late
                and iso_thr is not None
                and len(ok) > 1
                and max(self.remaining(v, s) for s in sids) > iso_thr
            ):
                return max(ok, key=lambda wid: (slots[wid], -wid))  # isolate a long unit
            if a == "first_free" or (len(ok) == 1 and a != "round_robin"):
                return ok[0]
            if a == "round_robin":
                n = len(workers)
                for k in range(n):
                    wid = (self.rr + k) % n
                    if wid in ok:
                        self.rr = wid + 1
                        return wid
                return None
            return min(ok, key=lambda wid: (get_load(wid), wid))

        def get_load(wid: int) -> float:
            if wid not in load:  # computed only when there is a choice to make
                load[wid] = self.worker_load(v, workers[wid])
            return load[wid]

        def place(gidx: int, sids: list[int], wid: int) -> None:
            if late:
                take(wid, gidx, sids)
            if wid in load:
                load[wid] += work(sids)
            if p["dispatch"] == "group":
                out.append(Place("group", gidx, wid))
            else:
                for s in sids:
                    out.append(Place("sample", s, wid))

        launched: set[int] = set()
        # (b) launches from the window
        new_units: list[tuple[int, list[int]]] = []
        if v.gate_open:
            capacity = sum(slots.values()) if late else None
            pending_n = sum(len(s) for _, s in units)
            while n_out + len(launched) < cap:
                if late and pending_n >= capacity:
                    break
                g = self._next_group(v, launched)
                if g is None:
                    break
                if (
                    float(p["deadline"]) > 0
                    and n_out + len(launched) >= v.B
                    and self._misses_deadline(v, g)
                ):
                    # deadline-aware admission governs only the surplus above B: do not launch a
                    # group predicted to go stale; the groups the next batch needs always launch
                    break
                launched.add(g.gidx)
                self._note_launch(g.gidx)
                us = self._units(v, g)
                new_units += us
                pending_n += sum(len(s) for _, s in us)
        all_units = units + new_units
        if not all_units:
            return []
        iso_thr = None
        if late and float(p["isolate"]) > 0 and len(all_units) > 1:
            rems = sorted(max(self.remaining(v, s) for s in u[1]) for u in all_units)
            iso_thr = float(p["isolate"]) * rems[(len(rems) - 1) // 2]
        if p["assign"] == "lpt":
            all_units = sorted(all_units, key=lambda u: (-work(u[1]), u[0], u[1][0]))
        placed_groups: set[int] = set()
        blocked_new = False
        for gidx, sids in all_units:
            is_new = gidx in launched and gidx not in placed_groups
            if is_new and blocked_new and p["assign"] != "lpt":
                continue
            wid = choose(gidx, sids)
            if wid is None:
                if late and is_new:
                    blocked_new = True
                continue
            place(gidx, sids, wid)
            placed_groups.add(gidx)
        # groups chosen for launch that got nothing placed were not launched: undo the note
        for _ in launched - placed_groups:
            if self.oldest_launches > 0:
                self.oldest_launches -= 1
        return out

    # ------------------------------------------------------------------ static engine
    def _act_static(self, v, dropped: set[int]) -> list:
        p = self.p
        idle = [w for w in v.workers() if w.accepting]
        if not idle:
            return []
        n_out = v.n_outstanding() - len(dropped)
        pool: list[int] = []
        for g in v.outstanding():
            if g.gidx not in dropped and (g.version is not None or v.gate_open):
                pool += g.unplaced()
        launched: set[int] = set()
        if v.gate_open:
            while n_out + len(launched) < self.cap(v):
                g = self._next_group(v, launched)
                if g is None:
                    break
                launched.add(g.gidx)
                self._note_launch(g.gidx)
                pool += g.unplaced()
        if not pool:
            return []
        cap = int(p["batch_size"]) or v.static_batch
        cap = min(cap, v.static_batch)
        est = {sid: self.est_sample(v, sid) for sid in pool}
        key = {sid: (v.sample(sid).gidx, v.sample(sid).sample_idx) for sid in pool}
        if p["batch"] == "fifo_chunk":
            order = sorted(pool, key=lambda s: key[s])
            batches = [order[i : i + cap] for i in range(0, len(order), cap)]
        elif p["batch"] == "sorted_chunk":
            order = sorted(pool, key=lambda s: (-est[s], key[s]))
            batches = [order[i : i + cap] for i in range(0, len(order), cap)]
        else:
            order = sorted(pool, key=lambda s: (est[s], key[s]))
            lens = [max(1, math.ceil(est[s])) for s in order]
            prompts = [v.group(v.sample(s).gidx).prompt_tokens for s in order]
            sysp = self._sys.engine

            def cost(b, lmax, i, j):
                return batch_cost_ms(sysp, b, max(prompts[i : j + 1]), lmax)

            _, segs = dp_partition(lens, cap, cost)
            batches = [order[i : j + 1] for i, j in segs]

            def bcost(bt):
                return batch_cost_ms(
                    sysp,
                    len(bt),
                    max(v.group(v.sample(s).gidx).prompt_tokens for s in bt),
                    max(max(1, math.ceil(est[s])) for s in bt),
                )

            batches.sort(key=lambda bt: (-bcost(bt), key[bt[0]]))
        out = []
        for w, bt in zip(idle, batches, strict=False):
            out.append(Place("batch", tuple(bt), w.wid))
        # groups selected for launch whose samples were not placed stay unlaunched
        placed = {v.sample(s).gidx for bt in batches[: len(idle)] for s in bt}
        for _ in launched - placed:
            if self.oldest_launches > 0:
                self.oldest_launches -= 1
        return out

    # ------------------------------------------------------------------ verifier order
    def pick_verification(self, v, queue) -> int:
        order = self.p["verifier"]
        if order == "fifo" or len(queue) == 1:
            return 0
        if order == "group_first":
            best, bk = 0, None
            for k, e in enumerate(queue):
                g = v.group(e.gidx)
                key = g.n_samples - g.n_verified
                if bk is None or key < bk:
                    best, bk = k, key
            return best
        best, bk = 0, None
        for k, e in enumerate(queue):
            key = v.verify_mean_s(e.task)
            if bk is None or key < bk:
                best, bk = k, key
        return best


# ---------------------------------------------------------------------- factory
PRESETS = {
    "reference": {
        "order": "fifo",
        "dispatch": "group",
        "binding": "early",
        "assign": "round_robin",
        "straggler": "wait",
        "estimator": "prior",
        "verifier": "fifo",
        "batch": "fifo_chunk",
    },
    "online_adaptive": {
        "order": "longest_first",
        "dispatch": "sample",
        "binding": "late",
        "assign": "lpt",
        "estimator": "group_evidence",
        "batch": "dp",
    },
    "oracle_lpt": {
        "order": "longest_first",
        "dispatch": "sample",
        "binding": "late",
        "assign": "lpt",
        "estimator": "oracle",
    },
    "oracle_dp": {"batch": "dp", "estimator": "oracle"},
    "dp": {"batch": "dp"},
    "lpt": {"order": "longest_first", "dispatch": "sample", "binding": "late", "assign": "lpt"},
    "quantile_lpt": {
        "order": "longest_first",
        "dispatch": "sample",
        "binding": "late",
        "assign": "lpt",
        "estimator": "eb",
        "quantile": 0.9,
        "batch": "dp",
    },
    "online_eb": {
        "order": "longest_first",
        "dispatch": "sample",
        "binding": "late",
        "assign": "lpt",
        "estimator": "eb",
        "isolate": 2.0,
        "batch": "dp",
    },
}


def make_policy(cfg: dict) -> ComposedPolicy:
    """Build a policy from ``{"name": ..., "params": {...}}``. ``name`` is a preset
    (reference, online_adaptive, oracle_lpt, oracle_dp, dp, lpt) or ``composed``; params
    override the preset's parts."""
    name = cfg.get("name", "composed")
    params = dict(cfg.get("params") or {})
    if name == "composed":
        base = {}
    elif name in PRESETS:
        base = dict(PRESETS[name])
    else:
        raise PolicyConfigError(f"unknown policy {name!r}")
    base.update(params)
    label = cfg.get("label")
    pol = ComposedPolicy(base, label=label)
    pol.preset = name
    return pol
