"""The sqlite3 store of the live service: one file (WAL) with groups, samples, attempts,
workers, steps, and an append-only event log with timestamps (the replay metadata).

Writes are buffered and committed once per controller instant, so a crash loses at most the
instant in progress. ``export_trace`` rebuilds a rollout trace (schema v1) from the store: the
lengths the workers reported and the verification durations the verifiers took, so a live run
can be replayed through the simulator.
"""

from __future__ import annotations

import json
import sqlite3

from rollout_engine.trace.schema import GroupRow, Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS groups (
  gidx INTEGER PRIMARY KEY, gid TEXT NOT NULL UNIQUE, task TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL, max_tokens INTEGER NOT NULL, n_samples INTEGER NOT NULL,
  est_tokens REAL NOT NULL, submitted_ms INTEGER NOT NULL, state TEXT NOT NULL,
  version INTEGER, step INTEGER, drop_reason TEXT);
CREATE TABLE IF NOT EXISTS samples (
  sid INTEGER PRIMARY KEY, gidx INTEGER NOT NULL, sidx INTEGER NOT NULL, state TEXT NOT NULL,
  worker INTEGER, attempts INTEGER NOT NULL DEFAULT 0, tokens INTEGER, finish_reason TEXT,
  version INTEGER, gen_ms INTEGER, ver_start_ms INTEGER, ver_end_ms INTEGER);
CREATE TABLE IF NOT EXISTS attempts (
  sid INTEGER NOT NULL, attempt INTEGER NOT NULL, worker INTEGER NOT NULL, placed_ms INTEGER NOT NULL,
  outcome TEXT, ended_ms INTEGER, PRIMARY KEY (sid, attempt));
CREATE TABLE IF NOT EXISTS workers (
  wid INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, max_seqs INTEGER, kv_tokens INTEGER, tp INTEGER,
  registered_ms INTEGER NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS steps (
  step INTEGER PRIMARY KEY, t_sel_ms INTEGER NOT NULL, groups TEXT NOT NULL, staleness TEXT NOT NULL,
  tokens INTEGER NOT NULL, t_train_end_ms INTEGER, t_pub_ms INTEGER);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, t_ms INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._events: list[tuple] = []
        self._groups: dict[int, tuple] = {}
        self._samples: dict[int, tuple] = {}
        self._attempts: list[tuple] = []
        self._attempt_end: list[tuple] = []
        self._steps: dict[int, tuple] = {}
        self._workers: dict[int, tuple] = {}

    # -- meta -----------------------------------------------------------------------------
    def set_meta(self, key: str, value) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value, sort_keys=True))
        )

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def has_data(self) -> bool:
        return self.db.execute("SELECT COUNT(*) FROM groups").fetchone()[0] > 0

    # -- buffered writes ------------------------------------------------------------------
    def event(self, t: int, kind: str, payload: dict) -> None:
        self._events.append((t, kind, json.dumps(payload, sort_keys=True)))

    def group(self, g, submitted_ms: int | None = None) -> None:
        sp = g.spec
        prev = self._groups.get(g.gidx)
        sub = submitted_ms if submitted_ms is not None else (prev[7] if prev else None)
        self._groups[g.gidx] = (
            g.gidx,
            sp.gid,
            sp.task,
            sp.prompt_tokens,
            sp.max_tokens,
            sp.n_samples,
            sp.est_tokens,
            sub,
            g.state,
            g.version,
            g.step,
            g.drop_reason or None,
        )

    def sample(self, s, finish_reason: str | None = None) -> None:
        self._samples[s.sid] = (
            s.sid,
            s.gidx,
            s.sidx,
            s.state,
            s.worker,
            s.attempts,
            s.tokens if s.gen_t >= 0 or s.removed else None,
            finish_reason,
            s.version if s.started else None,
            s.gen_t if s.gen_t >= 0 else None,
            s.ver_start if s.ver_start >= 0 else None,
            s.ver_end if s.ver_end >= 0 else None,
        )

    def attempt(self, sid: int, attempt: int, worker: int, placed_ms: int) -> None:
        self._attempts.append((sid, attempt, worker, placed_ms))

    def attempt_end(self, sid: int, attempt: int, outcome: str, ended_ms: int) -> None:
        self._attempt_end.append((outcome, ended_ms, sid, attempt))

    def step(self, st) -> None:
        self._steps[st.s] = (
            st.s,
            st.t_sel,
            json.dumps(st.groups),
            json.dumps(st.staleness),
            st.tokens,
            st.t_train_end if st.t_train_end >= 0 else None,
            st.t_pub if st.t_pub >= 0 else None,
        )

    def worker(self, wid: int, name: str, info: dict, registered_ms: int, state: str) -> None:
        self._workers[wid] = (
            wid,
            name,
            info.get("max_seqs"),
            info.get("kv_tokens"),
            info.get("tp"),
            registered_ms,
            state,
        )

    def flush(self) -> None:
        if not (
            self._events
            or self._groups
            or self._samples
            or self._attempts
            or self._attempt_end
            or self._steps
            or self._workers
        ):
            return
        db = self.db
        db.execute("BEGIN")
        try:
            if self._groups:
                db.executemany(
                    "INSERT INTO groups VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(gidx) DO UPDATE SET "
                    "state=excluded.state, version=excluded.version, step=excluded.step, drop_reason=excluded.drop_reason",
                    [
                        r if r[7] is not None else r[:7] + (0,) + r[8:]
                        for r in self._groups.values()
                    ],
                )
            if self._samples:
                db.executemany(
                    "INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    list(self._samples.values()),
                )
            if self._attempts:
                db.executemany(
                    "INSERT OR REPLACE INTO attempts (sid, attempt, worker, placed_ms) VALUES (?,?,?,?)",
                    self._attempts,
                )
            if self._attempt_end:
                db.executemany(
                    "UPDATE attempts SET outcome=?, ended_ms=? WHERE sid=? AND attempt=?",
                    self._attempt_end,
                )
            if self._steps:
                db.executemany(
                    "INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?,?)",
                    list(self._steps.values()),
                )
            if self._workers:
                db.executemany(
                    "INSERT OR REPLACE INTO workers VALUES (?,?,?,?,?,?,?)",
                    list(self._workers.values()),
                )
            if self._events:
                db.executemany(
                    "INSERT INTO events (t_ms, kind, payload) VALUES (?,?,?)", self._events
                )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        self._events, self._attempts, self._attempt_end = [], [], []
        self._groups, self._samples, self._steps, self._workers = {}, {}, {}, {}

    # -- reads ----------------------------------------------------------------------------
    def load(self) -> dict:
        db = self.db
        cols = lambda table: [d[1] for d in db.execute(f"PRAGMA table_info({table})")]  # noqa: E731
        out = {}
        for table, order in (
            ("groups", "gidx"),
            ("samples", "sid"),
            ("steps", "step"),
            ("workers", "wid"),
        ):
            c = cols(table)
            out[table] = [
                dict(zip(c, r, strict=True))
                for r in db.execute(f"SELECT * FROM {table} ORDER BY {order}")
            ]
        row = db.execute("SELECT MAX(t_ms) FROM events").fetchone()
        out["last_ms"] = row[0] or 0
        return out

    def last_ms(self) -> int:
        row = self.db.execute("SELECT MAX(t_ms) FROM events").fetchone()
        return row[0] or 0

    def events(self, kind: str | None = None) -> list[tuple]:
        q = (
            "SELECT t_ms, kind, payload FROM events"
            + (" WHERE kind = ?" if kind else "")
            + " ORDER BY seq"
        )
        return [(t, k, json.loads(p)) for t, k, p in self.db.execute(q, (kind,) if kind else ())]

    def close(self) -> None:
        self.flush()
        self.db.close()


def export_trace(store: Store, only_complete: bool = True) -> Trace:
    """A rollout trace of the groups whose samples all have a reported length and a measured
    verification time (lengths and verification durations as observed)."""
    data = store.load()
    by_g: dict[int, list[dict]] = {}
    for s in data["samples"]:
        by_g.setdefault(s["gidx"], []).append(s)
    groups = []
    for g in data["groups"]:
        ss = sorted(by_g.get(g["gidx"], []), key=lambda s: s["sidx"])
        ok = len(ss) == g["n_samples"] and all(
            s["tokens"] is not None
            and s["gen_ms"] is not None
            and s["ver_end_ms"] is not None
            and s["ver_start_ms"] is not None
            for s in ss
        )
        if not ok:
            if only_complete:
                continue
            raise ValueError(f"group {g['gid']} is incomplete")
        groups.append(
            GroupRow(
                g["gid"],
                g["task"],
                g["prompt_tokens"],
                g["max_tokens"],
                g["est_tokens"],
                tuple(s["tokens"] for s in ss),
                tuple(s["ver_end_ms"] - s["ver_start_ms"] for s in ss),
            )
        )
    return Trace(tuple(groups))
