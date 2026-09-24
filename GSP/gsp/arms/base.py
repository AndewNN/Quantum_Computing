"""The common interface of the arms (PLAN §5 S4): `Arm.run(instance, cell, effort, seed) -> RunRecord`.

* `RunConfig` is what gets hashed into the run_id (PLAN §3.2): the REQUIRED_CONFIG_KEYS of `store.records` plus
  arm-specific scalar extras (ring_order, sector_source, init, schedule numbers, ...). Nothing else is hashed.
* `RunRecord` = run.json (flat), the trajectory arrays (trajectory.npz, PLAN §1.6), counts.json and, at
  n <= 14, the final statevector (final_state.npy, classical order).
* `Arm.run` is idempotent: a run whose run.json has status "done" is loaded, not recomputed (unless force).
  A failure writes status "failed" with the traceback, then re-raises.

Seeds come from the seed table (`restart_seed`), never from arithmetic at run time (PLAN §3.3).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .._version import HARNESS_VERSION
from ..store.ids import run_id as _run_id
from ..store.io import atomic_write_bytes, load_npz, save_npz
from ..store.paths import run_dir
from ..store.records import (check_config, is_done, mark_done, mark_failed, new_record, read_record,
                             run_json_path, write_record)

FINAL_STATE_MAX_N = 14


@lru_cache(maxsize=4)
def _seed_table(root=None):
    from ..instances.instance import load_seed_table
    return load_seed_table(root)


@lru_cache(maxsize=4)
def _restart_seeds(root=None) -> dict:
    """draw_id -> (restart_seed_r0 .. r4), read once from the seed table (S6: `gsp plan` asks for ~40k seeds)."""
    st = _seed_table(root)
    cols = sorted(c for c in st.columns if c.startswith("restart_seed_r"))
    if st["draw_id"].duplicated().any():
        raise ValueError("seed table has duplicate draw_ids")
    return {str(d): tuple(int(v) for v in row) for d, row in zip(st["draw_id"], st[cols].itertuples(index=False))}


def restart_seed(draw_id: str, r: int, root=None) -> int:
    """The restart seed r of a draw, read from results/instances/seed_table.csv (PLAN §1.1)."""
    seeds = _restart_seeds(root)
    if draw_id not in seeds:
        raise KeyError(f"{draw_id} is not in the seed table")
    return seeds[draw_id][int(r)]


def _arm_table() -> dict:
    """name in run.json -> factory (S7 added A3, A3d; S8 added A4, A6); `gsp plan` / `gsp run` / the post-run step
    find every arm through this table (nothing else needs to change)."""
    from .dbqite import A4, A6
    from .qaoa import A0, A1
    from .ramp import A2
    from .varqite import A3, A3d
    return {"A0": A0, "A1": A1, "A2p": lambda: A2("penalty"), "A2c": lambda: A2("confined"), "A3": A3, "A3d": A3d,
            "A4": A4, "A6": A6}


def registered_arms() -> tuple:
    return tuple(_arm_table())


def make_arm(name: str) -> "Arm":
    """An arm instance by its name in run.json (the post-run step rebuilds circuits through it; `_arm_table`)."""
    table = _arm_table()
    if name not in table:
        raise KeyError(f"no arm class registered for {name!r}")
    return table[name]()


def draw_of(inst_id: str) -> str:
    return inst_id.split("q")[0]


@dataclass(frozen=True)
class RunConfig:
    arm: str
    encoding: str
    inst_id: str
    effort_kind: str
    effort: int
    K: int | None = None
    rule: str | None = None
    connectivity: str | None = None
    restart: int = 0
    lam: float | None = None
    schedule: str | None = None
    seed: int | None = None
    seed_ga: int | None = None
    extras: tuple = ()                    # sorted ((key, scalar), ...), hashed
    harness_version: str = HARNESS_VERSION

    def to_dict(self) -> dict:
        d = {"arm": self.arm, "encoding": self.encoding, "inst_id": self.inst_id, "K": self.K, "rule": self.rule,
             "connectivity": self.connectivity, "effort_kind": self.effort_kind, "effort": int(self.effort),
             "restart": int(self.restart), "lam": None if self.lam is None else float(self.lam),
             "schedule": self.schedule, "seed": self.seed, "seed_ga": self.seed_ga,
             "harness_version": self.harness_version}
        for k, v in self.extras:
            if k in d:
                raise KeyError(f"extra {k!r} shadows a required key")
            d[k] = v
        return check_config(d)

    @property
    def run_id(self) -> str:
        return _run_id(self.to_dict())

    def extra(self, key, default=None):
        return dict(self.extras).get(key, default)


def make_extras(**kw) -> tuple:
    return tuple(sorted((k, v) for k, v in kw.items() if v is not None))


# S9a: informational runs (evidence for Sensei's open decisions, PLAN §4.1) carry this hashed extra, e.g.
# "O-2" or "O-11". No rule reads a run that has it (`stats.d1._match` drops them; S9b's lambda* must too).
EVIDENCE_KEY = "evidence"


def evidence_tag(evidence) -> str | None:
    """Validate the evidence label of an informational run (None = a normal run: the key is not written, so every
    default run_id is unchanged)."""
    if evidence is None:
        return None
    ev = str(evidence).strip()
    if not ev or ev != evidence:
        raise ValueError(f"evidence tag must be a non-empty stripped string, got {evidence!r}")
    return ev


def is_evidence(rec) -> bool:
    """True for a run.json / registry row / config dict that carries an evidence tag."""
    v = rec.get(EVIDENCE_KEY) if hasattr(rec, "get") else None
    if v is None:
        return False
    try:
        import math
        return not (isinstance(v, float) and math.isnan(v))
    except TypeError:                                  # pragma: no cover
        return True


_CORE_KEYS = ("arm", "encoding", "inst_id", "effort_kind", "effort", "K", "rule", "connectivity", "restart", "lam",
              "schedule", "seed", "seed_ga", "harness_version")


def runconfig_from_record(rec: dict) -> RunConfig:
    """The RunConfig of a stored run.json (its hash must reproduce the stored run_id)."""
    from ..store.records import config_of
    cfg = config_of(rec)
    extras = tuple(sorted((k, v) for k, v in cfg.items() if k not in _CORE_KEYS))
    rc = RunConfig(arm=cfg["arm"], encoding=cfg["encoding"], inst_id=cfg["inst_id"], effort_kind=cfg["effort_kind"],
                   effort=int(cfg["effort"]), K=cfg.get("K"), rule=cfg.get("rule"),
                   connectivity=cfg.get("connectivity"), restart=int(cfg.get("restart", 0)), lam=cfg.get("lam"),
                   schedule=cfg.get("schedule"), seed=cfg.get("seed"), seed_ga=cfg.get("seed_ga"), extras=extras,
                   harness_version=cfg["harness_version"])
    if rec.get("run_id") is not None and rc.run_id != rec["run_id"]:
        raise ValueError(f"run.json {rec['run_id']}: the rebuilt config hashes to {rc.run_id}")
    return rc


@dataclass
class RunRecord:
    record: dict
    trajectory: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    path: Path | None = None
    skipped: bool = False                 # loaded from a done run, not recomputed

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    @property
    def status(self) -> str:
        return self.record["status"]

    def metric(self, key: str):
        return self.record.get(key if key.startswith("metric_") else "metric_" + key)


@dataclass
class Outcome:
    """What an arm's `execute` returns."""
    metrics: dict
    timings: dict
    diagnostics: dict
    trajectory: dict
    counts: dict
    final_state: np.ndarray | None = None


def load_run(arm: str, rid: str, root=None) -> RunRecord:
    d = run_dir(arm, rid, root)
    rec = read_record(d / "run.json")
    traj = load_npz(d / "trajectory.npz") if (d / "trajectory.npz").exists() else {}
    counts = json.loads((d / "counts.json").read_text()) if (d / "counts.json").exists() else {}
    return RunRecord(record=rec, trajectory=traj, counts=counts, path=d, skipped=True)


class Arm:
    """Subclasses set `name`, `encoding`, `effort_kind`, and implement `config` and `execute`."""

    name = "?"
    encoding = "?"
    effort_kind = "?"

    def config(self, instance, cell, effort, seed, **kw) -> RunConfig:  # pragma: no cover (abstract)
        raise NotImplementedError

    def execute(self, cfg: RunConfig, instance, rulers, cell, logger: bool = True, root=None) -> Outcome:  # pragma: no cover
        raise NotImplementedError

    def ansatz(self, cfg: RunConfig, inst, root=None) -> tuple:  # pragma: no cover (abstract)
        """(Ansatz, sector_idx or None) of a config: what `execute` runs, rebuilt by the post-run step (S5)."""
        raise NotImplementedError

    def run(self, instance, cell, effort, seed=None, *, rulers=None, root=None, runs_root=None,
            store: bool = True, logger: bool = True, force: bool = False, finalize: bool = True,
            on_done=None, **kw) -> RunRecord:
        """Run (or load) one configuration. `instance` is an `Instance` (frozen or ad hoc) or an inst_id.
        `root` is where instances / sectors / the seed table are read; `runs_root` (default: root) is where the
        run directory is written. `finalize` (stored runs only): the S5 post-run step -- the 1000-shot sample
        (samples.npz) and the final-state metrics (postrun.json), `gsp.metrics.postrun.finalize_run`.
        `on_done(path, record)` (stored runs only; S6 queue runner) is called after run.json says "done" and before
        the post-run step: the runner marks its progress phase there. A run stopped inside it or inside the
        post-run step stays "done" without postrun.json; `gsp run` / `gsp metrics finalize` complete it."""
        from ..instances.adhoc import load_any
        from ..sim import backend
        from ..store.paths import inst_path

        if isinstance(instance, str):
            instance, rulers, adhoc = load_any(instance, root)
        else:
            adhoc = not inst_path(instance.inst_id, root).exists()
            if rulers is None:
                if adhoc:
                    raise ValueError("an ad hoc instance needs its rulers")
                from ..instances.instance import load_rulers
                rulers = load_rulers(instance.inst_id, root)
        out_root = root if runs_root is None else runs_root
        cfg = self.config(instance, cell, effort, seed, adhoc=adhoc, root=root, **kw)
        rid = cfg.run_id
        if store and not force and is_done(self.name, rid, out_root):
            return load_run(self.name, rid, out_root)
        backend.ensure_target()
        rec = new_record(cfg.to_dict(), backend.runtime_info())
        assert rec["run_id"] == rid
        if store:
            write_record(rec, out_root)
        t0 = time.perf_counter()
        try:
            out = self.execute(cfg, instance, rulers, cell, logger=logger, root=root)
        except BaseException as exc:
            mark_failed(rec, exc)
            if store:
                write_record(rec, out_root)
            raise
        wall = time.perf_counter() - t0
        mark_done(rec, out.metrics, out.timings, out.diagnostics, wall_s=wall)
        path = None
        if store:
            path = run_dir(self.name, rid, out_root)
            save_npz(path / "trajectory.npz", out.trajectory)
            atomic_write_bytes(path / "counts.json", json.dumps(out.counts, indent=1, sort_keys=True).encode())
            if out.final_state is not None and instance.n <= FINAL_STATE_MAX_N:
                buf = __import__("io").BytesIO()
                np.save(buf, np.asarray(out.final_state, dtype=np.complex128), allow_pickle=False)
                atomic_write_bytes(path / "final_state.npy", buf.getvalue())
            write_record(rec, out_root)
            assert run_json_path(rec, root=out_root).exists()
            if on_done is not None:
                on_done(path, rec)
            if finalize:
                from ..metrics.postrun import finalize_run
                finalize_run(path, root=root, final_state=out.final_state, catch=True)
        return RunRecord(record=rec, trajectory=out.trajectory, counts=out.counts, path=path)


def validate_run_dir(path) -> list:
    """Problems of a stored run (empty = valid): run.json schema + status done, trajectory fields of PLAN §1.6
    with consistent lengths, counts.json present."""
    from ..store.records import validate_record
    path = Path(path)
    probs = []
    try:
        rec = read_record(path / "run.json")
        validate_record(rec)
    except Exception as exc:
        return [f"run.json: {exc!r}"]
    if rec["status"] != "done":
        probs.append(f"status {rec['status']}")
    if not (path / "counts.json").exists():
        probs.append("counts.json missing")
    if not (path / "trajectory.npz").exists():
        return probs + ["trajectory.npz missing"]
    tr = load_npz(path / "trajectory.npz")
    need = ("t", "params", "energy", "ar_f", "p_feas", "eps_tilde", "p_opt", "p_top10", "circuits_charged",
            "g2q_ii", "g2q_iii", "wall")
    miss = [k for k in need if k not in tr]
    if miss:
        probs.append(f"trajectory misses {miss}")
        return probs
    T = tr["t"].size
    for k in need:
        if tr[k].shape[0] != T:
            probs.append(f"trajectory {k} has {tr[k].shape[0]} rows, t has {T}")
    for k in ("ar_f", "p_feas", "p_opt", "p_top10"):
        v = tr[k]
        if np.any(~np.isfinite(v)) or np.any(v < -1e-9) or np.any(v > 1 + 1e-9):
            probs.append(f"{k} outside [0, 1]")
    if np.any(np.diff(tr["circuits_charged"]) < 0):
        probs.append("circuits_charged decreases")
    return probs
