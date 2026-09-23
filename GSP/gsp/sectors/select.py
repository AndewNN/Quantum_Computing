"""Top-K sector lists at the confined cells, the brute-force reference, selection loss, and the
controllability check (PLAN §1.2, §5 S2, D-15).

Which sector files exist (the cells of `configs/envelope.yaml`):
  violation rule   one file per (draw, K): K in {6, 8, 12} at N = 4..7 on all 30 accepted draws, and
                   K = 24 at N = 5, 6 on the `k24_eligible` draws. The three q of a draw share it.
  objective rule   one file per (instance, K = 12) at N = 4..7 (90 instances per N).
One GA run per (draw, violation) and per (instance, objective) yields every K of that scope: the
lists of different K are the top-K prefixes of the same final ranking, as in the old
`get_top_n_individuals(K)`. The GA seed is `ga_seed_violation` / `ga_seed_objective` of the seed
table; the three q of a draw share the objective seed (PLAN §1.1 has no q in the formula).

Production = the GA list (D-15). The brute-force top-K under the same total order is the
reference; a difference is **selection loss** and the GA list is still used. Every sector file
stores both lists, the GA counters (WP5), and per instance: H_obj on the kept strings, the
best-in-sector energy of both lists against the band optimum, and the Altafini check for the ring
(lexicographic order, D-9) and the complete graph (A4's condition).

Stored order: `idx` is the kept list sorted ascending by classical index (x_0 = MSB), which is the
lexicographic order of the bitstrings (x_0 first) and the ring order of PLAN §2.2. `rank_idx` keeps
the GA's own ranking (best first).

Beyond the cells, `build(extension=True)` also runs both rules at N = 8..10 (every accepted draw /
instance) for the timing and GA-vs-BF tables only; no sector files are written for them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .._version import HARNESS_VERSION
from ..instances.bits import bitstring
from ..instances.instance import load_instance, load_instances_table, load_rulers, load_seed_table
from ..instances.rulers import objective_on
from ..store.ids import canonical_json
from ..store.io import atomic_write_bytes, load_npz, save_npz, sha256_bytes
from ..store.paths import configs_dir, sector_jobs_dir, sector_path, sectors_dir
from .control import altafini, closure_dim, edges_for
from .ga import TABLE_4_1, GAParams, brute_force, problem_from_instance, run_ga

SECTOR_SCHEMA = 1
K_ALL = (6, 8, 12, 24)
KEEP = max(K_ALL)
CELL_N = (4, 5, 6, 7)
EXTENSION_N = (8, 9, 10)


# --- the cell list -------------------------------------------------------------------------------
def load_cells() -> list[dict]:
    env = yaml.safe_load((configs_dir() / "envelope.yaml").read_text())
    return list(env["confined_cells"])


def cell_label(c: dict) -> str:
    return f"{c['connectivity']}|{c['rule']}|K{int(c['K'])}|N{int(c['N'])}"


def _accepted(seed: pd.DataFrame, N: int, k24_only: bool = False) -> pd.DataFrame:
    a = seed[(seed["N"] == N) & seed["accepted"].astype(bool)]
    if k24_only:
        a = a[a["k24_eligible"].astype(bool)]
    return a.sort_values("accept_rank")


def _q_ids(inst_table: pd.DataFrame, draw_id: str) -> list[str]:
    g = inst_table[inst_table["draw_id"] == draw_id].sort_values("q")
    return g["inst_id"].tolist()


@dataclass
class Job:
    """One GA run (and its BF reference): a draw under the violation rule or an instance under the
    objective rule."""

    rule: str
    scope_id: str            # draw_id (violation) or inst_id (objective)
    draw_id: str
    N: int
    e: int
    n: int
    seed: int
    inst_ids: list[str]      # instances using the files (3 q for violation, 1 for objective)
    F_size: int
    file_Ks: list[int]       # K with a sector file (the cells); empty for extension jobs
    cells: dict = field(default_factory=dict)   # K -> [cell labels]
    in_cell: bool = True

    @property
    def job_id(self) -> str:
        return f"{self.scope_id}_{self.rule}"

    @property
    def check_Ks(self) -> list[int]:
        """K with a GA-vs-BF comparison: every K of K_ALL the band can host."""
        return [K for K in K_ALL if K <= self.F_size]


def plan_jobs(root=None, extension: bool = True, N_values=None) -> list[Job]:
    """The GA jobs of the §1.2 cells (and, with `extension`, the N = 8..10 timing jobs)."""
    seed = load_seed_table(root)
    it = load_instances_table(root)
    cells = load_cells()
    jobs: dict[str, Job] = {}

    def job_for(rule, row, inst_ids, in_cell=True):
        scope = row.draw_id if rule == "violation" else inst_ids[0]
        key = f"{scope}_{rule}"
        if key not in jobs:
            jobs[key] = Job(rule=rule, scope_id=scope, draw_id=row.draw_id, N=int(row.N), e=int(row.e),
                            n=int(row.n), seed=int(row.ga_seed_violation if rule == "violation"
                                                   else row.ga_seed_objective),
                            inst_ids=list(inst_ids), F_size=int(row.F_eps), file_Ks=[], in_cell=in_cell)
        return jobs[key]

    for c in cells:
        N, K, rule = int(c["N"]), int(c["K"]), c["rule"]
        rows = _accepted(seed, N, k24_only=c.get("draws") == "k24_eligible")
        for row in rows.itertuples():
            qids = _q_ids(it, row.draw_id)
            groups = [qids] if rule == "violation" else [[i] for i in qids]
            for ids in groups:
                j = job_for(rule, row, ids)
                if K not in j.file_Ks:
                    j.file_Ks.append(K)
                j.cells.setdefault(K, []).append(cell_label(c))
    if extension:
        for N in EXTENSION_N:
            for row in _accepted(seed, N).itertuples():
                qids = _q_ids(it, row.draw_id)
                job_for("violation", row, qids, in_cell=False)
                for i in qids:
                    job_for("objective", row, [i], in_cell=False)
    out = sorted(jobs.values(), key=lambda j: (j.N, j.e, j.rule, j.scope_id))
    for j in out:
        j.file_Ks.sort()
    if N_values is not None:
        out = [j for j in out if j.N in set(N_values)]
    return out


def job_config(job: Job, params: GAParams, inst_sha: dict) -> dict:
    return {"schema": SECTOR_SCHEMA, "harness_version": HARNESS_VERSION, "rule": job.rule,
            "scope_id": job.scope_id, "seed": job.seed, "file_Ks": job.file_Ks, "keep": KEEP,
            "ga": params.as_dict(), "inst_sha256": {i: inst_sha[i] for i in job.inst_ids}}


def config_hash(cfg: dict) -> str:
    return sha256_bytes(canonical_json(cfg).encode())[:16]


# --- per-instance quantities of one kept list ----------------------------------------------------
def _control(E_sorted: np.ndarray, connectivity: str) -> dict:
    K = E_sorted.size
    r = altafini(E_sorted, edges_for(connectivity, K))
    d = r.d_eff if r.holds else closure_dim(E_sorted, edges_for(connectivity, K))
    return {"holds": r.holds, "connected": r.connected, "gaps_nonzero": r.gaps_nonzero,
            "gaps_distinct": r.gaps_distinct, "min_gap_rel": r.min_gap_rel, "min_sep_rel": r.min_sep_rel,
            "d_eff": int(d), "d_eff_source": "altafini" if r.holds else "closure"}


def instance_view(inst_id: str, idx_sorted: np.ndarray, rank_idx: np.ndarray, bf_idx: np.ndarray,
                  root=None) -> dict:
    """H_obj on the kept strings and everything derived from it, for one instance."""
    inst = load_instance(inst_id, root)
    rul = load_rulers(inst_id, root)
    E = objective_on(inst.QU_obj, idx_sorted)              # the rulers' exact formula
    E_bf = objective_on(inst.QU_obj, bf_idx)
    rng_ = rul.E_max - rul.E_min
    best_ga, best_bf = float(E.min()), float(E_bf.min())
    ring = _control(E, "ring")
    comp = _control(E, "complete")
    # the completed work's ring order (GA / BF rank order), for Sensei's D-9 note only
    pos = {int(x): k for k, x in enumerate(idx_sorted)}
    E_rank = E[[pos[int(x)] for x in rank_idx]]
    ring_rank = altafini(E_rank, edges_for("ring", E_rank.size))
    xstar = set(int(x) for x in rul.xstar_idx)
    return {
        "inst_id": inst_id, "q": float(inst.q), "E": E, "E_min": rul.E_min, "E_max": rul.E_max,
        "best_f_ga": best_ga, "best_f_bf": best_bf,
        "ar_best_ga": (rul.E_max - best_ga) / rng_, "ar_best_bf": (rul.E_max - best_bf) / rng_,
        "xstar_in_ga": bool(xstar & set(int(x) for x in idx_sorted)),
        "xstar_in_bf": bool(xstar & set(int(x) for x in bf_idx)),
        "ring": ring, "complete": comp, "ring_rank_holds": ring_rank.holds,
    }


# --- one job -------------------------------------------------------------------------------------
def run_job(job: Job, params: GAParams = TABLE_4_1, root=None, out_root=None) -> dict:
    """Run the GA and the BF reference of one job, write its sector files and its job record."""
    inst0 = load_instance(job.inst_ids[0], root)
    rul0 = load_rulers(job.inst_ids[0], root)
    problem = problem_from_instance(inst0, job.rule)
    assert problem.scope_id == job.scope_id, (problem.scope_id, job.scope_id)
    ga = run_ga(problem, job.seed, params, keep=KEEP, track=True)
    bf = brute_force(problem, keep=KEEP)

    # the GA's band test against the frozen band (every string)
    band = np.zeros(1 << job.n, bool)
    band[rul0.band_idx] = True
    n_band_disagree = int(np.count_nonzero(band != bf["in_band"]))
    # the BF ranking against the rulers, inside the band
    if job.rule == "violation":
        ruler_rank = rul0.band_idx[np.lexsort((rul0.band_idx, rul0.band_pen))]
    else:
        ruler_rank = rul0.band_idx[np.lexsort((rul0.band_idx, rul0.f_band))]
    bf_matches_rulers = all(set(bf["rank_idx"][:K].tolist()) == set(ruler_rank[:K].tolist())
                            for K in job.check_Ks)

    rec = {
        "job_id": job.job_id, "rule": job.rule, "scope_id": job.scope_id, "draw_id": job.draw_id,
        "N": job.N, "e": job.e, "n": job.n, "q": float(inst0.q) if job.rule == "objective" else float("nan"),
        "seed": job.seed, "in_cell": job.in_cell, "F_size": job.F_size, "file_Ks": job.file_Ks,
        **{f"ga_{k}": v for k, v in params.as_dict().items()},
        "ga_mutation_rate": params.mutation_rate(job.n),
        "ga_wall_s": ga.wall_s, "ga_wall_track_s": ga.wall_track_s, "ga_cpu_s": ga.cpu_s,
        "ga_n_evals": ga.n_evals, "ga_n_violation_evals": ga.n_violation_evals,
        "ga_n_objective_evals": ga.n_objective_evals, "ga_n_unique": ga.n_unique,
        "ga_n_distinct_final": ga.n_distinct_final,
        "bf_wall_s": bf["wall_s"], "bf_n_evals": bf["n_evals"],
        "n_band_disagree": n_band_disagree, "bf_matches_rulers": bool(bf_matches_rulers),
        "harness_version": HARNESS_VERSION,
    }
    for K in K_ALL:
        if K in job.check_Ks:
            g, b = set(ga.top(K).tolist()), set(bf["rank_idx"][:K].tolist())
            rec[f"agree_K{K}"] = g == b
            rec[f"missed_K{K}"] = len(b - g)
        else:
            rec[f"agree_K{K}"] = None
            rec[f"missed_K{K}"] = None
    if job.rule == "objective":
        # the standalone objective-aware GA as a classical solver (WP5)
        from ..baselines.ga_solver import solver_record
        rec.update(solver_record(ga, inst0, rul0))

    sectors = []
    for K in job.file_Ks:
        rank_idx = ga.top(K).astype(np.int64)
        if rank_idx.size < K:
            raise RuntimeError(f"{job.job_id}: GA returned {rank_idx.size} < {K} distinct strings")
        idx = np.sort(rank_idx)
        bf_rank = bf["rank_idx"][:K].astype(np.int64)
        bf_idx = np.sort(bf_rank)
        missing = np.array(sorted(set(bf_idx.tolist()) - set(idx.tolist())), np.int64)
        extra = np.array(sorted(set(idx.tolist()) - set(bf_idx.tolist())), np.int64)
        viol = bf["viol"][idx]
        views = [instance_view(i, idx, rank_idx, bf_idx, root) for i in job.inst_ids]
        arrays = {
            "schema": SECTOR_SCHEMA, "harness_version": HARNESS_VERSION, "source": "GA",
            "scope_id": job.scope_id, "rule": job.rule, "K": K, "N": job.N, "e": job.e, "n": job.n,
            "draw_id": job.draw_id, "seed": job.seed, "config_hash": "",
            "idx": idx, "rank_idx": rank_idx,
            "bitstrings": np.array([bitstring(int(x), job.n) for x in idx]),
            "bf_idx": bf_idx, "bf_rank_idx": bf_rank, "identical": bool(missing.size == 0),
            "n_missing": int(missing.size), "missing_idx": missing, "extra_idx": extra,
            "viol": viol, "max_delta": float(np.sqrt(viol.max())),
            "n_out_of_band": int(np.count_nonzero(~band[idx])),
            **{f"ga_{k}": v for k, v in params.as_dict().items()},
            "ga_mutation_rate": params.mutation_rate(job.n),
            "ga_wall_s": ga.wall_s, "ga_cpu_s": ga.cpu_s, "ga_n_evals": ga.n_evals,
            "ga_n_violation_evals": ga.n_violation_evals, "ga_n_objective_evals": ga.n_objective_evals,
            "ga_n_unique": ga.n_unique, "ga_n_distinct_final": ga.n_distinct_final,
            "bf_wall_s": bf["wall_s"], "bf_n_evals": bf["n_evals"],
            "inst_ids": np.array(job.inst_ids), "q": np.array([v["q"] for v in views]),
            "E": np.stack([v["E"] for v in views]),
            "E_min": np.array([v["E_min"] for v in views]), "E_max": np.array([v["E_max"] for v in views]),
            "best_f_ga": np.array([v["best_f_ga"] for v in views]),
            "best_f_bf": np.array([v["best_f_bf"] for v in views]),
            "ar_best_ga": np.array([v["ar_best_ga"] for v in views]),
            "ar_best_bf": np.array([v["ar_best_bf"] for v in views]),
        }
        for conn in ("ring", "complete"):
            for k in ("holds", "min_gap_rel", "min_sep_rel", "d_eff"):
                arrays[f"{conn}_{k}"] = np.array([v[conn][k] for v in views])
        arrays["ring_rank_holds"] = np.array([v["ring_rank_holds"] for v in views])
        sectors.append((K, arrays, views, missing, extra))
    return {"record": rec, "sectors": sectors, "ga": ga, "bf": bf}


def write_job(job: Job, out: dict, cfg: dict, out_root=None) -> dict:
    """Write the sector files and the job JSON; returns the JSON record."""
    h = config_hash(cfg)
    rows = []
    for K, arrays, views, missing, extra in out["sectors"]:
        arrays = dict(arrays)
        arrays["config_hash"] = h
        p = sector_path(job.scope_id, job.rule, K, out_root)
        save_npz(p, arrays)
        for v in views:
            rows.append({
                "sector_file": p.name, "scope_id": job.scope_id, "inst_id": v["inst_id"], "q": v["q"],
                "rule": job.rule, "K": K, "N": job.N, "e": job.e, "n": job.n, "draw_id": job.draw_id,
                "cells": job.cells.get(K, []), "identical": bool(missing.size == 0),
                "n_missing": int(missing.size), "max_delta": float(arrays["max_delta"]),
                "n_out_of_band": int(arrays["n_out_of_band"]),
                "E_min": v["E_min"], "E_max": v["E_max"], "best_f_ga": v["best_f_ga"],
                "best_f_bf": v["best_f_bf"], "ar_best_ga": v["ar_best_ga"], "ar_best_bf": v["ar_best_bf"],
                "xstar_in_ga": v["xstar_in_ga"], "xstar_in_bf": v["xstar_in_bf"],
                **{f"ring_{k}": v["ring"][k] for k in v["ring"]},
                **{f"complete_{k}": v["complete"][k] for k in v["complete"]},
                "ring_rank_holds": v["ring_rank_holds"],
            })
    rec = {**out["record"], "config_hash": h, "config": cfg, "sectors": rows}
    p = sector_jobs_dir(out_root) / f"{job.job_id}.json"
    atomic_write_bytes(p, (json.dumps(rec, indent=1, default=_json_default) + "\n").encode())
    return rec


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def job_is_current(job: Job, cfg: dict, out_root=None) -> bool:
    p = sector_jobs_dir(out_root) / f"{job.job_id}.json"
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return False
    if rec.get("config_hash") != config_hash(cfg):
        return False
    return all(sector_path(job.scope_id, job.rule, K, out_root).exists() for K in job.file_Ks)


def build(root=None, out_root=None, params: GAParams = TABLE_4_1, extension: bool = True,
          N_values=None, force: bool = False, log=print) -> dict:
    """Run every job that is missing or stale, then rebuild the tables. Idempotent."""
    out_root = root if out_root is None else out_root
    it = load_instances_table(root)
    inst_sha = dict(zip(it["inst_id"], it["inst_sha256"]))
    jobs = plan_jobs(root, extension=extension, N_values=N_values)
    ran = skipped = 0
    for j in jobs:
        cfg = job_config(j, params, inst_sha)
        if not force and job_is_current(j, cfg, out_root):
            skipped += 1
            continue
        out = run_job(j, params, root, out_root)
        write_job(j, out, cfg, out_root)
        ran += 1
        if log and ran % 50 == 0:
            log(f"  {ran} jobs run ({j.job_id})")
    tabs = build_tables(out_root)
    if log:
        log(f"sectors: {ran} jobs run, {skipped} current; {len(tabs['sectors'])} (sector, instance) rows")
    return tabs


# --- tables --------------------------------------------------------------------------------------
def load_job_records(out_root=None) -> list[dict]:
    d = sector_jobs_dir(out_root)
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def build_tables(out_root=None) -> dict:
    recs = load_job_records(out_root)
    runs = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sectors", "config", "file_Ks")}
                         | {"file_Ks": " ".join(str(k) for k in r["file_Ks"])} for r in recs])
    rows = [s | {"cells": " ".join(s["cells"])} for r in recs for s in r["sectors"]]
    sec = pd.DataFrame(rows)
    d = sectors_dir(out_root)
    d.mkdir(parents=True, exist_ok=True)
    if not runs.empty:
        runs.to_parquet(d / "ga_runs.parquet", index=False)
    if not sec.empty:
        sec.to_parquet(d / "sectors.parquet", index=False)
    return {"runs": runs, "sectors": sec}


def load_tables(out_root=None) -> dict:
    d = sectors_dir(out_root)
    return {"runs": pd.read_parquet(d / "ga_runs.parquet"), "sectors": pd.read_parquet(d / "sectors.parquet")}


# --- the read side for the arms ------------------------------------------------------------------
@dataclass(frozen=True)
class Sector:
    """One instance's view of a sector file: what A1, A2c and A4 load (PLAN §1.2, slide 30)."""

    inst_id: str
    rule: str
    K: int
    n: int
    idx: np.ndarray           # kept strings, ascending classical index (lexicographic; the ring order)
    rank_idx: np.ndarray      # the GA ranking, best first
    bitstrings: np.ndarray
    E: np.ndarray             # H_obj(idx) of this instance, un-boosted
    identical_to_bf: bool
    seed: int
    path: Path
    arrays: dict


def sector_scope(inst_id: str, rule: str) -> str:
    return inst_id.split("q")[0] if rule == "violation" else inst_id


def load_sector(inst_id: str, rule: str, K: int, root=None) -> Sector:
    p = sector_path(sector_scope(inst_id, rule), rule, K, root)
    z = load_npz(p)
    ids = [str(s) for s in z["inst_ids"]]
    if inst_id not in ids:
        raise KeyError(f"{inst_id} is not an instance of {p.name}")
    k = ids.index(inst_id)
    return Sector(inst_id=inst_id, rule=rule, K=int(z["K"]), n=int(z["n"]), idx=z["idx"],
                  rank_idx=z["rank_idx"], bitstrings=z["bitstrings"], E=z["E"][k],
                  identical_to_bf=bool(z["identical"]), seed=int(z["seed"]), path=p, arrays=z)
