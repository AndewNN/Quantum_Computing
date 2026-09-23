"""The S4 checks of PLAN §5 S4: the N = 4 smoke run, the anneal check and the reproduction check. Each writes a
table under results/tables/ and stores its runs under results/runs/ (idempotent: done runs are loaded).

smoke        every §1.2 cell at N = 4, one instance (the first accepted draw, q = 1.5): A1 and A2c on the 5
             confined cells, A0 and A2p on the penalty side (lam = 0.005, a placeholder until S9's lambda*);
             A0 / A1 at depths 5, 7, 9 (restart 0) plus restarts 1-4 at depth 5 on A0 and the A1 baseline;
             A2 at every ramp depth x both schedules. Every run directory is validated.
anneal       A2 at the primary schedule, p = 5 ... 300, on 3 instances per encoding (N = 7, q = 1.5: A2p at
             lam = 0.005, A2c on the baseline ring violation K = 12 sector), with RAMP_SIGN = -1 (production) and
             +1: r(p) = (E(p) - E_min) / (E_max - E_min) over what the arm can reach (all 2^n strings of H(lam) /
             the sector's strings of H_obj) must fall toward 0 with -1 and rise toward 1 with +1.
repro        A1 ring K = 12, N = 5, depth 5, BF sector, rank-order ring, legacy_init, restart 0, e = 0, 1, 2 (e = 2 is
             a rejected draw, built ad hoc) vs the completed run's AR2; A0 at lam = 0.005 the same way.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..store.paths import tables_dir
from .base import validate_run_dir
from .qaoa import A0, A1
from .ramp import A2, RAMP_DEPTHS

SMOKE_LAM = 0.005
DEPTHS = (5, 7, 9)
N_RESTARTS = 5
COMPLETED_EXP = Path(__file__).resolve().parents[3] / "CUDA" / "experiments_approx_Q2_RAND_S1.0_W0.01_Jh"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_table(df: pd.DataFrame, name: str, root=None) -> Path:
    d = tables_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    df.to_parquet(p, index=False)
    return p


def _row(rec, **extra) -> dict:
    r = rec.record
    out = {"arm": r["arm"], "run_id": r["run_id"], "inst_id": r["inst_id"], "K": r["K"], "rule": r["rule"],
           "connectivity": r["connectivity"], "effort": r["effort"], "restart": r["restart"], "lam": r["lam"],
           "schedule": r["schedule"], "status": r["status"], "skipped": rec.skipped, "wall_s": r["wall_s"],
           "path": str(rec.path) if rec.path else None}
    for k, v in r.items():
        if k.startswith(("metric_", "time_", "diag_")):
            out[k] = v
    out.update(extra)
    return out


def first_instance(N: int, q: float = 1.5, root=None) -> str:
    from ..instances.instance import load_instances_table
    it = load_instances_table(root)
    g = it[(it["N"] == N) & (it["q"] == q)].sort_values("accept_rank")
    return str(g.iloc[0]["inst_id"])


def n4_cells(root=None) -> list:
    from ..sectors.select import load_cells
    return [c for c in load_cells() if int(c["N"]) == 4]


# --- smoke -------------------------------------------------------------------------------------------------
def smoke(root=None, runs_root=None, log=print) -> pd.DataFrame:
    iid = first_instance(4, 1.5, root)
    rows = []
    t0 = time.perf_counter()

    def rec(r, **kw):
        probs = validate_run_dir(r.path) if r.path else ["not stored"]
        rows.append(_row(r, valid=not probs, problems="; ".join(probs), **kw))
        if log:
            log(f"  {r.record['arm']:<3} {r.run_id} {kw.get('cell', '')} effort={r.record['effort']} "
                f"r={r.record['restart']} AR_F={r.record.get('metric_ar_f'):.4f} valid={not probs}"
                f"{' (loaded)' if r.skipped else ''}")

    for L in DEPTHS:
        for rr in (range(N_RESTARTS) if L == DEPTHS[0] else (0,)):
            rec(A0().run(iid, None, L, restart=rr, lam=SMOKE_LAM, root=root, runs_root=runs_root), cell="penalty")
    for p in RAMP_DEPTHS:
        for sch in ("primary", "secondary"):
            rec(A2("penalty").run(iid, None, p, schedule_tag=sch, lam=SMOKE_LAM, root=root, runs_root=runs_root),
                cell="penalty")
    for c in n4_cells(root):
        cell = {"connectivity": c["connectivity"], "rule": c["rule"], "K": int(c["K"])}
        label = f"{c['connectivity']}|{c['rule']}|K{int(c['K'])}"
        baseline = c.get("axis") == "baseline"
        for L in DEPTHS:
            for rr in (range(N_RESTARTS) if (baseline and L == DEPTHS[0]) else (0,)):
                rec(A1().run(iid, cell, L, restart=rr, root=root, runs_root=runs_root), cell=label)
        for p in RAMP_DEPTHS:
            for sch in ("primary", "secondary"):
                rec(A2("confined").run(iid, cell, p, schedule_tag=sch, root=root, runs_root=runs_root), cell=label)
    df = pd.DataFrame(rows)
    _write_table(df, "s4_smoke.parquet", runs_root or root)
    if log:
        log(f"smoke: {len(df)} runs, {int(df.valid.sum())} valid, {time.perf_counter() - t0:.0f} s")
    return df


# --- anneal ------------------------------------------------------------------------------------------------
def _reach_range(inst, lam, sector_idx):
    """(E_min, E_max, E_start) over what the arm reaches; E_start = the start state's energy (the mean over all
    strings for |+>^n, over the kept strings for the uniform sector state)."""
    from ..instances.encode import qubo_energies
    if sector_idx is None:
        diag = -qubo_energies(inst.qubo(lam), lam)
        return float(diag.min()), float(diag.max()), float(diag.mean())
    from ..instances.rulers import objective_on
    E = objective_on(inst.QU_obj, np.asarray(sector_idx))
    return float(E.min()), float(E.max()), float(E.mean())


def anneal(root=None, runs_root=None, N: int = 7, n_inst: int = 3, log=print) -> pd.DataFrame:
    from ..instances.instance import load_instance, load_instances_table
    from .qaoa import sector_view
    it = load_instances_table(root)
    ids = it[(it["N"] == N) & (it["q"] == 1.5)].sort_values("accept_rank")["inst_id"].tolist()[:n_inst]
    cell = {"connectivity": "ring", "rule": "violation", "K": 12}
    rows = []
    for iid in ids:
        inst = load_instance(iid, root)
        sv = sector_view(inst, "violation", 12, "ga", root)
        for enc in ("penalty", "confined"):
            arm = A2(enc)
            lo, hi, st = _reach_range(inst, SMOKE_LAM, None) if enc == "penalty" else _reach_range(inst, None, sv.idx)
            for sign in (-1, 1):
                for p in RAMP_DEPTHS:
                    kw = dict(schedule_tag="primary", sign=sign, root=root, runs_root=runs_root)
                    if enc == "penalty":
                        r = arm.run(iid, None, p, lam=SMOKE_LAM, **kw)
                    else:
                        r = arm.run(iid, cell, p, **kw)
                    E = r.record["metric_energy"]
                    rows.append({"inst_id": iid, "encoding": enc, "arm": arm.name, "sign": sign, "p": p, "energy": E,
                                 "E_min": lo, "E_max": hi, "r": (E - lo) / (hi - lo), "r_start": (st - lo) / (hi - lo),
                                 "ar_f": r.record["metric_ar_f"],
                                 "p_feas": r.record["metric_p_feas"], "run_id": r.run_id})
            if log:
                sub = pd.DataFrame([x for x in rows if x["inst_id"] == iid and x["encoding"] == enc])
                for sign in (-1, 1):
                    s = sub[sub.sign == sign].sort_values("p")
                    log(f"  {iid} {enc:<8} sign {sign:+d}: r(p) = " + " ".join(f"{v:.3f}" for v in s.r))
    df = pd.DataFrame(rows)
    _write_table(df, "s4_anneal.parquet", runs_root or root)
    return df


def anneal_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """Per (instance, encoding, sign), with r_start = r of the start state (p = 0):
    plan  (PLAN §5 S4, "energy decreases toward E_min as p grows; flipped, it goes toward E_max"):
          sign -1: r(300) < r(5) < r_start and Spearman(p, r) <= -0.8; sign +1: r(300) > r(5) > r_start;
    strict (set by S4 before the data, stricter than the plan): additionally r(300) <= 0.1 (sign -1) or
          r(300) >= 0.9 and Spearman >= 0.8 (sign +1)."""
    out = []
    for (iid, enc, sign), g in df.groupby(["inst_id", "encoding", "sign"]):
        g = g.sort_values("p")
        r5, r300 = float(g.r.iloc[0]), float(g.r.iloc[-1])
        r0 = float(g.r_start.iloc[0]) if "r_start" in g else float("nan")
        rho = float(pd.Series(g.p.values).rank().corr(pd.Series(g.r.values).rank()))
        if sign < 0:
            plan = r300 < r5 < r0 and rho <= -0.8
            strict = plan and r300 <= 0.1
        else:
            plan = r300 > r5 > r0
            strict = plan and r300 >= 0.9 and rho >= 0.8
        out.append({"inst_id": iid, "encoding": enc, "sign": int(sign), "r_start": r0, "r5": r5, "r300": r300,
                    "r_min": float(g.r.min()), "r_max": float(g.r.max()), "p_at_max": int(g.p.iloc[int(np.argmax(g.r.values))]),
                    "spearman": rho, "pass": bool(plan), "strict": bool(strict)})
    return pd.DataFrame(out)


# --- reproduction ------------------------------------------------------------------------------------------
def completed_ar2(kind: str, e: int, L: int = 5, N: int = 5) -> float:
    f = (COMPLETED_EXP / "exp_L1_q1.5" / "report_Preserving12_boost_Jh_AR2.csv" if kind == "A1"
         else COMPLETED_EXP / "exp_L0.005_q1.5" / "report_X_boost_Jh_AR2.csv")
    d = pd.read_csv(f)
    row = d[(d.Assets == N) & (d.Layer == L) & (d.Exp == e) & (d.Seed == 0) & (d.Point == 0)]
    return float(row.iloc[0]["Approximate_ratio"])


def completed_band(kind: str, L: int = 5, N: int = 5) -> tuple:
    f = (COMPLETED_EXP / "exp_L1_q1.5" / "report_Preserving12_boost_Jh_AR2.csv" if kind == "A1"
         else COMPLETED_EXP / "exp_L0.005_q1.5" / "report_X_boost_Jh_AR2.csv")
    d = pd.read_csv(f)
    v = d[(d.Assets == N) & (d.Layer == L) & (d.Seed == 0) & (d.Point == 0)]["Approximate_ratio"]
    return float(v.min()), float(v.max())


def repro(root=None, runs_root=None, es=(0, 1, 2), log=print) -> pd.DataFrame:
    rows = []
    cell = {"connectivity": "ring", "rule": "violation", "K": 12}
    for e in es:
        iid = f"N05e{e:03d}q1.5"
        r1 = A1().run(iid, cell, 5, restart=0, init="legacy", ring_order="rank", sector_source="bf", root=root,
                      runs_root=runs_root)
        r0 = A0().run(iid, None, 5, restart=0, lam=SMOKE_LAM, root=root, runs_root=runs_root)
        for kind, r in (("A1", r1), ("A0", r0)):
            ref = completed_ar2(kind, e)
            lo, hi = completed_band(kind)
            ar = r.record["metric_ar_f"]
            rows.append({"arm": kind, "e": e, "inst_id": iid, "adhoc": bool(r.record.get("inst_adhoc", False)),
                         "ar_f": ar, "ar2_completed": ref, "delta": ar - ref, "abs_delta": abs(ar - ref),
                         "completed_band_lo": lo, "completed_band_hi": hi, "in_plan_band": 0.25 <= ar <= 0.8,
                         "iterations": r.record["metric_iterations"], "converged": r.record["metric_converged"],
                         "per_iter_s": r.record["time_per_iter_s"], "run_id": r.run_id})
            if log:
                log(f"  {kind} e={e}: AR_F {ar:.4f} vs completed AR2 {ref:.4f} (delta {ar - ref:+.4f}), "
                    f"{r.record['metric_iterations']} it")
    df = pd.DataFrame(rows)
    _write_table(df, "s4_repro.parquet", runs_root or root)
    return df


def write_json(obj, name: str, root=None) -> Path:
    d = tables_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps(obj, indent=1, default=float))
    return p
