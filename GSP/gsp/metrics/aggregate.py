"""`gsp aggregate`: one row per done run -> results/tables/metrics.parquet (read back with
`gsp.store.load_metrics()`), built on the S1 registry (`store.index.build_index` / `load_registry`).

Row = the registry row (config, run.json metrics `metric_*`, timings `time_*`, diagnostics `diag_*`) plus:
  N, q, draw_id                          from the inst_id
  conv_*, ar_f_final                     `convergence.convergence` on trajectory.npz
  per_unit_* / per_circuit_* / layer_* / start_* / conv_exec_* / total_*   `resources.resources` on counts.json
  pre_*                                  `preprocessing.preprocessing` (sector file join; confined arms)
  st_* (recomputed state metrics), ar_best_S, p_any_feasible_S, p_opt_seen_S, sd_* (simulation difficulty),
  ar_best_S_sampled, p_feas_sampled, sample_*   from postrun.json (`postrun.finalize_run`); without it, the state
                                         metrics are recomputed on the CPU from final_state.npy (no sample columns)
  chk_* and `anomalies`                  consistency checks (below); `anomalies` is "" for a clean run.
Checks (tolerances are engineering choices, S5): the replayed final state vs final_state.npy (<= 1e-10); the state
metrics vs run.json's metric_* (<= 1e-10); the trajectory's last row vs run.json (<= 1e-12); validate_run_dir; the
conv g2q identity; the sample's tail probability under the exact best-of-S distribution (>= 1e-4) and the binomial
z of its feasible count (|z| <= 5); confined arms: 1 - p_sector <= 1e-12; the postrun step's status.
Incremental: rows whose (run_id, finished_at, postrun file state, METRICS_VERSION) key is unchanged are reused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..store.io import load_npz
from ..store.paths import runs_dir, tables_dir
from .convergence import convergence
from .postrun import POSTRUN_FILE, SAMPLES_FILE, read_postrun, rebuild, state_metrics
from .preprocessing import preprocessing
from .resources import resources
from .state import METRIC_KEYS

METRICS_VERSION = 1
STATE_KEYS = METRIC_KEYS + ("p_sector", "norm")
TOL_REPLAY = 1e-10
TOL_STATE_VS_RECORD = 1e-10
TOL_TRAJ_VS_RECORD = 1e-12
TOL_SAMPLE_TAIL = 1e-4
TOL_FEAS_Z = 5.0
TOL_LEAK = 1e-12


def metrics_path(root=None) -> Path:
    return tables_dir(root) / "metrics.parquet"


def _agg_key(rec_row: dict, d: Path) -> str:
    p = d / POSTRUN_FILE
    st = p.stat() if p.exists() else None
    return json.dumps([rec_row.get("run_id"), rec_row.get("finished_at"),
                       None if st is None else [st.st_size, int(st.st_mtime_ns)], METRICS_VERSION])


def _clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def run_row(reg_row: dict, root=None) -> dict:
    """Every derived metric of one done run (module doc)."""
    from ..arms.base import validate_run_dir
    from ..store.ids import parse_inst_id
    from ..store.records import read_record
    d = runs_dir(root) / reg_row["run_dir"]
    row = {k: _clean(v) for k, v in reg_row.items()}
    rec = read_record(d / "run.json")            # exact values (the registry is a union of every arm's keys)
    pi = parse_inst_id(rec["inst_id"])
    row.update({"N": pi["N"], "q": pi["q"], "draw_id": rec["inst_id"].split("q")[0], "run_path": str(d)})
    anomalies = []
    probs = validate_run_dir(d)
    row["chk_run_dir_problems"] = "; ".join(probs)
    if probs:
        anomalies.append("invalid_run_dir")
    tr = load_npz(d / "trajectory.npz")
    counts = json.loads((d / "counts.json").read_text())
    conv = convergence(tr)
    row.update(conv)
    row.update(resources(counts, conv, tr))
    if row.get("chk_conv_g2q") is False:
        anomalies.append("conv_g2q")
    row.update(preprocessing(rec, root))
    # the trajectory's last row against run.json
    dl = [abs(float(tr[k][-1]) - float(rec[f"metric_{k}"])) for k in METRIC_KEYS
          if k in tr and rec.get(f"metric_{k}") is not None and np.isfinite(tr[k][-1])]
    row["chk_traj_vs_record"] = max(dl) if dl else None
    if dl and max(dl) > TOL_TRAJ_VS_RECORD:
        anomalies.append("traj_vs_record")
    # final-state metrics: postrun.json, else recomputed from final_state.npy
    pr = read_postrun(d)
    if pr is not None and pr.get("status") == "done":
        sm = pr
        row["postrun_source"] = "postrun"
    else:
        if pr is not None:
            anomalies.append("postrun_failed")
        sm = None
        if (d / "final_state.npy").exists():
            _, _, _, A, _, ctx = rebuild(rec, root)
            K = int(rec["K"]) if rec.get("K") is not None else None
            sm = state_metrics(np.load(d / "final_state.npy"), ctx, K=K)
            row["postrun_source"] = "final_state"
        else:
            row["postrun_source"] = None
            anomalies.append("no_final_state_metrics")
    if sm is not None:
        for k, v in sm.items():
            if k in ("status", "run_id", "arm", "postrun_version", "error"):
                continue
            row[f"st_{k}" if k in STATE_KEYS else k] = v
        ds = [abs(float(sm[k]) - float(rec[f"metric_{k}"])) for k in METRIC_KEYS
              if sm.get(k) is not None and rec.get(f"metric_{k}") is not None]
        row["chk_state_vs_record"] = max(ds) if ds else None
        if ds and max(ds) > TOL_STATE_VS_RECORD:
            anomalies.append("state_vs_record")
        if rec.get("encoding") == "confined" and sm.get("p_sector") is not None and 1.0 - sm["p_sector"] > TOL_LEAK:
            anomalies.append("leakage")
    if row.get("replay_max_abs") is not None and row["replay_max_abs"] > TOL_REPLAY:
        anomalies.append("replay")
    if row.get("sample_tail_p") is not None and row["sample_tail_p"] < TOL_SAMPLE_TAIL:
        anomalies.append("sample_tail")
    if row.get("sample_feas_z") is not None and abs(row["sample_feas_z"]) > TOL_FEAS_Z:
        anomalies.append("sample_feas_z")
    row["has_samples"] = (d / SAMPLES_FILE).exists()
    row["anomalies"] = ",".join(anomalies)
    row["metrics_version"] = METRICS_VERSION
    row["agg_key"] = _agg_key(row, d)
    return row


def aggregate(root=None, rebuild_all: bool = False, write: bool = True, log=None) -> pd.DataFrame:
    from ..store.index import build_index
    reg = build_index(root)
    if reg.empty:
        df = pd.DataFrame()
    else:
        done = reg[reg["status"] == "done"]
        old = {}
        p = metrics_path(root)
        if p.exists() and not rebuild_all:
            prev = pd.read_parquet(p)
            if "agg_key" in prev:
                old = {r["run_id"]: r for r in prev.to_dict("records")}
        rows, n_new = [], 0
        for r in done.to_dict("records"):
            d = runs_dir(root) / r["run_dir"]
            key = _agg_key({k: _clean(v) for k, v in r.items()}, d)
            o = old.get(r["run_id"])
            if o is not None and o.get("agg_key") == key:
                rows.append(o)
                continue
            rows.append(run_row(r, root))
            n_new += 1
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["arm", "run_id"], kind="stable").reset_index(drop=True)
        if log:
            log(f"{len(df)} done runs, {n_new} (re)computed, {len(df) - n_new} reused")
    if write:
        out = metrics_path(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)
    return df
