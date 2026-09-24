"""`gsp missing`: which cells / instances / efforts of a plan or a queue lack done runs, in one call (S6).

Every spec gets one state, from its run.json (matched by run_id) and its post-run files:
  done          status done, postrun.json (status done) + samples.npz present
  unfinalized   status done, but the post-run step did not complete (stopped between "done" and the post-run step,
                or postrun.json says failed): `gsp run` on the queue or `gsp metrics finalize` completes it
  failed        status failed (the traceback is in run.json)
  running       status running (in flight, or stale after a SIGKILL; see `gsp progress`)
  absent        no run.json
  placeholder   not runnable yet (arm not registered, or an S9 input missing: lambda*, L0, the recursion cap)
Anything but "done" is missing. The summary groups by (arm, cell) and lists, per instance, the missing efforts
(L = depth, p/schedule = ramp depth, k = recursion steps; r = restart). `gsp missing` exits 0 when every planned run
is done and 1 otherwise, so scripts can test it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..store.paths import run_dir
from .plan import effort_label, spec_line
from .queue import postrun_state, run_state

STATES = ("done", "unfinalized", "failed", "running", "absent", "placeholder")


def spec_state(s: dict, out_root=None) -> str:
    if s.get("placeholder"):
        return "placeholder"
    if not s.get("run_id"):
        return "absent"
    st, _ = run_state(s["arm"], s["run_id"], out_root)
    if st == "done":
        return "done" if postrun_state(run_dir(s["arm"], s["run_id"], out_root)) == "finalized" else "unfinalized"
    if st in ("failed", "running"):
        return st
    return "absent"


def coverage(specs: list[dict], out_root=None) -> pd.DataFrame:
    rows = []
    for s in specs:
        rows.append({"arm": s["arm"], "cell": s.get("cell_label") or _cell_str(s), "N": s.get("N"),
                     "inst_id": s["inst_id"], "effort": effort_label(s), "restart": s.get("restart", 0),
                     "run_id": s.get("run_id"), "placeholder": s.get("placeholder", ""),
                     "state": spec_state(s, out_root), "_spec": s})
    return pd.DataFrame(rows, columns=["arm", "cell", "N", "inst_id", "effort", "restart", "run_id", "placeholder",
                                       "state", "_spec"])


def _cell_str(s: dict) -> str:
    c = s.get("cell")
    if not c:
        return f"penalty|N{s.get('N', '?')}"
    return f"{c['connectivity']}|{c['rule']}|K{c['K']}"


def summarize(df: pd.DataFrame, max_lines: int = 40) -> str:
    if df.empty:
        return "no specs"
    tot = df["state"].value_counts()
    n_missing = int((df["state"] != "done").sum())
    lines = [f"{len(df)} planned runs: " + ", ".join(f"{s} {int(tot.get(s, 0))}" for s in STATES)
             + f"  ->  {n_missing} missing (not done)"]
    g = df.groupby(["arm", "cell"], sort=False)["state"].value_counts().unstack(fill_value=0)
    for s in STATES:
        if s not in g:
            g[s] = 0
    g = g[list(STATES)]
    g["planned"] = g.sum(axis=1)
    inc = g[g["done"] < g["planned"]]
    if inc.empty:
        lines.append("every (arm, cell) is complete")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"(arm, cell) with missing runs: {len(inc)} of {len(g)}")
    w = max(len(c) for c in inc.index.get_level_values(1)) + 1
    lines.append(f"{'arm':<5} {'cell':<{w}} {'planned':>8} " + " ".join(f"{s:>11}" for s in STATES))
    shown = 0
    for (a, c), r in inc.iterrows():
        if shown >= max_lines:
            lines.append(f"... {len(inc) - shown} more (arm, cell) rows")
            break
        lines.append(f"{a:<5} {c:<{w}} {int(r['planned']):>8} " + " ".join(f"{int(r[s]):>11}" for s in STATES))
        shown += 1
    miss = df[df["state"] != "done"]
    lines.append("")
    lines.append("missing, per (arm, cell) -> instance: efforts (state):")
    per = defaultdict(lambda: defaultdict(list))
    for r in miss.itertuples(index=False):
        eff = f"{r.effort} r{r.restart}" if r.restart else r.effort
        per[(r.arm, r.cell)][r.inst_id].append(f"{eff} [{r.state}]")
    shown = 0
    for (a, c), insts in per.items():
        if shown >= max_lines:
            lines.append(f"... {len(per) - shown} more (arm, cell) groups (use --list or --out)")
            break
        n_inst = len(insts)
        items = list(insts.items())
        if n_inst <= 4:
            body = "; ".join(f"{i}: {', '.join(e)}" for i, e in items)
        else:
            effs = list(dict.fromkeys(x for _, e in items for x in e))      # plan order (numeric efforts)
            body = (f"{n_inst} instances ({items[0][0]} ... {items[-1][0]}); efforts: "
                    + ", ".join(effs[:12]) + (" ..." if len(effs) > 12 else ""))
        lines.append(f"  {a} | {c} | {body}")
        shown += 1
    return "\n".join(lines)


def as_json(df: pd.DataFrame) -> dict:
    by = df.groupby(["arm", "cell"], sort=False)["state"].value_counts().unstack(fill_value=0)
    return {"total": int(len(df)), "states": {s: int((df["state"] == s).sum()) for s in STATES},
            "missing": int((df["state"] != "done").sum()),
            "cells": [{"arm": a, "cell": c, **{s: int(r.get(s, 0)) for s in STATES}} for (a, c), r in by.iterrows()],
            "missing_runs": [{k: v for k, v in r.items() if k != "_spec"}
                             for r in df[df["state"] != "done"].to_dict("records")]}


def write_missing_queue(df: pd.DataFrame, path) -> int:
    """The runnable missing specs (every state but done / placeholder) as a new queue file."""
    from ..store.io import atomic_write_bytes
    rows = [r for r in df.to_dict("records") if r["state"] not in ("done", "placeholder")]
    atomic_write_bytes(Path(path), ("".join(spec_line(r["_spec"]) + "\n" for r in rows)).encode())
    return len(rows)


def list_lines(df: pd.DataFrame) -> str:
    miss = df[df["state"] != "done"]
    return "\n".join(f"{r.state:<12} {r.arm:<4} {r.cell:<28} {r.inst_id} {r.effort} r{r.restart} "
                     f"{r.run_id or '-'} {r.placeholder}" for r in miss.itertuples(index=False))


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
