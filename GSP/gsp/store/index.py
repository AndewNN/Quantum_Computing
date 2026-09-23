"""The registry: one row per run.json, rebuilt by scanning (PLAN §3.2).

Shards from any host merge by rsync of results/runs plus `gsp index`. Notebooks and reports read
through `load_registry()` / `load_metrics()` and never parse paths.
"""

from __future__ import annotations

import json

import pandas as pd

from .paths import registry_path, runs_dir, tables_dir
from .records import read_record


def scan_runs(root=None) -> list[dict]:
    rows = []
    base = runs_dir(root)
    if not base.exists():
        return rows
    for p in sorted(base.glob("*/*/run.json")):
        try:
            rec = read_record(p)
        except (OSError, json.JSONDecodeError) as exc:
            rec = {"run_id": p.parent.name, "arm": p.parent.parent.name, "status": "unreadable",
                   "error": repr(exc)}
        rec["run_dir"] = str(p.parent.relative_to(base))
        rows.append(rec)
    return rows


def build_index(root=None) -> pd.DataFrame:
    df = pd.DataFrame(scan_runs(root))
    if not df.empty:
        df = df.sort_values(["arm", "run_id"], kind="stable").reset_index(drop=True)
    out = registry_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return df


def load_registry(root=None, rebuild: bool = False) -> pd.DataFrame:
    p = registry_path(root)
    if rebuild or not p.exists():
        return build_index(root)
    return pd.read_parquet(p)


def load_metrics(root=None) -> pd.DataFrame:
    p = tables_dir(root) / "metrics.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist yet (written by `gsp aggregate`, S5)")
    return pd.read_parquet(p)
