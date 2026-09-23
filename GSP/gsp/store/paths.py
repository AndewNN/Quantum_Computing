"""Where things live (PLAN §3.1). Notebooks and reports never build these paths themselves.

`GSP_RESULTS` (env) overrides the results root, e.g. for tests.
"""

from __future__ import annotations

import os
from pathlib import Path

GSP_ROOT = Path(__file__).resolve().parents[2]          # .../GSP


def results_root(root=None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("GSP_RESULTS")
    return Path(env) if env else GSP_ROOT / "results"


def reports_dir() -> Path:
    return GSP_ROOT / "reports"


def configs_dir() -> Path:
    return GSP_ROOT / "configs"


def instances_dir(root=None) -> Path:
    return results_root(root) / "instances"


def seed_table_path(root=None) -> Path:
    return instances_dir(root) / "seed_table.csv"


def instances_parquet_path(root=None) -> Path:
    return instances_dir(root) / "instances.parquet"


def checksums_path(root=None) -> Path:
    return instances_dir(root) / "CHECKSUMS"


def inst_path(inst_id: str, root=None) -> Path:
    return instances_dir(root) / f"inst_{inst_id}.npz"


def rulers_path(inst_id: str, root=None) -> Path:
    return instances_dir(root) / f"rulers_{inst_id}.npz"


def sectors_dir(root=None) -> Path:
    return results_root(root) / "sectors"


def runs_dir(root=None) -> Path:
    return results_root(root) / "runs"


def run_dir(arm: str, run_id: str, root=None) -> Path:
    return runs_dir(root) / arm / run_id


def index_dir(root=None) -> Path:
    return results_root(root) / "index"


def registry_path(root=None) -> Path:
    return index_dir(root) / "registry.parquet"


def tables_dir(root=None) -> Path:
    return results_root(root) / "tables"


def figures_dir(root=None) -> Path:
    return results_root(root) / "figures"
