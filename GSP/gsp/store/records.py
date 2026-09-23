"""The run.json schema (PLAN §3.2).

run.json is **flat** (one level of scalar values), so the registry is a plain table:
  config keys  (hashed into run_id; REQUIRED_CONFIG_KEYS plus any arm-specific scalar keys)
  run_id, status (running | done | failed), error (traceback, failed runs only)
  started_at, finished_at (ISO-8601 UTC), wall_s
  host, python, git_sha, git_dirty          (recorded, never hashed)
  cudaq_version, target, target_option, driver_version, gpu_name   (from gsp.sim.backend)
  metric_*    final metrics            time_*    timings            diag_*    arm diagnostics
Heavy arrays never go here: trajectory.npz / counts.json / samples.npz sit next to run.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import socket
import subprocess
import sys
import traceback
from functools import lru_cache
from pathlib import Path

from .ids import _canon, run_id as _run_id, with_version
from .io import atomic_write_bytes
from .paths import GSP_ROOT, run_dir

SCHEMA_VERSION = 1
STATUSES = ("running", "done", "failed")

# Every RunConfig carries these (None where an arm has no such axis).
REQUIRED_CONFIG_KEYS = (
    "arm",            # A0 .. A6, A3d
    "encoding",       # penalty | confined
    "inst_id",
    "K",              # sector size (confined) or None
    "rule",           # violation | objective | None
    "connectivity",   # ring | complete | adaptive | None
    "effort_kind",    # depth | ramp_depth | steps
    "effort",         # L, p or step cap
    "restart",        # r
    "lam",            # lambda (penalty arms) or None
    "schedule",       # ramp schedule tag or None
    "seed",           # restart seed from the seed table, or None
    "seed_ga",        # GA seed of the sector, or None
    "harness_version",
)

RECORD_KEYS = (
    "schema_version", "run_id", "status", "error", "started_at", "finished_at", "wall_s",
    "host", "python", "git_sha", "git_dirty",
    "cudaq_version", "target", "target_option", "driver_version", "gpu_name",
)
NON_CONFIG_PREFIXES = ("metric_", "time_", "diag_")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@lru_cache(maxsize=1)
def git_info() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(["git", "-C", str(GSP_ROOT), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(GSP_ROOT), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return sha, dirty
    except Exception:
        return None, None


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def check_config(config: dict) -> dict:
    """Canonicalize and validate a RunConfig; returns the canonical copy (with harness_version)."""
    cfg = _canon(with_version(config))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"RunConfig misses {missing}")
    for k, v in cfg.items():
        if k in RECORD_KEYS or k.startswith(NON_CONFIG_PREFIXES):
            raise KeyError(f"{k!r} is a reserved run.json key, not a config key")
        if not _is_scalar(v):
            raise TypeError(f"config value {k}={v!r} is not a scalar (run.json is flat)")
    return cfg


def config_of(record: dict) -> dict:
    return {k: v for k, v in record.items()
            if k not in RECORD_KEYS and not k.startswith(NON_CONFIG_PREFIXES)}


def new_record(config: dict, runtime: dict | None = None) -> dict:
    cfg = check_config(config)
    sha, dirty = git_info()
    rec = dict(cfg)
    rec.update({
        "schema_version": SCHEMA_VERSION,
        "run_id": _run_id(cfg),
        "status": "running",
        "error": None,
        "started_at": _now(),
        "finished_at": None,
        "wall_s": None,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "git_sha": sha,
        "git_dirty": dirty,
        "cudaq_version": None, "target": None, "target_option": None,
        "driver_version": None, "gpu_name": None,
    })
    if runtime:
        for k in ("cudaq_version", "target", "target_option", "driver_version", "gpu_name"):
            rec[k] = runtime.get(k)
    return rec


def _prefixed(prefix: str, values: dict) -> dict:
    out = {}
    for k, v in (values or {}).items():
        v = _canon(v)
        if not _is_scalar(v):
            raise TypeError(f"{prefix}{k}={v!r} is not a scalar (heavy data goes to trajectory.npz)")
        out[k if k.startswith(prefix) else prefix + k] = v
    return out


def mark_done(record: dict, metrics: dict | None = None, timings: dict | None = None,
              diagnostics: dict | None = None, wall_s: float | None = None) -> dict:
    record.update(_prefixed("metric_", metrics))
    record.update(_prefixed("time_", timings))
    record.update(_prefixed("diag_", diagnostics))
    record["status"] = "done"
    record["finished_at"] = _now()
    record["wall_s"] = None if wall_s is None else float(wall_s)
    return record


def mark_failed(record: dict, exc: BaseException | None = None) -> dict:
    record["status"] = "failed"
    record["finished_at"] = _now()
    record["error"] = ("".join(traceback.format_exception(exc)) if exc is not None
                       else "".join(traceback.format_exception(*sys.exc_info())))
    return record


def validate_record(record: dict) -> None:
    for k in ("schema_version", "run_id", "status"):
        if k not in record:
            raise KeyError(f"run.json misses {k!r}")
    if record["status"] not in STATUSES:
        raise ValueError(f"bad status {record['status']!r}")
    for k, v in record.items():
        if not _is_scalar(v):
            raise TypeError(f"run.json is flat; {k} is {type(v).__name__}")
    cfg = check_config(config_of(record))
    if _run_id(cfg) != record["run_id"]:
        raise ValueError("run_id does not match the hashed config")
    if record["status"] == "failed" and not record.get("error"):
        raise ValueError("a failed run must record its traceback")


def run_json_path(record_or_arm, run_id: str | None = None, root=None) -> Path:
    if isinstance(record_or_arm, dict):
        return run_dir(record_or_arm["arm"], record_or_arm["run_id"], root) / "run.json"
    return run_dir(record_or_arm, run_id, root) / "run.json"


def write_record(record: dict, root=None) -> Path:
    validate_record(record)
    path = run_json_path(record, root=root)
    data = json.dumps(record, sort_keys=True, indent=1, allow_nan=True).encode()
    atomic_write_bytes(path, data)
    return path


def read_record(path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_done(arm: str, rid: str, root=None) -> bool:
    p = run_json_path(arm, rid, root)
    if not p.exists():
        return False
    try:
        return read_record(p).get("status") == "done"
    except (OSError, json.JSONDecodeError):
        return False
