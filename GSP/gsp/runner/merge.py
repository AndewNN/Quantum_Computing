"""`gsp merge --src DIR`: fold a pulled shard of results/runs (a remote host's, mirrored by
`scripts/remote/pull.sh` into results/incoming/<tag>/runs) into the local store (S6, PLAN §3.2).

Runs are directories keyed by run_id, so shards merge by copying directories. Rules:
  * only runs whose run.json is valid (schema + the run_id hashes its config + the directory is named by it) and
    says "done" are copied; running / failed / unreadable shard runs are counted and left alone;
  * a local run that is already "done" is never overwritten (a "done" pair that differs is reported as a
    conflict, the local copy wins);
  * a local run that is absent, "running" or "failed" is replaced by the shard's done run: the copy is staged
    in a temporary sibling directory and swapped in with renames, so a crash leaves either version whole.
The registry is rebuilt afterwards (`gsp index`).
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
from pathlib import Path

from ..store.paths import runs_dir
from ..store.records import validate_record


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def merge_runs(src, out_root=None, dry_run: bool = False, log=None) -> dict:
    src = Path(src)
    dst_base = runs_dir(out_root)
    out = {"copied": 0, "replaced": 0, "same": 0, "conflict_kept_local": 0, "not_done_skipped": 0,
           "invalid_skipped": 0, "conflicts": [], "invalid": []}
    for rj in sorted(src.glob("*/*/run.json")):
        d = rj.parent
        arm, rid = d.parent.name, d.name
        rec = _read(rj)
        try:
            if rec is None:
                raise ValueError("unreadable run.json")
            validate_record(rec)
            if rec["run_id"] != rid or rec.get("arm") != arm:
                raise ValueError(f"directory {arm}/{rid} holds run {rec.get('arm')}/{rec['run_id']}")
        except Exception as exc:
            out["invalid_skipped"] += 1
            out["invalid"].append(f"{arm}/{rid}: {exc}")
            continue
        if rec["status"] != "done":
            out["not_done_skipped"] += 1
            continue
        target = dst_base / arm / rid
        local = _read(target / "run.json") if (target / "run.json").exists() else None
        if local is not None and local.get("status") == "done":
            same = filecmp.cmp(rj, target / "run.json", shallow=False)
            if same:
                out["same"] += 1
            else:
                out["conflict_kept_local"] += 1
                out["conflicts"].append(f"{arm}/{rid}: local done {local.get('finished_at')} on {local.get('host')}"
                                        f" vs shard {rec.get('finished_at')} on {rec.get('host')}")
            continue
        if dry_run:
            out["replaced" if target.exists() else "copied"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.parent / f".{rid}.merge-{os.getpid()}"
        if stage.exists():
            shutil.rmtree(stage)
        shutil.copytree(d, stage)
        if target.exists():
            old = target.parent / f".{rid}.old-{os.getpid()}"
            os.replace(target, old)
            os.replace(stage, target)
            shutil.rmtree(old)
            out["replaced"] += 1
        else:
            os.replace(stage, target)
            out["copied"] += 1
        if log:
            log(f"merged {arm}/{rid} ({'replaced ' + str(local.get('status')) if local else 'new'})")
    return out
