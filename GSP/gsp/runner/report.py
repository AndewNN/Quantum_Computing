"""`gsp report`: the harness report STUB (S6). S15 builds the real one (figures, the second manuscript's tables).

Today it states what the store holds: runs per arm x status, the metrics table (rows, anomalies), and the queues
with their last progress. Printed; `write=True` also writes reports/runs.md.
"""

from __future__ import annotations

import datetime as _dt
import json

from ..store.paths import logs_dir, queues_dir, reports_dir


def markdown(root=None) -> str:
    from ..store.index import build_index
    reg = build_index(root)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    lines = ["# Harness report (stub; S15 builds it out)", "", f"Generated {now} by `gsp report`.", "",
             "## Runs in the store (registry)", ""]
    if reg.empty:
        lines.append("No runs.")
    else:
        t = reg.groupby(["arm", "status"]).size().unstack(fill_value=0)
        cols = list(t.columns)
        lines.append("| arm | " + " | ".join(cols) + " | total |")
        lines.append("|---" * (len(cols) + 2) + "|")
        for a, r in t.iterrows():
            lines.append(f"| {a} | " + " | ".join(str(int(r[c])) for c in cols) + f" | {int(r.sum())} |")
    lines += ["", "## Metrics table", ""]
    try:
        from ..store.index import load_metrics
        m = load_metrics(root)
        bad = int((m["anomalies"] != "").sum()) if "anomalies" in m else 0
        lines.append(f"`tables/metrics.parquet`: {len(m)} rows, {bad} with anomalies (`gsp aggregate`).")
    except FileNotFoundError:
        lines.append("`tables/metrics.parquet` does not exist yet (`gsp aggregate`).")
    lines += ["", "## Queues", ""]
    qs = sorted(queues_dir(root).glob("*.jsonl")) if queues_dir(root).exists() else []
    if not qs:
        lines.append("No queue files.")
    for q in qs:
        n = sum(1 for ln in q.read_text().splitlines() if ln.strip())
        pp = logs_dir(root) / f"queue_{q.stem}.progress.json"
        st = ""
        if pp.exists():
            pr = json.loads(pp.read_text())
            c = pr.get("counts", {})
            st = (f"; last runner: {pr.get('state')} at {pr.get('updated_at')}, done {c.get('done_total')}, "
                  f"failed now {c.get('failed_now')}")
        lines.append(f"- `{q.name}`: {n} specs{st} (`gsp missing --queue {q}` for the gaps)")
    return "\n".join(lines) + "\n"


def write(root=None):
    p = reports_dir() / "runs.md"
    p.write_text(markdown(root))
    return p
