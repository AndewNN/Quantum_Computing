"""`gsp q`: edit a queue file while a dynamic runner (`gsp run --dynamic`) works through it (TQE rerun).

The runner re-reads its file before every run, so every edit here is one atomic rewrite (temp + rename): it sees
the old or the new file, never half of one. Lines are matched by `--match` (a substring of the raw JSON line, e.g.
a run_id, "Exp3", "N10", '"exp":2') or by position. The in-flight run is never touched; a removed line that is
already running simply finishes.

    ls   QUEUE                     counts (done / pending / running) and the first pending lines
    add  QUEUE SRC [--top]         append (or prepend) SRC's lines whose run_id is not in QUEUE yet
    rm   QUEUE --match S           drop the matching lines
    top  QUEUE --match S           move the matching lines to the front (their order kept)
    take QUEUE OUT (--match S | --tail N)   move pending lines out of QUEUE into OUT (appended; for rebalancing
                                   between machines: take -> scp -> add on the other host)
"""

from __future__ import annotations

import json
from pathlib import Path

from ..store.io import atomic_write_bytes
from .queue import run_state


def _lines(path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _write(path, lines: list[str]) -> None:
    atomic_write_bytes(Path(path), ("".join(ln + "\n" for ln in lines)).encode())


def _rid(line: str):
    try:
        return json.loads(line).get("run_id")
    except json.JSONDecodeError:
        return None


def _state(line: str, out_root=None) -> str:
    try:
        s = json.loads(line)
    except json.JSONDecodeError:
        return "bad"
    if not s.get("run_id"):
        return "absent"
    return run_state(s["arm"], s["run_id"], out_root)[0]


def ls(path, out_root=None, show: int = 10) -> dict:
    lines = _lines(path)
    st = [_state(ln, out_root) for ln in lines]
    counts = {k: st.count(k) for k in sorted(set(st))}
    pend = [json.loads(ln).get("label") or _rid(ln) for ln, s in zip(lines, st) if s != "done"][:show]
    return {"queue": str(path), "lines": len(lines), "states": counts,
            "pending": sum(s != "done" for s in st), "next": pend}


def add(path, src, top: bool = False) -> int:
    cur = _lines(path)
    have = {_rid(ln) for ln in cur}
    new = [ln for ln in _lines(src) if _rid(ln) not in have]
    _write(path, (new + cur) if top else (cur + new))
    return len(new)


def rm(path, match: str) -> int:
    cur = _lines(path)
    keep = [ln for ln in cur if match not in ln]
    _write(path, keep)
    return len(cur) - len(keep)


def top(path, match: str) -> int:
    cur = _lines(path)
    hit = [ln for ln in cur if match in ln]
    _write(path, hit + [ln for ln in cur if match not in ln])
    return len(hit)


def take(path, out, match: str | None = None, tail: int | None = None, out_root=None) -> int:
    """Move PENDING lines (not done) from `path` to the end of `out`: those matching `match`, or the last `tail`
    pending ones. `out` is written first, so a crash in between duplicates a line (harmless: run_ids dedup) rather
    than losing it."""
    if (match is None) == (tail is None):
        raise ValueError("take needs exactly one of match / tail")
    cur = _lines(path)
    pending = [i for i, ln in enumerate(cur) if _state(ln, out_root) not in ("done", "running")]
    if match is not None:
        pick = {i for i in pending if match in cur[i]}
    else:
        pick = set(pending[-int(tail):]) if int(tail) > 0 else set()
    moved = [cur[i] for i in sorted(pick)]
    have = {_rid(ln) for ln in _lines(out)}
    _write(out, _lines(out) + [ln for ln in moved if _rid(ln) not in have])
    _write(path, [ln for i, ln in enumerate(cur) if i not in pick])
    return len(moved)
