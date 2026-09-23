"""Ids (PLAN §1.1, §3.2).

draw_id = N{N:02d}e{e:03d};  inst_id = {draw_id}q{q}  (e.g. N05e003q1.5); a changed instance gets
a new id with suffix `v2` (never an overwrite).
run_id  = first 16 hex digits of sha256(canonical JSON of the RunConfig), where the RunConfig
always carries `harness_version`. The git SHA is recorded in run.json but never hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from .._version import HARNESS_VERSION
from ..instances.draws import draw_id, inst_id, q_tag  # noqa: F401  (re-exported)

_INST_RE = re.compile(r"^N(?P<N>\d{2})e(?P<e>\d{3})q(?P<q>\d+\.\d+)(?P<v>v\d+)?$")


def parse_inst_id(iid: str) -> dict:
    m = _INST_RE.match(iid)
    if not m:
        raise ValueError(f"not an inst_id: {iid!r}")
    return {"N": int(m["N"]), "e": int(m["e"]), "q": float(m["q"]), "version": m["v"]}


def _canon(v):
    """JSON-safe canonical scalar/list/dict (numpy scalars -> python, tuples -> lists)."""
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):  # numpy scalar / 0-d array
        v = v.item()
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite float in a RunConfig")
        return v
    raise TypeError(f"unsupported RunConfig value {v!r} ({type(v).__name__})")


def canonical_json(config: dict) -> str:
    return json.dumps(_canon(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def with_version(config: dict) -> dict:
    """Return a copy carrying `harness_version` (the current one unless already set)."""
    out = dict(config)
    out.setdefault("harness_version", HARNESS_VERSION)
    return out


def run_id(config: dict) -> str:
    if "harness_version" not in config:
        raise KeyError("a RunConfig must carry harness_version (use with_version)")
    return hashlib.sha256(canonical_json(config).encode("ascii")).hexdigest()[:16]
