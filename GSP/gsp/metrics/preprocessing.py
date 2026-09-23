"""The classical preprocessing cost of a confined run, joined from its sector file (PLAN §1.7, WP5).

The sector a run used is (sector scope, rule, K): per draw for the violation rule (the three q share one GA run,
`pre_scope = "draw"`), per instance for the objective rule (`pre_scope = "instance"`). Joined columns: the GA
wall-clock / CPU time and its objective and violation evaluation counts (Table 4.1 GA, S2), the brute-force
reference timing, and the selection loss (GA list identical to the BF list? strings missing). A run on the BF list
(sector_source = "bf") carries the same columns plus `pre_list = "bf"`: the reader charges the BF cost then.
Penalty arms and ad hoc instances without a sector file get no columns.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

_FIELDS = ("ga_wall_s", "ga_cpu_s", "ga_n_evals", "ga_n_violation_evals", "ga_n_objective_evals", "ga_n_unique",
           "bf_wall_s", "bf_n_evals", "identical", "n_missing", "seed")


@lru_cache(maxsize=4096)
def _sector_scalars(scope: str, rule: str, K: int, root) -> dict | None:
    from ..store.io import load_npz
    from ..store.paths import sector_path
    p = sector_path(scope, rule, K, root)
    if not p.exists():
        return None
    z = load_npz(p)
    out = {}
    for k in _FIELDS:
        if k in z:
            v = np.asarray(z[k]).item()
            out[k] = bool(v) if k == "identical" else v
    out["file"] = p.name
    return out


def preprocessing(rec: dict, root=None) -> dict:
    if rec.get("encoding") != "confined" or rec.get("rule") in (None, "") or rec.get("K") in (None, ""):
        return {}
    from ..sectors.select import sector_scope
    rule, K = rec["rule"], int(rec["K"])
    s = _sector_scalars(sector_scope(rec["inst_id"], rule), rule, K, None if root is None else str(root))
    if s is None:
        return {"pre_available": False}
    out = {f"pre_{k}": v for k, v in s.items()}
    out.update({"pre_available": True, "pre_scope": "draw" if rule == "violation" else "instance",
                "pre_list": rec.get("sector_source", "ga")})
    return out
