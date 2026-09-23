"""Instances built in memory, never frozen (S4): the completed work's draws that the §1.1 acceptance rejects
but a reproduction check needs (PLAN §5 S4: N = 5, e = 2, |F_eps| = 4 < 12). Same recipe as the freeze
(`make_draw`, `encode`, `band`, `rulers_from_band`), so a frozen id rebuilt here equals its file.

`load_any(inst_id)` returns (Instance, Rulers, adhoc) for a frozen id or, if the id is not frozen, the ad hoc
build (adhoc = True; run configs carry `inst_adhoc = True`).
"""

from __future__ import annotations

from functools import lru_cache

from ..store.paths import inst_path
from .data import load_market
from .draws import EPS, inst_id as make_inst_id, make_draw
from .encode import encode
from .freeze import instance_arrays
from .instance import instance_from_arrays, load_instance, load_rulers
from .rulers import band, rulers_from_band


@lru_cache(maxsize=None)
def _market():
    return load_market()


def adhoc_instance(N: int, e: int, q: float):
    market = _market()
    d = make_draw(market, N, e)
    enc = encode(d.B, d.P, d.ret, d.cov, q)
    iid = make_inst_id(N, e, q)
    inst = instance_from_arrays(instance_arrays(d, enc, iid, market.sources))
    rul = rulers_from_band(band(enc.QU_pen, enc.P_bb, EPS), enc.QU_obj)
    return inst, rul


def load_any(inst_id: str, root=None):
    if inst_path(inst_id, root).exists():
        return load_instance(inst_id, root), load_rulers(inst_id, root), False
    from ..store.ids import parse_inst_id
    p = parse_inst_id(inst_id)
    inst, rul = adhoc_instance(p["N"], p["e"], p["q"])
    return inst, rul, True
