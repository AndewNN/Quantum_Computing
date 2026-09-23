"""The linear ramp of arm A2 (PLAN §1.5, D-10; memory `qaoa-ramp-sign-and-scale`).

Ported from CUDA/PO_new_ApproxRatio.py:855-859 / 900-903 (mode "Ramp" and --LR_init) with gamma in Jh-boosted
units (`VarQITE/varqite_routes.py` RAMP_GAMMA_UNITS = "boosted", route RAMPB; PLAN §1.5):

    gamma_l = dgamma * alpha * l / p                 (l = 1..p; circuit units, the circuit carries un-boosted H)
    beta_l  = RAMP_SIGN * dbeta * (1 - (l - 1)/p),   RAMP_SIGN = -1

so the applied rotation is exp(-i dgamma (l/p) alpha H). On both mixers a positive ramp follows the TOP
eigenvector of +H_M (|+>^n for the X mixer, the uniform sector state for the ring) and anneals to the MAXIMUM of
H; beta is negated so the ramp follows -H_M, whose ground state is the start state, down to the minimum (the S4
anneal check measures both signs).
"""

from __future__ import annotations

import numpy as np

RAMP_SIGN = -1
SCHEDULES = {"primary": (0.2, 3.0), "secondary": (1.5, 3.0)}     # (dbeta, dgamma), D-10


def ramp_params(p: int, dbeta: float, dgamma: float, alpha: float, sign: int = RAMP_SIGN) -> np.ndarray:
    """[gamma_1..gamma_p, beta_1..beta_p] of the linear ramp in circuit units (alpha = the Jh boost)."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    out = np.zeros(2 * p)
    for itt in range(p):
        out[itt] = dgamma * alpha * (itt + 1) / p
        out[p + itt] = (-dbeta if sign < 0 else dbeta) * (1 - itt / p)
    return out


def schedule(tag: str) -> tuple:
    if tag not in SCHEDULES:
        raise KeyError(f"unknown ramp schedule {tag!r} (one of {sorted(SCHEDULES)})")
    return SCHEDULES[tag]
