"""Synthetic data with a known answer for Rules D1, D2, C1 and C2 (tests/test_stats.py and scripts/s5_checks.py).

D1  `d1_curves(shift, ...)`: per-draw AR_F trajectories of the D1 arms on one N with 30 draws x 3 q, R restarts
    for the trained arms, a saturating curve per configuration and a planted per-draw difference `shift[arm]`
    (a constant, or a callable rng -> per-draw array) between each arm and A1's value.
D2  `d2_data(kind, exponent)`: log value = c + exponent * log d_eff (kind "deff"), = c + exponent * n ("n"), on the
    confined grid n in {8, 10, 12, 14} x K in {6, 8, 12}; kind "collinear" ties K to n.
C1  `leakage_series(kind)`: the max over 10 random-walk rounding accumulations (sqrt(D) growth) or a systematic
    rate per gate (linear growth), both with the eps_num-style max over circuits.
C2  Haar states come from `c2.haar_states`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .d1 import R_EXPECTED

Q = (1.0, 1.5, 3.0)


def _curve(final: float, init: float, T: int, tau: float) -> np.ndarray:
    t = np.arange(T + 1, dtype=np.float64)
    return final - (final - init) * np.exp(-t / tau)


def d1_curves(shift: dict, N: int = 5, n_draws: int = 30, seed: int = 0, configs: dict | None = None,
              base: float = 0.5, draw_sd: float = 0.1, restart_sd: float = 0.002, q_sd: float = 0.002,
              T: int = 60) -> pd.DataFrame:
    """shift: {arm: constant or callable(rng) -> (n_draws,) array}; A1 is the reference (shift 0). configs:
    {arm: [(label, cost_per_unit, ceiling_offset, tau)]}: a configuration's final value = the arm's per-draw value
    + ceiling_offset; its trajectory saturates with time constant tau (units) from 0.1; g2q = cost_per_unit x t.
    A2c / A4-like single-row arms can be given T = 0 via tau = 0 (one row at cost_per_unit)."""
    rng = np.random.default_rng(seed)
    configs = configs or {}
    draw_base = base + draw_sd * rng.standard_normal(n_draws)
    rows = []
    arms = ["A1"] + [a for a in shift if a != "A1"]
    for arm in arms:
        s = shift.get(arm, 0.0)
        per_draw = draw_base + (s(rng) if callable(s) else float(s))
        cfgs = configs.get(arm, [("L5", 100, 0.0, 10.0)])
        R = R_EXPECTED.get(arm, 1)
        for (lab, cost, off, tau) in cfgs:
            for i in range(n_draws):
                for q in Q:
                    qv = per_draw[i] + off + q_sd * rng.standard_normal()
                    for r in range(R):
                        fin = float(np.clip(qv + restart_sd * rng.standard_normal(), 0.0, 1.0))
                        if tau <= 0:
                            g2q = np.array([int(cost)], dtype=np.int64)
                            ar = np.array([fin])
                        else:
                            ar = _curve(fin, 0.1, T, tau)
                            g2q = (np.arange(T + 1) * int(cost)).astype(np.int64)
                        rows.append({"arm": arm, "N": N, "draw_id": f"N{N:02d}e{i:03d}",
                                     "inst_id": f"N{N:02d}e{i:03d}q{q}", "q": q, "cfg": f"{arm}|{lab}",
                                     "cfg_label": lab, "restart": r, "g2q": g2q, "ar": ar})
    return pd.DataFrame(rows)


def d2_data(kind: str = "deff", exponent: float = -1.0, noise: float = 0.05, per_cell: int = 10, seed: int = 0,
            n_values=(8, 10, 12, 14), K_values=(6, 8, 12)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    if kind == "collinear":
        grid = [(n, K) for j, n in enumerate(n_values[:3]) for K in (4 * 2 ** j, 5 * 2 ** j, 6 * 2 ** j)]
    else:
        grid = [(n, K) for n in n_values for K in K_values]
    for n, K in grid:
        d = K * K - 1
        for _ in range(per_cell):
            if kind in ("deff", "collinear"):
                lv = 1.0 + exponent * np.log(d)
            elif kind == "n":
                lv = 1.0 + exponent * n
            else:
                raise ValueError(kind)
            rows.append({"cell": f"n{n}K{K}", "n": n, "K": K, "d_eff": d,
                         "value": float(np.exp(lv + noise * rng.standard_normal()))})
    return pd.DataFrame(rows)


def leakage_series(kind: str, D=None, n_circuits: int = 10, seed: int = 0, unit: float = 1.1e-16,
                   rate: float = 3e-15) -> tuple[np.ndarray, np.ndarray]:
    """(D, eps(D)): kind "sqrt" = max over circuits of |sum of D rounding errors| (random walk, ~ sqrt(D));
    kind "linear" = a systematic leak rate per gate plus the same rounding."""
    rng = np.random.default_rng(seed)
    D = np.unique(np.round(np.logspace(0, 3, 25)).astype(int)) if D is None else np.asarray(D, dtype=int)
    Dmax = int(D.max())
    walks = np.cumsum(unit * rng.standard_normal((n_circuits, Dmax)), axis=1)
    noise = np.abs(walks[:, D - 1]).max(axis=0)
    if kind == "sqrt":
        return D, noise
    if kind == "linear":
        return D, rate * D + noise
    raise ValueError(kind)
