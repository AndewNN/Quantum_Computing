"""Solution-quality metrics of PLAN §1.7 on arrays (the per-state core is `state.MetricContext`).

Conventions (`example_metrics.py`, proposal §6.3.2):
  quality(x) = (E_max - f(x)) / (E_max - E_min) on the band F_eps, with E_min / E_max over the FULL band, so
               1 = the best band string and 0 = the worst; strings outside the band are infeasible.
  AR_F       = sum_{x in F_eps} p(x) quality(x)                                  (`MetricContext.evaluate`)
  AR_best_S  = E[ max quality over the feasible shots among S | at least one feasible shot ]   (Eq. plan-arbest)
  P(optimum seen in S shots) = 1 - (1 - p_opt)^S.

`ar_best_exact` computes AR_best_S exactly from the probabilities: sort the band strings by quality, best
first, with cumulative mass C_j; then P(max >= v_j) = 1 - (1 - C_j)^S and
  E[max ; any feasible] = sum_j v_j [P(max >= v_j) - P(max >= v_{j-1})],
divided by P(any feasible) = 1 - (1 - p_feas)^S. Equal qualities need no grouping (their terms add up).
`ar_best_from_shots` is the Monte-Carlo estimator of `example_metrics.py` (the mean over shot rows that hold a
feasible string of the best feasible quality in that row); `best_of_shots` is one realization (one row), which
is what the stored `cudaq.sample(1000)` check gives.
"""

from __future__ import annotations

import numpy as np

AR_BEST_SHOTS = 1000          # S of PLAN §1.7 (deck: S = 1000 for the classical comparison)


def normalized_quality(f, E_min: float, E_max: float) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    rng = float(E_max) - float(E_min)
    return (float(E_max) - f) / rng if rng > 0 else np.ones_like(f)


def _pow_complement(c, S: int) -> np.ndarray:
    """(1 - c)^S, stable for c near 0 and exact 0 at c = 1."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return np.exp(S * np.log1p(-c))


def p_any_feasible(p_feas: float, S: int) -> float:
    return float(1.0 - _pow_complement(p_feas, S))


def ar_best_exact(prob_feasible, quality_feasible, S: int = AR_BEST_SHOTS) -> float:
    """AR_best_S from the probabilities of the feasible strings and their qualities (any order)."""
    p = np.asarray(prob_feasible, dtype=np.float64)
    v = np.asarray(quality_feasible, dtype=np.float64)
    if p.shape != v.shape:
        raise ValueError("prob and quality must have the same shape")
    if p.size == 0:
        return float("nan")
    order = np.argsort(-v, kind="stable")
    p, v = np.clip(p[order], 0.0, None), v[order]
    C = np.cumsum(p)
    G = 1.0 - _pow_complement(C, S)                  # P(max >= v_j)
    dG = np.diff(np.r_[0.0, G])
    num = float(np.dot(v, dG))
    den = float(G[-1])
    return num / den if den > 0 else float("nan")


def best_quality_distribution(prob_feasible, quality_feasible, S: int = AR_BEST_SHOTS):
    """The distribution of the best feasible quality among S shots, conditioned on >= 1 feasible shot:
    (distinct values descending, probabilities). Used for the tail check of a sampled value."""
    p = np.asarray(prob_feasible, dtype=np.float64)
    v = np.asarray(quality_feasible, dtype=np.float64)
    vals, inv = np.unique(-v, return_inverse=True)            # ascending in -v = descending in v
    mass = np.bincount(inv, weights=np.clip(p, 0.0, None), minlength=vals.size)
    G = 1.0 - _pow_complement(np.cumsum(mass), S)
    den = G[-1]
    pm = np.diff(np.r_[0.0, G]) / den if den > 0 else np.full(vals.size, np.nan)
    return -vals, pm


def sample_tail_probability(observed: float, prob_feasible, quality_feasible, S: int = AR_BEST_SHOTS,
                            tol: float = 1e-12) -> float:
    """min(P(M >= obs), P(M <= obs)) for the best feasible quality M among S shots (exact distribution).
    A small value flags a sample that is inconsistent with the stored state."""
    if not np.isfinite(observed):
        return float("nan")
    vals, pm = best_quality_distribution(prob_feasible, quality_feasible, S)
    ge = float(pm[vals >= observed - tol].sum())
    le = float(pm[vals <= observed + tol].sum())
    return min(ge, le)


def ar_best_from_shots(shots, quality, feasible) -> float:
    """The Monte-Carlo AR_best_S of `example_metrics.py`: `shots` is (rows, S) of string indices; the mean over
    rows with a feasible shot of the best feasible quality in the row."""
    shots = np.asarray(shots)
    quality = np.asarray(quality, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    best = [quality[row[feasible[row]]].max() for row in shots if feasible[row].any()]
    return float(np.mean(best)) if best else float("nan")


def band_positions(band_idx, idx) -> tuple[np.ndarray, np.ndarray]:
    """(in_band mask, position in band_idx) of classical indices idx; band_idx must be ascending."""
    band_idx = np.asarray(band_idx, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    pos = np.searchsorted(band_idx, idx)
    ok = pos < band_idx.size
    ok[ok] = band_idx[pos[ok]] == idx[ok]
    return ok, np.where(ok, pos, -1)


def best_of_shots(observed_idx, band_idx, band_quality) -> float:
    """Best feasible quality among the observed strings of one set of shots; NaN if none is feasible."""
    ok, pos = band_positions(band_idx, observed_idx)
    if not ok.any():
        return float("nan")
    return float(np.asarray(band_quality, dtype=np.float64)[pos[ok]].max())


def p_seen(p: float, S: int = AR_BEST_SHOTS) -> float:
    """P(at least one of S shots lands in a set of mass p) = 1 - (1 - p)^S."""
    return float(1.0 - _pow_complement(p, S))
