"""Exhaustive rulers for n <= 20 (PLAN §1.1, §1.7, §5 S1).

For one encoded instance, enumerate all 2^n strings chunk by chunk and return:
  band        F_eps = {x : (P_bb.x - 1)^2 <= eps^2}, evaluated exactly as the completed work did
              (`state_penalty = -all_state_to_return(n, 1, QU_pen)`, `|state_penalty| <= 1 * eps^2`,
              the confined runs' lambda = 1 band); a direct |P_bb.x - 1| <= eps cross-check counts
              disagreements and near-boundary strings.
  f           the un-boosted objective H_obj(x) = -(x^T QU_obj x) over the band (the old
              `-state_eval`), E_min / E_max over the band, X* (band strings at E_min), the top-10
              (ties at the 10th kept, set size recorded), the ladder spacing (median adjacent gap of
              the sorted distinct normalized values over the band), |F_eps|.
Memory: one chunk of `chunk` strings at a time plus the band arrays; at n = 20 the peak stays in
the tens of MB (the full 2^20 penalty vector alone is 8 MB), far below the ~2 GB cap.

Ties: two energies are "equal" when they differ by at most TIE_RTOL * (E_max - E_min). This
only merges rounding-level differences (1e-16 relative); genuine ladder gaps are >= 1e-9 of the
range at every size here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bits import bit_matrix
from .encode import qubo_energies

TIE_RTOL = 1e-12
BOUNDARY_ATOL = 1e-12   # |Delta - eps| below this counts as a near-boundary string
MAX_N = 20
DEFAULT_CHUNK = 1 << 16


@dataclass(frozen=True)
class Band:
    n: int
    eps: float
    idx: np.ndarray           # uint32/int64 classical indices (x_0 = MSB), ascending
    pen: np.ndarray           # (P.x - 1)^2 on the band, the old state_penalty at lambda = 1
    n_direct_disagree: int    # strings where |P.x - 1| <= eps disagrees with the QUBO test
    n_near_boundary: int      # strings with ||P.x - 1| - eps| < BOUNDARY_ATOL

    @property
    def size(self) -> int:
        return int(self.idx.size)


def band(QU_pen: np.ndarray, P_bb: np.ndarray, eps: float, chunk: int = DEFAULT_CHUNK) -> Band:
    n = QU_pen.shape[0]
    if n > MAX_N:
        raise ValueError(f"rulers are exhaustive only up to n = {MAX_N} (got {n})")
    total = 1 << n
    idx_parts, pen_parts = [], []
    n_dis = n_near = 0
    eps_t = 1.0 * eps ** 2
    P_bb = np.asarray(P_bb, dtype=np.float64)
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        pen = -qubo_energies(QU_pen, 1.0, start, stop, chunk=chunk)
        mask = np.abs(pen) <= eps_t
        delta = np.abs(bit_matrix(start, stop, n, dtype=np.float64) @ P_bb - 1.0)
        n_dis += int(np.count_nonzero(mask != (delta <= eps)))
        n_near += int(np.count_nonzero(np.abs(delta - eps) < BOUNDARY_ATOL))
        sel = np.nonzero(mask)[0]
        idx_parts.append(sel.astype(np.int64) + start)
        pen_parts.append(pen[sel])
    idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64)
    return Band(n=n, eps=float(eps), idx=idx, pen=np.concatenate(pen_parts),
                n_direct_disagree=n_dis, n_near_boundary=n_near)


def objective_on(QU_obj: np.ndarray, idx: np.ndarray, chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """H_obj(x) = -(x^T QU_obj x) on the given classical indices (un-boosted, MIN problem).

    Evaluated with the `all_state_to_return` formula on the full enumeration chunk that holds
    each index, so every value is computed exactly as in a full enumeration.
    """
    n = QU_obj.shape[0]
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty(idx.size, dtype=np.float64)
    if idx.size == 0:
        return out
    total = 1 << n
    blocks = idx // chunk
    for b in np.unique(blocks):
        start = int(b) * chunk
        stop = min(total, start + chunk)
        vals = qubo_energies(QU_obj, 0.0, start, stop, chunk=chunk)
        sel = blocks == b
        out[sel] = -vals[idx[sel] - start]
    return out


@dataclass(frozen=True)
class Rulers:
    n: int
    eps: float
    band_idx: np.ndarray
    band_pen: np.ndarray
    f_band: np.ndarray
    E_min: float
    E_max: float
    xstar_idx: np.ndarray
    top10_idx: np.ndarray      # sorted by (f, index)
    top10_size: int
    gap_band: float            # median adjacent spacing of distinct normalized values (NaN if < 2)
    n_distinct: int
    F_size: int
    f_all_min: float           # unconstrained extremes over all 2^n strings (for reference)
    f_all_max: float
    n_direct_disagree: int
    n_near_boundary: int


def ladder_gap(f: np.ndarray, E_min: float, E_max: float) -> tuple[float, int]:
    """Median adjacent spacing of the sorted distinct normalized values (E_max - f)/(E_max - E_min)."""
    rng = E_max - E_min
    if f.size < 2 or not rng > 0:
        return float("nan"), int(min(f.size, 1))
    v = np.unique((f - E_min) / rng)          # sorted, exact duplicates merged
    if v.size > 1 and np.any(np.diff(v) <= TIE_RTOL):
        keep = [v[0]]                          # greedy merge of rounding-level neighbours (rare)
        for x in v[1:]:
            if x - keep[-1] > TIE_RTOL:
                keep.append(x)
        v = np.array(keep)
    if v.size < 2:
        return float("nan"), int(v.size)
    return float(np.median(np.diff(v))), int(v.size)


def rulers_from_band(b: Band, QU_obj: np.ndarray, chunk: int = DEFAULT_CHUNK) -> Rulers:
    n = b.n
    total = 1 << n
    # full enumeration of f for the unconstrained extremes and the band values
    f_band = np.empty(b.size, dtype=np.float64)
    f_all_min, f_all_max = np.inf, -np.inf
    blocks = b.idx // chunk
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        vals = -qubo_energies(QU_obj, 0.0, start, stop, chunk=chunk)
        f_all_min = min(f_all_min, float(vals.min()))
        f_all_max = max(f_all_max, float(vals.max()))
        sel = blocks == (start // chunk)
        f_band[sel] = vals[b.idx[sel] - start]
    if b.size == 0:
        nan = float("nan")
        empty = np.zeros(0, np.int64)
        return Rulers(n, b.eps, b.idx, b.pen, f_band, nan, nan, empty, empty, 0, nan, 0, 0,
                      f_all_min, f_all_max, b.n_direct_disagree, b.n_near_boundary)
    E_min, E_max = float(f_band.min()), float(f_band.max())
    tol = TIE_RTOL * (E_max - E_min)
    xstar = b.idx[f_band <= E_min + tol]
    order = np.lexsort((b.idx, f_band))
    if b.size <= 10:
        top = order
    else:
        f10 = f_band[order[9]]
        top = order[f_band[order] <= f10 + tol]
    gap, n_distinct = ladder_gap(f_band, E_min, E_max)
    return Rulers(n=n, eps=b.eps, band_idx=b.idx, band_pen=b.pen, f_band=f_band,
                  E_min=E_min, E_max=E_max, xstar_idx=np.sort(xstar), top10_idx=b.idx[top],
                  top10_size=int(top.size), gap_band=gap, n_distinct=n_distinct, F_size=b.size,
                  f_all_min=f_all_min, f_all_max=f_all_max,
                  n_direct_disagree=b.n_direct_disagree, n_near_boundary=b.n_near_boundary)
