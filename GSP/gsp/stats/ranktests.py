"""The exact Wilcoxon signed-rank test and Holm's step-down correction (Rule D1, PLAN §1.7).

`signed_rank_exact(d)`: the exact two-sided test on paired differences d.
  * Zeros: |d| <= zero_tol are dropped (Wilcoxon's original treatment, scipy's zero_method="wilcox").
    zero_tol = 1e-12 merges rounding-level differences (two arms returning the same state up to float noise).
  * Ties: midranks of |d| (values equal to within tie_tol = 1e-12 share a rank), and the null distribution is the
    EXACT conditional distribution of T+ = sum of the ranks of the positive differences over the 2^m equally likely
    sign patterns of those ranks. Doubled midranks are integers, so the distribution is a subset-sum count
    (dynamic programming, exact integers), valid with or without ties.
  * p = min(1, 2 min(P(T+ <= t), P(T+ >= t))). Without ties or zeros this is scipy's `method="exact"` p-value
    (tested to 1e-12).
`holm(p, m)`: Holm-adjusted p-values over a family of fixed size m (default len(p)); missing entries (NaN) count
as members that are never rejected (p = 1), so a pair without data does not shrink the family of Rule D1's four
primary comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

ZERO_TOL = 1e-12
TIE_TOL = 1e-12


@dataclass(frozen=True)
class SignedRankResult:
    n: int              # differences given (finite)
    m: int              # non-zero differences used
    t_plus: float       # sum of the (mid)ranks of the positive differences
    t_minus: float
    p: float            # exact two-sided p
    ties: bool


def _midranks(a: np.ndarray, tie_tol: float) -> np.ndarray:
    """Midranks of a (ascending), treating values within tie_tol of the previous distinct value as equal."""
    order = np.argsort(a, kind="stable")
    s = a[order]
    groups = np.zeros(s.size, dtype=np.int64)
    for i in range(1, s.size):
        groups[i] = groups[i - 1] + (1 if s[i] - s[i - 1] > tie_tol else 0)
    r = rankdata(groups, method="average")
    out = np.empty_like(r)
    out[order] = r
    return out


def _null_counts(r2: np.ndarray) -> np.ndarray:
    """counts[s] = number of subsets of the doubled ranks r2 with sum s (python ints: exact)."""
    total = int(r2.sum())
    counts = [0] * (total + 1)
    counts[0] = 1
    acc = 0
    for x in (int(v) for v in r2):
        for s in range(acc, -1, -1):
            if counts[s]:
                counts[s + x] += counts[s]
        acc += x
    return np.array(counts, dtype=object)


def signed_rank_exact(d, zero_tol: float = ZERO_TOL, tie_tol: float = TIE_TOL) -> SignedRankResult:
    d = np.asarray(d, dtype=np.float64).ravel()
    d = d[np.isfinite(d)]
    n = int(d.size)
    nz = d[np.abs(d) > zero_tol]
    m = int(nz.size)
    if m == 0:
        return SignedRankResult(n=n, m=0, t_plus=0.0, t_minus=0.0, p=1.0, ties=False)
    r = _midranks(np.abs(nz), tie_tol)
    r2 = np.rint(2 * r).astype(np.int64)                  # doubled midranks are integers
    ties = bool(np.unique(r2).size < m)
    t2 = int(r2[nz > 0].sum())
    counts = _null_counts(r2)
    total = sum(counts)                                    # = 2^m
    lower = sum(counts[: t2 + 1])
    upper = sum(counts[t2:])
    p = min(1.0, 2.0 * min(lower, upper) / total) if total else 1.0
    tp = t2 / 2.0
    return SignedRankResult(n=n, m=m, t_plus=tp, t_minus=float(r.sum()) - tp, p=float(p), ties=ties)


def holm(p, m: int | None = None) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN in -> NaN out, counted in the family as never rejected."""
    p = np.asarray(p, dtype=np.float64)
    m = int(p.size if m is None else m)
    if m < p.size:
        raise ValueError("family size m smaller than the number of p-values")
    out = np.full(p.shape, np.nan)
    have = np.nonzero(np.isfinite(p))[0]
    order = have[np.argsort(p[have], kind="stable")]
    running = 0.0
    for j, i in enumerate(order):
        adj = min(1.0, (m - j) * p[i])
        running = max(running, adj)
        out[i] = running
    return out
