"""Rule D2 (PLAN §1.7, proposal §6.5): does log d_eff predict a progress diagnostic better than n?

Data for one diagnostic: rows (cell, n, K, d_eff, value), one row per observation (e.g. per instance); a
configuration is one confined (n, K) cell.
  * Models: log value ~ a + b n (the n model) and log value ~ a + b log d_eff (the d_eff model), ordinary least
    squares on every observation. Non-positive values cannot enter a log and are dropped (counted).
  * Preconditions (checked before any fit): at least 3 distinct K at each of at least 3 distinct n, and the
    Spearman |rho(n, log d_eff)| over the fitted cells <= 0.8. Otherwise the verdict is "not separable".
  * Leave-one-configuration-out folds: each cell is predicted by both models fitted on the other cells; RMSE over
    every held-out observation; ratio = RMSE_deff / RMSE_n.
  * 2000 bootstrap resamples over folds (cells drawn with replacement; the held-out squared errors of the drawn
    folds are pooled) give the 95 % percentile interval of the ratio. The d_eff model is preferred iff
    ratio <= 0.8 and the interval's upper end < 1.
  * Exponent: the slope b of the d_eff model on all cells; its 95 % interval and standard error come from the same
    fold resamples (cells with replacement, refit; resamples that leave fewer than 2 distinct d_eff are skipped).
Verdict over the two voting diagnostics (V on A1, energy drop per step on A4):
  supported            d_eff preferred for both and both exponent intervals contain the inverse-variance-weighted
                       mean of the two exponents (weights 1 / SE^2);
  partially supported  preferred for both, homogeneity fails;
  refuted              preferred for at most one;
  not separable        a voting diagnostic fails its preconditions (reported per diagnostic as well).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RATIO_MAX = 0.8
RHO_MAX = 0.8
MIN_K_PER_N = 3
MIN_N = 3
N_BOOT = 2000
BOOT_SEED = 0
VERDICTS = ("supported", "partially supported", "refuted", "not separable")


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def preconditions(df: pd.DataFrame) -> dict:
    cells = df.drop_duplicates("cell")
    k_per_n = cells.groupby("n")["K"].nunique()
    n_ok = int((k_per_n >= MIN_K_PER_N).sum())
    if cells["n"].nunique() > 1 and cells["d_eff"].nunique() > 1:
        rho = float(spearmanr(cells["n"], np.log(cells["d_eff"].astype(float))).statistic)
    else:
        rho = float("nan")
    ok_design = n_ok >= MIN_N
    ok_rho = bool(np.isfinite(rho) and abs(rho) <= RHO_MAX)
    return {"n_with_3K": n_ok, "spearman_rho": rho, "design_ok": ok_design, "rho_ok": ok_rho,
            "separable": bool(ok_design and ok_rho), "n_cells": int(len(cells))}


@dataclass
class D2Result:
    name: str
    separable: bool
    precond: dict
    rmse_n: float = float("nan")
    rmse_d: float = float("nan")
    ratio: float = float("nan")
    ratio_ci: tuple = (float("nan"), float("nan"))
    preferred: bool = False
    exponent: float = float("nan")
    exponent_ci: tuple = (float("nan"), float("nan"))
    exponent_se: float = float("nan")
    exponent_n: float = float("nan")
    n_obs: int = 0
    n_dropped: int = 0
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        d = {"diagnostic": self.name, "separable": self.separable, "rmse_n": self.rmse_n, "rmse_deff": self.rmse_d,
             "ratio": self.ratio, "ratio_lo": self.ratio_ci[0], "ratio_hi": self.ratio_ci[1],
             "preferred": self.preferred, "exponent": self.exponent, "exponent_lo": self.exponent_ci[0],
             "exponent_hi": self.exponent_ci[1], "exponent_se": self.exponent_se, "slope_n": self.exponent_n,
             "n_obs": self.n_obs, "n_dropped": self.n_dropped}
        d.update({f"pre_{k}": v for k, v in self.precond.items()})
        return d


def d2_diagnostic(df: pd.DataFrame, name: str = "diagnostic", n_boot: int = N_BOOT,
                  seed: int = BOOT_SEED) -> D2Result:
    """df columns: cell, n, K, d_eff, value."""
    df = df.copy()
    pos = df["value"] > 0
    n_drop = int((~pos).sum())
    df = df[pos & np.isfinite(df["value"])]
    pre = preconditions(df)
    res = D2Result(name=name, separable=pre["separable"], precond=pre, n_obs=int(len(df)), n_dropped=n_drop)
    if not pre["separable"]:
        return res
    y = np.log(df["value"].to_numpy(dtype=np.float64))
    xn = df["n"].to_numpy(dtype=np.float64)
    xd = np.log(df["d_eff"].to_numpy(dtype=np.float64))
    cell = df["cell"].to_numpy()
    cells = np.array(sorted(set(cell), key=str))
    sse_n = np.zeros(cells.size)
    sse_d = np.zeros(cells.size)
    cnt = np.zeros(cells.size)
    idx_of = {c: np.nonzero(cell == c)[0] for c in cells}
    for j, c in enumerate(cells):
        te = idx_of[c]
        tr = np.setdiff1d(np.arange(y.size), te)
        an, bn = _ols(xn[tr], y[tr])
        ad, bd = _ols(xd[tr], y[tr])
        sse_n[j] = float(((y[te] - (an + bn * xn[te])) ** 2).sum())
        sse_d[j] = float(((y[te] - (ad + bd * xd[te])) ** 2).sum())
        cnt[j] = te.size
    rmse_n = float(np.sqrt(sse_n.sum() / cnt.sum()))
    rmse_d = float(np.sqrt(sse_d.sum() / cnt.sum()))
    ratio = rmse_d / rmse_n if rmse_n > 0 else float("inf")
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_boot)
    slopes = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, cells.size, cells.size)
        sn, sd_, cc = sse_n[pick].sum(), sse_d[pick].sum(), cnt[pick].sum()
        ratios[b] = np.sqrt(sd_ / cc) / np.sqrt(sn / cc) if sn > 0 else np.inf
        rows = np.concatenate([idx_of[cells[j]] for j in pick])
        if np.unique(xd[rows]).size >= 2:
            slopes[b] = _ols(xd[rows], y[rows])[1]
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    _, b_d = _ols(xd, y)
    _, b_n = _ols(xn, y)
    s = slopes[np.isfinite(slopes)]
    e_lo, e_hi = np.percentile(s, [2.5, 97.5]) if s.size else (np.nan, np.nan)
    res.rmse_n, res.rmse_d, res.ratio, res.ratio_ci = rmse_n, rmse_d, float(ratio), (float(lo), float(hi))
    res.preferred = bool(ratio <= RATIO_MAX and hi < 1.0)
    res.exponent, res.exponent_ci = float(b_d), (float(e_lo), float(e_hi))
    res.exponent_se = float(s.std(ddof=1)) if s.size > 1 else float("nan")
    res.exponent_n = float(b_n)
    res.extra = {"n_boot_slopes": int(s.size), "seed": seed, "n_boot": n_boot}
    return res


def ivw_mean(exponents, ses) -> float:
    e = np.asarray(exponents, dtype=np.float64)
    w = 1.0 / np.asarray(ses, dtype=np.float64) ** 2
    return float((w * e).sum() / w.sum())


def verdict(results: list[D2Result]) -> dict:
    """The RQ2 verdict over the voting diagnostics (module doc)."""
    if any(not r.separable for r in results):
        return {"verdict": "not separable", "ivw_mean": float("nan"), "homogeneous": None,
                "n_preferred": int(sum(r.preferred for r in results if r.separable))}
    n_pref = int(sum(r.preferred for r in results))
    m = ivw_mean([r.exponent for r in results], [r.exponent_se for r in results])
    homo = bool(all(r.exponent_ci[0] <= m <= r.exponent_ci[1] for r in results))
    if n_pref < len(results):
        v = "refuted"
    else:
        v = "supported" if homo else "partially supported"
    return {"verdict": v, "ivw_mean": m, "homogeneous": homo, "n_preferred": n_pref}


def run_d2(diagnostics: dict, voting=("V_A1", "drop_A4"), n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """diagnostics: {name: df}. Returns {"table": per-diagnostic rows, "verdict": dict over the voting ones}."""
    res = {k: d2_diagnostic(v, k, n_boot=n_boot, seed=seed) for k, v in diagnostics.items()}
    vote = [res[k] for k in voting if k in res]
    out = verdict(vote) if len(vote) == len(voting) else {"verdict": "incomplete", "missing":
                                                          [k for k in voting if k not in res]}
    table = pd.DataFrame([r.row() | {"voting": r.name in voting} for r in res.values()])
    return {"table": table, "verdict": out, "results": res}
