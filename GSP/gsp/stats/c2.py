"""Rule C2, expressibility on the reachable subspace (PLAN §1.7, proposal §6.5; Sim et al. 2019 with the Haar
reference of dimension d = the reachable dimension):

  KL(P_emp || P_Haar^(d)),   P_Haar^(d)(F) = (d - 1)(1 - F)^(d - 2)   on F in [0, 1],

from N_PAIRS = 5000 fidelities F = |<psi(theta)|psi(theta')>|^2 of independent Eq. 4.11 parameter pairs, a fixed
histogram of 75 equal bins on [0, 1], and 1000 bootstrap resamples of the fidelity pairs (95 % percentile
interval). The Haar bin masses are the exact integrals (1 - a)^(d-1) - (1 - b)^(d-1) over each bin [a, b],
evaluated in log space (no underflow at d ~ 600); bins with no empirical mass contribute 0.
Completion (Rule C2): >= 4 effort levels for both connectivity endpoints of A1 (ring, complete) and, where it
applies, along A4, every point with its d and its interval (`completion`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PAIRS = 5000
N_BINS = 75
N_BOOT = 1000
BOOT_SEED = 0
MIN_LEVELS = 4


def haar_bin_log_probs(d: int, bins: int = N_BINS) -> np.ndarray:
    """log of the Haar^(d) mass of each of `bins` equal bins on [0, 1]."""
    if d < 2:
        raise ValueError("d >= 2")
    e = np.linspace(0.0, 1.0, bins + 1)
    a, b = e[:-1], e[1:]
    la = (d - 1) * np.log1p(-a)                          # log (1 - a)^(d-1)
    with np.errstate(divide="ignore"):
        lb = (d - 1) * np.log1p(-b)                      # -inf at b = 1
    return la + np.log1p(-np.exp(lb - la))


def histogram(fids, bins: int = N_BINS) -> np.ndarray:
    f = np.clip(np.asarray(fids, dtype=np.float64), 0.0, 1.0)
    c, _ = np.histogram(f, bins=bins, range=(0.0, 1.0))
    return c / max(1, c.sum())


def kl_to_haar(fids, d: int, bins: int = N_BINS) -> float:
    p = histogram(fids, bins)
    lq = haar_bin_log_probs(d, bins)
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - lq[m])))


def kl_bootstrap(fids, d: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED, bins: int = N_BINS) -> dict:
    fids = np.asarray(fids, dtype=np.float64)
    rng = np.random.default_rng(seed)
    lq = haar_bin_log_probs(d, bins)
    ks = np.empty(n_boot)
    for b in range(n_boot):
        p = histogram(fids[rng.integers(0, fids.size, fids.size)], bins)
        m = p > 0
        ks[b] = np.sum(p[m] * (np.log(p[m]) - lq[m]))
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return {"kl": kl_to_haar(fids, d, bins), "lo": float(lo), "hi": float(hi), "d": int(d), "n_pairs": int(fids.size),
            "bins": bins, "n_boot": n_boot, "seed": seed}


def pair_fidelities(states_a, states_b) -> np.ndarray:
    """|<a_i|b_i>|^2 for two stacks of normalized states (rows)."""
    a = np.asarray(states_a)
    b = np.asarray(states_b)
    return np.abs(np.einsum("ij,ij->i", a.conj(), b)) ** 2


def sample_fidelities(state_fn, param_fn, n_pairs: int = N_PAIRS) -> np.ndarray:
    """F for n_pairs independent parameter pairs: state_fn(theta) -> statevector, param_fn() -> theta (the caller
    owns the RNG, seeded from the seed table)."""
    out = np.empty(n_pairs)
    for i in range(n_pairs):
        a, b = state_fn(param_fn()), state_fn(param_fn())
        out[i] = abs(np.vdot(a, b)) ** 2
    return out


def haar_states(d: int, m: int, rng) -> np.ndarray:
    """m Haar-random states in C^d (reference and tests)."""
    z = rng.normal(size=(m, d)) + 1j * rng.normal(size=(m, d))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def completion(points: pd.DataFrame) -> dict:
    """Rule C2's completion check. points: rows (arm, connectivity, level, kl, lo, hi, d). Complete when A1 has
    >= 4 levels at both ring and complete and every point carries d and an interval; A4 counted if present."""
    def levels(arm, conn=None):
        s = points[points["arm"] == arm]
        if conn is not None:
            s = s[s["connectivity"] == conn]
        s = s.dropna(subset=["kl", "lo", "hi", "d"])
        return int(s["level"].nunique())
    ring, comp = levels("A1", "ring"), levels("A1", "complete")
    a4 = levels("A4") if (points["arm"] == "A4").any() else None
    ok = ring >= MIN_LEVELS and comp >= MIN_LEVELS and (a4 is None or a4 >= MIN_LEVELS)
    return {"A1_ring_levels": ring, "A1_complete_levels": comp, "A4_levels": a4, "complete": bool(ok)}
