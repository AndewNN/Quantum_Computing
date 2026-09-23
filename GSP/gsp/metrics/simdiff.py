"""Empirical simulation difficulty of a prepared state (PLAN §1.7, proposal §6.3.2): exact MPS by sequential SVD.

For a normalized state psi of n qubits in classical order (x_0 = MSB = site 0):
  F(chi)       the fidelity |<psi_chi|psi>|^2 of the left-to-right sequential-SVD MPS truncated to bond dimension
               chi (singular values absorbed to the right, psi_chi normalized), for chi = 1, 2, 4, ... up to
               2^floor(n/2), where F = 1 exactly. With left-isometric tensors psi_chi is the projection of psi on
               the kept left bases, so F(chi) = the norm^2 of the final right tensor = the product over the cuts of
               the kept weight fraction.
  chi_star     the smallest integer chi with F(chi) >= 1 - 1e-3, found by bisection inside a rigorous bracket from
               the exact spectra s_k of the cuts k: F(chi) <= U(chi) = min_k sum_{i<chi} s_{k,i}^2 (Eckart-Young:
               psi_chi has Schmidt rank <= chi at every cut) and F(chi) >= L(chi) = 1 - sum_k sum_{i>=chi} s_{k,i}^2
               (the weight discarded at a cut of the truncated state is at most the exact one). The bisection
               assumes F non-decreasing in chi inside the bracket (the grid is checked; tests compare with a full
               scan). `chi_star_grid` = the smallest power of two; `chi_star_bound` = the smallest chi with
               L(chi) >= 1 - 1e-3 (the cheap, conservative reading of "the weight discarded bounds the loss").
               F(chi) = 1 exactly once chi >= the largest exact rank (no truncation anywhere).
  mem_repr     32 n chi_star^2 bytes (proposal: the complex-double bound), beside mem_dense = 16 * 2^n and the
               actual size of the truncated MPS (`mem_mps_bytes`).
  S_half       the von Neumann entropy (ebits) of the half cut A = {0, ..., ceil(n/2) - 1}, from the exact Schmidt
               spectrum there; `S_over_log2K` = S_half / log2 K for the confined arms (S <= log2 K by construction).
  discarded    the exact spectra also give the textbook bound 1 - F(chi) <= sum over cuts of the discarded weight
               (`F_bound_*`), stored as information.
  wall_chi_star  the wall-clock of one truncated decomposition at chi_star; peak RSS of the process during the
               whole computation (Linux: /proc/self/clear_refs resets the high-water mark, VmHWM is read after;
               `work_bytes` = VmHWM - VmRSS before; the process's own peak before the reset is kept as
               `process_peak_rss_before_bytes`, i.e. the run's peak when called by the post-run step). Elsewhere
               these are None.

Engines: "numpy" (np.linalg.svd) and "quimb" (`MatrixProductState.from_dense(max_bond=chi, cutoff=0)`, fidelity by
contraction to a dense vector). They agree to <= 1e-10 (tests/test_simdiff.py and the S5 cross-test over the stored
runs, reports/metrics.md). DEFAULT_ENGINE is the one the aggregate uses.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np

F_THRESHOLD = 1e-3
ENGINES = ("numpy", "quimb")
DEFAULT_ENGINE = "quimb"


# --- peak RSS (Linux) ---------------------------------------------------------------------------------------
def _status_kb(key: str) -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


@contextmanager
def peak_rss():
    """Yields a dict that receives {"rss_before_bytes", "peak_rss_bytes", "work_bytes", "hwm_before_bytes"} (None
    if unsupported). Resetting the high-water mark also lowers what getrusage's ru_maxrss reports afterwards, so the
    process's peak BEFORE the reset is kept as hwm_before_bytes (for a post-run step: the run's own peak)."""
    out = {"rss_before_bytes": None, "peak_rss_bytes": None, "work_bytes": None, "hwm_before_bytes": None}
    hwm0 = _status_kb("VmHWM")
    out["hwm_before_bytes"] = None if hwm0 is None else 1024 * hwm0
    ok = False
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")                 # reset the peak-RSS high-water mark to the current RSS
        ok = True
    except OSError:
        ok = False
    before = _status_kb("VmRSS") if ok else None
    try:
        yield out
    finally:
        if ok and before is not None:
            hwm = _status_kb("VmHWM")
            if hwm is not None:
                out["rss_before_bytes"] = 1024 * before
                out["peak_rss_bytes"] = 1024 * hwm
                out["work_bytes"] = 1024 * max(0, hwm - before)


# --- exact spectra --------------------------------------------------------------------------------------------
def _as_state(psi, n: int | None) -> tuple[np.ndarray, int]:
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    if n is None:
        n = int(round(math.log2(psi.size)))
    if psi.size != 1 << n:
        raise ValueError(f"state of size {psi.size} is not 2^{n}")
    nrm = np.linalg.norm(psi)
    if not nrm > 0:
        raise ValueError("zero state")
    return psi / nrm, n


def schmidt_spectra(psi, n: int | None = None) -> list[np.ndarray]:
    """Exact singular values at every cut k = 1 .. n-1 (cut after site k-1), by one sequential exact SVD. Values
    below the numerical-rank floor S_max * max(shape) * eps (numpy's matrix_rank rule) are rounding, not rank, and
    are dropped."""
    psi, n = _as_state(psi, n)
    rem = psi.reshape(1, -1)
    out = []
    for _ in range(n - 1):
        M = rem.reshape(rem.shape[0] * 2, -1)
        _, S, Vh = np.linalg.svd(M, full_matrices=False)
        keep = S > S[0] * max(M.shape) * np.finfo(np.float64).eps
        S, Vh = S[keep], Vh[keep]
        out.append(S)
        rem = S[:, None] * Vh
    return out


def half_cut(n: int) -> int:
    """|A| = ceil(n / 2): the cut after site ceil(n/2) - 1, i.e. index ceil(n/2) - 1 of `schmidt_spectra`."""
    return (n + 1) // 2


def entropy_ebits(S: np.ndarray) -> float:
    p = np.asarray(S, dtype=np.float64) ** 2
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def half_cut_entropy(psi, n: int | None = None) -> float:
    psi, n = _as_state(psi, n)
    a = half_cut(n)
    S = np.linalg.svd(psi.reshape(1 << a, -1), compute_uv=False)
    return entropy_ebits(S)


# --- truncation ------------------------------------------------------------------------------------------------
def fidelity_numpy(psi, n: int, chi: int) -> tuple[float, list[int]]:
    """F(chi) and the bond dimensions of the truncated left-to-right sequential SVD (numpy engine)."""
    rem = psi.reshape(1, -1)
    bonds = []
    for _ in range(n - 1):
        M = rem.reshape(rem.shape[0] * 2, -1)
        _, S, Vh = np.linalg.svd(M, full_matrices=False)
        r = int(min(chi, S.size))
        bonds.append(r)
        rem = S[:r, None] * Vh[:r]
    return float(np.vdot(rem, rem).real), bonds


def fidelity_quimb(psi, n: int, chi: int) -> tuple[float, list[int]]:
    import quimb.tensor as qtn
    mps = qtn.MatrixProductState.from_dense(psi, 2, max_bond=int(chi), cutoff=0.0)
    v = np.asarray(mps.to_dense()).ravel()
    nv = float(np.vdot(v, v).real)
    return float(abs(np.vdot(v, psi)) ** 2 / nv), [int(b) for b in mps.bond_sizes()]


def fidelity(psi, n: int, chi: int, engine: str = DEFAULT_ENGINE) -> tuple[float, list[int]]:
    if engine == "numpy":
        return fidelity_numpy(psi, n, chi)
    if engine == "quimb":
        return fidelity_quimb(psi, n, chi)
    raise ValueError(engine)


def chi_grid(n: int) -> list[int]:
    """1, 2, 4, ..., 2^floor(n/2) (the largest exact bond dimension of n qubits)."""
    return [1 << k for k in range(n // 2 + 1)]


def mem_mps_bytes(bonds: list[int]) -> int:
    """16 bytes x sum over sites of chi_left * 2 * chi_right."""
    b = [1] + list(bonds) + [1]
    return int(16 * sum(b[i] * 2 * b[i + 1] for i in range(len(b) - 1)))


def _bracket(spectra: list[np.ndarray], target: float, chi_max: int) -> tuple[int, int, callable, callable]:
    """(lo, hi, U, L): F(chi) < target for chi < lo and F(hi) >= target (module doc)."""
    cum = [np.cumsum(s.astype(np.float64) ** 2) for s in spectra]
    tot = [c[-1] for c in cum]

    def kept(k, chi):
        c = cum[k]
        return c[min(chi, c.size) - 1] / tot[k]

    def U(chi):
        return min(kept(k, chi) for k in range(len(cum))) if cum else 1.0

    def L(chi):
        return 1.0 - sum(1.0 - kept(k, chi) for k in range(len(cum))) if cum else 1.0

    lo = next((c for c in range(1, chi_max + 1) if U(c) >= target), chi_max)
    hi = next((c for c in range(lo, chi_max + 1) if L(c) >= target), chi_max)
    return lo, hi, U, L


def simdiff(psi, n: int | None = None, K: int | None = None, engine: str = DEFAULT_ENGINE,
            threshold: float = F_THRESHOLD, measure_rss: bool = True) -> dict:
    """All simulation-difficulty numbers of one state (module doc). Flat dict of scalars; the F(chi) curve as
    F_chi{chi} keys (and the spectral lower bound as F_bound_chi{chi})."""
    psi, n = _as_state(psi, n)
    ctx = peak_rss() if measure_rss else None
    rss = {"rss_before_bytes": None, "peak_rss_bytes": None, "work_bytes": None, "hwm_before_bytes": None}
    t0 = time.perf_counter()
    if ctx is not None:
        cm = ctx.__enter__()
    n_sweeps = 0
    try:
        spectra = schmidt_spectra(psi, n)
        chi_max = int(max((s.size for s in spectra), default=1))
        target = 1.0 - threshold
        cache: dict = {}

        def F_of(chi):
            nonlocal n_sweeps
            if chi >= chi_max:
                return 1.0
            if chi not in cache:
                cache[chi] = fidelity(psi, n, chi, engine)[0]
                n_sweeps += 1
            return cache[chi]

        grid = chi_grid(n)
        F = {chi: F_of(chi) for chi in grid}
        chi_g = next(c for c in grid if F[c] >= target)
        monotone = all(F[grid[i + 1]] >= F[grid[i]] - 1e-12 for i in range(len(grid) - 1))
        lo, hi, U, Lb = _bracket(spectra, target, chi_max)
        hi = min(hi, chi_g)                         # F(chi_g) >= target as well
        lo = max(lo, chi_g // 2 + 1 if chi_g > 1 else 1)
        a, b = lo - 1, hi                           # invariant: F(a) < target (or a = lo - 1), F(b) >= target
        while b - a > 1:
            mid = (a + b) // 2
            if F_of(mid) >= target:
                b = mid
            else:
                a = mid
        chi_star = b
        chi_bound = next((c for c in range(1, chi_max + 1) if Lb(c) >= target), chi_max)
        t1 = time.perf_counter()
        F_star, bonds = fidelity(psi, n, chi_star, engine)
        wall_star = time.perf_counter() - t1
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
            rss = cm
    a_half = half_cut(n)
    S_half = entropy_ebits(spectra[a_half - 1]) if n > 1 else 0.0
    out = {
        "n": n, "engine": engine, "threshold": threshold,
        "chi_star": int(chi_star), "chi_star_grid": int(chi_g), "chi_star_bound": int(chi_bound),
        "chi_bracket_lo": int(lo), "chi_bracket_hi": int(hi), "F_chi_star": float(F_star),
        "chi_exact_max": chi_max, "chi_half_exact": int(spectra[a_half - 1].size) if n > 1 else 1,
        "mem_repr_bytes": int(32 * n * chi_star ** 2), "mem_dense_bytes": int(16 * (1 << n)),
        "mem_mps_bytes": mem_mps_bytes(bonds),
        "S_half": S_half, "S_over_log2K": (S_half / math.log2(K)) if (K is not None and K > 1) else None,
        "F_monotone_grid": bool(monotone), "n_sweeps": n_sweeps, "wall_chi_star_s": wall_star,
        "wall_total_s": time.perf_counter() - t0,
        "peak_rss_bytes": rss["peak_rss_bytes"], "rss_before_bytes": rss["rss_before_bytes"],
        "work_bytes": rss["work_bytes"], "process_peak_rss_before_bytes": rss["hwm_before_bytes"],
    }
    for chi in grid:
        out[f"F_chi{chi}"] = float(F[chi])
        out[f"F_bound_chi{chi}"] = float(max(0.0, Lb(chi)))
    return out


def chi_star_scan(psi, n: int | None = None, threshold: float = F_THRESHOLD, engine: str = "numpy") -> int:
    """The smallest chi with F(chi) >= 1 - threshold by a full scan chi = 1, 2, 3, ... (tests)."""
    psi, n = _as_state(psi, n)
    chi = 1
    while fidelity(psi, n, chi, engine)[0] < 1.0 - threshold:
        chi += 1
    return chi


def cross_check(psi, n: int | None = None) -> float:
    """max over the chi grid of |F_numpy - F_quimb| (the S5 cross-test)."""
    psi, n = _as_state(psi, n)
    return max(abs(fidelity_numpy(psi, n, c)[0] - fidelity_quimb(psi, n, c)[0]) for c in chi_grid(n))
