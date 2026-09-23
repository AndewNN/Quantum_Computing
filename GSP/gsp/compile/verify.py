"""Numerical verification of the explicit decompositions (PLAN §2.4, §5 S3): each gate list's unitary
against the ideal gate it claims to implement, on `npsim`. Used by the tests and by the report.
"""

from __future__ import annotations

import numpy as np

from . import decompose as dc
from . import npsim
from .decompose import Angle


def mc_su2_error(kind: str, k: int, angle: float = 0.7351, open_controls: bool = False) -> dict:
    """Vale's C^k(R_kind(angle)) on k + 1 qubits (controls 0..k-1, target k) against the ideal gate.
    open_controls: X-conjugate the controls (the way the transitions use it)."""
    ctl = list(range(k))
    gl = dc.mc_su2(ctl, k, kind, Angle(angle))
    if open_controls:
        flips = [dc.Gate("x", (c,)) for c in ctl]
        gl = flips + gl + flips
    U = npsim.unitary(gl, k + 1)
    ref = npsim.ideal_controlled(k + 1, ctl, k, npsim.rot(kind, angle), open_controls)
    return {"kind": kind, "k": k, "err": float(np.abs(U - ref).max()), "cx": dc.cnot_count(gl),
            "cx_formula": dc.vale_cnots(k)}


def transition_errors(n: int, n_cases: int = 10, seed: int = 0, angle: float = 0.4123) -> dict:
    """Random transitions on n qubits (random u != v, random S within [n] \\ {k0}, Rx and Ry):
    max |U_(iii) - U_native| over the full 2^n x 2^n unitaries."""
    rng = np.random.default_rng(seed)
    worst, sizes = 0.0, []
    for case in range(n_cases):
        u, v = rng.choice(1 << n, size=2, replace=False)
        D = [k for k in range(n) if ((int(u) ^ int(v)) >> (n - 1 - k)) & 1]
        k0 = D[0]
        others = [k for k in range(n) if k != k0]
        s = int(rng.integers(1, min(5, len(others)) + 1)) if others else 0
        S = tuple(sorted(rng.choice(others, size=s, replace=False).tolist())) if s else ()
        xm = tuple(k for k in range(n) if (int(u) >> (n - 1 - k)) & 1)
        ladder = [(k0, k) for k in D[1:]]
        kind = "rx" if case % 2 == 0 else "ry"
        a = Angle(angle * (1 + case))
        U3 = npsim.unitary(dc.transition_iii(xm, ladder, S, k0, kind, a), n)
        Un = npsim.unitary(dc.transition_native(xm, ladder, S, k0, kind, a), n)
        worst = max(worst, float(np.abs(U3 - Un).max()))
        sizes.append(s)
    return {"n": n, "cases": n_cases, "max_err": worst, "S_sizes": sizes}


def ii_errors(n: int, n_cases: int = 4, seed: int = 1, angle: float = 0.37) -> dict:
    """V18's (ii) construction (n system + n - 2 clean ancillas) against the native transition with
    all n - 1 non-pivot qubits as open controls: the ancilla-zero block equals it, and nothing leaves
    the ancilla-zero subspace."""
    rng = np.random.default_rng(seed)
    N = 2 * n - 2
    worst_blk, worst_leak = 0.0, 0.0
    for case in range(n_cases):
        u, v = rng.choice(1 << n, size=2, replace=False)
        D = [k for k in range(n) if ((int(u) ^ int(v)) >> (n - 1 - k)) & 1]
        k0 = D[0]
        xm = tuple(k for k in range(n) if (int(u) >> (n - 1 - k)) & 1)
        ladder = [(k0, k) for k in D[1:]]
        kind = "rx" if case % 2 == 0 else "ry"
        a = Angle(angle * (1 + case))
        cols = np.arange(1 << n, dtype=np.int64) << (n - 2)          # ancillas (low bits) = 0
        psi = npsim.columns(N, cols)
        npsim.apply(dc.transition_ii(n, xm, ladder, k0, kind, a), psi)
        U = npsim.flat(psi)
        blk = U[cols, :]
        mask = np.ones(1 << N, dtype=bool)
        mask[cols] = False
        S = tuple(k for k in range(n) if k != k0)
        Un = npsim.unitary(dc.transition_native(xm, ladder, S, k0, kind, a), n)
        worst_blk = max(worst_blk, float(np.abs(blk - Un).max()))
        worst_leak = max(worst_leak, float(np.abs(U[mask]).max()))
    return {"n": n, "cases": n_cases, "block_err": worst_blk, "leak": worst_leak}
