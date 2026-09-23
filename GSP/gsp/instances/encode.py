"""Encoding of a draw: normalization, QUBOs, Ising Hamiltonians, the Jh boost (PLAN §1.1).

Ports (copied, not imported) from Utils/qaoaCUDAQ.py: `po_normalize` (147-169),
`ret_cov_to_QUBO` (171-174), `all_state_to_return` (653-668, here chunked) and `to_sig`
(746-750). `qubo_to_ising` (176-184) and `process_ansatz_values` (186-210) built a
`cudaq.SpinOperator`; here they are replayed in numpy so that the coefficients are **bit-identical**
to the old operator without calling CUDA-Q: the old operator accumulated each (i, j) product
term into keyed terms in loop order and `canonicalize()` then merged terms that differ only by
identity factors, in insertion order. `tests/test_legacy_repro.py` checks 0-ulp agreement.

Conventions (the old code's, verbatim):
  QUBO of the MAX problem:  QU(lam, q) = diag(ret + 2 lam P) - (lam P P^T + q cov)
  x^T QU x - lam  = ret.x - q x^T cov x - lam (P.x - 1)^2          (all_state_to_return)
  Hamiltonian of the MIN problem: H = -qubo_to_ising(QU, lam), i.e. H(x) = -(x^T QU x - lam)
  H_obj = H at lam = 0 (q-dependent);  Pen = H of the pure penalty QUBO at lam = 1
          ((P.x - 1)^2);  H(lam) is built from QU(lam, q) directly, as the old code did.
  Jh boost: alpha = to_sig(1 / max(|J_ij|, |h_i|), 4) over the Hamiltonian being run.
  Qubit i <-> x_i through (1 - Z_i)/2. Every stored energy is un-boosted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bits import bit_matrix


# --- Utils/qaoaCUDAQ.py:147-174, copied verbatim (debug prints removed) ---
def po_normalize(B, P, ret, cov):
    P_b = P / B
    ret_b = ret * P_b
    cov_b = np.diag(P_b) @ cov @ np.diag(P_b)

    n_max = np.int32(np.floor(np.log2(B/P))) + 1
    n_qs = np.cumsum(n_max)
    n_qs = np.insert(n_qs, 0, 0)
    n_qubit = n_qs[-1]
    C = np.zeros((len(P), n_qubit))
    for i in range(len(P)):
         for j in range(n_max[i]):
              C[i, n_qs[i] + j] = 2**j

    P_bb = C.T @ P_b
    ret_bb = C.T @ ret_b
    cov_bb = C.T @ cov_b @ C
    return P_bb, ret_bb, cov_bb, int(n_qubit), n_max, C


def ret_cov_to_QUBO(ret: np.ndarray, cov: np.ndarray, P: np.ndarray, lamb: float, q:float) -> np.ndarray: # Max return, Min variance
    di = np.diag(ret + 2*lamb*P)
    mat = lamb * np.outer(P, P) + q * cov
    return di - mat


# --- Utils/qaoaCUDAQ.py:746-750, copied verbatim ---
def to_sig(x, sig=3):
    x = float(x)
    res = float(f"{x:.{sig}}")
    res = int(res) if res.is_integer() else res
    return res
# --- end of copies ---


@dataclass(frozen=True)
class Ising:
    """H = const + sum_i h[i] Z_i + sum_{i<j} J[i, j] Z_i Z_j  (J strictly upper triangular).

    `has_h[i]` / `has_J[i, j]` mark the terms the old SpinOperator carried (a carried term may
    still have a zero coefficient; `terms()` drops zeros exactly like `process_ansatz_values`).
    """

    n: int
    const: float
    h: np.ndarray
    J: np.ndarray
    has_h: np.ndarray
    has_J: np.ndarray

    def scaled(self, alpha: float) -> "Ising":
        """Multiply every coefficient by alpha (the boost), like `H * hamiltonian_boost`."""
        return Ising(self.n, self.const * alpha, self.h * alpha, self.J * alpha, self.has_h, self.has_J)

    def terms(self):
        """(idx_1, coeff_1, idx_2_a, idx_2_b, coeff_2) in the order of `process_ansatz_values`:
        one-body terms by qubit, then two-body terms by (a, b) lexicographically; zeros dropped."""
        idx_1, coeff_1, a2, b2, c2 = [], [], [], [], []
        for i in range(self.n):
            if self.has_h[i] and self.h[i] != 0:
                idx_1.append(i)
                coeff_1.append(float(self.h[i]))
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if self.has_J[a, b] and self.J[a, b] != 0:
                    a2.append(a)
                    b2.append(b)
                    c2.append(float(self.J[a, b]))
        return idx_1, coeff_1, a2, b2, c2

    def diagonal(self, idx) -> np.ndarray:
        """Ising energy of classical indices `idx` (x_0 = MSB). For tests and checks; the
        rulers use the QUBO form (`qubo_energies`), as the old metrics did."""
        from .bits import index_to_bits
        z = 1.0 - 2.0 * index_to_bits(idx, self.n).astype(np.float64)
        return self.const + z @ self.h + np.einsum("...i,ij,...j->...", z, self.J, z)


def qubo_to_ising(qubo: np.ndarray, lamb: float) -> Ising:
    """Numpy replay of the old `-qubo_to_ising(qubo, lamb).canonicalize()` (the MIN-problem H).

    Replays CUDA-Q's bookkeeping: the operator starts as -lamb * I_0; each (i, j) with a nonzero
    entry adds qubo[i, j] * (I - Z_i)/2 (I - Z_j)/2, whose four product terms go to keys over the
    qubit pair (a, b) = (min, max) in the order II, IZ, ZI, ZZ (diagonal: I_i, Z_i); keys are kept
    in first-insertion order; canonicalize() then sums keys with the same Z support in that order;
    finally the sign is flipped (exact).
    """
    qubo = np.asarray(qubo, dtype=np.float64)
    n = qubo.shape[0]
    terms: dict = {}

    def add(key, v):
        terms[key] = terms[key] + v if key in terms else v

    add(((0, 0),), -lamb * 1.0)           # (qubit, is_Z)
    for i in range(n):
        for j in range(n):
            v = qubo[i, j]
            if i != j and v != 0:
                a, b = (i, j) if i < j else (j, i)
                add(((a, 0), (b, 0)), v * 0.25)
                add(((a, 0), (b, 1)), v * -0.25)
                add(((a, 1), (b, 0)), v * -0.25)
                add(((a, 1), (b, 1)), v * 0.25)
            elif i == j and v != 0:
                add(((i, 0),), v * 0.5)
                add(((i, 1),), v * -0.5)
    const = None
    h = np.zeros(n)
    J = np.zeros((n, n))
    has_h = np.zeros(n, dtype=bool)
    has_J = np.zeros((n, n), dtype=bool)
    for key, c in terms.items():
        zs = tuple(qb for qb, isz in key if isz)
        if len(zs) == 0:
            const = c if const is None else const + c
        elif len(zs) == 1:
            (i,) = zs
            h[i] = h[i] + c if has_h[i] else c
            has_h[i] = True
        else:
            a, b = zs
            J[a, b] = J[a, b] + c if has_J[a, b] else c
            has_J[a, b] = True
    return Ising(n=n, const=-(const if const is not None else 0.0), h=-h, J=-J, has_h=has_h, has_J=has_J)


def jh_boost(H: Ising) -> float:
    """The completed work's `-norm Jh` boost: to_sig(1 / max(max|J|, max|h|), 4)
    (PO_new_ApproxRatio.py:606-617, learning_rate_scale = 1)."""
    _, c1, _, _, c2 = H.terms()
    max_J = np.max(np.abs(np.array(c2)))
    max_h = np.max(np.abs(np.array(c1)))
    max_J_h = max(max_J, max_h)
    return to_sig(1 / max_J_h * 1.0, 4)


def qubo_energies(qubo: np.ndarray, lam: float, start: int = 0, stop: int | None = None,
                  chunk: int = 1 << 16) -> np.ndarray:
    """x^T qubo x - lam for classical indices start..stop-1 (MAX-problem convention).

    Port of `all_state_to_return` (Utils/qaoaCUDAQ.py:653-668): the same float32 0/1 bit rows
    and the same two matmuls, evaluated chunk by chunk so n = 20 never materialises the full
    (2^n, n) matrices. Chunks start at multiples of `chunk`.
    """
    qubo = np.asarray(qubo)
    qb = qubo.shape[0]
    stop = (1 << qb) if stop is None else stop
    out = np.empty(stop - start, dtype=np.float64)
    pos = start
    while pos < stop:
        end = min(stop, (pos // chunk + 1) * chunk)
        l = bit_matrix(pos, end, qb, dtype=np.float32)
        ss = l @ qubo
        ss = (ss.reshape(-1, 1, qb) @ l.reshape(-1, qb, 1))
        out[pos - start:end - start] = ss.reshape(-1) - lam
        pos = end
    return out


@dataclass(frozen=True)
class Encoding:
    """Everything the old code derived from (B, P, ret, cov) for one q."""

    q: float
    P_bb: np.ndarray
    ret_bb: np.ndarray
    cov_bb: np.ndarray
    n: int
    n_max: np.ndarray
    C: np.ndarray
    QU_obj: np.ndarray        # ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, 0, q)   (the old QU_eval)
    QU_pen: np.ndarray        # ret_cov_to_QUBO(0, 0, P_bb, 1, 0)              (the old QU_lamb at lam = 1)
    H_obj: Ising              # -qubo_to_ising(QU_obj, 0)
    Pen: Ising                # -qubo_to_ising(QU_pen, 1): (P.x - 1)^2
    boost_obj: float          # jh_boost(H_obj): the boost of the confined arms


def encode(B: float, P, ret, cov, q: float) -> Encoding:
    P_bb, ret_bb, cov_bb, n, n_max, C = po_normalize(B, P, ret, cov)
    QU_obj = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, 0.0, q)
    QU_pen = ret_cov_to_QUBO(np.zeros_like(ret_bb), np.zeros_like(cov_bb), P_bb, 1.0, 0.0)
    H_obj = qubo_to_ising(QU_obj, 0.0)
    Pen = qubo_to_ising(QU_pen, 1.0)
    return Encoding(q=float(q), P_bb=P_bb, ret_bb=ret_bb, cov_bb=cov_bb, n=n, n_max=n_max, C=C,
                    QU_obj=QU_obj, QU_pen=QU_pen, H_obj=H_obj, Pen=Pen, boost_obj=jh_boost(H_obj))


def qubo_lambda(ret_bb, cov_bb, P_bb, lam: float, q: float) -> np.ndarray:
    """QU(lam, q): the old `QU = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)`."""
    return ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lam, q)


def hamiltonian(ret_bb, cov_bb, P_bb, lam: float, q: float) -> Ising:
    """H(lam) = H_obj + lam Pen, built with the old code's exact convention
    (`-qubo_to_ising(QU, lamb)` with QU at lam), un-boosted."""
    return qubo_to_ising(qubo_lambda(ret_bb, cov_bb, P_bb, lam, q), lam)
