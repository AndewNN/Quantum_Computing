"""A CPU statevector reference engine for the A3 McLachlan loop (S7; tests and checks only, never a harness engine).

`NumpyEngine(A)` offers the primitives of `gsp.arms.varqite.CircuitEngine` computed exactly from numpy states
(`gsp.compile.npsim` on the ansatz's simulation gates), so the loop can run on the CPU:
  energy(theta)        <alpha H> (constant included, as `Ansatz.op`);
  var_diag(theta)      Var(G_k) on parameter k's truncated state (gamma_k: k + 1 cost layers, k mixers, Z basis;
                       beta_k: k + 1 layers, X basis), G_gamma in circuit units;
  fidelity(a, b, m)    |<psi_m(a)|psi_m(b)>|^2 on the first m layers;
plus, for metric "exact" (M6, D-3: tests only):
  exact_metric(theta)  M = Re<d_i|d_j> - Re<d_i|psi><psi|d_j>, |d_k> = U_{>k} (-i G_k) U_{<=k} |+>, gates in circuit
                       order gamma_0, beta_0, gamma_1, ... (the example_varqite.py construction);
  exact_C(theta)       C_k = -Re<d_k| alpha H |psi> = -1/2 dE/dtheta_k.
Dense 2^n matrices: n <= 10.
"""

from __future__ import annotations

import numpy as np

from gsp.circuits.program import unroll_layered
from gsp.compile import npsim
from gsp.compile.decompose import Gate


def _bits(n):
    return ((np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.float64)


class NumpyEngine:
    def __init__(self, A):
        self.A = A
        self.n = int(A.n)
        self.L = int(A.L)
        self.n_params = 2 * self.L
        c = [abs(v) for v in A.ct.coeff_1 + A.ct.coeff_2]
        self.coef_max = float(max(c)) if c else 1.0
        self.gen_scale = np.array([self.coef_max] * self.L + [1.0] * self.L)
        idx = np.arange(1 << self.n)
        self.diag_H = A.H.diagonal(idx)                                  # un-boosted, constant included
        circ_alpha = A.alpha if A.meta.get("circuit_boosted") else 1.0
        self.g_gamma = circ_alpha * self.diag_H                           # G_gamma in circuit units (old H_ansatz / boost)
        z = 1.0 - 2.0 * _bits(self.n)
        self.g_beta_x = z.sum(axis=1)                                     # sum_j X_j in the X basis
        self.hall = [Gate("h", (q,)) for q in range(self.n)]
        # exact metric: the cost generator without the constant (a global phase: no effect on M or C)
        from gsp.circuits.cost import diagonal
        self.cost_diag = diagonal(A.ct, idx)

    def _state(self, gates, params) -> np.ndarray:
        psi = npsim.apply(gates, npsim.columns(self.n, [0]), params)
        return npsim.flat(psi)[:, 0].copy()

    def state(self, params, m: int | None = None) -> np.ndarray:
        m = self.L if m is None else int(m)
        g = [float(params[i]) for i in range(m)]
        b = [float(params[self.L + i]) for i in range(m)]
        return self._state(unroll_layered(self.A.start, self.A.layer, m), g + b)

    def energy(self, params) -> float:
        p = np.abs(self.state(params)) ** 2
        return float(self.A.alpha * (p @ self.diag_H))

    def var_diag(self, params) -> np.ndarray:
        out = np.zeros(self.n_params)
        for k in range(self.n_params):
            ell = k if k < self.L else k - self.L
            pr = np.array(params, dtype=np.float64).copy()
            if k < self.L:
                pr[self.L + ell] = 0.0
            psi = self.state(pr, ell + 1)
            if k < self.L:
                prob, vals = np.abs(psi) ** 2, self.g_gamma
            else:
                prob = np.abs(self._rotate_x(psi)) ** 2
                vals = self.g_beta_x
            m1 = prob @ vals
            out[k] = prob @ vals ** 2 - m1 ** 2
        return np.maximum(out, 0.0)

    def _rotate_x(self, psi):
        s = npsim.apply(self.hall, psi.reshape((2,) * self.n + (1,)).copy())
        return npsim.flat(s)[:, 0]

    def fidelity(self, pa, pb, m: int) -> float:
        return float(abs(np.vdot(self.state(pa, m), self.state(pb, m))) ** 2)

    # --- exact metric and C (M6) -------------------------------------------------------------------------
    def _derivs(self, params):
        n, L = self.n, self.L
        dim = 1 << n
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        HX = np.zeros((dim, dim), dtype=complex)
        for j in range(n):
            op = np.array([[1.0 + 0j]])
            for k in range(n):
                op = np.kron(op, X if k == j else np.eye(2))
            HX += op
        wX, VX = np.linalg.eigh(HX)

        def U_cost(t):
            return np.exp(-1j * t * self.cost_diag)

        def apply(kind, t, v):
            if kind == "c":
                return U_cost(t) * v
            return VX @ (np.exp(-1j * wX * t) * (VX.conj().T @ v))

        def gen(kind, v):
            return self.cost_diag * v if kind == "c" else HX @ v

        seq = []                                   # (kind, parameter index) in circuit order
        for ell in range(L):
            seq += [("c", ell), ("x", L + ell)]
        plus = np.ones(dim, dtype=complex) / np.sqrt(dim)
        layers = [plus]
        for kind, i in seq:
            layers.append(apply(kind, params[i], layers[-1]))
        phi = layers[-1]
        d = np.zeros((2 * L, dim), dtype=complex)
        for pos, (kind, i) in enumerate(seq):
            v = -1j * gen(kind, layers[pos + 1])
            for kind2, i2 in seq[pos + 1:]:
                v = apply(kind2, params[i2], v)
            d[i] = v
        return phi, d

    def exact_metric(self, params) -> np.ndarray:
        phi, d = self._derivs(params)
        A = np.real(d.conj() @ d.T)
        a = np.imag(d.conj() @ phi)                # <d_j|phi> ... Im part; M = A - a a^T
        return A - np.outer(a, a)

    def exact_C(self, params) -> np.ndarray:
        phi, d = self._derivs(params)
        Hphi = self.A.alpha * self.diag_H * phi
        return -np.real(d.conj() @ Hphi)
