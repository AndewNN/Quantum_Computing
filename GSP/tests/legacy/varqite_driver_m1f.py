"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for the S7 A3 / A3d reproduction.

Provenance
----------
source : VarQITE/experiment/driver_4routes_20260916.py (repo Quantum_Computing, untracked in git,
         sha256 8a6d83281c5b3e233f631ed1f7633004b4d0589683c0644d7c51c1eaabae3b0d); the driver that produced the
         stored 2026-09-16 route sweep (VarQITE/experiment/exp_Q2_L*_q1.5/*_{M1fC1,M4C1,...}_Ramp_boost_Jh.*).
lines  : 66-88    DEFAULTS, ROUTES
         90-102   _target_set, set_target
         105-147  kernel_qaoa_X_overlap, kernel_qaoa_X_trunc
         150-479  class VarQITE: __init__, _dataset, _hamiltonians, _exact, _ansatz, make_points_init, get_psi,
                  observe_energy, metrics, vtau, dpsi_all, build_A_dpsi, psd, fidelity, trunc_args,
                  generator_of_bits, var_diag, kappa_delta, build_A_fidelity, n_circuits_A, n_circuits_C, build_A,
                  build_C, solve_theta_dot, run_mcLachlan
copied : 2026-09-24 (GSP session S7), byte-for-byte via sed; only this header, the import block and
         `legacy_cfg` were added. `Utils.qaoaCUDAQ` names come from the frozen S1 / S4 copies
         (tests/legacy/qaoaCUDAQ_instance.py, qaoaCUDAQ_kernels.py); DATA_COV / DATA_RET point at the same dataset.
         The stored runs ran this under CUDA-Q 0.13.0 (env cudaq13) with PRECISION fp64, EXACT_MEASURE "prob" (the
         stored cfg); here it runs under the gsp env (CUDA-Q 0.15.1). It calls cudaq.get_state inside its loop
         (EXACT_MEASURE "prob"): a reference for the checks only, never harness code (PLAN §3.3).
"""
# --- import block (added) ---
import os
import time
from typing import List

import numpy as np
import pandas as pd
import cudaq
from cudaq import spin
import torch
from tqdm import tqdm

from tests.legacy.qaoaCUDAQ_instance import (po_normalize, ret_cov_to_QUBO, qubo_to_ising, process_ansatz_values,
                                             all_state_to_return, find_budget, to_sig)
from tests.legacy.qaoaCUDAQ_kernels import kernel_qaoa_X

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_COV = os.path.join(_REPO, "dataset", "top_50_us_stocks_data_20250526_011226_covariance.csv")
DATA_RET = os.path.join(_REPO, "dataset", "top_50_us_stocks_returns_price.csv")


def legacy_cfg(route, **grid):
    """DEFAULTS + route preset + grid, the merge rule of the driver's `route_cfg` (lines 597-605)."""
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in ROUTES[route].items() if k != "desc"})
    cfg.update(grid)
    return cfg


# --- driver_4routes_20260916.py:66-88 (verbatim) ---
DEFAULTS = dict(
    e=0, SEED=0, N_ASSETS=7, TARGET_QUBIT_IN=2, q=1.5, lamb=0.005, LAYER=5, eps=0.1,
    INIT="Ramp", DELTA_GAMMA=3.0, DELTA_BETA=1.5,
    OPTIMIZE_METHOD="mcLachlan",
    DTAU=0.1, N_STEPS=300, FD_SHIFT=1e-4, GRAD_METHOD="fd", PHASE_CORRECTION=True,
    A_METHOD="fidelity_kappa", FD_SCHEME="forward", FID_STENCIL="central", FID_SHIFT=1e-2,
    KAPPA=0.01, CAP_ANGLE=0.02, SHOTS=None, EXACT_MEASURE="observe",
    INVERSE_METHOD="tikhonov", TIKHONOV_LAMBDA=1e-7, EIG_CUTOFF=1e-12,
    MAX_ITER=300, LR=0.01, SHIFT=1e-4, WEIGHT_DECAY=0.0,
    F_TOL=1e-4, HAM_BOOST_MODE="Jh", HAM_BOOST=1.0, PRECISION="fp64", TARGET="nvidia",
    min_P=108, max_P=216, DUPLICATE_ASSET=False,
)

ROUTES = {
    "M6C2": dict(OPTIMIZE_METHOD="mcLachlan", A_METHOD="dpsi", FD_SCHEME="central",
                 desc="M6 statevector FD (reference, not measurable) + C2 central"),
    "GRAD": dict(OPTIMIZE_METHOD="gradient", GRAD_METHOD="fd",
                 desc="Adam + forward-difference gradient (old method, reference 2)"),
    "M1fC1": dict(OPTIMIZE_METHOD="mcLachlan", A_METHOD="fidelity_kappa", FID_STENCIL="forward", KAPPA=0.01, CAP_ANGLE=0.02, FD_SCHEME="forward",
                  desc="M1' overlap forward stencil, kappa shifts, Var(G_k) diagonal + C1 forward"),
    "M4C1": dict(OPTIMIZE_METHOD="mcLachlan", A_METHOD="diag", FD_SCHEME="forward",
                 desc="M4 diagonal Var(G_k) only + C1 forward"),
}

# --- driver_4routes_20260916.py:90-102 (verbatim) ---
_target_set = None


def set_target(target, precision):
    """nvidia (GPU statevector, fp64 option) or qpp-cpu (CPU statevector, always fp64). Same numbers to roundoff;
    at 14 qubits the GPU is launch-overhead-bound, so many CPU processes in parallel can give more throughput."""
    global _target_set
    if _target_set != (target, precision):
        if target == "nvidia":
            cudaq.set_target("nvidia", option="fp64") if precision == "fp64" else cudaq.set_target("nvidia")
        else:
            cudaq.set_target(target)
        _target_set = (target, precision)


# --- driver_4routes_20260916.py:105-147 (verbatim) ---
@cudaq.kernel
def kernel_qaoa_X_overlap(thetas_a: List[float], thetas_b: List[float], qubit_count: int, layer_count: int, idx_1: List[int], coeff_1: List[float], idx_2_a: List[int], idx_2_b: List[int], coeff_2: List[float]):
    # U(thetas_b) followed by U(thetas_a)^dagger: P(|0...0>) = |<psi(a)|psi(b)>|^2
    qreg = cudaq.qvector(qubit_count)
    h(qreg)
    for i in range(layer_count):
        for j in range(len(idx_1)):
            rz(2 * coeff_1[j] * thetas_b[i], qreg[idx_1[j]])
        for j in range(len(idx_2_a)):
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
            rz(2 * coeff_2[j] * thetas_b[i], qreg[idx_2_b[j]])
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
        for j in range(qubit_count):
            rx(2.0 * thetas_b[layer_count + i], qreg[j])
    for i in range(layer_count - 1, -1, -1):
        for j in range(qubit_count):
            rx(-2.0 * thetas_a[layer_count + i], qreg[j])
        for j in range(len(idx_2_a) - 1, -1, -1):
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
            rz(-2 * coeff_2[j] * thetas_a[i], qreg[idx_2_b[j]])
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
        for j in range(len(idx_1) - 1, -1, -1):
            rz(-2 * coeff_1[j] * thetas_a[i], qreg[idx_1[j]])
    h(qreg)


@cudaq.kernel
def kernel_qaoa_X_trunc(thetas: List[float], qubit_count: int, layer_count: int, idx_1: List[int], coeff_1: List[float], idx_2_a: List[int], idx_2_b: List[int], coeff_2: List[float], n_cost: int, n_mix: int, x_basis: int):
    # first n_cost cost layers and n_mix mixer layers (n_mix in {n_cost - 1, n_cost}); x_basis=1 rotates to the X basis before measuring
    qreg = cudaq.qvector(qubit_count)
    h(qreg)
    for i in range(n_cost):
        for j in range(len(idx_1)):
            rz(2 * coeff_1[j] * thetas[i], qreg[idx_1[j]])
        for j in range(len(idx_2_a)):
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
            rz(2 * coeff_2[j] * thetas[i], qreg[idx_2_b[j]])
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
        if i < n_mix:
            for j in range(qubit_count):
                rx(2.0 * thetas[layer_count + i], qreg[j])
    if x_basis == 1:
        h(qreg)


# --- driver_4routes_20260916.py:150-479 (verbatim) ---
class VarQITE:
    """One PO instance (e, N_ASSETS, lamb, q) plus the estimators and optimizers of cudaq.ipynb."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        for k, v in self.cfg.items():
            setattr(self, k, v)
        assert self.INIT in ["Zero", "Random", "Ramp"]
        assert self.OPTIMIZE_METHOD in ["gradient", "mcLachlan"]
        assert self.INVERSE_METHOD in ["diagonalize", "tikhonov"]
        assert self.GRAD_METHOD == "fd", "psr is notebook-only (17-62x slower, identical in fp64)"
        assert self.A_METHOD in ["dpsi", "fidelity", "fidelity_kappa", "diag"]
        assert self.FD_SCHEME in ["forward", "central"]
        assert self.FID_STENCIL in ["forward", "central"]
        assert self.HAM_BOOST_MODE in ["J", "Jh", "h", "fixed"]
        assert self.EXACT_MEASURE in ["observe", "prob"]
        set_target(self.TARGET, self.PRECISION)
        self.device = torch.device("cuda:0" if self.TARGET == "nvidia" else "cpu")
        self.rng = np.random.default_rng(self.SEED)
        self._dataset()
        self._hamiltonians()
        self._exact()
        self._ansatz()

    # ---------------------------------------------------------------- instance
    def _dataset(self):
        st = time.perf_counter()
        data_cov_pd = pd.read_csv(DATA_COV)
        data_ret_p_pd = pd.read_csv(DATA_RET)
        data_ret_p_pd = data_ret_p_pd[(data_ret_p_pd["Price"] > self.min_P) & (data_ret_p_pd["Price"] < self.max_P)]
        data_cov_pd = data_cov_pd.loc[data_cov_pd["Ticker"].isin(data_ret_p_pd["Ticker"])].reset_index(drop=True)
        data_cov_pd = data_cov_pd[["Ticker"] + data_cov_pd["Ticker"].tolist()]

        e, N_ASSETS = self.e, self.N_ASSETS
        np.random.seed(911 + 991 * e + 997 * N_ASSETS)
        rng_state = np.random.get_state()
        asset_idx = np.random.choice(data_cov_pd.shape[0], N_ASSETS, replace=self.DUPLICATE_ASSET)
        self.data_cov = data_cov_pd.drop("Ticker", axis=1).to_numpy()[asset_idx, :][:, asset_idx]
        self.stock_names = data_ret_p_pd["Company_Name"].to_numpy()[asset_idx]
        data_ret_p = data_ret_p_pd.drop("Ticker", axis=1)
        self.asset_idx_raw = data_ret_p.index[asset_idx].to_numpy()
        data_ret_p = data_ret_p.drop("Company_Name", axis=1).to_numpy()[asset_idx, :]
        self.data_ret = data_ret_p[:, 0]
        self.data_p = data_ret_p[:, 1]

        np.random.set_state(rng_state)
        weighted = np.random.uniform(0, 1)
        B_mi, B_ma = find_budget(self.TARGET_QUBIT_IN * N_ASSETS, self.data_p, self.min_P, self.max_P, min_mix_mode=True)
        self.B = B_mi * weighted + B_ma * (1 - weighted)
        self.data_time = time.perf_counter() - st

    def _hamiltonians(self):
        st = time.perf_counter()
        lamb, q = self.lamb, self.q
        P, ret, cov = self.data_p[:self.N_ASSETS], self.data_ret[:self.N_ASSETS], self.data_cov[:self.N_ASSETS, :self.N_ASSETS]
        P_bb, ret_bb, cov_bb, n_qubit, n_max, C_enc = po_normalize(self.B, P, ret, cov)
        self.n_qubit = int(n_qubit)

        self.QU = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
        self.QU_lamb = ret_cov_to_QUBO(np.zeros_like(ret_bb), np.zeros_like(cov_bb), P_bb, lamb, 0.0)
        self.QU_eval = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, 0.0, q)
        QU_return = ret_cov_to_QUBO(ret_bb, np.zeros_like(cov_bb), np.zeros_like(P_bb), 0.0, 0.0)
        QU_risk = ret_cov_to_QUBO(np.zeros_like(ret_bb), cov_bb, np.zeros_like(P_bb), 0.0, q)

        H_ansatz = -qubo_to_ising(self.QU, lamb).canonicalize()
        idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use = process_ansatz_values(H_ansatz)
        self.idx_1_use, self.idx_2_a_use, self.idx_2_b_use = idx_1_use, idx_2_a_use, idx_2_b_use
        self.coeff_1_use, self.coeff_2_use = np.array(coeff_1_use), np.array(coeff_2_use)

        max_J = np.max(np.abs(self.coeff_2_use))
        max_h = np.max(np.abs(self.coeff_1_use))
        self.max_J_h = max(max_J, max_h)
        mode = self.HAM_BOOST_MODE
        use_norm = max_J if mode == "J" else self.max_J_h if mode == "Jh" else max_h if mode == "h" else 1.0
        self.hamiltonian_boost = to_sig(1 / use_norm if mode != "fixed" else self.HAM_BOOST, 4)
        self.H_ansatz = H_ansatz * self.hamiltonian_boost
        self.H_eval = -qubo_to_ising(self.QU_eval, 0.0).canonicalize() * self.hamiltonian_boost
        self.H_lamb = -qubo_to_ising(self.QU_lamb, lamb).canonicalize() * self.hamiltonian_boost
        self.H_return = -qubo_to_ising(QU_return, 0.0).canonicalize() * self.hamiltonian_boost
        self.H_risk = -qubo_to_ising(QU_risk, 0.0).canonicalize() * self.hamiltonian_boost
        self.ham_time = time.perf_counter() - st

    def _exact(self):
        st = time.perf_counter()
        n_qubit, lamb = self.n_qubit, self.lamb
        self.state_eval = all_state_to_return(n_qubit, 0.0, self.QU_eval)
        state_optim = -all_state_to_return(n_qubit, lamb, self.QU)
        state_penalty = -all_state_to_return(n_qubit, lamb, self.QU_lamb)
        self.state_penalty = state_penalty
        self.diag_H = self.hamiltonian_boost * state_optim.astype(np.float64)

        order = np.argsort(state_optim)
        self.E0, self.E1 = self.diag_H[order[0]], self.diag_H[order[1]]
        self.idx_ground = np.where(np.isclose(state_optim, state_optim[order[0]]))[0]
        self.idx_optimal = int(np.argsort(self.state_eval)[-1])

        eps_t = lamb * self.eps ** 2
        self.idx_feasible = np.where(np.abs(state_penalty) <= eps_t)[0]
        self.mi_r, self.ma_r = (self.state_eval[self.idx_feasible].min(), self.state_eval[self.idx_feasible].max()) if len(self.idx_feasible) >= 2 else (np.nan, np.nan)
        self.exact_time = time.perf_counter() - st

    def _ansatz(self):
        self.layer_count = self.LAYER
        self.parameter_count = 2 * self.layer_count
        self.ansatz_fixed_param = (int(self.n_qubit), self.layer_count, self.idx_1_use, self.coeff_1_use, self.idx_2_a_use, self.idx_2_b_use, self.coeff_2_use)
        self.axes_flip = tuple(range(self.n_qubit - 1, -1, -1))
        self.dim = 1 << self.n_qubit

        n_qubit, layer_count = self.n_qubit, self.layer_count
        P_zero = 1.0
        for j in range(n_qubit):
            P_zero = P_zero * (0.5 * (spin.i(j) + spin.z(j)))
        self.P_zero = P_zero
        self.G_gamma = self.H_ansatz * (1.0 / self.hamiltonian_boost)   # generator of gamma in circuit units: rz(2 c gamma) = exp(-i c gamma Z)
        self.G_gamma2 = self.G_gamma * self.G_gamma
        G_beta = spin.x(0)                                              # generator of beta: rx(2 beta) = exp(-i beta X)
        for j in range(1, n_qubit):
            G_beta = G_beta + spin.x(j)
        self.G_beta = G_beta
        self.G_beta2 = G_beta * G_beta
        self.gen_scale = np.array([self.max_J_h] * layer_count + [1.0] * layer_count)   # largest gate angle per unit of parameter
        # EXACT_MEASURE == "prob": measured-basis values of the generators in cudaq's state ordering (q0 = LSB)
        self.G_gamma_diag = (self.diag_H / self.hamiltonian_boost).reshape([2] * n_qubit).transpose(self.axes_flip).ravel()   # G_gamma in circuit units (Z basis)
        idx = np.arange(self.dim)
        self.G_beta_diag = n_qubit - 2.0 * np.array([bin(i).count("1") for i in idx])                                         # sum_j Z_j after h (X basis)

    def make_points_init(self):
        layer_count, parameter_count = self.layer_count, self.parameter_count
        mm_1 = np.min(np.abs(self.coeff_1_use)) if len(self.coeff_1_use) > 0 else 1e9
        mm_2 = np.min(np.abs(self.coeff_2_use)) if len(self.coeff_2_use) > 0 else 1e9
        mm_i = np.pi / min(mm_1, mm_2)
        np.random.seed(4001 + 4099 * self.e + 4999 * self.N_ASSETS + 5099 * self.SEED)
        points_init = np.zeros(parameter_count)
        if self.INIT == "Random":
            points_init = np.random.uniform(-1, 1, parameter_count)
            points_init[:layer_count] *= mm_i
            points_init[layer_count:] *= np.pi
        elif self.INIT == "Ramp":
            for i in range(layer_count):
                points_init[i] = self.DELTA_GAMMA * (i + 1) / layer_count
                points_init[layer_count + i] = self.DELTA_BETA * (1 - i / layer_count)
        return points_init

    # ---------------------------------------------------------------- quantum values
    def get_psi(self, params):     # diagnostics / M6 reference only
        psi = np.array(cudaq.get_state(kernel_qaoa_X, params, *self.ansatz_fixed_param))
        return psi.reshape([2] * self.n_qubit).transpose(self.axes_flip).ravel().astype(np.complex128)

    def observe_energy(self, H, params):
        return float(cudaq.observe(kernel_qaoa_X, H, params, *self.ansatz_fixed_param).expectation())

    def metrics(self, params, psi):
        prob = np.abs(psi) ** 2
        E_obj = self.observe_energy(self.H_ansatz, params) / self.hamiltonian_boost
        E_eval = self.observe_energy(self.H_eval, params) / self.hamiltonian_boost
        E_lamb = self.observe_energy(self.H_lamb, params) / self.hamiltonian_boost
        P_ground = float(prob[self.idx_ground].sum())
        P_optimal = float(prob[self.idx_optimal])
        approx_ratio = (float(prob @ self.state_eval) - self.mi_r) / (self.ma_r - self.mi_r) if len(self.idx_feasible) >= 2 else float("nan")
        return E_obj, E_eval, E_lamb, P_ground, P_optimal, approx_ratio

    def vtau(self, psi):
        prob = np.abs(psi) ** 2
        return float(prob @ self.diag_H ** 2 - (prob @ self.diag_H) ** 2)

    # ---------------------------------------------------------------- A (metric)
    def dpsi_all(self, params):
        dps = np.zeros((self.parameter_count, self.dim), dtype=np.complex128)
        for i in range(self.parameter_count):
            pp, pm = params.copy(), params.copy()
            pp[i] += self.FD_SHIFT
            pm[i] -= self.FD_SHIFT
            dps[i] = (self.get_psi(pp) - self.get_psi(pm)) / (2 * self.FD_SHIFT)
        return dps

    def build_A_dpsi(self, psi, dps):
        A = np.real(dps.conj() @ dps.T)
        if self.PHASE_CORRECTION:
            v = dps.conj() @ psi
            A -= np.real(np.outer(v, v.conj()))
        return A

    @staticmethod
    def psd(A):
        w, V = np.linalg.eigh(A)
        return V @ (np.abs(w) * V.T)

    def fidelity(self, thetas_a, thetas_b):
        # F = P(|0...0>) of the compute-uncompute circuit. "observe": <P_0> with P_0 = prod (I+Z_j)/2 expanded into 2^n Pauli
        # terms (notebook route, 16k terms at 14 qubits); "prob": the same probability read off the simulated circuit (S -> inf limit of sample)
        if self.EXACT_MEASURE == "observe":
            F = float(cudaq.observe(kernel_qaoa_X_overlap, self.P_zero, thetas_a, thetas_b, *self.ansatz_fixed_param).expectation())
        else:
            amp0 = np.array(cudaq.get_state(kernel_qaoa_X_overlap, thetas_a, thetas_b, *self.ansatz_fixed_param))[0]
            F = float(abs(amp0) ** 2)
        return self.rng.binomial(self.SHOTS, min(max(F, 0.0), 1.0)) / self.SHOTS if self.SHOTS else F

    def trunc_args(self, k):
        lc = self.layer_count
        return (k + 1, k, self.G_gamma, self.G_gamma2) if k < lc else (k - lc + 1, k - lc + 1, self.G_beta, self.G_beta2)

    def generator_of_bits(self, k, bitstr):          # G_k evaluated on a measured bitstring (char j = qubit j)
        z = 1 - 2 * np.array([int(ch) for ch in bitstr])
        if k >= self.layer_count:
            return float(z.sum())
        return float(self.coeff_1_use @ z[self.idx_1_use] + self.coeff_2_use @ (z[self.idx_2_a_use] * z[self.idx_2_b_use]))

    def var_diag(self, params):                      # A_kk = Var_{psi_k}(G_k): 2 observes (exact) or 1 sampled circuit (SHOTS) per parameter
        out = np.zeros(self.parameter_count)
        for k in range(self.parameter_count):
            n_cost, n_mix, G, G2 = self.trunc_args(k)
            if self.SHOTS:
                res = cudaq.sample(kernel_qaoa_X_trunc, params, *self.ansatz_fixed_param, n_cost, n_mix, 1 if k >= self.layer_count else 0, shots_count=int(self.SHOTS))
                vals = np.array([self.generator_of_bits(k, b) for b in res])
                w = np.array([res.count(b) for b in res], dtype=float)
                m1 = w @ vals / w.sum()
                out[k] = w @ (vals - m1) ** 2 / w.sum()
            elif self.EXACT_MEASURE == "observe":
                m1 = cudaq.observe(kernel_qaoa_X_trunc, G, params, *self.ansatz_fixed_param, n_cost, n_mix, 0).expectation()
                m2 = cudaq.observe(kernel_qaoa_X_trunc, G2, params, *self.ansatz_fixed_param, n_cost, n_mix, 0).expectation()
                out[k] = m2 - m1 ** 2
            else:   # exact measured-basis distribution of the one truncated circuit (S -> inf limit of the sample branch)
                x_basis = 1 if k >= self.layer_count else 0
                prob = np.abs(np.array(cudaq.get_state(kernel_qaoa_X_trunc, params, *self.ansatz_fixed_param, n_cost, n_mix, x_basis))) ** 2
                vals = self.G_beta_diag if x_basis else self.G_gamma_diag
                m1 = prob @ vals
                out[k] = prob @ vals ** 2 - m1 ** 2
        return np.maximum(out, 0.0)

    def kappa_delta(self, diag):
        return np.minimum(self.KAPPA / np.sqrt(np.maximum(diag, 1e-300)), 0.5 * self.CAP_ANGLE / self.gen_scale)

    def build_A_fidelity(self, params, delta=None, diag=None):
        p = self.parameter_count
        delta = np.full(p, self.FID_SHIFT) if delta is None else delta
        E = np.eye(p) * delta[:, None]
        F = lambda s: self.fidelity(params, params + s)  # noqa: E731
        A = np.zeros((p, p))
        if self.FID_STENCIL == "forward":
            F_i = np.array([F(E[i]) for i in range(p)])
            for i in range(p):
                A[i, i] = (1 - F_i[i]) / delta[i] ** 2
                for j in range(i + 1, p):
                    A[i, j] = A[j, i] = (F_i[i] + F_i[j] - F(E[i] + E[j]) - 1) / (2 * delta[i] * delta[j])
        else:
            for i in range(p):
                if diag is None:
                    A[i, i] = (2 - F(E[i]) - F(-E[i])) / (2 * delta[i] ** 2)
                for j in range(i + 1, p):
                    A[i, j] = A[j, i] = -(F(E[i] + E[j]) - F(E[i] - E[j]) - F(-E[i] + E[j]) + F(-E[i] - E[j])) / (8 * delta[i] * delta[j])
        if diag is not None:
            A[np.diag_indices(p)] = diag
        return self.psd(A) if self.SHOTS else A

    def n_circuits_A(self):          # circuits per step charged to A
        p, method, stencil = self.parameter_count, self.A_METHOD, self.FID_STENCIL
        if method in ("dpsi", "diag"):
            return 2 * p
        off = p + p * (p - 1) // 2 if stencil == "forward" else 2 * p * (p - 1)
        return off + (2 * p if method == "fidelity_kappa" else (0 if stencil == "forward" else 2 * p))

    def n_circuits_C(self):          # circuits per step charged to C (or to the gradient)
        p = self.parameter_count
        if self.OPTIMIZE_METHOD == "gradient":
            return p + 1
        return p + 1 if self.FD_SCHEME == "forward" else 2 * p

    def build_A(self, params, psi=None):
        if self.A_METHOD == "fidelity":
            return self.build_A_fidelity(params)
        if self.A_METHOD == "fidelity_kappa":
            d = self.var_diag(params)
            return self.build_A_fidelity(params, self.kappa_delta(d), d)
        if self.A_METHOD == "diag":
            return np.diag(self.var_diag(params))
        return self.build_A_dpsi(self.get_psi(params) if psi is None else psi, self.dpsi_all(params))

    # ---------------------------------------------------------------- C and the solve
    def build_C(self, params):
        sigma = np.sqrt(self.vtau(self.get_psi(params)) / self.SHOTS) if self.SHOTS else 0.0   # shot noise on each energy (noise model only)
        E = lambda p: self.observe_energy(self.H_ansatz, p) + (self.rng.normal(0.0, sigma) if self.SHOTS else 0.0)  # noqa: E731
        C = np.zeros(self.parameter_count)
        E_0 = E(params) if self.FD_SCHEME == "forward" else None
        for i in range(self.parameter_count):
            pp, pm = params.copy(), params.copy()
            pp[i] += self.FD_SHIFT
            pm[i] -= self.FD_SHIFT
            if self.FD_SCHEME == "forward":
                C[i] = -0.5 * (E(pp) - E_0) / self.FD_SHIFT
            else:
                C[i] = -0.5 * (E(pp) - E(pm)) / (2 * self.FD_SHIFT)
        return C

    def solve_theta_dot(self, A, C):
        if self.INVERSE_METHOD == "diagonalize":
            w, V = np.linalg.eigh(A)
            w_inv = np.zeros_like(w)
            keep = w > self.EIG_CUTOFF * np.max(np.abs(w))
            w_inv[keep] = 1.0 / w[keep]
            return V @ (w_inv * (V.T @ C))
        return np.linalg.solve(A + self.TIKHONOV_LAMBDA * np.eye(self.parameter_count), C)

    # ---------------------------------------------------------------- optimizers
    def run_mcLachlan(self, params_init, pbar=True):
        params = params_init.copy()
        history, iter_times = [], []
        last_f, cou_con, num_iter = None, 0, 0
        psi = self.get_psi(params)  # diagnostics only (P_ground etc.); feeds the update only when A_METHOD == "dpsi"
        pbar_it = tqdm(range(self.N_STEPS), disable=not pbar)
        for it in pbar_it:
            st_it = time.perf_counter()
            A = self.build_A(params, psi)
            C = self.build_C(params)
            theta_dot = self.solve_theta_dot(A, C)
            cooling = float(C @ theta_dot)                     # C^T A^+ C
            R_min = self.vtau(psi) - cooling                   # McLachlan minimum residual, must be >= 0
            params += self.DTAU * theta_dot
            psi = self.get_psi(params)
            iter_times.append(time.perf_counter() - st_it)
            E_obj, E_eval, E_lamb, P_ground, P_optimal, approx_ratio = self.metrics(params, psi)
            history.append([E_obj, E_eval, E_lamb, P_ground, P_optimal, approx_ratio, params[0], params[self.layer_count], R_min, -2 * cooling])
            num_iter += 1
            expectation = E_obj * self.hamiltonian_boost
            cou_con = cou_con + 1 if last_f is not None and abs(expectation - last_f) < self.F_TOL else 0
            if cou_con >= 3:
                break
            last_f = expectation
            if pbar:
                pbar_it.set_description(f"E {E_obj:.6f}, E_eval {E_eval:.6f}, P_gs {P_ground:.4f}")
        return params, np.array(history), np.array(iter_times), num_iter
