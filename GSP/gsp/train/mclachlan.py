"""The McLachlan (VarQITE) update loop of arms A3 / A3d (PLAN §1.5, D-3), ported from the completed work's
`VarQITE/varqite_routes.py` (DEFAULTS, `var_diag`, `build_C`, `theta_dot`, the Ramp init) and the M1f fidelity
route (`M1fC1`) of `VarQITE/experiment/driver_4routes_20260916.py` (`kappa_delta`, `build_A_fidelity` with
FID_STENCIL = "forward", `solve_theta_dot`, `run_mcLachlan`).

Per step, at theta = theta_{t-1} (every quantity through the engine's observe-only primitives):
  d_k   = Var_{psi_k}(G_k)                         p variance circuits (M1 and diag)
  M     M1 ("M1", A3): kappa shifts delta_k = min(kappa / sqrt(d_k), cap / 2 / gen_scale_k), forward stencil
            F_i  = |<psi(theta)|psi(theta + delta_i e_i)>|^2,  F_ij = |<psi(theta)|psi(theta + delta_i e_i + delta_j e_j)>|^2
            M_ij = (F_i + F_j - F_ij - 1) / (2 delta_i delta_j)  (i != j),   M_ii = d_i
                                                   p + p(p-1)/2 overlap circuits
        diag ("diag", A3d = M4C1): M = diag(d)
        exact ("exact"): the engine's statevector metric (M6; tests only, D-3)
  C_i   = -1/2 (E(theta + fd e_i) - E(theta)) / fd  p energies (E(theta) is the previous step's post-update energy)
  thetadot = solve(M + tikhonov I, C);  theta_t = theta + dtau thetadot;  E(theta_t)   1 energy
  stop (verbatim `run_mcLachlan`): |E(theta_t) - E(theta_{t-1})| < f_tol for 3 consecutive steps, the first
  comparison at t = 2; else n_steps. E is the loop's energy (boosted units, the old `H_ansatz`).
Circuits per step: M1 p(p+1)/2 + p + (p + 1); diag p + (p + 1). The first step also measures E(theta_0), once.

Settings (verbatim, `McLachlanConfig` defaults): dtau 0.1, 300 steps, FD shift 1e-4 (forward), Tikhonov 1e-6,
kappa 0.01, cap angle 0.02 rad, f_tol 1e-4, patience 3. `psd` (S9a, O-11 evidence only, default off): M is replaced
by its projection onto the PSD cone (negative eigenvalues clipped at 0) before the Tikhonov solve; the spectrum
diagnostics stay those of the estimate, and psd_clip_sum / psd_clip_n record what was clipped.

Diagnostics per step (PLAN §1.6; the arm adds V_tau from the logger's state, never used by the update):
cond(M) and the singular values of the unregularized M, its numerical rank (numpy's default tolerance) and the
largest consecutive singular-value ratio (the SV gap and where it sits), lambda_min(M), the Tikhonov used,
cooling = C . thetadot (the old history column; dE/dtau = -2 cooling), thetadot^T M thetadot (for the McLachlan
residual of the applied step), |thetadot|, the largest gate-angle change of the step max_k dtau |thetadot_k| gen_scale_k
(`max_angle_step`; the arm counts steps above 1 rad as jumps), C, thetadot, d, the kappa shifts.

The post-update `logger(t, params)` receives a COPY at t = 0 and after every update; nothing it does reaches the
loop (a test runs with and without it). Wall clock excludes the logger.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

METRICS = ("M1", "diag", "exact")


@dataclass(frozen=True)
class McLachlanConfig:
    metric: str = "M1"
    dtau: float = 0.1
    n_steps: int = 300
    fd_shift: float = 1e-4
    tikhonov: float = 1e-6
    kappa: float = 0.01
    cap_angle: float = 0.02
    f_tol: float = 1e-4
    patience: int = 3
    psd: bool = False              # S9a (O-11 evidence only): project M onto its PSD cone before the Tikhonov solve

    def __post_init__(self):
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")


def ramp_init(L: int, delta_gamma: float = 3.0, delta_beta: float = 1.5) -> np.ndarray:
    """The Ramp init of `varqite_routes.make_points_init` for the McLachlan routes (positive ramp, circuit units:
    no sign flip, no boost; those apply to the RAMP routes only): gamma_i = 3 (i + 1) / L, beta_i = 1.5 (1 - i / L)."""
    x = np.zeros(2 * L)
    for i in range(L):
        x[i] = delta_gamma * (i + 1) / L
        x[L + i] = delta_beta * (1 - i / L)
    return x


def kappa_delta(diag: np.ndarray, gen_scale: np.ndarray, kappa: float, cap_angle: float) -> np.ndarray:
    """Per-parameter overlap shifts (driver `kappa_delta`)."""
    return np.minimum(kappa / np.sqrt(np.maximum(diag, 1e-300)), 0.5 * cap_angle / gen_scale)


def metric_m1(fid, params: np.ndarray, delta: np.ndarray, diag: np.ndarray, L: int) -> tuple:
    """M1 forward stencil (driver `build_A_fidelity`, FID_STENCIL "forward", exact branch: no PSD projection).
    fid(pa, pb, m) = |<psi(pa)|psi(pb)>|^2 on the first m layers. Returns (M, number of overlap circuits)."""
    p = params.size
    E = np.eye(p) * delta[:, None]
    lay = np.array([k if k < L else k - L for k in range(p)])
    F_i = np.array([fid(params, params + E[i], int(lay[i]) + 1) for i in range(p)])
    A = np.zeros((p, p))
    n_fid = p
    for i in range(p):
        A[i, i] = (1 - F_i[i]) / delta[i] ** 2
        for j in range(i + 1, p):
            F_ij = fid(params, params + (E[i] + E[j]), int(max(lay[i], lay[j])) + 1)
            n_fid += 1
            A[i, j] = A[j, i] = (F_i[i] + F_i[j] - F_ij - 1) / (2 * delta[i] * delta[j])
    A[np.diag_indices(p)] = diag
    return A, n_fid


def build_C(energy, params: np.ndarray, E_0: float, fd_shift: float) -> np.ndarray:
    """C_i = -1/2 dE/dtheta_i by forward differences (driver `build_C`, FD_SCHEME "forward")."""
    C = np.zeros(params.size)
    for i in range(params.size):
        pp = params.copy()
        pp[i] += fd_shift
        C[i] = -0.5 * (energy(pp) - E_0) / fd_shift
    return C


def psd_project(A: np.ndarray) -> tuple:
    """(V max(w, 0) V^T of the symmetrized A, sum of the clipped |negative eigenvalues|, how many were clipped): the
    old SHOTS branch's `psd` option (O-11 evidence; never used by the plan's A3). A matrix with no negative eigenvalue
    is returned unchanged (the eigen-reconstruction would only add rounding)."""
    S = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(S)
    neg = w < 0
    if not neg.any():
        return A, 0.0, 0
    return (V * np.maximum(w, 0.0)) @ V.T, float(-w[neg].sum()), int(neg.sum())


def spectrum(A: np.ndarray) -> dict:
    """cond, singular values, numerical rank, SV gap (largest s_i / s_{i+1}) and where it sits, lambda_min."""
    s = np.linalg.svd(A, compute_uv=False)
    p = s.size
    tol = s.max() * p * np.finfo(np.float64).eps if p else 0.0
    rank = int((s > tol).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = s[:-1] / s[1:] if p > 1 else np.array([np.nan])
    ratios = np.where(np.isfinite(ratios), ratios, np.inf)
    gi = int(np.argmax(ratios)) if p > 1 else 0
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return {"sv": s, "cond": cond, "rank": rank, "sv_gap": float(ratios[gi]) if p > 1 else float("nan"),
            "sv_gap_at": gi + 1, "eig_min": float(np.linalg.eigvalsh(0.5 * (A + A.T))[0])}


@dataclass
class McLachlanResult:
    params: np.ndarray            # theta_T
    params_hist: np.ndarray       # (T + 1, p)
    E_loop: np.ndarray            # (T + 1,): E(theta_t) as the loop measured it (loop units)
    step: dict                    # per-step diagnostics, (T + 1, ...) arrays, row 0 = NaN (row t = the step to theta_t)
    wall_hist: np.ndarray         # (T + 1,)
    n_steps: int                  # T
    converged: bool
    calls: dict                   # circuits per step by class: {"variance": p, "overlap": .., "energy": p + 1}
    first_extra: dict             # circuits measured once, in step 1: {"energy": 1}
    logger_s: float
    step_calls: list = field(default_factory=list)

    @property
    def circuits_per_step(self) -> int:
        return int(sum(self.calls.values()))

    @property
    def circuits_charged(self) -> np.ndarray:
        t = np.arange(self.n_steps + 1, dtype=np.int64)
        return t * self.circuits_per_step + (t >= 1) * int(sum(self.first_extra.values()))


def run_mclachlan(engine, x0, cfg: McLachlanConfig = McLachlanConfig(), logger=None) -> McLachlanResult:
    """The loop (module doc). `engine` provides n_params, L, gen_scale, energy(params), var_diag(params),
    fidelity(pa, pb, m); for metric "exact" also exact_metric(params) and exact_C(params) (test engines)."""
    x = np.array(x0, dtype=np.float64)
    p = x.size
    L = int(engine.L)
    gen_scale = np.asarray(engine.gen_scale, dtype=np.float64)
    t_start = time.perf_counter()
    logger_s = 0.0

    def log(t, params):
        nonlocal logger_s
        if logger is None:
            return
        tl = time.perf_counter()
        logger(t, np.array(params, dtype=np.float64, copy=True))
        logger_s += time.perf_counter() - tl

    nanp = np.full(p, np.nan)
    keys_vec = ("sv", "C", "thetadot", "diag", "delta")
    keys_sc = ("cond", "rank", "sv_gap", "sv_gap_at", "eig_min", "tikhonov", "cooling", "tdMtd", "thetadot_norm",
               "max_angle_step")
    step = {k: [nanp.copy()] for k in keys_vec}
    step.update({k: [np.nan] for k in keys_sc})
    params_hist = [x.copy()]
    wall = [0.0]
    log(0, x)
    E_prev = float(engine.energy(x))
    E_hist = [E_prev]
    calls = None
    step_calls = []
    last_f = None
    cou_con = 0
    converged = False
    for _it in range(cfg.n_steps):
        n_var = n_fid = 0
        if cfg.metric == "exact":
            A = np.asarray(engine.exact_metric(x), dtype=np.float64)
            C = np.asarray(engine.exact_C(x), dtype=np.float64)
            d = np.diag(A).copy()
            delta = nanp.copy()
        else:
            d = np.asarray(engine.var_diag(x), dtype=np.float64)
            n_var = p
            if cfg.metric == "M1":
                delta = kappa_delta(d, gen_scale, cfg.kappa, cfg.cap_angle)
                A, n_fid = metric_m1(engine.fidelity, x, delta, d, L)
            else:
                delta = nanp.copy()
                A = np.diag(d)
            C = build_C(engine.energy, x, E_prev, cfg.fd_shift)
        sp = spectrum(A)                                   # of the estimate (before any PSD projection)
        if cfg.psd:
            A, clip_sum, clip_n = psd_project(A)
            step.setdefault("psd_clip_sum", [np.nan]).append(clip_sum)
            step.setdefault("psd_clip_n", [np.nan]).append(float(clip_n))
        td = np.linalg.solve(A + cfg.tikhonov * np.eye(p), C)
        cooling = float(C @ td)
        x += cfg.dtau * td
        E_new = float(engine.energy(x))
        this = {"variance": n_var, "overlap": n_fid, "energy": p + 1 if cfg.metric != "exact" else 1}
        step_calls.append(this)
        calls = this if calls is None else calls
        if this != calls:
            raise AssertionError(f"circuits per step changed: {this} != {calls}")
        for k, v in (("sv", sp["sv"]), ("C", C), ("thetadot", td), ("diag", d), ("delta", delta)):
            step[k].append(np.asarray(v, dtype=np.float64).copy())
        for k, v in (("cond", sp["cond"]), ("rank", sp["rank"]), ("sv_gap", sp["sv_gap"]),
                     ("sv_gap_at", sp["sv_gap_at"]), ("eig_min", sp["eig_min"]), ("tikhonov", cfg.tikhonov),
                     ("cooling", cooling), ("tdMtd", float(td @ A @ td)), ("thetadot_norm", float(np.linalg.norm(td))),
                     ("max_angle_step", float(np.max(np.abs(cfg.dtau * td) * gen_scale)))):
            step[k].append(float(v))
        params_hist.append(x.copy())
        E_hist.append(E_new)
        wall.append(time.perf_counter() - t_start - logger_s)
        log(len(E_hist) - 1, x)
        cou_con = cou_con + 1 if last_f is not None and abs(E_new - last_f) < cfg.f_tol else 0
        if cou_con >= cfg.patience:
            converged = True
            break
        last_f = E_new
        E_prev = E_new
    T = len(E_hist) - 1
    out = {k: np.array(v, dtype=np.float64) for k, v in step.items()}
    return McLachlanResult(params=x.copy(), params_hist=np.array(params_hist), E_loop=np.array(E_hist), step=out,
                           wall_hist=np.array(wall), n_steps=T, converged=converged,
                           calls=calls or {"variance": 0, "overlap": 0, "energy": 0},
                           first_extra={"energy": 1}, logger_s=logger_s, step_calls=step_calls)
