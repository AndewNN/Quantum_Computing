"""Arms A3 (penalty VarQITE, the M1 metric in the loop, D-3) and A3d (the M4C1 diagonal-metric ablation, optional
tier), PLAN §1.5-§1.7, S7.

Both run on A0's circuit (`qaoa.penalty_arm_ansatz`: H(lam) cost layer + X mixer from H^n, depth L, p = 2L
parameters, the un-boosted coefficients in the circuit and the Jh-boosted energy <alpha H> as the loop's E: the
completed work's convention, O-2, which `varqite_routes.py` also used), from the Ramp init of `varqite_routes.py`
(`train.mclachlan.ramp_init`), with the McLachlan loop of `train.mclachlan` (dtau 0.1, 300 steps, C1 forward FD,
Tikhonov 1e-6; stop rule verbatim).

The observe-only primitives (`CircuitEngine`, backend calls only; kernels in `gsp/circuits/`):
  energy(theta)       <alpha H(lam)> on the layered kernel (the A0 energy circuit);
  var_diag(theta)     Var(G_k) = <G_k^2> - <G_k>^2 by two observes on parameter k's truncated circuit
                      (`circuits.overlap.truncated_args`); G_gamma = the cost operator in circuit units (the old
                      `H_ansatz / boost`, constant included), G_beta = sum_j X_j;
  fidelity(a, b, m)   P(0...0) of the overlap circuit on the first m layers through the flag qubit:
                      (1 - <Z_flag>) / 2 (`circuits.overlap_kernel`). Never the 2^n-term projector, never get_state.
The post-update logger (`VarLogger`) reads the state (get_state) for the §1.7 metrics and V_tau; V_tau enters only
the diagnostics R_min = V_tau(theta_{t-1}) - C . thetadot and the residual of the applied step
V_tau - 2 C . thetadot + thetadot^T M thetadot (both in loop units, alpha^2 x un-boosted; the old history's R_min).

Charging (PLAN §1.5 / §1.7; `a3_counts`): per step M1 = p(p+1)/2 overlap circuits + p variance circuits, plus
(p + 1) energy circuits (A3d: p variance + (p + 1) energy); the first step also measures E(theta_0) once
(`first_unit_extra`). Gate counts per circuit class from the abstract A0 layer (`compile.transpile`): energy = L
layers; variance of gamma_k = (k + 1) cost layers + k mixers, of beta_k = k + 1 layers (the old truncated circuit);
overlap on m layers = 2m layers (flag gate not charged: `circuits.overlap` doc). The trajectory's g2q is the
cumulative sum of these, so counts.json carries `charge_model: per_unit` (`metrics.resources`).

Config extras (hashed): init "ramp", metric ("M1" A3 | "diag" A3d), grad "fd_forward", dtau, n_steps, tikhonov.
A3 is deterministic (R = 1): `restart` must be 0; `seed` = the draw's restart seed r = 0 (used only by the post-run
sample). Effort = depth L.
"""

from __future__ import annotations

import time

import numpy as np

from ..circuits import overlap as ov
from ..circuits.ansatz import Ansatz
from ..compile import tcount
from ..compile.transpile import cost_counts
from ..metrics.state import METRIC_KEYS, StateLogger, metric_context
from ..train.mclachlan import McLachlanConfig, McLachlanResult, ramp_init, run_mclachlan
from .base import Arm, Outcome, RunConfig, draw_of, make_extras, restart_seed
from .qaoa import penalty_arm_ansatz

A3_DEFAULTS = {"dtau": 0.1, "n_steps": 300, "tikhonov": 1e-6}
JUMP_ANGLE = 1.0            # rad: steps whose largest gate-angle change exceeds this are counted as jumps (S7).
#                             Blow-ups in Sensei's sense are the steps with R_min < 0 (metric R_min_neg); the 15
#                             S7 smoke / timing records carry this count under its first name, metric_blowup_steps.
COUNT_KEYS = ("cx_ii", "cx_iii", "t_ii", "t_iii", "tdepth_ii", "tdepth_iii")


# --- the observe-only primitives --------------------------------------------------------------------------------
class CircuitEngine:
    """energy / var_diag / fidelity of A0's circuit through `backend.observe` (module doc)."""

    def __init__(self, A: Ansatz):
        if A.kind != "penalty":
            raise ValueError("A3 runs on A0's circuit (a penalty ansatz)")
        from ..sim import backend
        self.A = A
        self.L = int(A.L)
        self.n_params = 2 * self.L
        self.n = int(A.n)
        self.oprog = ov.encode_overlap(A.start, A.layer, A.n, A.L)
        c = [abs(v) for v in A.ct.coeff_1 + A.ct.coeff_2]
        self.coef_max = float(max(c)) if c else 1.0
        self.gen_scale = np.array([self.coef_max] * self.L + [1.0] * self.L)
        circ_alpha = A.alpha if A.meta.get("circuit_boosted") else 1.0
        self.G_gamma = backend.ising_op(A.H, circ_alpha)
        self.G_gamma2 = backend.op_product(self.G_gamma, self.G_gamma)
        self.G_beta = backend.x_sum_op(self.n)
        self.G_beta2 = backend.op_product(self.G_beta, self.G_beta)
        self.z_flag = backend.z_op(self.n)
        self._backend = backend

    def energy(self, params) -> float:
        return self.A.energy(params)

    def var_diag(self, params) -> np.ndarray:
        from ..circuits.program import layered_kernel
        be, kern = self._backend, layered_kernel()
        out = np.zeros(self.n_params)
        for k in range(self.n_params):
            args = ov.truncated_args(self.A.prog, params, k)
            G, G2 = (self.G_gamma, self.G_gamma2) if k < self.L else (self.G_beta, self.G_beta2)
            m1 = be.observe(kern, G, *args)
            m2 = be.observe(kern, G2, *args)
            out[k] = m2 - m1 ** 2
        return np.maximum(out, 0.0)

    def fidelity(self, pa, pb, m: int) -> float:
        ev = self._backend.observe(ov.overlap_kernel(), self.z_flag, *self.oprog.args(pa, pb, m, 1))
        return (1.0 - ev) / 2.0

    # statevector references (tests and checks only; never called by the loop)
    def fidelity_statevector(self, pa, pb, m: int) -> float:
        psi = self._backend.get_state(ov.overlap_kernel(), *self.oprog.args(pa, pb, m, 0))
        return float(abs(psi[0]) ** 2)


# --- the logger ------------------------------------------------------------------------------------------------------
class VarLogger(StateLogger):
    """StateLogger + V_tau (un-boosted variance of H(lam)) from the same state. With ctx None it logs the energy and
    V_tau of the ansatz's H only (the 3-qubit example check)."""

    def __init__(self, ansatz: Ansatz, ctx=None):
        super().__init__(ansatz, ctx)
        self.diag = ctx.diag if ctx is not None else ansatz.H.diagonal(np.arange(1 << ansatz.n))
        self.v_tau: list = []

    def __call__(self, t: int, params: np.ndarray) -> None:
        psi = self.ansatz.state(params)
        self.last_state = psi
        self.t.append(int(t))
        prob = np.abs(psi) ** 2
        e = float(prob @ self.diag)
        self.v_tau.append(float(prob @ self.diag ** 2 - e ** 2))
        self.rows.append(self.ctx.evaluate(prob) if self.ctx is not None else {"energy": e})

    def arrays(self) -> dict:
        out = super().arrays()
        out["V_tau"] = np.array(self.v_tau, dtype=np.float64)
        return out


# --- counts ------------------------------------------------------------------------------------------------------------
def _circ(C: dict, X: dict, n_cost: int, n_mix: int) -> dict:
    t = n_cost * C["t_ii"] + n_mix * X["t"]
    td = n_cost * C["tdepth_ii"] + n_mix * X["tdepth"]
    cx = n_cost * C["cx_ii"]
    return {"cx_ii": cx, "cx_iii": cx, "t_ii": t, "t_iii": t, "tdepth_ii": td, "tdepth_iii": td}


def _add(a: dict, b: dict, k: int = 1) -> dict:
    return {key: int(a.get(key, 0)) + k * int(b[key]) for key in COUNT_KEYS}


def a3_counts(A: Ansatz, metric: str) -> dict:
    """counts.json of an A3 / A3d run (module doc): per circuit class, per step, the one-off first-step energy."""
    ts = tcount.t_syn()
    C = cost_counts(A.ct, ts)
    X = {"t": tcount.x_mixer_t(A.n, ts), "tdepth": tcount.x_mixer_tdepth(ts)}
    L, p = A.L, 2 * A.L
    zero = {k: 0 for k in COUNT_KEYS}
    energy = _circ(C, X, L, L)
    var = dict(zero)
    for k in range(L):
        var = _add(var, _circ(C, X, k + 1, k))           # gamma_k
        var = _add(var, _circ(C, X, k + 1, k + 1))       # beta_k
    ovl = dict(zero)
    n_ovl = 0
    if metric == "M1":
        lay = [ov.layer_of(k, L) for k in range(p)]
        for i in range(p):
            ovl = _add(ovl, _circ(C, X, 2 * (lay[i] + 1), 2 * (lay[i] + 1)))
            n_ovl += 1
            for j in range(i + 1, p):
                m = max(lay[i], lay[j]) + 1
                ovl = _add(ovl, _circ(C, X, 2 * m, 2 * m))
                n_ovl += 1
    classes = {"energy": {"circuits": p + 1, **{k: (p + 1) * v for k, v in energy.items()}},
               "variance": {"circuits": p, **var},
               "overlap": {"circuits": n_ovl, **ovl}}
    per_unit = dict(zero)
    for c in classes.values():
        per_unit = _add(per_unit, c)
    layer = _circ(C, X, 1, 1)
    return {"effort_unit": "step", "charge_model": "per_unit", "metric": metric,
            "circuits_per_unit": int(sum(c["circuits"] for c in classes.values())),
            "per_unit": per_unit, "per_circuit": energy, "layer": layer, "start": dict(zero),
            "first_unit_extra": {"circuits": 1, **energy}, "classes": classes,
            "n": A.n, "L": A.L, "n_params": p,
            "flag_gate_charged": False,
            "note": "per_circuit = the energy circuit (A0's circuit); overlap / variance circuits in classes"}


def charged_series(counts: dict, T: int) -> dict:
    """Cumulative circuits and (ii) / (iii) two-qubit executions up to theta_t, t = 0..T."""
    t = np.arange(T + 1, dtype=np.int64)
    first = (t >= 1).astype(np.int64)
    pu, fx = counts["per_unit"], counts["first_unit_extra"]
    return {"circuits_charged": t * int(counts["circuits_per_unit"]) + first * int(fx["circuits"]),
            "g2q_ii": t * int(pu["cx_ii"]) + first * int(fx["cx_ii"]),
            "g2q_iii": t * int(pu["cx_iii"]) + first * int(fx["cx_iii"])}


# --- one run -------------------------------------------------------------------------------------------------------
def run_a3(A: Ansatz, mc: McLachlanConfig, ctx=None, x0=None, logger: bool = True):
    """(McLachlanResult, logger arrays or None, final state or None, engine) of one A3 / A3d trajectory."""
    eng = CircuitEngine(A)
    x0 = ramp_init(A.L) if x0 is None else np.asarray(x0, dtype=np.float64)
    log = VarLogger(A, ctx) if logger else None
    res = run_mclachlan(eng, x0, mc, logger=log)
    rows = log.arrays() if log is not None else None
    return res, rows, (log.last_state if log is not None else None), eng


def step_diagnostics(res: McLachlanResult, v_tau_loop: np.ndarray | None) -> dict:
    """The per-step A3 diagnostics of PLAN §1.6 as (T + 1)-row arrays (row 0 = NaN)."""
    s = res.step
    out = {"E_loop": res.E_loop, "cond_M": s["cond"], "rank_M": s["rank"], "sv_gap": s["sv_gap"],
           "sv_gap_at": s["sv_gap_at"], "eig_min_M": s["eig_min"], "tikhonov": s["tikhonov"],
           "cooling": s["cooling"], "dE_dtau": -2.0 * s["cooling"], "tdMtd": s["tdMtd"],
           "thetadot_norm": s["thetadot_norm"], "max_angle_step": s["max_angle_step"], "sv": s["sv"], "C": s["C"],
           "thetadot": s["thetadot"],
           "M_diag": s["diag"], "fid_delta": s["delta"]}
    T = res.n_steps
    if v_tau_loop is not None:
        prev = np.r_[np.nan, v_tau_loop[:T]]
        out["R_min"] = prev - s["cooling"]
        out["residual"] = prev - 2.0 * s["cooling"] + s["tdMtd"]
    else:
        out["R_min"] = np.full(T + 1, np.nan)
        out["residual"] = np.full(T + 1, np.nan)
    return out


class A3(Arm):
    name = "A3"
    encoding = "penalty"
    effort_kind = "depth"
    metric = "M1"

    def config(self, inst, cell, effort, seed, *, lam=None, restart: int = 0, n_steps: int | None = None,
               adhoc: bool = False, root=None) -> RunConfig:
        """`n_steps` (default 300) is only for smoke / timing runs (S7, S9); the planner never passes it."""
        if int(restart) != 0:
            raise ValueError(f"{self.name} is deterministic (R = 1): restart must be 0")
        if lam is None:
            raise ValueError(f"{self.name} needs lam (lambda*(N) from S9)")
        seed = restart_seed(draw_of(inst.inst_id), 0, root) if seed is None else int(seed)
        settings = dict(A3_DEFAULTS)
        if n_steps is not None:
            settings["n_steps"] = int(n_steps)
        return RunConfig(arm=self.name, encoding=self.encoding, inst_id=inst.inst_id, effort_kind=self.effort_kind,
                         effort=int(effort), restart=0, lam=float(lam), seed=seed,
                         extras=make_extras(init="ramp", metric=self.metric, grad="fd_forward",
                                            inst_adhoc=True if adhoc else None, **settings))

    def ansatz(self, cfg, inst, root=None):
        return penalty_arm_ansatz(inst, cfg.lam, cfg.effort), None

    def mc_config(self, cfg: RunConfig) -> McLachlanConfig:
        if cfg.extra("init") != "ramp" or cfg.extra("grad") != "fd_forward":
            raise ValueError(f"unsupported A3 config: init {cfg.extra('init')}, grad {cfg.extra('grad')}")
        return McLachlanConfig(metric=cfg.extra("metric"), dtau=float(cfg.extra("dtau")),
                               n_steps=int(cfg.extra("n_steps")), tikhonov=float(cfg.extra("tikhonov")))

    def execute(self, cfg: RunConfig, inst, rulers, cell, logger: bool = True, root=None) -> Outcome:
        t0 = time.perf_counter()
        A, _ = self.ansatz(cfg, inst, root)
        mc = self.mc_config(cfg)
        ctx = metric_context(inst, rulers, cfg.lam)
        counts = a3_counts(A, mc.metric)
        setup_s = time.perf_counter() - t0
        res, rows, final_psi, eng = run_a3(A, mc, ctx=ctx, logger=logger)
        t1 = time.perf_counter()
        T = res.n_steps
        if rows is not None:
            rows.pop("t_logged")
            v_tau = rows.pop("V_tau")
        else:                                     # post-run metrics only
            final_psi = A.state(res.params)
            last = ctx.evaluate_state(final_psi)
            rows = {k: np.r_[np.full(T, np.nan), v] for k, v in last.items()}
            v_tau = None
        final = {k: float(v[-1]) for k, v in rows.items()}
        post_s = time.perf_counter() - t1
        ch = charged_series(counts, T)
        if not np.array_equal(ch["circuits_charged"], res.circuits_charged):
            raise AssertionError("circuit count of the loop != counts.json")
        traj = {"t": np.arange(T + 1, dtype=np.int64), "params": res.params_hist, **ch, "wall": res.wall_hist}
        for k, v in rows.items():
            traj[k] = np.asarray(v, dtype=np.float64)
        traj["V_tau"] = np.full(T + 1, np.nan) if v_tau is None else v_tau
        diag_steps = step_diagnostics(res, None if v_tau is None else v_tau * A.alpha ** 2)
        traj.update(diag_steps)
        loop_vs_log = (float(np.max(np.abs(res.E_loop / A.alpha - rows["energy"]))) if v_tau is not None else None)
        rmin = diag_steps["R_min"][1:]
        metrics = dict(final)
        ar0 = float(rows["ar_f"][0])
        metrics.update({"ar_f_init": ar0 if np.isfinite(ar0) else None, "iterations": T,
                        "converged": bool(res.converged), "circuits_charged": int(ch["circuits_charged"][-1]),
                        "g2q_ii": int(ch["g2q_ii"][-1]), "g2q_iii": int(ch["g2q_iii"][-1]),
                        "E_loop_final": float(res.E_loop[-1]),
                        "R_min_neg": int(np.sum(rmin < 0)) if v_tau is not None else None,
                        "R_min_min": float(np.nanmin(rmin)) if (v_tau is not None and T) else None,
                        "thetadot_norm_max": float(np.max(diag_steps["thetadot_norm"][1:])) if T else None,
                        "cond_M_max": float(np.max(diag_steps["cond_M"][1:])) if T else None,
                        "eig_min_M_min": float(np.min(diag_steps["eig_min_M"][1:])) if T else None,
                        "jump_steps": int(np.sum(diag_steps["max_angle_step"][1:] > JUMP_ANGLE)),
                        "max_angle_step_max": float(np.max(diag_steps["max_angle_step"][1:])) if T else None})
        per_step = np.diff(res.wall_hist)
        timings = {"setup_s": setup_s, "train_s": float(res.wall_hist[-1]), "logger_s": res.logger_s,
                   "post_s": post_s, "per_step_s": float(res.wall_hist[-1] / T) if T else None,
                   "per_step_median_s": float(np.median(per_step)) if T else None}
        diag = {"n": A.n, "L": A.L, "n_params": A.n_params, "alpha": A.alpha, "kappa_min": A.kappa_min(),
                "metric": mc.metric, "coef_max": eng.coef_max, "circuits_per_step": int(counts["circuits_per_unit"]),
                "overlap_per_step": int(counts["classes"]["overlap"]["circuits"]),
                "variance_per_step": int(counts["classes"]["variance"]["circuits"]),
                "energy_per_step": int(counts["classes"]["energy"]["circuits"]),
                "cx_ii_step": int(counts["per_unit"]["cx_ii"]), "cx_ii_energy_circuit": int(counts["per_circuit"]["cx_ii"]),
                "sim_gates": int(A.prog.n_gates), "loop_vs_logger": loop_vs_log}
        return Outcome(metrics=metrics, timings=timings, diagnostics=diag, trajectory=traj, counts=counts,
                       final_state=final_psi)


class A3d(A3):
    """The M4C1 ablation: McLachlan with the diagonal metric Var(G_k) (optional tier, N <= 7)."""
    name = "A3d"
    metric = "diag"


ARMS = {"A3": A3, "A3d": A3d}
__all__ = ["A3", "A3d", "CircuitEngine", "VarLogger", "a3_counts", "charged_series", "run_a3", "METRIC_KEYS"]
