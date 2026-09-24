"""Arms A0 (penalty QAOA, X mixer) and A1 (confined QAOA, compiled preserving mixer), PLAN §1.5-§1.6.

Both: random init (Eq. 4.11; `legacy_init` for the S4 equivalence test), AdamW with forward FD on the boosted
energy f = <alpha H> (observe only), the post-update logger, R = 5 restarts with seeds from the seed table.
Effort unit: one iteration, charged 2L + 1 circuits.

Config extras (hashed): init ("random" | "legacy"), grad ("fd_forward"); A1 also ring_order ("lex" default, D-9
open; "rank" = the completed work's order) and sector_source ("ga" = the production GA list, D-15; "bf" = the
brute-force reference list, used by the reproduction check). An ad hoc (not frozen) instance adds
inst_adhoc = True. S9a (O-2 evidence only, written only when set): circuit_boosted = True (the circuit carries
alpha x the coefficients; kappa_min, and so the Eq. 4.11 gamma range, follow the circuit) and evidence = "O-2"
(no rule reads an evidence run).

Also used by A2 (`ramp.py`): `sector_view`, `penalty_arm_ansatz`, `confined_arm_ansatz`, `trajectory_common`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..circuits import preserving as pr
from ..circuits.ansatz import Ansatz, confined_ansatz, penalty_ansatz
from ..metrics.state import METRIC_KEYS, StateLogger, metric_context
from ..train.adamw import AdamWConfig, train_adamw
from ..train.gradients import ForwardFD
from ..train.init import init_params, mm_i_eq411, mm_i_legacy
from .base import Arm, Outcome, RunConfig, draw_of, evidence_tag, make_extras, restart_seed

FD_DELTA = 1e-4
SECTOR_SOURCES = ("ga", "bf")


# --- sectors -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SectorView:
    """The kept strings of a confined cell as `preserving.build_circuit` reads them."""
    n: int
    idx: np.ndarray          # ascending classical indices (lexicographic)
    rank_idx: np.ndarray     # ranking, best first
    source: str
    seed_ga: int | None


def legacy_rank(inst, K: int) -> np.ndarray:
    """The completed work's brute-force sector: `np.argsort(state_penalty)[:K]` with state_penalty =
    -all_state_to_return(n, 1, QU_lamb) (PO_new_ApproxRatio.py:660, 768-795; `get_init_states`)."""
    from ..instances.encode import qubo_energies
    state_penalty = -qubo_energies(inst.QU_pen, 1.0)
    return np.argsort(state_penalty)[:K].astype(np.int64)


def sector_view(inst, rule: str, K: int, source: str = "ga", root=None) -> SectorView:
    if source not in SECTOR_SOURCES:
        raise ValueError(f"sector_source must be one of {SECTOR_SOURCES}")
    from ..sectors.select import load_sector
    from ..store.paths import sector_path
    from ..sectors.select import sector_scope
    have_file = sector_path(sector_scope(inst.inst_id, rule), rule, K, root).exists()
    if not have_file:
        if source != "bf" or rule != "violation":
            raise FileNotFoundError(f"no {rule} K={K} sector file for {inst.inst_id} (only the violation BF list "
                                    "can be rebuilt ad hoc)")
        r = legacy_rank(inst, K)
        return SectorView(n=int(inst.n), idx=np.sort(r), rank_idx=r, source="bf", seed_ga=None)
    sec = load_sector(inst.inst_id, rule, K, root)
    if source == "ga":
        return SectorView(n=sec.n, idx=np.asarray(sec.idx, np.int64), rank_idx=np.asarray(sec.rank_idx, np.int64),
                          source="ga", seed_ga=int(sec.seed))
    bf_rank = np.asarray(sec.arrays["bf_rank_idx"], np.int64)[:K]
    return SectorView(n=sec.n, idx=np.sort(bf_rank), rank_idx=bf_rank, source="bf", seed_ga=None)


# --- ansatz of a config --------------------------------------------------------------------------------
def penalty_arm_ansatz(inst, lam: float, L: int, circuit_boosted: bool = False) -> Ansatz:
    """`circuit_boosted` (S9a, O-2 evidence only): the circuit carries alpha x the coefficients (a different arm
    version, PLAN §1.5); default = the completed work's un-boosted circuit."""
    if lam is None:
        raise ValueError("a penalty arm needs lam (lambda*(N) from S9; S4 checks use 0.005)")
    return penalty_ansatz(inst.hamiltonian(float(lam)), L, circuit_boosted=bool(circuit_boosted))


def confined_arm_ansatz(inst, cfg: RunConfig, L: int, root=None) -> tuple:
    sv = sector_view(inst, cfg.rule, cfg.K, cfg.extra("sector_source", "ga"), root)
    circ = pr.build_circuit(sv, cfg.connectivity, cfg.extra("ring_order", "lex"))
    return confined_ansatz(inst.H_obj, inst.boost_obj, circ, L,
                           circuit_boosted=bool(cfg.extra("circuit_boosted", False))), sv


def unit_counts(A: Ansatz, circuits_per_unit: int, unit: str) -> dict:
    pc = A.counts["per_circuit"]
    out = dict(A.counts)
    out.update({"effort_unit": unit, "circuits_per_unit": int(circuits_per_unit),
                "per_unit": {k: int(v) * int(circuits_per_unit) for k, v in pc.items()},
                "sim_gates_per_circuit": int(A.prog.n_gates), "abstract_gates_per_circuit": len(A.abstract_gates()),
                "n": A.n, "L": A.L})
    return out


def trajectory_common(A: Ansatz, t, params, rows: dict, circuits_charged, wall) -> dict:
    """The PLAN §1.6 fields shared by every arm (arm diagnostics are added by the caller)."""
    cc = np.asarray(circuits_charged, dtype=np.int64)
    out = {"t": np.asarray(t, dtype=np.int64), "params": np.asarray(params, dtype=np.float64),
           "circuits_charged": cc,
           "g2q_ii": cc * int(A.counts["per_circuit"]["cx_ii"]),
           "g2q_iii": cc * int(A.counts["per_circuit"]["cx_iii"]),
           "wall": np.asarray(wall, dtype=np.float64)}
    for k, v in rows.items():
        out[k] = np.asarray(v, dtype=np.float64)
    return out


# --- the trained arms ----------------------------------------------------------------------------------
class TrainedArm(Arm):
    effort_kind = "depth"

    def ansatz(self, cfg: RunConfig, inst, root=None) -> tuple:  # pragma: no cover (abstract)
        raise NotImplementedError

    def execute(self, cfg: RunConfig, inst, rulers, cell, logger: bool = True, root=None) -> Outcome:
        t0 = time.perf_counter()
        A, sector_idx = self.ansatz(cfg, inst, root)
        legacy = cfg.extra("init", "random") == "legacy"
        x0 = init_params(A, cfg.seed, legacy=legacy)
        if legacy:
            from ..circuits.legacy_pauli import legacy_mm_p
            mm_p = legacy_mm_p(A.circ.order, A.n) if A.kind == "confined" else 1e9
            gamma_range = float(mm_i_legacy(A.ct.coeff_1, A.ct.coeff_2, mm_p))
        else:
            gamma_range = float(mm_i_eq411(A.kappa_min()))
        ctx = metric_context(inst, rulers, cfg.lam if A.kind == "penalty" else None, sector_idx=sector_idx)
        log = StateLogger(A, ctx) if logger else None
        setup_s = time.perf_counter() - t0
        res = train_adamw(A.energy, x0, AdamWConfig(), ForwardFD(FD_DELTA), logger=log)
        t1 = time.perf_counter()
        T = res.n_iter
        if log is not None:
            rows = log.arrays()
            rows.pop("t_logged")
            final_psi = log.last_state
        else:                                    # post-run metrics only (get_state after the run)
            final_psi = A.state(res.params)
            last = ctx.evaluate_state(final_psi)
            rows = {k: np.r_[np.full(T, np.nan), v] for k, v in last.items()}
        final = {k: float(v[-1]) for k, v in rows.items()}
        post_s = time.perf_counter() - t1
        traj = trajectory_common(A, np.arange(T + 1), res.params_hist, rows, res.circuits_charged, res.wall_hist)
        nanpad = np.full(1, np.nan)
        traj["f_loop"] = np.r_[res.f_hist, nanpad]            # f(theta_t) as the loop measured it (boosted)
        traj["lr"] = np.r_[res.lr_hist, nanpad]
        traj["grad_norm"] = np.r_[res.grad_norm, nanpad]
        loop_vs_log = float(np.max(np.abs(res.f_hist / A.alpha - rows["energy"][:T]))) if (log is not None and T) else None
        metrics = dict(final)
        ar0 = float(rows["ar_f"][0])
        metrics.update({"ar_f_init": ar0 if np.isfinite(ar0) else None, "iterations": T, "converged": bool(res.converged),
                        "circuits_charged": int(res.circuits_charged[-1]), "g2q_ii": int(traj["g2q_ii"][-1]),
                        "g2q_iii": int(traj["g2q_iii"][-1]), "f_final_loop": float(res.f_hist[-1]) if T else None})
        timings = {"setup_s": setup_s, "train_s": float(res.wall_hist[-1]), "logger_s": res.logger_s,
                   "post_s": post_s, "per_iter_s": float(res.wall_hist[-1] / T) if T else None}
        diag = {"n": A.n, "L": A.L, "n_params": A.n_params, "alpha": A.alpha, "kappa_min": A.kappa_min(),
                "gamma_range": gamma_range, "sim_gates": int(A.prog.n_gates),
                "cx_ii_circuit": int(A.counts["per_circuit"]["cx_ii"]),
                "cx_iii_circuit": int(A.counts["per_circuit"]["cx_iii"]), "loop_vs_logger": loop_vs_log}
        if A.kind == "confined":
            diag.update({"K_actual": int(A.circ.K), "S_max": int(A.counts["mixer"]["S_max"])})
        counts = unit_counts(A, res.circuits_per_iter, "iteration")
        return Outcome(metrics=metrics, timings=timings, diagnostics=diag, trajectory=traj, counts=counts,
                       final_state=final_psi)


class A0(TrainedArm):
    name = "A0"
    encoding = "penalty"

    def config(self, inst, cell, effort, seed, *, restart: int = 0, lam=None, init: str = "random",
               circuit_boosted: bool = False, evidence: str | None = None, adhoc: bool = False,
               root=None) -> RunConfig:
        """`circuit_boosted` / `evidence` (S9a, O-2 evidence): hashed only when set, so every default run_id is
        unchanged; the planner never passes them."""
        if init not in ("random", "legacy"):
            raise ValueError(init)
        seed = restart_seed(draw_of(inst.inst_id), restart, root) if seed is None else int(seed)
        return RunConfig(arm=self.name, encoding=self.encoding, inst_id=inst.inst_id, effort_kind=self.effort_kind,
                         effort=int(effort), restart=int(restart), lam=float(lam), seed=seed,
                         extras=make_extras(init=init, grad=ForwardFD.name, inst_adhoc=True if adhoc else None,
                                            circuit_boosted=True if circuit_boosted else None,
                                            evidence=evidence_tag(evidence)))

    def ansatz(self, cfg, inst, root=None):
        return penalty_arm_ansatz(inst, cfg.lam, cfg.effort, bool(cfg.extra("circuit_boosted", False))), None


class A1(TrainedArm):
    name = "A1"
    encoding = "confined"

    def config(self, inst, cell, effort, seed, *, restart: int = 0, init: str = "random", ring_order: str = "lex",
               sector_source: str = "ga", circuit_boosted: bool = False, evidence: str | None = None,
               adhoc: bool = False, root=None) -> RunConfig:
        """`circuit_boosted` / `evidence`: as A0 (S9a, O-2 evidence; hashed only when set)."""
        if init not in ("random", "legacy"):
            raise ValueError(init)
        if ring_order not in pr.RING_ORDERS:
            raise ValueError(ring_order)
        seed = restart_seed(draw_of(inst.inst_id), restart, root) if seed is None else int(seed)
        sv = sector_view(inst, cell["rule"], int(cell["K"]), sector_source, root)
        return RunConfig(arm=self.name, encoding=self.encoding, inst_id=inst.inst_id, effort_kind=self.effort_kind,
                         effort=int(effort), K=int(cell["K"]), rule=cell["rule"], connectivity=cell["connectivity"],
                         restart=int(restart), seed=seed, seed_ga=sv.seed_ga,
                         extras=make_extras(init=init, grad=ForwardFD.name, ring_order=ring_order,
                                            sector_source=sector_source, inst_adhoc=True if adhoc else None,
                                            circuit_boosted=True if circuit_boosted else None,
                                            evidence=evidence_tag(evidence)))

    def ansatz(self, cfg, inst, root=None):
        A, sv = confined_arm_ansatz(inst, cfg, cfg.effort, root)
        return A, sv.idx


ARMS = {"A0": A0, "A1": A1}
__all__ = ["A0", "A1", "SectorView", "sector_view", "legacy_rank", "METRIC_KEYS"]
