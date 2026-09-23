"""Arm A2: the linear ramp, no training (PLAN §1.5, D-10). A2p runs A0's circuit on H(lam*) from H^n; A2c runs
A1's circuit (same sector file, star start) on H_obj. One circuit per ramp depth p, charged 1.

Schedules (D-10): primary (dbeta, dgamma) = (0.2, 3.0), secondary (1.5, 3.0), both in Jh-boosted units with beta
negated (`train.schedules`, RAMP_SIGN = -1). `ramp_sign = +1` exists only for the S4 anneal check (it must
anneal to the maximum). Depths {5, 7, 9, 15, 21, 30, 50, 100, 200, 300}.

Config: effort_kind "ramp_depth", effort p, schedule tag, restart 0, seed None; extras ramp_dbeta, ramp_dgamma,
ramp_sign (and ring_order / sector_source for A2c).
"""

from __future__ import annotations

import time

import numpy as np

from ..circuits import preserving as pr
from ..metrics.state import metric_context
from ..train.schedules import RAMP_SIGN, ramp_params, schedule
from .base import Arm, Outcome, RunConfig, make_extras
from .qaoa import confined_arm_ansatz, penalty_arm_ansatz, sector_view, trajectory_common, unit_counts

RAMP_DEPTHS = (5, 7, 9, 15, 21, 30, 50, 100, 200, 300)


class A2(Arm):
    effort_kind = "ramp_depth"

    def __init__(self, encoding: str):
        if encoding not in ("penalty", "confined"):
            raise ValueError(encoding)
        self.encoding = encoding
        self.name = "A2p" if encoding == "penalty" else "A2c"

    def config(self, inst, cell, effort, seed=None, *, schedule_tag: str = "primary", sign: int = RAMP_SIGN,
               lam=None, ring_order: str = "lex", sector_source: str = "ga", adhoc: bool = False,
               root=None) -> RunConfig:
        db, dg = schedule(schedule_tag)
        extras = dict(ramp_dbeta=float(db), ramp_dgamma=float(dg), ramp_sign=int(sign),
                      inst_adhoc=True if adhoc else None)
        common = dict(arm=self.name, encoding=self.encoding, inst_id=inst.inst_id, effort_kind=self.effort_kind,
                      effort=int(effort), restart=0, schedule=schedule_tag, seed=None)
        if self.encoding == "penalty":
            return RunConfig(**common, lam=float(lam), extras=make_extras(**extras))
        if ring_order not in pr.RING_ORDERS:
            raise ValueError(ring_order)
        sv = sector_view(inst, cell["rule"], int(cell["K"]), sector_source, root)
        return RunConfig(**common, K=int(cell["K"]), rule=cell["rule"], connectivity=cell["connectivity"],
                         seed_ga=sv.seed_ga,
                         extras=make_extras(**extras, ring_order=ring_order, sector_source=sector_source))

    def ansatz(self, cfg: RunConfig, inst, root=None) -> tuple:
        p = int(cfg.effort)
        if self.encoding == "penalty":
            return penalty_arm_ansatz(inst, cfg.lam, p), None
        A, sv = confined_arm_ansatz(inst, cfg, p, root)
        return A, sv.idx

    def execute(self, cfg: RunConfig, inst, rulers, cell, logger: bool = True, root=None) -> Outcome:
        t0 = time.perf_counter()
        p = int(cfg.effort)
        A, sector_idx = self.ansatz(cfg, inst, root)
        params = ramp_params(p, cfg.extra("ramp_dbeta"), cfg.extra("ramp_dgamma"), A.alpha, cfg.extra("ramp_sign"))
        ctx = metric_context(inst, rulers, cfg.lam if A.kind == "penalty" else None, sector_idx=sector_idx)
        setup_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        f = A.energy(params)                    # the one charged circuit (observe)
        circ_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        psi = A.state(params)                   # post-run metrics
        m = ctx.evaluate_state(psi)
        post_s = time.perf_counter() - t2
        rows = {k: [v] for k, v in m.items()}
        traj = trajectory_common(A, [0], params[None, :], rows, [1], [circ_s])
        traj["f_loop"] = np.array([f])
        metrics = dict(m)
        metrics.update({"f_observe": float(f), "circuits_charged": 1,
                        "g2q_ii": int(traj["g2q_ii"][-1]), "g2q_iii": int(traj["g2q_iii"][-1])})
        diag = {"n": A.n, "p": p, "alpha": A.alpha, "sim_gates": int(A.prog.n_gates),
                "cx_ii_circuit": int(A.counts["per_circuit"]["cx_ii"]),
                "cx_iii_circuit": int(A.counts["per_circuit"]["cx_iii"]),
                "observe_vs_state": abs(f / A.alpha - m["energy"])}
        timings = {"setup_s": setup_s, "circuit_s": circ_s, "post_s": post_s}
        return Outcome(metrics=metrics, timings=timings, diagnostics=diag, trajectory=traj,
                       counts=unit_counts(A, 1, "circuit"), final_state=psi)


A2p = lambda: A2("penalty")   # noqa: E731
A2c = lambda: A2("confined")  # noqa: E731
