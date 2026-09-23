"""Convergence effort (PLAN §1.7, proposal §6.3.2): the first effort unit t with AR_F(t) >= tau AR_F(final),
tau = 0.95, converted to cumulative charged circuits and two-qubit gate executions ((ii) and (iii)) through the
trajectory's own counters, and always reported with AR_F(final).

It is computed per run (= per restart); aggregation over restarts is the reader's choice. For the untrained A2
(one row) it is trivially t = 0 at the cost of its one circuit (A2's convergence measure is the ramp depth at
which quality saturates, read from the p sweep, RQ2). A trajectory without logged AR_F (logger off) has none.
"""

from __future__ import annotations

import numpy as np

TAU = 0.95


def convergence(traj: dict, tau: float = TAU) -> dict:
    ar = np.asarray(traj["ar_f"], dtype=np.float64)
    out = {"conv_tau": tau, "ar_f_final": float(ar[-1]) if ar.size else float("nan")}
    keys = ("conv_t", "conv_circuits", "conv_g2q_ii", "conv_g2q_iii", "conv_frac_units")
    if ar.size == 0 or not np.all(np.isfinite(ar)):
        out.update({k: None for k in keys})
        return out
    thr = tau * ar[-1]
    i = int(np.argmax(ar >= thr))            # exists: the last row satisfies it (AR_F >= 0)
    t = np.asarray(traj["t"])
    T = int(t[-1]) if t.size else 0
    out.update({"conv_t": int(t[i]), "conv_circuits": int(traj["circuits_charged"][i]),
                "conv_g2q_ii": int(traj["g2q_ii"][i]), "conv_g2q_iii": int(traj["g2q_iii"][i]),
                "conv_frac_units": (int(t[i]) / T) if T > 0 else None})
    return out
