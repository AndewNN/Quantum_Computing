"""Per-state metrics of PLAN §1.7 from a statevector (S4: what the post-update logger needs; S5 builds the rest
of the library -- AR_best_S, convergence, resources, simulation difficulty -- on top of this).

For probabilities prob[x] = |psi_x|^2 in classical order (x_0 = MSB, `gsp.instances.bits`):
  energy     sum_x prob[x] H(x), the un-boosted energy of the Hamiltonian the arm runs
             (H(lam) = -(x^T QU(lam) x - lam) for penalty arms, H_obj for confined arms);
  p_feas     sum over the band F_eps (Delta(x) = |P.x - 1| <= eps);
  eps_tilde  sqrt(sum_x prob[x] Delta(x)^2), Delta^2 = (P.x - 1)^2 over all 2^n strings;
  ar_f       sum_{x in F_eps} prob[x] (E_max - f(x)) / (E_max - E_min), f = H_obj, E_min / E_max over the band
             (`example_metrics.py`; the completed work's AR2);
  p_opt      mass on X* (every band string at E_min);
  p_top10    mass on the top-10 band strings (ties at the 10th kept; the set size is the rulers' top10_size);
  p_sector   (confined arms) mass on the kept strings: 1 - leakage.
The diagonals come from the same QUBO formula as the rulers (`encode.qubo_energies`, the old
`all_state_to_return`), chunked, so they agree with `observe` to 1e-12 (S1 test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..instances.encode import qubo_energies

METRIC_KEYS = ("energy", "ar_f", "p_feas", "eps_tilde", "p_opt", "p_top10")


@dataclass
class MetricContext:
    n: int
    band_idx: np.ndarray
    ar_weight: np.ndarray           # (E_max - f)/(E_max - E_min) on the band
    delta2: np.ndarray              # (P.x - 1)^2 on all 2^n strings
    diag: np.ndarray                # H(x) of the arm's Hamiltonian, un-boosted, all 2^n strings
    xstar_idx: np.ndarray
    top10_idx: np.ndarray
    sector_idx: np.ndarray | None = None

    def evaluate(self, prob: np.ndarray) -> dict:
        prob = np.asarray(prob, dtype=np.float64)
        pb = prob[self.band_idx]
        out = {
            "energy": float(prob @ self.diag),
            "ar_f": float(pb @ self.ar_weight),
            "p_feas": float(pb.sum()),
            "eps_tilde": float(np.sqrt(max(0.0, float(prob @ self.delta2)))),
            "p_opt": float(prob[self.xstar_idx].sum()),
            "p_top10": float(prob[self.top10_idx].sum()),
        }
        if self.sector_idx is not None:
            out["p_sector"] = float(prob[self.sector_idx].sum())
        return out

    def evaluate_state(self, psi: np.ndarray) -> dict:
        return self.evaluate(np.abs(np.asarray(psi)) ** 2)


def metric_context(inst, rul, lam: float | None, sector_idx=None) -> MetricContext:
    """For an `Instance` and its `Rulers`; lam = None or 0 means the arm runs H_obj (confined arms)."""
    lam = 0.0 if lam is None else float(lam)
    QU = inst.QU_obj if lam == 0.0 else inst.qubo(lam)
    diag = -qubo_energies(QU, lam)
    delta2 = np.maximum(-qubo_energies(inst.QU_pen, 1.0), 0.0)
    rng = rul.E_max - rul.E_min
    w = (rul.E_max - rul.f_band) / rng if rng > 0 else np.ones_like(rul.f_band)
    return MetricContext(n=int(inst.n), band_idx=np.asarray(rul.band_idx, dtype=np.int64), ar_weight=w,
                         delta2=delta2, diag=diag, xstar_idx=np.asarray(rul.xstar_idx, dtype=np.int64),
                         top10_idx=np.asarray(rul.top10_idx, dtype=np.int64),
                         sector_idx=None if sector_idx is None else np.asarray(sector_idx, dtype=np.int64))


class StateLogger:
    """The post-update logger of PLAN §1.6: `logger(t, params)` reads the state (get_state; never used by an
    update) and appends the metrics of theta_t. Structurally separate from the trainer: it receives a copy of
    the parameters and returns nothing."""

    def __init__(self, ansatz, ctx: MetricContext):
        self.ansatz = ansatz
        self.ctx = ctx
        self.t: list = []
        self.rows: list = []
        self.last_state: np.ndarray | None = None

    def __call__(self, t: int, params: np.ndarray) -> None:
        psi = self.ansatz.state(params)
        self.last_state = psi
        self.t.append(int(t))
        self.rows.append(self.ctx.evaluate_state(psi))

    def arrays(self) -> dict:
        keys = list(self.rows[0].keys()) if self.rows else list(METRIC_KEYS)
        out = {"t_logged": np.array(self.t, dtype=np.int64)}
        for k in keys:
            out[k] = np.array([r[k] for r in self.rows], dtype=np.float64)
        return out
