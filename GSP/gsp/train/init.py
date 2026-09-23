"""Initial parameters of the trained arms (PLAN §1.5). params = [gamma_1..gamma_L, beta_1..beta_L].

Random init (Eq. 4.11), ported from CUDA/PO_new_ApproxRatio.py:812-818 and 839-845:

    np.random.seed(seed); points = np.random.uniform(-1, 1, 2L)
    points[:L] *= mm_i;   points[L:] *= np.pi,          mm_i = np.pi / kappa_min

The seed is the restart seed of the seed table (never derived at run time). A private
`np.random.RandomState(seed)` gives the same Mersenne-Twister numbers as the old global `np.random.seed`.

  kappa_min (Eq. 4.11, the default): the smallest nonzero |coefficient| of the cost Hamiltonian as applied in
      the circuit (un-boosted, `ansatz.py`), the mixer counting as 1 (`Ansatz.kappa_min`). For A0 this equals the
      old range exactly (the un-boosted coefficients are all < 1).
  legacy (`legacy_init=True`, the S4 equivalence test only): the old mm_i = pi / min(mm_1, mm_2, mm_p), with the
      old types -- mm_p is the float32 minimum of the old merged Pauli coefficients (`legacy_pauli.legacy_mm_p`)
      for the preserving mixer and 1e9 for the X mixer, so numpy 2 makes mm_i float32 in the preserving case,
      exactly as it was in the completed runs.
"""

from __future__ import annotations

import numpy as np


def random_points(seed: int, L: int, mm_i) -> np.ndarray:
    """The old init with a given gamma range mm_i (lines 839-845, verbatim arithmetic)."""
    rng = np.random.RandomState(int(seed))
    points = rng.uniform(-1, 1, (2 * L))
    points[:L] *= mm_i
    points[L:] *= np.pi
    return points


def mm_i_eq411(kappa_min: float) -> float:
    return np.pi / kappa_min


def mm_i_legacy(coeff_1, coeff_2, mm_p=1e9):
    """Lines 812-818 verbatim (coeff_1 / coeff_2: the cost coefficients as the circuit carries them)."""
    coeff_1_use, coeff_2_use = np.array(coeff_1), np.array(coeff_2)
    mm_1 = np.min(np.abs(coeff_1_use)) if len(coeff_1_use) > 0 else 1e9
    mm_2 = np.min(np.abs(coeff_2_use)) if len(coeff_2_use) > 0 else 1e9
    mm_i = np.pi / min(mm_1, mm_2, mm_p)
    return mm_i


def init_params(ansatz, seed: int, legacy: bool = False) -> np.ndarray:
    """Random init of an A0 / A1 ansatz (PLAN §1.5); `legacy` reproduces the completed work's gamma range."""
    L = ansatz.L
    if not legacy:
        return random_points(seed, L, mm_i_eq411(ansatz.kappa_min()))
    mm_p = 1e9
    if ansatz.kind == "confined":
        from ..circuits.legacy_pauli import legacy_mm_p
        if ansatz.circ.connectivity != "ring":
            raise ValueError("legacy_init exists only for the ring (the completed work's mixer)")
        mm_p = legacy_mm_p(ansatz.circ.order, ansatz.n)
    return random_points(seed, L, mm_i_legacy(ansatz.ct.coeff_1, ansatz.ct.coeff_2, mm_p))
