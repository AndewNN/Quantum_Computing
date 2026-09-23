"""Gradient back ends for the trained arms (PLAN §1.6, D-7): pluggable, observe-only.

`ForwardFD(delta)` is the completed work's forward difference (CUDA/PO_new_ApproxRatio.py:888-896):

    grad_j = (f(theta + delta e_j) - f(theta)) / delta,   delta = SHIFT = 1e-4,

evaluated with the same numpy arithmetic (theta + shift with a zero vector carrying delta at j), so the
gradients are bit-identical to the old loop given the same energies. Charged circuits per evaluation:
P + 1 (the base circuit and P shifted ones), i.e. 2L + 1.
"""

from __future__ import annotations

import numpy as np


class ForwardFD:
    name = "fd_forward"

    def __init__(self, delta: float = 1e-4):
        self.delta = float(delta)

    def circuits(self, n_params: int) -> int:
        return int(n_params) + 1

    def __call__(self, energy, params: np.ndarray, f0: float) -> list:
        P = params.size
        out = []
        for j in range(P):
            shift = np.zeros(P)
            shift[j] = self.delta
            forward = float(energy(params + shift))
            out.append((forward - f0) / self.delta)
        return out
