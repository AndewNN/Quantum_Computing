"""The X mixer and the A0 circuit (PLAN §1.5), a port of `kernel_qaoa_X` (Utils/qaoaCUDAQ.py:462-488).

`kernel_qaoa_X` applies H on every qubit, then per layer l: the cost layer rz(2 c gamma_l) / cx-rz-cx with the
boosted coefficients c of the Hamiltonian being run, then rx(2 beta_l) on every qubit, i.e. exp(-i beta_l X_k).
Parameters thetas = [gamma_1..gamma_L, beta_1..beta_L].

  a0_gates(ct, L)         the counted (abstract) circuit, gate for gate the old kernel;
  x_layer_gates(n, pidx)  the mixer layer: rx(2 beta) on every qubit (counts: 0 CNOTs, n synthesized rotations);
  the simulation form (`ansatz.penalty_ansatz`) runs the cost layer as `cost.cost_gates_sim`.
The verbatim kernel itself lives in `xkernel.py` (it imports cudaq), as the reference and the "before" engine.
"""

from __future__ import annotations

from ..compile.decompose import Angle, Gate
from .cost import CostTerms, cost_gates


def h_start(n: int) -> list:
    return [Gate("h", (q,)) for q in range(n)]


def x_layer_gates(n: int, pidx: int) -> list:
    """exp(-i beta X) on every qubit: rx(2 beta), beta = params[pidx]."""
    return [Gate("rx", (q,), Angle(2.0, pidx)) for q in range(n)]


def a0_gates(ct: CostTerms, L: int) -> list:
    """The whole A0 circuit as `kernel_qaoa_X` builds it (params = [gamma_1..gamma_L, beta_1..beta_L])."""
    out = h_start(ct.n)
    for ell in range(L):
        out += cost_gates(ct, pidx=ell) + x_layer_gates(ct.n, L + ell)
    return out
