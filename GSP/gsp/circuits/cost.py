"""The cost layer exp(-i gamma alpha H) of a diagonal Ising H (PLAN §1.5, §2.4).

Ported from the completed work's `kernel_qaoa_X` / `kernel_qaoa_Preserved` (Utils/qaoaCUDAQ.py): per
field rz(2 c gamma) on qubit i, per ZZ term cx(a, b) rz(2 c gamma)_b cx(a, b), in the order of
`Ising.terms()` (fields by qubit, then pairs lexicographically), zeros dropped, c = alpha * coefficient.
The arms pass alpha = 1 (S4): the completed runs' circuits carried the un-boosted coefficients (the boost
scaled only the observable; `ansatz.py` has the evidence). The constant is a global phase and is not applied.

Counts: 2 CNOTs per ZZ term (V18); one arbitrary rotation per term (V20).
S3 builds it for the timing and the counts; S4 owns the arm conventions (sign, boost) and their tests.

Two forms of the same layer (S4):
  `cost_gates`      the counted (abstract) circuit, cx-rz-cx per ZZ term: what (ii) / (iii) / T count;
  `cost_gates_sim`  what the simulator runs: one controlled rz per ZZ term plus one rz per qubit,
                    exp(-i g c Z_a Z_b) = rz(2 c g)_b crz(-4 c g)_{a -> b}, with the rz's of every term on
                    qubit b and its field merged into one rz(2 g (h_b + sum_a c_ab))_b. All gates are diagonal,
                    so they commute and the merge is exact (the unitary is the same up to the rounding of the
                    summed angle, <= 1e-15 in the tests). 1 + n(n-1)/2 + n gates instead of n + 3 n(n-1)/2,
                    and cusvsim fuses diagonal gates: the layer runs ~3x faster (STATUS S4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..compile.decompose import Angle, Gate


@dataclass(frozen=True)
class CostTerms:
    n: int
    idx_1: tuple
    coeff_1: tuple
    idx_2a: tuple
    idx_2b: tuple
    coeff_2: tuple

    @property
    def n_zz(self) -> int:
        return len(self.idx_2a)

    @property
    def n_rot(self) -> int:
        return len(self.idx_1) + len(self.idx_2a)


def cost_terms(H, alpha: float = 1.0) -> CostTerms:
    """From a `gsp.instances.encode.Ising`, every coefficient times alpha."""
    i1, c1, a2, b2, c2 = H.terms()
    return CostTerms(n=H.n, idx_1=tuple(i1), coeff_1=tuple(float(alpha) * c for c in c1),
                     idx_2a=tuple(a2), idx_2b=tuple(b2), coeff_2=tuple(float(alpha) * c for c in c2))


def cost_gates(ct: CostTerms, pidx: int = 0) -> list:
    """The layer as a gate list (for counts and the numpy simulator); gamma = params[pidx]."""
    out = [Gate("rz", (i,), Angle(2.0 * c, pidx)) for i, c in zip(ct.idx_1, ct.coeff_1)]
    for a, b, c in zip(ct.idx_2a, ct.idx_2b, ct.coeff_2):
        out += [Gate("cx", (a, b)), Gate("rz", (b,), Angle(2.0 * c, pidx)), Gate("cx", (a, b))]
    return out


def cost_gates_sim(ct: CostTerms, pidx: int = 0) -> list:
    """The simulation form of the layer (module docstring): crz per ZZ term (term order), then rz per qubit
    (ascending, zeros dropped). Exactly the unitary of `cost_gates(ct, pidx)`."""
    hz = np.zeros(ct.n)
    for i, c in zip(ct.idx_1, ct.coeff_1):
        hz[i] += 2.0 * c
    for a, b, c in zip(ct.idx_2a, ct.idx_2b, ct.coeff_2):
        hz[b] += 2.0 * c
    out = [Gate("crz", (a, b), Angle(-4.0 * c, pidx)) for a, b, c in zip(ct.idx_2a, ct.idx_2b, ct.coeff_2)]
    out += [Gate("rz", (q,), Angle(float(hz[q]), pidx)) for q in range(ct.n) if hz[q] != 0.0]
    return out


def min_abs_coeff(ct: CostTerms) -> float:
    """Smallest nonzero |coefficient| of the layer as applied in the circuit (boosted); inf if none."""
    vals = [abs(c) for c in ct.coeff_1 + ct.coeff_2 if c != 0.0]
    return float(min(vals)) if vals else float("inf")


def diagonal(ct: CostTerms, idx) -> np.ndarray:
    """sum_i c_i z_i + sum c_ab z_a z_b on classical indices (no constant)."""
    from ..instances.bits import index_to_bits
    z = 1.0 - 2.0 * index_to_bits(idx, ct.n).astype(np.float64)
    out = np.zeros(z.shape[:-1])
    for i, c in zip(ct.idx_1, ct.coeff_1):
        out = out + c * z[..., i]
    for a, b, c in zip(ct.idx_2a, ct.idx_2b, ct.coeff_2):
        out = out + c * z[..., a] * z[..., b]
    return out


def append_cost(kern, q, ct: CostTerms, gamma) -> int:
    """Emit the layer into a builder kernel; gamma is a kernel parameter (QuakeValue) or a float."""
    count = 0
    for i, c in zip(ct.idx_1, ct.coeff_1):
        kern.rz(gamma * (2.0 * c), q[i])
        count += 1
    for a, b, c in zip(ct.idx_2a, ct.idx_2b, ct.coeff_2):
        kern.cx(q[a], q[b])
        kern.rz(gamma * (2.0 * c), q[b])
        kern.cx(q[a], q[b])
        count += 3
    return count
