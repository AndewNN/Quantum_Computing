"""The cost layer exp(-i gamma alpha H) of a diagonal Ising H (PLAN §1.5, §2.4).

Ported from the completed work's `kernel_qaoa_X` / `kernel_qaoa_Preserved` (Utils/qaoaCUDAQ.py): per
field rz(2 c gamma) on qubit i, per ZZ term cx(a, b) rz(2 c gamma)_b cx(a, b), in the order of
`Ising.terms()` (fields by qubit, then pairs lexicographically), zeros dropped, c = alpha * coefficient
(the Jh boost, applied in the circuit). The constant is a global phase and is not applied.

Counts: 2 CNOTs per ZZ term (V18); one arbitrary rotation per term (V20).
S3 builds it for the timing and the counts; S4 owns the arm conventions (sign, boost) and their tests.
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
