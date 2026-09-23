"""The T-count / T-depth companion (PLAN §0 "resource axis", §2.4), a port of V20 in
`~/Desktop/Quantum_Master_Proposal/Lecture_Notes/code/verify_notes.py` (~l. 2509-2600).

Rules (V20, every count read off the V18 circuits):
  * T_syn = ceil(3 log2(1/eps_syn)) T gates per arbitrary rotation (Ross-Selinger): 30 at 1e-3.
  * Clifford gates (H, S, X, CNOT) are free; Rx / Ry cost what Rz costs.
  * Cost layer: one arbitrary Rz per term, n(n-1)/2 + n for a dense QUBO; T-depth (chi'(K_n) + 1) T_syn
    (a round-robin edge colouring runs the ZZ rotations in chi'(K_n) rounds, the fields in one more;
    chi' = n - 1 for even n, n for odd n). For a sparser ZZ graph the same depth is an upper bound.
  * X mixer: n R_x, n T_syn, T-depth T_syn.
  * One compiled transition, route (ii): 2(n - 2) Toffolis (computed and uncomputed) and 2 synthesized
    rotations: 2(n - 2) c_Tof + 2 T_syn T gates, T-depth 2(n - 2) d_Tof + 2 T_syn. Headline constants
    c_Tof = 7 (textbook Toffoli), d_Tof = 1 (V20; the relative-phase Toffoli would be c = 4). The Hamming
    distance drops out (W is Clifford).
  * A layer's K transitions serialize on the shared register (each touches every system qubit), so a
    layer's T-depth is K times one transition's.
For route (iii) the T-count and ASAP T-depth come from the explicit gate list (`decompose.t_count`,
`decompose.t_depth`), with the same T_syn per synthesized rotation.
"""

from __future__ import annotations

import math

EPS_SYN = 1e-3
C_TOF = 7
D_TOF = 1


def t_syn(eps: float = EPS_SYN) -> int:
    return int(math.ceil(3 * math.log2(1 / eps)))


def chi_edge(n: int) -> int:
    """Chromatic index of the complete graph K_n."""
    return n - 1 if n % 2 == 0 else n


def cost_layer_t(n_rot: int, ts: int | None = None) -> int:
    return int(n_rot) * (t_syn() if ts is None else ts)


def cost_layer_tdepth(n: int, ts: int | None = None, has_fields: bool = True) -> int:
    ts = t_syn() if ts is None else ts
    return (chi_edge(n) + (1 if has_fields else 0)) * ts


def x_mixer_t(n: int, ts: int | None = None) -> int:
    return n * (t_syn() if ts is None else ts)


def x_mixer_tdepth(ts: int | None = None) -> int:
    return t_syn() if ts is None else ts


def transition_t_ii(n: int, c_tof: int = C_TOF, ts: int | None = None) -> int:
    ts = t_syn() if ts is None else ts
    return 2 * (n - 2) * c_tof + 2 * ts


def transition_tdepth_ii(n: int, d_tof: int = D_TOF, ts: int | None = None) -> int:
    ts = t_syn() if ts is None else ts
    return 2 * (n - 2) * d_tof + 2 * ts
