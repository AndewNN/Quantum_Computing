"""The circuits of arm A3 besides the energy circuit (S7, PLAN §1.5): the M1 overlap circuit with its flag qubit,
and the truncated circuits of the diagonal Var(G_k). All run A0's circuit (an `Ansatz` of kind "penalty").

Overlap (`kernel_overlap` in `overlap_kernel.py`): H^n, the first m layers at pb, their inverse at pa, H^n, then the
flag, so P(0...0) = |<psi_m(pa)|psi_m(pb)>|^2 = (1 - <Z_flag>)/2 is an `observe` of ONE Pauli term (never the
2^n-term projector, never get_state in the loop). The layers are the simulation form of A0's layer
(`cost.cost_gates_sim` + `xmixer.x_layer_gates`); the inverse layer is the reversed list with negated angles.

Truncated variance circuits (the layered kernel, `program.layered_kernel`), verbatim the old
`kernel_qaoa_X_trunc(n_cost, n_mix)`:
  gamma_k: k + 1 cost layers and k mixers, measured in Z (G_gamma = the cost operator in circuit units);
  beta_k:  k + 1 full layers, measured in X (G_beta = sum_j X_j).
Run as the layered program with L' = k + 1 layers and the parameters of those layers; for gamma_k the last mixer
angle is set to 0 (rx(0) is the identity exactly).

Counts (the device circuits charged, PLAN §1.7; counted, never simulated, D-1): the overlap circuit at m layers is
2m layers (m forward, m inverse, no cross-layer cancellation, like `compile.transpile`); the H gates are Clifford.
The flag gate is NOT charged: on a device P(0...0) is the frequency of the all-zeros string among the shots of the
same circuit measured in Z, which needs no gate; the flag is how the simulator reads that frequency with `observe`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..compile.decompose import Angle, Gate
from . import program

OVERLAP_OPS = ("x", "h", "cx", "rx", "ry", "rz", "crz")


def inverse_layer(layer: list) -> list:
    """U_l^dag as a gate list: the layer reversed, every rotation angle negated (x / h / cx are self-inverse)."""
    return [g.inverse() for g in reversed(layer)]


@dataclass(frozen=True)
class OverlapProgram:
    """Arguments of `overlap_kernel.kernel_overlap` (docstring there)."""
    n: int
    L: int
    code: list
    coef: list
    ctl: list
    seg: list

    def args(self, pa, pb, m: int, flag: int = 1) -> tuple:
        a = [float(x) for x in pa]
        b = [float(x) for x in pb]
        if len(a) != 2 * self.L or len(b) != 2 * self.L:
            raise ValueError(f"overlap program needs 2L = {2 * self.L} parameters per side")
        if not 1 <= int(m) <= self.L:
            raise ValueError(f"m = {m} outside 1..{self.L}")
        return (self.n, self.L, int(m), self.code, self.coef, self.ctl, self.seg, a, b, int(flag))


def encode_overlap(start: list, layer: list, n: int, L: int) -> OverlapProgram:
    """Pack start (H^n), one layer (pidx 0 = gamma, 1 = beta), its inverse and the end segment (H^n)."""
    end = [Gate("h", (q,)) for q in range(n)]
    code, coef, ctl, seg = [], [], [], []
    for seg_idx, gates in enumerate((start, layer, inverse_layer(layer), end)):
        seg.append(len(code))
        for g in gates:
            if g.name not in OVERLAP_OPS:
                raise ValueError(f"{g.name} is not an overlap-kernel gate")
            if len(g.qubits) - 1 > (1 if g.name in ("cx", "crz") else 0):
                raise ValueError(f"{g.name} with {len(g.qubits) - 1} controls")
            if any(q >= n for q in g.qubits):
                raise ValueError(f"qubit out of range in {g}")
            if g.angle is None or g.angle.pidx < 0:
                slot = program.SLOT_CONST
            elif seg_idx in (0, 3):
                raise ValueError("start / end gates must have constant angles")
            elif g.angle.pidx in (0, 1):
                slot = program.SLOT_GAMMA if g.angle.pidx == 0 else program.SLOT_BETA
            else:
                raise ValueError(f"layer angles bind pidx 0 (gamma) or 1 (beta), got {g.angle.pidx}")
            code.append(program._pack(g, slot, len(ctl)))
            ctl.extend(int(c) for c in g.qubits[:-1])
            coef.append(0.0 if g.angle is None else float(g.angle.coef))
        seg.append(len(code))
    return OverlapProgram(n=n, L=int(L), code=code, coef=coef, ctl=ctl or [0], seg=seg)


def overlap_gates(start: list, layer: list, L: int, m: int) -> list:
    """The flat gate list of the overlap circuit without the flag, for the numpy reference: parameters are
    [pb (2L) | pa (2L)], i.e. forward layer l binds (l, L + l), inverse layer l binds (2L + l, 3L + l)."""
    out = list(start)
    for ell in range(m):
        for g in layer:
            out.append(_bind(g, ell, L, 0))
    for ell in range(m - 1, -1, -1):
        for g in inverse_layer(layer):
            out.append(_bind(g, ell, L, 2 * L))
    out += [Gate("h", (q,)) for q in range(program_n(start, layer))]
    return out


def program_n(start, layer) -> int:
    return 1 + max(q for g in list(start) + list(layer) for q in g.qubits)


def _bind(g: Gate, ell: int, L: int, off: int) -> Gate:
    if g.angle is None or g.angle.pidx < 0:
        return g
    return Gate(g.name, g.qubits, Angle(g.angle.coef, off + (ell if g.angle.pidx == 0 else L + ell)))


def overlap_kernel():
    """The overlap kernel (imports cudaq on first use)."""
    from .overlap_kernel import kernel_overlap
    return kernel_overlap


# --- the truncated variance circuits (layered kernel) ---------------------------------------------------------
def truncated_args(prog: program.LayeredProgram, params, k: int) -> tuple:
    """Layered-kernel arguments of parameter k's variance circuit (module doc): L' = k + 1 layers."""
    from dataclasses import replace
    L = prog.L
    ell = k if k < L else k - L
    g = [float(params[i]) for i in range(ell + 1)]
    b = [float(params[L + i]) for i in range(ell + 1)]
    if k < L:
        b[ell] = 0.0                     # gamma_k: k + 1 cost layers, k mixers
    return replace(prog, L=ell + 1).args(g + b)


def layer_of(k: int, L: int) -> int:
    """Layer index (0-based) of parameter k in [gamma_0..gamma_{L-1}, beta_0..beta_{L-1}]."""
    return k if k < L else k - L
