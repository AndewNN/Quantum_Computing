"""Gate lists as data for the two data-driven kernels.

Flat lists -> the interpreter kernel (`interp.kernel_program`, S3; the C1 checks and any flat list):

    prog = encode(gates, n)                  # gates from `preserving.a1_gates` / `cost.cost_gates` / ...
    backend.observe(kernel(), op, *prog.args(params))
    backend.get_state(kernel(), *prog.args(params))

Layered circuits (start + L identical layers) -> the layered kernel (`layered.kernel_layered`, S4; the arms):

    lp = encode_layered(start, layer, n, L)  # layer angles: pidx 0 = gamma_l, 1 = beta_l
    backend.observe(layered_kernel(), op, *lp.args(params))    # params = [gamma_1..gamma_L, beta_1..beta_L]

Both kernels compile once per process; any list runs without recompiling, and `get_state` costs what
`observe` costs. Kernel arguments cost ~0.8 us per list element per call (STATUS S4), so the layered form
(one layer's data, one packed int per gate) is ~2-4x faster than the flat one on the same circuit.
Controls per gate: up to MAX_CONTROLS (n <= 20).
"""

from __future__ import annotations

from dataclasses import dataclass

OPCODES = {"x": 0, "h": 1, "s": 2, "sdg": 3, "t": 4, "tdg": 5, "cx": 6, "rx": 7, "ry": 8, "rz": 9,
           "mcrx": 10, "mcry": 11, "crz": 12}
MAX_CONTROLS = 19


@dataclass(frozen=True)
class Program:
    n: int
    op: list
    tgt: list
    cs: list
    nc: list
    ctl: list
    coef: list
    pidx: list
    n_params: int

    @property
    def n_gates(self) -> int:
        return len(self.op)

    def args(self, params=()) -> tuple:
        p = [float(x) for x in params]
        if len(p) < self.n_params:
            raise ValueError(f"program needs {self.n_params} parameters, got {len(p)}")
        return (self.n, self.op, self.tgt, self.cs, self.nc, self.ctl, self.coef, self.pidx, p or [0.0])


def encode(gates, n: int) -> Program:
    op, tgt, cs, nc, ctl, coef, pidx = [], [], [], [], [], [], []
    n_params = 0
    for g in gates:
        if g.name not in OPCODES:
            raise ValueError(g.name)
        controls = g.qubits[:-1]
        if len(controls) > MAX_CONTROLS:
            raise ValueError(f"{len(controls)} controls > {MAX_CONTROLS}")
        if any(q >= n for q in g.qubits):
            raise ValueError(f"qubit out of range in {g}")
        op.append(OPCODES[g.name])
        tgt.append(int(g.qubits[-1]))
        cs.append(len(ctl))
        nc.append(len(controls))
        ctl.extend(int(c) for c in controls)
        if g.angle is None:
            coef.append(0.0)
            pidx.append(-1)
        else:
            coef.append(float(g.angle.coef))
            pidx.append(int(g.angle.pidx))
            if g.angle.pidx >= 0:
                n_params = max(n_params, g.angle.pidx + 1)
    return Program(n=n, op=op, tgt=tgt, cs=cs, nc=nc, ctl=ctl or [0], coef=coef, pidx=pidx,
                   n_params=n_params)


def kernel():
    """The interpreter kernel (imports cudaq on first use)."""
    from .interp import kernel_program
    return kernel_program


# --- the layered program (S4): a start segment once, then one layer segment L times -----------------------
LAYERED_OPS = ("x", "h", "cx", "rx", "ry", "rz", "mcrx", "mcry", "crz")
SLOT_CONST, SLOT_GAMMA, SLOT_BETA = 0, 1, 2
MAX_QUBITS_LAYERED = 32            # 5 bits of target / arity in the packed code


@dataclass(frozen=True)
class LayeredProgram:
    """Arguments of `layered.kernel_layered` (docstring there). In the layer gates, `Angle.pidx` 0 means
    gamma_l and 1 means beta_l; start gates carry constant angles only."""

    n: int
    L: int
    code: list
    coef: list
    ctl: list
    seg: list
    n_start: int
    n_layer: int

    @property
    def n_gates(self) -> int:
        """Gates the simulator applies: start + L layers."""
        return self.n_start + self.L * self.n_layer

    @property
    def n_params(self) -> int:
        return 2 * self.L

    def args(self, params) -> tuple:
        p = [float(x) for x in params]
        if len(p) != 2 * self.L:
            raise ValueError(f"layered program needs 2L = {2 * self.L} parameters, got {len(p)}")
        return (self.n, self.L, self.code, self.coef, self.ctl, self.seg, p or [0.0])


def _pack(g, slot: int, ctl_start: int) -> int:
    controls = g.qubits[:-1]
    return OPCODES[g.name] + 16 * (int(g.qubits[-1]) + 32 * (len(controls) + 32 * (slot + 4 * ctl_start)))


def encode_layered(start, layer, n: int, L: int) -> LayeredProgram:
    """Pack a start segment (constant angles) and one layer (pidx 0 = gamma, 1 = beta) for `kernel_layered`."""
    if n > MAX_QUBITS_LAYERED:
        raise ValueError(f"n = {n} > {MAX_QUBITS_LAYERED}")
    code, coef, ctl = [], [], []
    for seg_idx, gates in enumerate((start, layer)):
        for g in gates:
            if g.name not in LAYERED_OPS:
                raise ValueError(f"{g.name} is not a layered-kernel gate")
            controls = g.qubits[:-1]
            if len(controls) > MAX_CONTROLS:
                raise ValueError(f"{len(controls)} controls > {MAX_CONTROLS}")
            if any(q >= n for q in g.qubits):
                raise ValueError(f"qubit out of range in {g}")
            if g.name in ("cx", "crz") and len(controls) != 1:
                raise ValueError(f"{g.name} needs one control")
            if g.angle is None or g.angle.pidx < 0:
                slot = SLOT_CONST
            elif seg_idx == 0:
                raise ValueError("start gates must have constant angles")
            elif g.angle.pidx in (0, 1):
                slot = SLOT_GAMMA if g.angle.pidx == 0 else SLOT_BETA
            else:
                raise ValueError(f"layer angles bind pidx 0 (gamma) or 1 (beta), got {g.angle.pidx}")
            code.append(_pack(g, slot, len(ctl)))
            ctl.extend(int(c) for c in controls)
            coef.append(0.0 if g.angle is None else float(g.angle.coef))
    ns, nl = len(start), len(layer)
    return LayeredProgram(n=n, L=int(L), code=code, coef=coef, ctl=ctl or [0], seg=[0, ns, ns, ns + nl],
                          n_start=ns, n_layer=nl)


def unroll_layered(start, layer, L: int) -> list:
    """The flat gate list a layered program runs (pidx 0 -> l, 1 -> L + l): for the numpy simulator and the
    interpreter, which the tests compare with the layered kernel."""
    from ..compile.decompose import Angle, Gate
    out = list(start)
    for ell in range(L):
        for g in layer:
            if g.angle is None or g.angle.pidx < 0:
                out.append(g)
            else:
                out.append(Gate(g.name, g.qubits, Angle(g.angle.coef, ell if g.angle.pidx == 0 else L + ell)))
    return out


def layered_kernel():
    """The layered kernel (imports cudaq on first use)."""
    from .layered import kernel_layered
    return kernel_layered
