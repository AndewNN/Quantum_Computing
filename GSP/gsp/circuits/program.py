"""Gate lists as data for the interpreter kernel (`interp.kernel_program`), the default engine (S3).

    prog = encode(gates, n)                  # gates from `preserving.a1_gates` / `cost.cost_gates` / ...
    backend.observe(kernel(), op, *prog.args(params))
    backend.get_state(kernel(), *prog.args(params))

The kernel is compiled once per process; any list runs without recompiling, and `get_state` costs what
`observe` costs (A1 at n = 14, L = 9, 4,300 gates: 53 ms each; see reports/mixer_counts.md "Timing").
Controls per gate: up to MAX_CONTROLS (n <= 20).
"""

from __future__ import annotations

from dataclasses import dataclass

OPCODES = {"x": 0, "h": 1, "s": 2, "sdg": 3, "t": 4, "tdg": 5, "cx": 6, "rx": 7, "ry": 8, "rz": 9,
           "mcrx": 10, "mcry": 11}
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
