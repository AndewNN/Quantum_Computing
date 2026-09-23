"""The circuits of arms A0, A1, A2p and A2c on one interface (PLAN §1.5, S4).

An `Ansatz` holds, for one instance and one depth L:
  * the simulation program: a start segment and one layer (cost layer then mixer), packed for the layered
    kernel (`program.encode_layered`); params = [gamma_1..gamma_L, beta_1..beta_L];
  * the Hamiltonian being run (un-boosted `Ising`), its Jh boost alpha, and the `CostTerms` the circuit carries;
  * the counts of the ABSTRACT circuit (PLAN §2.4): (ii) / (iii) CNOTs and T per circuit, from
    `compile.transpile` (A1) or the cost-layer rule (A0). The simulation program is an exact rewrite of the
    abstract circuit (`cost.cost_gates_sim`, `simopt.merge_x`) and is never counted.

The completed work's convention AS IT RAN (S4, verified against the stored runs; STATUS S4): the circuit applies
the UN-BOOSTED coefficients, exp(-i gamma H), while the loop's objective is the boosted f = <alpha H> (so the
boost scales f, the gradient and the f_tol test, not the circuit). The code text boosts `H_ansatz` before
`process_ansatz_values`, but on CUDA-Q 0.13 the scalar never reached `get_raw_data`: the stored first iterates
equal an init with pi / min|un-boosted coefficient| (not the boosted one), and replaying the stored final
parameters reproduces the stored AR2 of all 10 N = 5 A0 runs to 5e-8 only with un-boosted circuit coefficients
(A1: 1e-3, the fp32 Pauli route). PLAN §1.5 says the same: kappa_min is taken "as applied in the circuit" and the
ramp's gamma_l = dgamma alpha l / p is "in circuit units". `circuit_boosted=True` builds the other
parametrization (a different arm version, PLAN §1.5; never used by the arms).

  penalty  (A0, A2p): H(lam) = H_obj + lam Pen, start H^n, X mixer (`xmixer`).
  confined (A1, A2c): H_obj, star start state over the sector, compiled preserving mixer (`preserving`).

`energy(params)` is the loop's f: <alpha H> by `backend.observe` (observe only, PLAN §1.6). `state(params)` is
`backend.get_state` in classical order, for the post-update logger and post-run metrics only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..compile import transpile as tp
from ..instances.encode import Ising, jh_boost
from . import preserving as pr
from . import program
from .cost import CostTerms, cost_gates_sim, cost_terms, min_abs_coeff
from .simopt import merge_x
from .xmixer import a0_gates, h_start, x_layer_gates


@dataclass
class Ansatz:
    kind: str                      # "penalty" | "confined"
    n: int
    L: int
    H: Ising                       # the Hamiltonian being run, un-boosted
    alpha: float                   # its Jh boost (scales the observable f = <alpha H>)
    ct: CostTerms                  # the cost terms as applied in the circuit (un-boosted by default)
    start: list                    # simulation gates of the start segment
    layer: list                    # simulation gates of one layer (pidx 0 = gamma, 1 = beta)
    prog: program.LayeredProgram
    counts: dict                   # abstract-circuit counts per circuit (see `circuit_counts`)
    circ: pr.PreservingCircuit | None = None
    meta: dict = field(default_factory=dict)
    _op: object = None

    @property
    def n_params(self) -> int:
        return 2 * self.L

    @property
    def op(self):
        """The boosted Hamiltonian alpha H as a cudaq SpinOperator (the old `H_ansatz`), built lazily."""
        if self._op is None:
            from ..sim import backend
            self._op = backend.ising_op(self.H, self.alpha)
        return self._op

    def args(self, params) -> tuple:
        return self.prog.args(params)

    def energy(self, params) -> float:
        """f(params) = <psi(params)| alpha H |psi(params)> (observe; what the update loop uses)."""
        from ..sim import backend
        return backend.observe(program.layered_kernel(), self.op, *self.prog.args(params))

    def state(self, params) -> np.ndarray:
        """The statevector in classical order (x_0 = MSB). Logger / post-run metrics only."""
        from ..sim import backend
        return backend.get_state_classical(program.layered_kernel(), self.n, *self.prog.args(params))

    def unrolled(self) -> list:
        """The flat simulation gate list (for the numpy simulator / the interpreter in tests)."""
        return program.unroll_layered(self.start, self.layer, self.L)

    def abstract_gates(self) -> list:
        """The counted circuit, gate for gate (A0: `kernel_qaoa_X`; A1: star prep + L (cost + mixer))."""
        if self.kind == "penalty":
            return a0_gates(self.ct, self.L)
        return pr.a1_gates(self.circ, self.L, cost=self.ct)

    def kappa_min(self) -> float:
        """PLAN §1.5 (Eq. 4.11): the smallest nonzero |coefficient| of the cost Hamiltonian as applied in the
        circuit, with the mixer counting as 1."""
        return float(min(min_abs_coeff(self.ct), 1.0))


def penalty_counts(ct: CostTerms, L: int) -> dict:
    """Per circuit of A0 / A2p: H^n (free) + L x (cost layer + X mixer)."""
    lay = tp.a0_layer_counts(ct)
    cx = int(lay["cx"])
    return {"per_circuit": {"cx_ii": L * cx, "cx_iii": L * cx, "t_ii": L * int(lay["t"]), "t_iii": L * int(lay["t"]),
                            "tdepth_ii": L * int(lay["tdepth"]), "tdepth_iii": L * int(lay["tdepth"])},
            "layer": {"cx_ii": cx, "cx_iii": cx, "t_ii": int(lay["t"]), "t_iii": int(lay["t"]),
                      "tdepth_ii": int(lay["tdepth"]), "tdepth_iii": int(lay["tdepth"]), "n_zz": ct.n_zz,
                      "n_rot": ct.n_rot},
            "start": {"cx_ii": 0, "cx_iii": 0, "t_ii": 0, "t_iii": 0, "tdepth_ii": 0, "tdepth_iii": 0}}


def confined_counts(circ: pr.PreservingCircuit, ct: CostTerms, L: int) -> dict:
    """Per circuit of A1 / A2c: star start state + L x (cost layer + preserving mixer), from `transpile`."""
    c = tp.circuit_counts(circ, ct)
    tot = tp.a1_totals(c, L)
    keys = ("cx_ii", "cx_iii", "t_ii", "t_iii", "tdepth_ii", "tdepth_iii")
    return {"per_circuit": {k: int(tot[k]) for k in keys},
            "layer": {k: int(c["layer"][k]) for k in keys},
            "start": {k: int(c["prep"][k]) for k in keys},
            "mixer": {k: int(c["mixer"][k]) for k in keys} | {"S_mean": c["mixer"]["S_mean"], "S_max": c["mixer"]["S_max"],
                                                              "d_mean": c["mixer"]["d_mean"]},
            "cx_ii_S_per_circuit": int(tot["cx_ii_S"])}


def penalty_ansatz(H: Ising, L: int, alpha: float | None = None, circuit_boosted: bool = False) -> Ansatz:
    """A0 / A2p on H = H(lam) (un-boosted); alpha defaults to the Jh boost of H (PLAN §1.1)."""
    alpha = jh_boost(H) if alpha is None else float(alpha)
    ct = cost_terms(H, alpha if circuit_boosted else 1.0)
    start = h_start(H.n)
    layer = cost_gates_sim(ct, 0) + x_layer_gates(H.n, 1)
    prog = program.encode_layered(start, layer, H.n, L)
    return Ansatz(kind="penalty", n=H.n, L=int(L), H=H, alpha=alpha, ct=ct, start=start, layer=layer, prog=prog,
                  counts=penalty_counts(ct, L), meta={"circuit_boosted": bool(circuit_boosted)})


def confined_ansatz(H_obj: Ising, alpha: float, circ: pr.PreservingCircuit, L: int,
                    circuit_boosted: bool = False) -> Ansatz:
    """A1 / A2c on H_obj with the instance's boost `boost_obj`, over a `PreservingCircuit`."""
    ct = cost_terms(H_obj, float(alpha) if circuit_boosted else 1.0)
    start = merge_x(pr.prep_gates(circ))
    layer = merge_x(cost_gates_sim(ct, 0) + pr.layer_gates(circ, 1))
    prog = program.encode_layered(start, layer, circ.n, L)
    return Ansatz(kind="confined", n=circ.n, L=int(L), H=H_obj, alpha=float(alpha), ct=ct, start=start, layer=layer,
                  prog=prog, counts=confined_counts(circ, ct, L), circ=circ,
                  meta={"K": circ.K, "connectivity": circ.connectivity, "ring_order": circ.ring_order,
                        "order": [int(x) for x in circ.order], "circuit_boosted": bool(circuit_boosted)})
