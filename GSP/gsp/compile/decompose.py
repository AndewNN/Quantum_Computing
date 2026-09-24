"""Explicit gate lists: the (ii) and (iii) compilations of a transition, and their building blocks
(PLAN §2.1, §2.4; D-1). Counts are read off these lists (`transpile.py`), never from a formula alone;
`npsim.py` multiplies them out so the tests can compare unitaries.

Gate set of the lists: x, h, s, sdg, t, tdg, cx (Clifford + T), rx / ry / rz (one-qubit rotations,
`rx(a) = exp(-i a X / 2)` as in CUDA-Q), and the *native* multi-controlled rotations `mcrx` / `mcry`
(closed controls; open controls are X-conjugated). A native gate appears only in the simulation
route; the (ii) and (iii) lists contain none.

Qubit k is bit x_k of a string (x_0 = MSB of the classical index, `gsp.instances.bits`).

Angles are affine in at most one circuit parameter: `Angle(coef, pidx)` has the value
`coef * params[pidx]`, or the constant `coef` when `pidx < 0`.

--- (iii): the ancilla-free multi-controlled SU(2) of Vale et al. 2024 (arXiv:2302.06377, TCAD 43(3)) ---
`mc_su2(controls, target, kind, angle)` builds C^k(R) for R = Rx / Ry / Rz(angle) on the k + 1 qubits
only (no ancilla), following the paper's Theorem 3 / Corollary 2 (the "real-valued diagonal" case,
Fig. 7; Rx, Ry, Rz all qualify):
  * the controls split into k1 = ceil(k/2) and k2 = floor(k/2);
  * time order  MCX(k1) A  MCX(k2) A†  MCX(k1) A  MCX(k2) A†  on the target, so the all-ones action
    is (A† X A X)^2 = R and every other control pattern gives the identity (Eqs. 3, 4, 6);
  * A = Rz(-a/4) for Rz(a); A = Ry(-a/4) for Ry(a); Rx(a) = H Rz(a) H (Lemma 2, the H pair on the
    target);
  * each MCX uses the other group as dirty (borrowed) ancillas: k_i >= 3 controls cost 8 k_i - 6 CNOTs
    (Iten et al. 2016, Lemma 8: Barenco Lemma 7.2's Toffoli chain, the two Toffolis on the target exact,
    every ancilla-targeting Toffoli the 3-CNOT Margolus gate, the pairs sharing their outer halves);
    k_i = 2 is one Toffoli (6 CNOTs), k_i = 1 one CNOT, k_i = 0 an unconditional X.
  CNOT count: C(k) = 2 c(k1) + 2 c(k2) with c(0) = 0, c(1) = 1, c(2) = 6, c(m >= 3) = 8m - 6, i.e.
  0, 2, 4, 14, 24, 48 for k = 0..5 and exactly 16k - 24 = 16(k + 1) - 40 for k >= 6 (the paper's
  Theorem 3 bound, n = k + 1 qubits).

--- (ii): the device count of Eq. 4.10 (verify_notes.py V18) ---
`transition_ii` rebuilds V18's construction: W, open controls on all n - 1 non-pivot qubits, a chain of
relative-phase (Margolus) Toffolis on n - 2 clean ancillas (qubits n .. 2n - 3), a controlled rotation
(2 CNOTs), and the inverses: 2(d - 1) + 6(n - 2) + 2 = 6n + 2d - 12 CNOTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

ROTATIONS = ("rx", "ry", "rz")
PHASES = ("r1",)                 # r1(a) = diag(1, e^{ia}) (CUDA-Q's r1); S8: in the MC phase's (ii) / (iii) lists
NATIVE_MC = ("mcrx", "mcry", "mcr1")   # mcr1: native multi-controlled phase (S8, DB-QITE's e^{ir|0..0><0..0|})
SIM_ONLY = ("crz",)              # controlled Rz: only in the simulation form of the cost layer (S4, cost.py)
CLIFFORD_1Q = ("x", "h", "s", "sdg")


@dataclass(frozen=True)
class Angle:
    coef: float
    pidx: int = -1                       # < 0: the constant `coef`

    def value(self, params=None) -> float:
        if self.pidx < 0:
            return float(self.coef)
        return float(self.coef) * float(params[self.pidx])

    def scaled(self, c: float) -> "Angle":
        return Angle(self.coef * c, self.pidx)

    @property
    def is_param(self) -> bool:
        return self.pidx >= 0


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple                        # 1q: (q,); cx: (c, t); mc*: (*controls, target)
    angle: Angle | None = None

    def inverse(self) -> "Gate":
        if self.name in ("x", "h", "cx"):
            return self
        if self.name in ("s", "sdg", "t", "tdg"):
            return Gate({"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}[self.name], self.qubits)
        if self.name in ROTATIONS or self.name in PHASES or self.name in NATIVE_MC or self.name in SIM_ONLY:
            return Gate(self.name, self.qubits, self.angle.scaled(-1.0))
        raise ValueError(self.name)


def inverse(gates) -> list:
    return [g.inverse() for g in reversed(gates)]


def _g(name, *qubits, angle=None):
    return Gate(name, tuple(int(q) for q in qubits), angle)


def _c(value: float) -> Angle:
    return Angle(float(value), -1)


# --- Toffolis -------------------------------------------------------------------------------------
def toffoli(a: int, b: int, t: int) -> list:
    """Exact Toffoli (controls a, b; target t): 6 CNOTs, 7 T/T† (Nielsen-Chuang Fig. 4.9)."""
    return [_g("h", t), _g("cx", b, t), _g("tdg", t), _g("cx", a, t), _g("t", t), _g("cx", b, t),
            _g("tdg", t), _g("cx", a, t), _g("t", b), _g("t", t), _g("h", t), _g("cx", a, b),
            _g("t", a), _g("tdg", b), _g("cx", a, b)]


def _F(c: int, t: int) -> list:
    """Outer half of the Margolus gate: Ry(pi/4) CX(c, t) Ry(pi/4) on t (time order)."""
    return [_g("ry", t, angle=_c(np.pi / 4)), _g("cx", c, t), _g("ry", t, angle=_c(np.pi / 4))]


def _Fdg(c: int, t: int) -> list:
    return [_g("ry", t, angle=_c(-np.pi / 4)), _g("cx", c, t), _g("ry", t, angle=_c(-np.pi / 4))]


def rtof(c1: int, c2: int, t: int) -> list:
    """Relative-phase (Margolus) Toffoli, a port of `rtof18` (verify_notes.py V18): a Toffoli up to a
    sign on one basis state, 3 CNOTs. Time order F(c2, t), CX(c1, t), F†(c2, t)."""
    return _F(c2, t) + [_g("cx", c1, t)] + _Fdg(c2, t)


# --- multi-controlled X with dirty ancillas (Iten et al. 2016, Lemma 8) -----------------------------
def _chain(controls, ancillas) -> list:
    """The compressed Barenco chain for k = len(controls) >= 3: it XORs c_1 ... c_{k-1} into
    a_{k-2} (up to a diagonal phase that the second copy removes). Pairs R_j ... R_j share their
    outer halves: F_j M_j [inner] M_j F_j†."""
    c, a = list(controls), list(ancillas)
    k = len(c)
    down, up = [], []
    for j in range(k - 1, 2, -1):             # j = k-1 .. 3 (1-based control index)
        tgt, mid, ctl = a[j - 2], a[j - 3], c[j - 1]
        down += _F(ctl, tgt) + [_g("cx", mid, tgt)]
        up = [_g("cx", mid, tgt)] + _Fdg(ctl, tgt) + up
    return down + rtof(c[0], c[1], a[0]) + up


def mcx(controls, target: int, ancillas=()) -> list:
    """C^k(X) on `target` (closed controls). k >= 3 needs k - 2 dirty ancillas and costs 8k - 6 CNOTs;
    k = 2 is an exact Toffoli, k = 1 a CNOT, k = 0 an X."""
    controls = list(controls)
    k = len(controls)
    if k == 0:
        return [_g("x", target)]
    if k == 1:
        return [_g("cx", controls[0], target)]
    if k == 2:
        return toffoli(controls[0], controls[1], target)
    anc = list(ancillas)[: k - 2]
    if len(anc) < k - 2:
        raise ValueError(f"C^{k}(X) needs {k - 2} dirty ancillas, got {len(anc)}")
    top = toffoli(controls[-1], anc[-1], target)
    ch = _chain(controls, anc)
    return top + ch + top + ch


def mcx_cnots(k: int) -> int:
    return 0 if k == 0 else 1 if k == 1 else 6 if k == 2 else 8 * k - 6


# --- Vale et al. 2024: ancilla-free C^k(SU(2)) with a real-valued diagonal --------------------------
def mc_su2(controls, target: int, kind: str, angle: Angle) -> list:
    """C^k(R_kind(angle)) on `target`, closed controls, on the k + 1 qubits only (see module doc)."""
    controls = [int(c) for c in controls]
    k = len(controls)
    if kind not in ROTATIONS:
        raise ValueError(kind)
    if k == 0:
        return [_g(kind, target, angle=angle)]
    k1 = (k + 1) // 2
    g1, g2 = controls[:k1], controls[k1:]
    m1, m2 = mcx(g1, target, g2), mcx(g2, target, g1)
    rot = "ry" if kind == "ry" else "rz"
    A, Adg = _g(rot, target, angle=angle.scaled(-0.25)), _g(rot, target, angle=angle.scaled(0.25))
    body = m1 + [A] + m2 + [Adg] + m1 + [A] + m2 + [Adg]
    if kind == "rx":
        return [_g("h", target)] + body + [_g("h", target)]
    return body


def vale_cnots(k: int) -> int:
    """Closed form of `mc_su2`'s CNOT count (the tests check it against the list)."""
    if k == 0:
        return 0
    k1, k2 = (k + 1) // 2, k // 2
    return 2 * mcx_cnots(k1) + 2 * mcx_cnots(k2)


# --- one transition: W, open controls, the rotation, and the inverses ---------------------------------
def _w_gates(xmask, ladder) -> list:
    return [_g("x", k) for k in xmask] + [_g("cx", c, t) for c, t in ladder]


def transition_native(xmask, ladder, S, k0: int, kind: str, angle: Angle) -> list:
    """The simulated circuit (D-1): W, X on S, one native multi-controlled rotation, X on S, W†."""
    W = _w_gates(xmask, ladder)
    flips = [_g("x", s) for s in S]
    if S:
        core = [Gate("mc" + kind, tuple(S) + (k0,), angle)]
    else:
        core = [_g(kind, k0, angle=angle)]
    return W + flips + core + flips + inverse(W)


def transition_iii(xmask, ladder, S, k0: int, kind: str, angle: Angle) -> list:
    """(iii): the same circuit with the multi-controlled rotation replaced by `mc_su2` (no ancilla)."""
    W = _w_gates(xmask, ladder)
    flips = [_g("x", s) for s in S]
    return W + flips + mc_su2(S, k0, kind, angle) + flips + inverse(W)


def transition_ii(n: int, xmask, ladder, k0: int, kind: str, angle: Angle) -> list:
    """(ii): V18's construction on n + (n - 2) qubits (clean ancillas n .. 2n - 3), with the n - 1
    open controls = every qubit but the pivot. Needs n >= 3."""
    ctl = [k for k in range(n) if k != k0]
    anc = list(range(n, 2 * n - 2))
    W = _w_gates(xmask, ladder)
    flips = [_g("x", k) for k in ctl]
    ch = rtof(ctl[0], ctl[1], anc[0])
    for j in range(2, len(ctl)):
        ch += rtof(anc[j - 2], ctl[j], anc[j - 1])
    half = angle.scaled(0.5)
    if kind == "rx":       # controlled Rx(a) = H Rz(a/2) CX Rz(-a/2) CX H   (V18's cr18)
        cr = [_g("h", k0), _g("rz", k0, angle=half), _g("cx", anc[-1], k0),
              _g("rz", k0, angle=half.scaled(-1.0)), _g("cx", anc[-1], k0), _g("h", k0)]
    else:                  # controlled Ry(a) = Ry(a/2) CX Ry(-a/2) CX
        cr = [_g("ry", k0, angle=half), _g("cx", anc[-1], k0),
              _g("ry", k0, angle=half.scaled(-1.0)), _g("cx", anc[-1], k0)]
    return W + flips + ch + cr + inverse(ch) + flips + inverse(W)


# --- S8: the multi-controlled phase e^{i a |0...0><0...0|} of DB-QITE (PLAN §1.5) ---------------------------
# The DB-QITE step needs e^{i r rho_k} = U_k e^{i r |0..0><0..0|} U_k^dagger: a phase on |0...0> of all n system
# qubits, i.e. X on every qubit, the (n-1)-controlled phase C^{n-1}(r1(a)) on qubit n - 1 (closed controls
# 0..n-2), X on every qubit (an open-controlled phase gate). Three forms of the same unitary:
#   mcphase_native   the simulated gate (one native mcr1; the kernel applies r1.ctrl on the qubit register);
#   mcphase_ii       the device count (ii) by S3's Eq. 4.10 rules with d = 1 (no W ladder): V18's chain of n - 2
#                    relative-phase Toffolis on n - 2 clean ancillas (computed and uncomputed) and a controlled
#                    phase CP(a) = CX r1(-a/2)_t CX r1(a/2)_t r1(a/2)_c (2 CNOTs): 6(n - 2) + 2 CNOTs;
#   mcphase_iii      the ancilla-free (iii) list. A phase gate is U(2), not SU(2), so Vale et al.'s linear
#                    C^k(SU(2)) does not apply to it directly; the exact ancilla-free route used here peels one
#                    control at a time: C^m(r1(a)) on (c_1..c_m -> t) = C^m(Rz(a)) on t times C^{m-1}(r1(a/2)) on
#                    (c_1..c_{m-1} -> c_m) (on |1..1>|t> the phases are e^{-+ia/2} e^{ia/2}), down to r1(a / 2^{n-1})
#                    on qubit 0: sum_{j=1}^{n-1} C_Vale(j) CNOTs (quadratic in n; STATUS S8 flags it for Sensei).
def mcphase_native(n: int, angle: Angle) -> list:
    """X on every qubit, C^{n-1}(r1(angle)) on qubit n - 1 (controls 0..n-2), X on every qubit (n >= 2)."""
    if n < 2:
        raise ValueError("the multi-controlled phase needs n >= 2")
    flips = [_g("x", k) for k in range(n)]
    return flips + [Gate("mcr1", tuple(range(n)), angle)] + flips


def mcphase_ii(n: int, angle: Angle) -> list:
    """(ii): the open-controlled phase on n system qubits with V18's ancilla chain (ancillas n .. 2n - 3)."""
    if n < 2:
        raise ValueError("the multi-controlled phase needs n >= 2")
    t = n - 1
    ctl = list(range(n - 1))
    flips = [_g("x", k) for k in range(n)]
    half = angle.scaled(0.5)
    if n == 2:
        chain, last = [], ctl[0]
    else:
        anc = list(range(n, 2 * n - 2))
        chain = rtof(ctl[0], ctl[1], anc[0])
        for j in range(2, len(ctl)):
            chain += rtof(anc[j - 2], ctl[j], anc[j - 1])
        last = anc[-1]
    cp = [_g("cx", last, t), _g("r1", t, angle=half.scaled(-1.0)), _g("cx", last, t), _g("r1", t, angle=half),
          _g("r1", last, angle=half)]
    return flips + chain + cp + inverse(chain) + flips


def mcphase_iii(n: int, angle: Angle) -> list:
    """(iii): the ancilla-free peeling construction (block comment above), Vale's C^j(Rz) at every level."""
    if n < 2:
        raise ValueError("the multi-controlled phase needs n >= 2")
    flips = [_g("x", k) for k in range(n)]
    body = []
    for j in range(n - 1, 0, -1):
        body += mc_su2(list(range(j)), j, "rz", angle.scaled(0.5 ** (n - 1 - j)))
    body.append(_g("r1", 0, angle=angle.scaled(0.5 ** (n - 1))))
    return flips + body + flips


def mcphase_iii_cnots(n: int) -> int:
    """Closed form of `mcphase_iii`'s CNOT count, sum_{j=1}^{n-1} C_Vale(j) (the tests check it against the list)."""
    return int(sum(vale_cnots(j) for j in range(1, n)))


# --- counting helpers -------------------------------------------------------------------------------
def cnot_count(gates) -> int:
    n_native = sum(g.name in NATIVE_MC or g.name in SIM_ONLY for g in gates)
    if n_native:
        raise ValueError("a native multi-controlled (or simulation-only) gate has no CNOT count; count the "
                         "abstract circuit instead")
    return sum(g.name == "cx" for g in gates)


def _rotation_t_cost(angle: Angle, t_syn: int) -> int:
    """T gates of one rotation: a multiple of pi/2 is Clifford (0), an odd multiple of pi/4 is one T,
    anything else (or a circuit parameter) is synthesized to eps_syn: T_syn (V20)."""
    if angle.is_param:
        return t_syn
    r = angle.coef / (np.pi / 4)
    if abs(r - round(r)) < 1e-12:
        return 0 if int(round(r)) % 2 == 0 else 1
    return t_syn


def gate_t_cost(g: Gate, t_syn: int) -> int:
    if g.name in ("t", "tdg"):
        return 1
    if g.name in ROTATIONS or g.name in PHASES:
        return _rotation_t_cost(g.angle, t_syn)
    if g.name in NATIVE_MC:
        raise ValueError("lower native gates before counting T")
    return 0


def t_count(gates, t_syn: int) -> int:
    return int(sum(gate_t_cost(g, t_syn) for g in gates))


def t_depth(gates, t_syn: int) -> int:
    """ASAP T-depth: Cliffords cost 0 but keep their order, a T gate 1, a synthesized rotation T_syn."""
    front: dict = {}
    depth = 0
    for g in gates:
        start = max((front.get(q, 0) for q in g.qubits), default=0)
        end = start + gate_t_cost(g, t_syn)
        for q in g.qubits:
            front[q] = end
        depth = max(depth, end)
    return int(depth)


@lru_cache(maxsize=None)
def canonical_iii(d: int, s: int, kind: str) -> tuple:
    """The (iii) list of a transition with Hamming distance d and |S| = s on a canonical layout
    (pivot 0, D = {0..d-1}, S = {1..s}, u = 0): its counts depend on (d, s, kind) only."""
    ladder = [(0, k) for k in range(1, d)]
    return tuple(transition_iii((), ladder, tuple(range(1, s + 1)), 0, kind, Angle(1.0, 0)))
