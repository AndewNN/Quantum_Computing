"""The linear-in-n preserving mixer and the star start state (PLAN §2.1-§2.4, D-1, D-9).

Descriptors (CPU only; cudaq is imported lazily by the kernel builders at the bottom):

* A **transition** of a kept pair (u, v): difference set D = {k : u_k != v_k}, pivot k0 = min D,
  x-mask = {k : u_k = 1}, CNOT ladder (k0 -> k) for k in D \\ {k0}, the open-control set S, and the
  rotation (Rx for the mixer, Ry for the start state). W = X on the x-mask, then the ladder, maps
  |u> -> |0...0> and |v> -> |e_k0>; W† C^{0}_S(R(angle))_{k0} W acts as exp(-i (angle/2) (|u><v| + h.c.))
  (Rx) or as the real rotation |u> -> cos |u> + sin |v> (Ry) on span{|u>, |v>}, and as the identity on
  every string whose W-image has a 1 on S.
* **S = a greedy minimum hitting set** of the W-images of the strings that must stay untouched: every
  other kept string for a mixer transition; the strings that already carry amplitude for a start-state
  step. Greedy picks the qubit (never k0) hitting most unhit images, ties to the lowest index.
  |S| <= min(n - 1, K - 2).
* The **sector order** (D-9, open): `ring_order="lex"` is the sector file's `idx` (ascending classical
  index = lexicographic bitstrings, PLAN §2.2 as written); `"rank"` is its `rank_idx` (GA ranking,
  what the completed work used). Every edge list and the star's centre u_1 follow this order.
* **Mixer layer** U_M(beta) := prod_{(i,j) in E} exp(-i beta A_{u_i u_j}) in a fixed edge order (a
  first-order product formula; a definition): ring = (i, i+1 mod K) for i = 0..K-1 (K = 2: one edge),
  complete = (i, j), i < j, lexicographic. `symmetrized=True` (second order, not used in the sweep) is
  the forward product at beta/2 followed by the reverse product at beta/2.
* **Star start state**: X on the ones of u_1, then for j = 2..K the pair (u_1, u_j) rotated by
  Ry(2 phi_j), sin phi_j = 1/sqrt(K - j + 2): the uniform superposition over the kept strings with all
  amplitudes +1/sqrt(K).

The dense references (`dense_layer`, `uniform_state`) live in the K-dimensional sector basis of the
same order. The C1 operator-level check compares the circuit against `dense_layer` (not expm(H_M)).

Simulation (D-1): each multi-controlled rotation is ONE native multi-controlled gate. `a1_gates` lays a
whole circuit out as one gate list; two engines run it: the interpreter kernel `interp.kernel_program`
(default; compiled once per process, `program.py`) and the builder API `build_kernel` (PLAN §2.4's route,
one `cudaq.make_kernel` per circuit with `crx` / `cry` on runtime control lists). `simulate_decomposed`
/ `decomposed=True` emits the explicit (iii) lists instead, for the C1 cross-check at n <= 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..compile.decompose import Angle, transition_iii, transition_native
from ..sectors.control import edges_for

RING_ORDERS = ("lex", "rank")
CONNECTIVITIES = ("ring", "complete")


# --- bit helpers (classical index, qubit k = bit n-1-k) --------------------------------------------
def _bit(x, k: int, n: int):
    return (np.asarray(x, dtype=np.int64) >> (n - 1 - k)) & 1


def _mask(qubits, n: int) -> int:
    m = 0
    for k in qubits:
        m |= 1 << (n - 1 - int(k))
    return m


def diff_set(u: int, v: int, n: int) -> tuple:
    return tuple(k for k in range(n) if ((u ^ v) >> (n - 1 - k)) & 1)


def w_image(x, u: int, v: int, n: int) -> np.ndarray:
    """W applied to classical indices x (vectorised): XOR u, then CNOT(k0 -> k) for k in D \\ {k0}."""
    D = diff_set(u, v, n)
    y = np.asarray(x, dtype=np.int64) ^ int(u)
    if len(D) > 1:
        b0 = _bit(y, D[0], n)
        y = y ^ (b0 * _mask(D[1:], n))
    return y


def hitting_set(images, k0: int, n: int) -> tuple:
    """Greedy minimum hitting set: qubits (never k0) such that every image has a 1 on one of them.
    Ties go to the lowest qubit index, so the result is deterministic."""
    left = np.asarray(images, dtype=np.int64).reshape(-1)
    S = []
    cand = [k for k in range(n) if k != k0]
    while left.size:
        counts = np.array([int(_bit(left, k, n).sum()) for k in cand])
        if counts.max() == 0:
            raise ValueError("an image has no 1 outside the pivot (it is |u> or |v>)")
        k = cand[int(np.argmax(counts))]            # argmax returns the first (lowest) maximum
        S.append(k)
        left = left[_bit(left, k, n) == 0]
    return tuple(sorted(S))


# --- descriptors ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Transition:
    """One two-level rotation between kept strings u = order[i] and v = order[j] (PLAN §2.1)."""

    n: int
    u: int
    v: int
    edge: tuple                  # (i, j): positions in the sector order
    D: tuple                     # difference set, ascending
    k0: int                      # pivot = min D
    xmask: tuple                 # qubits with u_k = 1
    S: tuple                     # open-control set, ascending
    kind: str                    # "rx" (mixer) or "ry" (start state)
    coef: float                  # rotation angle = coef * beta (mixer) or the constant coef (start state)
    param: bool                  # True: the angle scales a circuit parameter

    @property
    def d(self) -> int:
        return len(self.D)

    @property
    def ladder(self) -> tuple:
        return tuple((self.k0, k) for k in self.D[1:])

    def angle(self, pidx: int = -1) -> Angle:
        return Angle(self.coef, pidx if self.param else -1)

    def native_gates(self, pidx: int = -1) -> list:
        return transition_native(self.xmask, self.ladder, self.S, self.k0, self.kind, self.angle(pidx))

    def iii_gates(self, pidx: int = -1) -> list:
        return transition_iii(self.xmask, self.ladder, self.S, self.k0, self.kind, self.angle(pidx))


def make_transition(order: np.ndarray, n: int, i: int, j: int, protect, kind: str, coef: float,
                    param: bool) -> Transition:
    order = np.asarray(order, dtype=np.int64)
    u, v = int(order[i]), int(order[j])
    D = diff_set(u, v, n)
    if not D:
        raise ValueError("u == v")
    k0 = D[0]
    prot = np.asarray([order[p] for p in protect], dtype=np.int64)
    S = hitting_set(w_image(prot, u, v, n), k0, n) if prot.size else ()
    xmask = tuple(k for k in range(n) if (u >> (n - 1 - k)) & 1)
    return Transition(n=n, u=u, v=v, edge=(int(i), int(j)), D=D, k0=k0, xmask=xmask, S=S, kind=kind,
                      coef=float(coef), param=param)


def sector_order(sector, ring_order: str = "lex") -> np.ndarray:
    """The kept strings in circuit order (D-9): lex = `idx` (sorted), rank = `rank_idx`."""
    if ring_order == "lex":
        out = np.asarray(sector.idx, dtype=np.int64)
    elif ring_order == "rank":
        out = np.asarray(sector.rank_idx, dtype=np.int64)
    else:
        raise ValueError(f"ring_order must be one of {RING_ORDERS}, got {ring_order!r}")
    if len(set(out.tolist())) != out.size:
        raise ValueError("repeated kept string")
    return out


def layer_edges(K: int, connectivity: str, symmetrized: bool = False) -> list:
    """[(i, j, coef)] in application order; the angle of each transition is coef * 2 beta."""
    E = edges_for(connectivity, K)
    if not symmetrized:
        return [(i, j, 1.0) for i, j in E]
    return [(i, j, 0.5) for i, j in E] + [(i, j, 0.5) for i, j in reversed(E)]


def mixer_layer(order, n: int, connectivity: str, symmetrized: bool = False) -> list:
    """One mixer layer: transitions with rotation angle Rx(2 coef beta), i.e. exp(-i coef beta A)."""
    K = len(order)
    out = []
    for i, j, c in layer_edges(K, connectivity, symmetrized):
        protect = [p for p in range(K) if p not in (i, j)]
        out.append(make_transition(order, n, i, j, protect, "rx", 2.0 * c, True))
    return out


def star_phis(K: int) -> np.ndarray:
    """phi_j for j = 2..K: sin phi_j = 1/sqrt(K - j + 2)."""
    j = np.arange(2, K + 1)
    return np.arcsin(1.0 / np.sqrt(K - j + 2.0))


def star_prep(order, n: int) -> tuple:
    """(x-mask of u_1, [K - 1 Ry transitions]) of the star start state (PLAN §2.3)."""
    order = np.asarray(order, dtype=np.int64)
    K = order.size
    u1 = int(order[0])
    xm = tuple(k for k in range(n) if (u1 >> (n - 1 - k)) & 1)
    steps = []
    for j, phi in zip(range(1, K), star_phis(K)):          # 0-based position j = 1..K-1
        protect = list(range(1, j))                          # u_2 .. u_{j-1} already carry amplitude
        steps.append(make_transition(order, n, 0, j, protect, "ry", 2.0 * float(phi), False))
    return xm, steps


@dataclass(frozen=True)
class PreservingCircuit:
    """Everything the kernel builder and the counter need for one sector and one mixer shape."""

    n: int
    order: np.ndarray
    connectivity: str
    ring_order: str
    symmetrized: bool
    layer: list                      # one mixer layer (angles in units of that layer's beta)
    prep_xmask: tuple
    prep: list                       # K - 1 start-state transitions
    meta: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return int(self.order.size)


def build_circuit(sector, connectivity: str = "ring", ring_order: str = "lex",
                  symmetrized: bool = False) -> PreservingCircuit:
    """From a `gsp.sectors.select.Sector` (or anything with n, idx, rank_idx)."""
    if connectivity not in CONNECTIVITIES:
        raise ValueError(connectivity)
    n = int(sector.n)
    order = sector_order(sector, ring_order)
    return circuit_from_order(order, n, connectivity, ring_order, symmetrized)


def circuit_from_order(order, n: int, connectivity: str = "ring", ring_order: str = "lex",
                       symmetrized: bool = False) -> PreservingCircuit:
    order = np.asarray(order, dtype=np.int64)
    xm, prep = star_prep(order, n)
    return PreservingCircuit(n=n, order=order, connectivity=connectivity, ring_order=ring_order,
                             symmetrized=symmetrized, layer=mixer_layer(order, n, connectivity, symmetrized),
                             prep_xmask=xm, prep=prep)


# --- gate lists of whole circuits (numpy side) -------------------------------------------------------
def prep_gates(circ: PreservingCircuit, decomposed: bool = False) -> list:
    from ..compile.decompose import Gate
    out = [Gate("x", (k,)) for k in circ.prep_xmask]
    for tr in circ.prep:
        out += tr.iii_gates() if decomposed else tr.native_gates()
    return out


def layer_gates(circ: PreservingCircuit, pidx: int, decomposed: bool = False) -> list:
    out = []
    for tr in circ.layer:
        out += tr.iii_gates(pidx) if decomposed else tr.native_gates(pidx)
    return out


# --- dense references in the K x K sector basis -------------------------------------------------------
def two_level(K: int, i: int, j: int, theta: float, kind: str = "rx") -> np.ndarray:
    G = np.eye(K, dtype=np.complex128)
    c, s = np.cos(theta), np.sin(theta)
    if kind == "rx":            # exp(-i theta (|i><j| + |j><i|))
        G[i, i] = G[j, j] = c
        G[i, j] = G[j, i] = -1j * s
    else:                       # |i> -> c|i> + s|j>
        G[i, i] = G[j, j] = c
        G[j, i], G[i, j] = s, -s
    return G


def dense_layer(K: int, connectivity: str, beta: float, symmetrized: bool = False) -> np.ndarray:
    """The ordered product prod_E exp(-i beta A_e) in K x K (the C1 reference block, PLAN §2.2)."""
    U = np.eye(K, dtype=np.complex128)
    for i, j, c in layer_edges(K, connectivity, symmetrized):
        U = two_level(K, i, j, c * beta) @ U
    return U


def ring_hamiltonian(K: int, connectivity: str = "ring") -> np.ndarray:
    H = np.zeros((K, K))
    for i, j in edges_for(connectivity, K):
        H[i, j] = H[j, i] = 1.0
    return H


def uniform_state(K: int) -> np.ndarray:
    return np.full(K, 1.0 / np.sqrt(K), dtype=np.complex128)


# --- whole A1-type circuits as one gate list ----------------------------------------------------------
def a1_gates(circ: PreservingCircuit, L: int = 1, *, cost=None, prep: bool = True, input_idx: int | None = None,
             decomposed: bool = False) -> list:
    """[X on the ones of input_idx] [star prep] then L x ([cost layer(gamma_l)] mixer layer(beta_l)).

    Parameter layout (the `Angle.pidx` of the list): params = [gamma_1..gamma_L, beta_1..beta_L] when a
    cost layer is present, else [beta_1..beta_L] (the completed work's thetas order).
    cost: a `gsp.circuits.cost.CostTerms` (already boosted) or None (mixer only).
    decomposed: the explicit (iii) lists instead of native multi-controlled gates (`simulate_decomposed`).
    """
    from ..compile.decompose import Gate
    from .cost import cost_gates

    n_gamma = L if cost is not None else 0
    out = []
    if input_idx is not None:
        out += [Gate("x", (k,)) for k in range(circ.n) if (int(input_idx) >> (circ.n - 1 - k)) & 1]
    if prep:
        out += prep_gates(circ, decomposed=decomposed)
    for ell in range(L):
        if cost is not None:
            out += cost_gates(cost, pidx=ell)
        out += layer_gates(circ, n_gamma + ell, decomposed=decomposed)
    return out


def n_angles(L: int, cost=None) -> int:
    return (2 * L) if cost is not None else L


# --- CUDA-Q kernels ----------------------------------------------------------------------------------
# Two engines run the same gate list (tests check they agree):
#   "interp"  (default) the data-driven `interp.kernel_program`, compiled once per process;
#   "builder" one `cudaq.make_kernel` per circuit (PLAN §2.4's route), angles as a runtime list.
@dataclass
class BuiltKernel:
    """A builder kernel and its parameter layout: params = [gamma_1..gamma_L, beta_1..beta_L]
    (gammas only when a cost layer is present), then `n_input` input angles (ry(pi * bit) per qubit)."""

    kernel: object
    n: int
    L: int
    n_gamma: int
    n_beta: int
    n_input: int
    n_gates: int = 0

    @property
    def n_params(self) -> int:
        return self.n_gamma + self.n_beta + self.n_input

    def params(self, gammas=(), betas=(), input_idx: int | None = None) -> list:
        out = [float(g) for g in gammas] + [float(b) for b in betas]
        if len(out) != self.n_gamma + self.n_beta:
            raise ValueError("wrong number of angles")
        if self.n_input:
            bits = [0] * self.n if input_idx is None else [int((input_idx >> (self.n - 1 - k)) & 1)
                                                           for k in range(self.n)]
            out += [np.pi * b for b in bits]
        return out


def _emit(kern, q, p, gates):
    """Emit a gate list into a builder kernel. Angles with pidx >= 0 become coef * p[pidx]."""
    def ang(a):
        return float(a.coef) if a.pidx < 0 else p[a.pidx] * float(a.coef)

    for g in gates:
        nm, qs = g.name, g.qubits
        if nm in ("x", "h", "s", "sdg", "t", "tdg"):
            getattr(kern, nm)(q[qs[0]])
        elif nm == "cx":
            kern.cx(q[qs[0]], q[qs[1]])
        elif nm in ("rx", "ry", "rz"):
            getattr(kern, nm)(ang(g.angle), q[qs[0]])
        elif nm == "mcrx":
            kern.crx(ang(g.angle), [q[c] for c in qs[:-1]], q[qs[-1]])
        elif nm == "mcry":
            kern.cry(ang(g.angle), [q[c] for c in qs[:-1]], q[qs[-1]])
        else:
            raise ValueError(nm)
    return len(gates)


def build_kernel(circ: PreservingCircuit, L: int = 1, *, cost=None, prep: bool = True,
                 input_bits: bool = False, simulate_decomposed: bool = False) -> BuiltKernel:
    """Builder-API kernel of `a1_gates(circ, L, cost=..., prep=...)` (PLAN §2.4's route).
    input_bits: prepend ry(p[..]) on every qubit so a basis input |x> is chosen at run time
    (ry(pi)|0> = |1> up to cos(pi/2) = 6e-17)."""
    import cudaq  # noqa: WPS433 (lazy: the descriptors above are CPU-only)

    n_gamma = L if cost is not None else 0
    kern, p = cudaq.make_kernel(list)
    q = kern.qalloc(circ.n)
    n_input = circ.n if input_bits else 0
    base_in = n_gamma + L
    for k in range(n_input):
        kern.ry(p[base_in + k], q[k])
    count = n_input + _emit(kern, q, p, a1_gates(circ, L, cost=cost, prep=prep, decomposed=simulate_decomposed))
    return BuiltKernel(kernel=kern, n=circ.n, L=L, n_gamma=n_gamma, n_beta=L, n_input=n_input, n_gates=count)
