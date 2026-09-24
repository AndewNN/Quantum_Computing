"""The DB-QITE circuits of arms A4 / A6 (PLAN §1.5, §1.7, D-2; S8).

The step (PLAN §1.5): |psi_{k+1}> = e^{i r H} e^{i r rho_k} e^{-i r H} |psi_k>, r = sqrt(s), with
e^{i r rho_k} = U_k e^{i r |0..0><0..0|} U_k^dagger; the leading e^{-i r rho_k} of the group commutator acts on |psi_k>
as a global phase and is dropped. So U_{k+1} = e^{i r H} U_k P(r) U_k^dagger e^{-i r H} U_k (operator order) with
P(a) = e^{i a |0..0><0..0|}: three copies of U_k per step, and U_k carries 3^k copies of U_0.

Two step conventions (`STEP_UNITS`; the r of the cost exponential r_H and of the phase r_rho, r_H r_rho = s always):
  "plan"        r_H = r_rho = sqrt(s), with s = g / sigma_H and the un-boosted H in the circuit: PLAN §1.5 as written
                (the default, the arms' `step_units` extra).
  "normalized"  the circuit carries H / sigma_H and r = sqrt(g) (r_H = sqrt(s / sigma_H), r_rho = sqrt(s sigma_H)):
                the example_dbqite.py regime (its sigma_H is 1.27) on every instance. Behind the flag (STATUS S8,
                O-13): with sigma_H ~ 1e-4 (A4, un-boosted H_obj) "plan" gives r = 8..78 rad, far outside the
                group-commutator regime (r << 1 and r sigma_H << 1).
Both give the ideal flow's leading term e^{s [rho, H]}; they differ at O(r^3).

Simulation (`DBCircuit`): the kernel `dbqite_kernel.kernel_dbqite` runs a TOKEN list (`recursion_tokens`) over two base
segments (U_0 and the cost exponential), so the per-call kernel arguments are O(3^k) tokens, not O(3^k |U_0|) gates.
Adjacent cost tokens are merged (e^{-iaH} e^{-ibH} = e^{-i(a+b)H}, exact up to the rounding of a + b) -- a
simulation-only rewrite like `simopt.merge_x`; the counts never merge (below).
  energy(s_list)      <H> by backend.observe (un-boosted H; the greedy grid argmin of the loop, observe only);
  state(s_list)       backend.get_state in classical order (logger, post-run, C1 checks; never the update);
  sample(s_list, S)   backend.sample (the post-run AR_best_S check).
References (CPU): `formula_state` = the exact reflection formula e^{i a rho} = I + (e^{ia} - 1) rho (D-2's C1
reference); `DBCircuit.gates(s_list)` = the unrolled simulation gate list for `compile.npsim` (small n).

Counts (`recursion_counts`; PLAN §1.7, S3 rules, abstract circuit, no cross-segment cancellation):
  c(U_0) = the start circuit (A4: the S3 star prep, `transpile.circuit_counts(circ)["prep"]`; A6: H^n = 0);
  c(U_{k+1}) = 3 c(U_k) + 2 c(cost) + c(P), with c(cost) = 2 CNOTs per ZZ term (+ V20 T) and c(P) from
  `decompose.mcphase_ii` / `mcphase_iii` ((ii) = 6(n - 2) + 2; (iii) ancilla-free, quadratic). T-depths add
  serially (an upper bound). Multi-controlled gates of U_k: 3^k mc(U_0) + (3^k - 1) / 2 phases (Rule C1's matching).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..compile import decompose as dc
from ..compile import tcount
from ..compile.decompose import Angle, Gate

TOK_U0, TOK_U0_INV, TOK_COST, TOK_PHASE = 0, 1, 2, 3
STEP_UNITS = ("plan", "normalized")
GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)            # PLAN §1.5: example_dbqite.py's grid, in units of 1 / sigma_H
COUNT_KEYS = ("cx_ii", "cx_iii", "t_ii", "t_iii", "tdepth_ii", "tdepth_iii")
DB_OPS = ("x", "h", "cx", "rx", "ry", "rz", "mcry", "crz")


# --- the Hamiltonian -------------------------------------------------------------------------------------------------
@dataclass
class DBHam:
    """A diagonal H = const + sum_T c_T Z_T (T = qubit tuples). `ising` (the arms) keeps the instance's `Ising` for the
    observe operator and the counts; the example check builds one from a dense diagonal (`from_diagonal`, which may
    carry 3-body terms)."""
    n: int
    const: float
    terms: tuple                                  # ((qubits, coef), ...), no identity, zeros dropped
    ising: object = None
    _diag: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_ising(cls, H) -> "DBHam":
        i1, c1, a2, b2, c2 = H.terms()
        terms = tuple(((int(i),), float(c)) for i, c in zip(i1, c1)) + tuple(
            ((int(a), int(b)), float(c)) for a, b, c in zip(a2, b2, c2))
        return cls(n=int(H.n), const=float(H.const), terms=terms, ising=H)

    @classmethod
    def from_diagonal(cls, d, tol: float = 1e-14) -> "DBHam":
        """Walsh-Hadamard expansion of a diagonal given on classical indices (x_0 = MSB): every Z-string term."""
        d = np.asarray(d, dtype=np.float64)
        n = int(round(np.log2(d.size)))
        if 1 << n != d.size:
            raise ValueError("the diagonal must have 2^n entries")
        x = np.arange(d.size)
        terms = []
        const = float(d.mean())
        for m in range(1, d.size):
            qs = tuple(k for k in range(n) if (m >> (n - 1 - k)) & 1)
            par = np.array([bin(v & m).count("1") & 1 for v in x])
            c = float(np.mean(d * (1.0 - 2.0 * par)))
            if abs(c) > tol * max(1.0, float(np.abs(d).max())):
                terms.append((qs, c))
        terms.sort(key=lambda t: (len(t[0]), t[0]))
        return cls(n=n, const=const, terms=tuple(terms))

    def diagonal(self) -> np.ndarray:
        """H(x) on all 2^n classical indices (constant included)."""
        if self._diag is None:
            if self.ising is not None:
                self._diag = self.ising.diagonal(np.arange(1 << self.n))
            else:
                x = np.arange(1 << self.n)
                out = np.full(x.size, self.const)
                for qs, c in self.terms:
                    m = 0
                    for k in qs:
                        m |= 1 << (self.n - 1 - k)
                    par = np.array([bin(v & m).count("1") & 1 for v in x])
                    out = out + c * (1.0 - 2.0 * par)
                self._diag = out
        return self._diag

    def cost_terms(self):
        """The `cost.CostTerms` of the <= 2-body part (un-boosted): what the cost segment's counts use."""
        from .cost import CostTerms
        one = [(qs[0], c) for qs, c in self.terms if len(qs) == 1]
        two = [(qs[0], qs[1], c) for qs, c in self.terms if len(qs) == 2]
        return CostTerms(n=self.n, idx_1=tuple(i for i, _ in one), coeff_1=tuple(c for _, c in one),
                         idx_2a=tuple(a for a, _, _ in two), idx_2b=tuple(b for _, b, _ in two),
                         coeff_2=tuple(c for _, _, c in two))

    def cost_gates_sim(self) -> list:
        """exp(-i a H) (no constant: its phases cancel between e^{-irH} and e^{irH}) as simulation gates with angle
        coef * a (pidx 0): the <= 2-body part as `cost.cost_gates_sim` (crz per ZZ term + one merged rz per qubit), each
        k >= 3 body term as a CNOT ladder around rz(2 c a) (the example's 3-qubit diagonal only)."""
        from .cost import cost_gates_sim
        out = cost_gates_sim(self.cost_terms(), 0)
        for qs, c in self.terms:
            if len(qs) < 3:
                continue
            lad = [Gate("cx", (q, qs[-1])) for q in qs[:-1]]
            out += lad + [Gate("rz", (qs[-1],), Angle(2.0 * c, 0))] + lad[::-1]
        return out

    @property
    def max_body(self) -> int:
        return max((len(qs) for qs, _ in self.terms), default=0)

    def op(self):
        """The cudaq SpinOperator of H (un-boosted, constant included), built lazily through the backend."""
        from ..sim import backend
        if self.ising is not None:
            return backend.ising_op(self.ising, 1.0)
        return backend.zterm_op(self.const, self.terms, self.n)


def sigma_H(diag, idx=None) -> float:
    """PLAN §1.5: the standard deviation (population, ddof 0) of H's diagonal over the strings the arm can reach
    (the sector for A4, all 2^n for A6)."""
    d = np.asarray(diag, dtype=np.float64)
    return float(np.std(d if idx is None else d[np.asarray(idx, dtype=np.int64)]))


def r_pair(s: float, units: str, sigma: float) -> tuple:
    """(r_H, r_rho) of one step of size s (module doc)."""
    s = float(s)
    if units == "plan":
        r = float(np.sqrt(s))
        return r, r
    if units == "normalized":
        return float(np.sqrt(s / sigma)), float(np.sqrt(s * sigma))
    raise ValueError(f"step_units must be one of {STEP_UNITS}")


# --- tokens ----------------------------------------------------------------------------------------------------------
def _inverse_tokens(toks: list) -> list:
    return [(TOK_U0_INV if k == TOK_U0 else TOK_U0 if k == TOK_U0_INV else k,
             0.0 if k in (TOK_U0, TOK_U0_INV) else -a) for k, a in reversed(toks)]


def merge_cost(toks: list) -> list:
    """Adjacent cost tokens -> one (angles summed in token order); a merged angle of exactly 0 is dropped."""
    out: list = []
    for k, a in toks:
        if k == TOK_COST and out and out[-1][0] == TOK_COST:
            out[-1] = (TOK_COST, out[-1][1] + a)
            if out[-1][1] == 0.0:
                out.pop()
        else:
            out.append((k, a))
    return out


def step_tokens(base: list, r_H: float, r_rho: float, merge: bool = True) -> list:
    """U_{k+1} from U_k's tokens (circuit order): U_k, cost(+r_H), U_k^dagger, P(r_rho), U_k, cost(-r_H)."""
    toks = list(base) + [(TOK_COST, float(r_H))] + _inverse_tokens(base) + [(TOK_PHASE, float(r_rho))] + list(base) \
        + [(TOK_COST, -float(r_H))]
    return merge_cost(toks) if merge else toks


def recursion_tokens(pairs, merge: bool = True) -> list:
    """Tokens of U_k for the steps' (r_H, r_rho) pairs, U_0 = [(TOK_U0, 0)]."""
    toks = [(TOK_U0, 0.0)]
    for rH, rr in pairs:
        toks = step_tokens(toks, rH, rr, merge)
    return toks


# --- the circuit -----------------------------------------------------------------------------------------------------
def _pack(g: Gate, slot: int, ctl_start: int) -> int:
    from .program import OPCODES
    controls = g.qubits[:-1]
    return OPCODES[g.name] + 16 * (int(g.qubits[-1]) + 32 * (len(controls) + 32 * (slot + 4 * ctl_start)))


@dataclass
class DBProgram:
    n: int
    code: list
    coef: list
    ctl: list
    seg: list
    n_u0: int
    n_cost: int


def encode_db(start: list, cost: list, n: int) -> DBProgram:
    """Pack U_0 (constant angles) and the cost segment (angles coef * a, pidx 0) for `kernel_dbqite`."""
    from .program import MAX_CONTROLS, MAX_QUBITS_LAYERED
    if n > MAX_QUBITS_LAYERED or n < 2:
        raise ValueError(f"n = {n} outside 2..{MAX_QUBITS_LAYERED}")
    code, coef, ctl = [], [], []
    for seg_idx, gates in enumerate((start, cost)):
        for g in gates:
            if g.name not in DB_OPS:
                raise ValueError(f"{g.name} is not a DB-QITE kernel gate")
            controls = g.qubits[:-1]
            if len(controls) > MAX_CONTROLS:
                raise ValueError(f"{len(controls)} controls > {MAX_CONTROLS}")
            if any(q >= n for q in g.qubits):
                raise ValueError(f"qubit out of range in {g}")
            if g.name in ("cx", "crz") and len(controls) != 1:
                raise ValueError(f"{g.name} needs one control")
            if g.name == "mcry" and not controls:
                raise ValueError("mcry needs controls")
            param = g.angle is not None and g.angle.pidx >= 0
            if param and (seg_idx == 0 or g.angle.pidx != 0):
                raise ValueError("U_0 gates take constant angles; cost gates bind pidx 0 (the token's angle)")
            code.append(_pack(g, 1 if param else 0, len(ctl)))
            ctl.extend(int(c) for c in controls)
            coef.append(0.0 if g.angle is None else float(g.angle.coef))
    ns, nc = len(start), len(cost)
    return DBProgram(n=n, code=code, coef=coef, ctl=ctl or [0], seg=[0, ns, ns, ns + nc], n_u0=ns, n_cost=nc)


class DBCircuit:
    """One arm's recursion: start circuit U_0, Hamiltonian, step convention (module doc). Params of every call =
    the list of chosen step sizes s_1..s_k (un-boosted units of H), the trajectory's `params` row."""

    def __init__(self, n: int, start: list, ham: DBHam, *, kind: str, sigma: float, units: str = "plan",
                 start_counts: dict | None = None, start_mc: int = 0, start_abstract: list | None = None,
                 meta: dict | None = None):
        if units not in STEP_UNITS:
            raise ValueError(f"step_units must be one of {STEP_UNITS}")
        self.n, self.kind, self.units, self.sigma = int(n), kind, units, float(sigma)
        self.ham = ham
        self.start = list(start)
        self.cost = ham.cost_gates_sim()
        self.prog = encode_db(self.start, self.cost, self.n)
        self.start_counts = start_counts
        self.start_mc = int(start_mc)
        self.start_abstract = start_abstract
        self.meta = dict(meta or {})
        self._op = None
        self._cache: tuple = ((), [(TOK_U0, 0.0)])

    # parameters -> tokens
    def pairs(self, s_list) -> list:
        return [r_pair(s, self.units, self.sigma) for s in s_list]

    def tokens(self, s_list) -> list:
        """Tokens of U_k for s_1..s_k; the last prefix is cached (the grid's 7 candidates share it)."""
        s = tuple(float(v) for v in s_list if np.isfinite(v))
        pre, toks = self._cache
        if len(pre) > len(s) or s[:len(pre)] != pre:
            pre, toks = (), [(TOK_U0, 0.0)]
        while len(pre) < len(s) - 1:
            toks = step_tokens(toks, *r_pair(s[len(pre)], self.units, self.sigma))
            pre = s[:len(pre) + 1]
        self._cache = (pre, toks)
        if len(pre) == len(s):
            return toks
        return step_tokens(toks, *r_pair(s[-1], self.units, self.sigma))

    def args(self, s_list) -> tuple:
        toks = self.tokens(s_list)
        p = self.prog
        return (p.n, p.code, p.coef, p.ctl, p.seg, [int(k) for k, _ in toks], [float(a) for _, a in toks])

    @property
    def op(self):
        if self._op is None:
            self._op = self.ham.op()
        return self._op

    # the simulator (backend only)
    def energy(self, s_list) -> float:
        """<psi_k| H |psi_k> (un-boosted) by observe: the loop's energy."""
        from ..sim import backend
        from .dbqite_kernel import kernel_dbqite
        return backend.observe(kernel_dbqite, self.op, *self.args(s_list))

    def state(self, s_list) -> np.ndarray:
        """Classical-order statevector of U_k|0> (logger / post-run / C1 checks only)."""
        from ..sim import backend
        from .dbqite_kernel import kernel_dbqite
        return backend.get_state_classical(kernel_dbqite, self.n, *self.args(s_list))

    def sample(self, s_list, shots: int = 1000) -> dict:
        from ..sim import backend
        from .dbqite_kernel import kernel_dbqite
        return backend.sample(kernel_dbqite, *self.args(s_list), shots_count=int(shots))

    # CPU references
    def gates(self, s_list) -> list:
        """The unrolled simulation gate list of U_k (what the kernel applies), for `compile.npsim` at small n."""
        out = []
        for k, a in self.tokens(s_list):
            if k == TOK_U0:
                out += self.start
            elif k == TOK_U0_INV:
                out += dc.inverse(self.start)
            elif k == TOK_COST:
                out += [g if g.angle is None or g.angle.pidx < 0 else Gate(g.name, g.qubits, Angle(g.angle.coef * a))
                        for g in self.cost]
            else:
                out += dc.mcphase_native(self.n, Angle(a))
        return out

    def formula_state(self, s_list, psi0: np.ndarray | None = None) -> np.ndarray:
        """psi_k by the exact reflection formula from psi_0 (default: U_0|0> by npsim)."""
        if psi0 is None:
            psi0 = start_state(self.start, self.n)
        return formula_state(self.ham.diagonal(), psi0, self.pairs([v for v in s_list if np.isfinite(v)]))

    # sizes and counts
    def sim_gates(self, s_list) -> int:
        n_phase = 2 * self.n + 1
        return int(sum(self.prog.n_u0 if k in (TOK_U0, TOK_U0_INV) else self.prog.n_cost if k == TOK_COST else n_phase
                       for k, _ in self.tokens(s_list)))

    def phase_counts(self) -> dict:
        return mcphase_counts(self.n)

    def cost_counts(self) -> dict:
        from ..compile.transpile import cost_counts
        if self.ham.max_body > 2:
            raise ValueError("counts are defined for Ising (<= 2-body) Hamiltonians only")
        c = cost_counts(self.ham.cost_terms())
        return {k: int(c[k]) for k in COUNT_KEYS} | {"n_zz": int(c["n_zz"]), "n_rot": int(c["n_rot"])}

    def counts(self, k_max: int) -> dict:
        """Per-circuit counts of U_0..U_{k_max} (`recursion_counts`)."""
        if self.start_counts is None:
            raise ValueError("no start-circuit counts")
        return recursion_counts(self.start_counts, self.cost_counts(), self.phase_counts(), self.start_mc, k_max)


def start_state(start: list, n: int) -> np.ndarray:
    from ..compile import npsim
    psi = npsim.columns(n, [0])
    npsim.apply(start, psi)
    return npsim.flat(psi)[:, 0].copy()


def formula_step(diag: np.ndarray, psi: np.ndarray, r_H: float, r_rho: float) -> np.ndarray:
    """e^{i r_H H} (I + (e^{i r_rho} - 1) |psi><psi|) e^{-i r_H H} |psi> (the compiled step, exact)."""
    ph = np.exp(-1j * r_H * diag)
    phi = ph * psi
    phi = phi + (np.exp(1j * r_rho) - 1.0) * np.vdot(psi, phi) * psi
    return np.conj(ph) * phi


def formula_state(diag, psi0, pairs) -> np.ndarray:
    psi = np.asarray(psi0, dtype=np.complex128).copy()
    d = np.asarray(diag, dtype=np.float64)
    for rH, rr in pairs:
        psi = formula_step(d, psi, rH, rr)
    return psi


# --- counts ----------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def mcphase_counts(n: int, ts: int | None = None) -> dict:
    """The open-controlled phase on n qubits (`decompose.mcphase_ii` / `mcphase_iii`): (ii) CNOTs from V18's list and
    T by the V20 rules (2 (n - 2) Toffolis at c_Tof / d_Tof, 3 synthesized rotations, T-depth 2 T_syn); (iii) from the
    ancilla-free list."""
    ts = tcount.t_syn() if ts is None else ts
    a = Angle(1.0, 0)
    ii = dc.mcphase_ii(n, a)
    iii = dc.mcphase_iii(n, a)
    tof = max(0, 2 * (n - 2))
    return {"cx_ii": dc.cnot_count(ii), "cx_iii": dc.cnot_count(iii),
            "t_ii": tof * tcount.C_TOF + 3 * ts, "tdepth_ii": tof * tcount.D_TOF + 2 * ts,
            "t_iii": dc.t_count(iii, ts), "tdepth_iii": dc.t_depth(iii, ts)}


def recursion_counts(start: dict, cost: dict, phase: dict, start_mc: int, k_max: int) -> dict:
    """c(U_k), k = 0..k_max: c(U_{k+1}) = 3 c(U_k) + 2 c(cost) + c(P) per key; plus the structural numbers."""
    out = {key: [int(start.get(key, 0))] for key in COUNT_KEYS}
    for _ in range(int(k_max)):
        for key in COUNT_KEYS:
            out[key].append(3 * out[key][-1] + 2 * int(cost[key]) + int(phase[key]))
    k = np.arange(int(k_max) + 1)
    out["u0_copies"] = [int(3 ** j) for j in k]
    out["cost_layers"] = [int(3 ** j - 1) for j in k]
    out["phases"] = [int((3 ** j - 1) // 2) for j in k]
    out["mc_gates"] = [int(3 ** j * start_mc + (3 ** j - 1) // 2) for j in k]
    return out


def charged_series(per_circuit: dict, grid_size: int, T: int) -> dict:
    """Cumulative charges up to psi_t, t = 0..T: step k (psi_{k-1} -> psi_k) runs |grid| energy circuits of U_k."""
    t = np.arange(T + 1)
    out = {"circuits_charged": (t * int(grid_size)).astype(np.int64)}
    for key, name in (("cx_ii", "g2q_ii"), ("cx_iii", "g2q_iii")):
        per = np.asarray(per_circuit[key][:T + 1], dtype=np.int64)
        step = np.r_[0, grid_size * per[1:]]
        out[name] = np.cumsum(step).astype(np.int64)
    return out
