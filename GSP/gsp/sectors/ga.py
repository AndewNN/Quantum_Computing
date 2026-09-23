"""Numpy genetic algorithm for sector selection, both fitness rules (PLAN §5 S2, D-5, D-6).

A reimplementation of the completed work's pybind11 `ga_solver`
(`MyLib/Genetic/genetic_solver.cpp`, commit 8634ca8 "Obj focus"; the call site is
`CUDA/PO_new_ApproxRatio.py:495-560`). Operators, hyperparameters and ranking are the C++ ones:

  population   N_pop random chromosomes, every bit uniform in {0, 1}
  generation   evaluate the whole population, sort it (total order below), then build the next one:
               the `elitism` best are copied unchanged; the rest come in pairs, each parent by a
               tournament of `tournament` individuals drawn uniformly with replacement (the best
               wins), single-point crossover with probability r_c at a point uniform in [1, n-1]
               (child1 = p1[:c] + p2[c:], child2 = p2[:c] + p1[c:]), then independent bit-flip
               mutation with probability r_m = 1.5 / n on every bit of both children
  end          after G generations: evaluate, sort, return the top-K *distinct* chromosomes
Table 4.1 (proposal §4.2): N_pop = 2000, G = 35, r_c = 0.85, r_m = 1.5/n, t = 5, elitism 2.

Fitness rules (the C++ `better` / sort comparator):
  violation    F(x) = (P_bb . x - 1)^2 (Eq. 4.x of §4.2; the C++ ranks (cost - B)^2 = B^2 F(x)).
  objective    D-6, lexicographic: strings inside the band |P_bb . x - 1| <= eps come first, ranked by
               H_obj(x) = -(x^T QU_obj x) (un-boosted, the confined arms' cost); strings outside are
               ranked by F(x). No weight between the terms. Same semantics as the old `--GA_OBJ`.
Ties (measure zero except identical chromosomes) are broken as in the C++: by the budget used
(P_bb . x), then by the chromosome read as a bit vector.

Chromosome layout. The C++ chromosome is asset-major with the **most significant** bit of each
asset's quantity first (`quantity = (quantity << 1) | bit`); the old call site reversed every
asset block to get the qubit string. Single-point crossover acts on chromosome positions, so the
layout matters: this GA keeps the C++ layout and maps it to the harness bit order (qubit i = x_i,
po_normalize weight 2^j on qubit n_qs[a] + j) with a fixed permutation (`chromosome_perm`).

Randomness: `numpy.random.default_rng(seed)` (PCG64) with the GA seed from the seed table. The
stream differs from the C++ mt19937, so runs agree with the pybind module in distribution, not
bit for bit (S2 validates the top-K lists against brute force and against the pybind module).

Counters (for WP5): wall-clock and CPU time of the run, fitness evaluations ((G + 1) N_pop: the
whole population is evaluated every generation, elites included, as in the C++), violation and
objective evaluations separately, distinct strings ever evaluated, and the evaluation index at
which each string was first evaluated (so "evaluations to hit" any string is a lookup).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

RULES = ("violation", "objective")
MAX_TRACK_N = 24          # first-evaluation bookkeeping uses a 2^n array


@dataclass(frozen=True)
class GAParams:
    """Table 4.1 of the proposal (§4.2); r_m = mutation_scale / n."""

    population: int = 2000
    generations: int = 35
    crossover_rate: float = 0.85
    mutation_scale: float = 1.5
    tournament: int = 5
    elitism: int = 2

    def mutation_rate(self, n: int) -> float:
        return self.mutation_scale / n

    def as_dict(self) -> dict:
        return asdict(self)


TABLE_4_1 = GAParams()


def chromosome_perm(n_max) -> np.ndarray:
    """perm[j] = the harness bit (qubit) index held by chromosome position j.

    Asset a occupies positions s .. s + L - 1 in both layouts (s = sum of the earlier lengths);
    chromosome position s + b carries weight 2^(L-1-b) (C++), which is qubit s + L - 1 - b
    (po_normalize puts weight 2^j on qubit s + j).
    """
    n_max = [int(L) for L in np.asarray(n_max).ravel()]
    perm = []
    s = 0
    for L in n_max:
        perm.extend(s + L - 1 - b for b in range(L))
        s += L
    return np.asarray(perm, dtype=np.int64)


def _weights(n: int) -> np.ndarray:
    return np.int64(1) << np.arange(n - 1, -1, -1, dtype=np.int64)


@dataclass(frozen=True)
class Problem:
    """What the GA and the brute-force reference rank: one draw (violation) or one instance (objective)."""

    rule: str
    n: int
    P_bb: np.ndarray            # normalized prices per qubit (P_bb . x = cost / B)
    eps: float
    QU_obj: np.ndarray | None   # the MAX-problem objective QUBO (objective rule only)
    perm: np.ndarray            # chromosome position -> qubit index
    scope_id: str = ""

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if self.rule == "objective" and self.QU_obj is None:
            raise ValueError("the objective rule needs QU_obj")
        if sorted(self.perm.tolist()) != list(range(self.n)):
            raise ValueError("perm is not a permutation of the qubits")

    # --- layouts -----------------------------------------------------------------------------
    def chrom_to_x(self, C: np.ndarray) -> np.ndarray:
        X = np.empty_like(C)
        X[:, self.perm] = C
        return X

    def x_to_chrom(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.perm]

    # --- fitness -----------------------------------------------------------------------------
    def evaluate(self, X: np.ndarray) -> dict:
        """Fitness fields of the rows of X (harness bit order, 0/1)."""
        Xf = X.astype(np.float64)
        cost = Xf @ self.P_bb                         # = sum quantity * price / B
        dev = cost - 1.0
        out = {"cost": cost, "viol": dev * dev, "in_band": np.abs(dev) <= self.eps}
        if self.rule == "objective":
            XQ = Xf @ self.QU_obj
            out["obj"] = -np.einsum("ij,ij->i", XQ, Xf)   # H_obj(x) = -(x^T QU_obj x), un-boosted
        return out

    def order(self, fit: dict, chrom_idx: np.ndarray) -> np.ndarray:
        """Indices that sort rows best-first by the rule's total order (the C++ comparator)."""
        if self.rule == "violation":
            return np.lexsort((chrom_idx, fit["cost"], fit["viol"]))
        key = np.where(fit["in_band"], fit["obj"], fit["viol"])
        return np.lexsort((chrom_idx, fit["cost"], key, ~fit["in_band"]))


def problem_from_instance(inst, rule: str) -> Problem:
    """Problem of an `Instance` (gsp.instances.instance). The violation rule uses only the draw's
    (P_bb, eps), so the three q of a draw give the same problem; `scope_id` is the draw id then."""
    n = int(inst.n)
    perm = chromosome_perm(inst.arrays["n_max"])
    if rule == "violation":
        return Problem("violation", n, np.asarray(inst.P_bb, np.float64), float(inst.eps), None, perm,
                       scope_id=inst.draw_id)
    return Problem("objective", n, np.asarray(inst.P_bb, np.float64), float(inst.eps),
                   np.asarray(inst.QU_obj, np.float64), perm, scope_id=inst.inst_id)


# --- operators (the C++ ones; vectorized over the whole generation) -------------------------------
def tournament_winners(rng, population: int, draws: int, size: int) -> np.ndarray:
    """Positions of `draws` tournament winners in a population sorted best-first: each tournament
    draws `size` positions uniformly with replacement and the best (the smallest position) wins."""
    return rng.integers(0, population, size=(draws, size)).min(axis=1)


def single_point_crossover(p1: np.ndarray, p2: np.ndarray, do_cx: np.ndarray, point: np.ndarray):
    """child1 = p1[:c] + p2[c:], child2 = p2[:c] + p1[c:] where `do_cx`; copies of the parents otherwise."""
    head = (np.arange(p1.shape[1])[None, :] < point[:, None]) | ~do_cx[:, None]
    return np.where(head, p1, p2), np.where(head, p2, p1)


def bit_flip(C: np.ndarray, flips: np.ndarray) -> np.ndarray:
    """Independent bit-flip mutation: flip C[i, j] where flips[i, j]."""
    return C ^ flips.astype(C.dtype)


@dataclass
class GAResult:
    rule: str
    n: int
    seed: int
    params: dict
    rank_idx: np.ndarray            # top distinct strings of the final population, best first (classical idx)
    n_distinct_final: int
    wall_s: float                   # the whole run (init .. top-K), bookkeeping excluded
    wall_track_s: float             # first-evaluation bookkeeping (not part of the algorithm)
    cpu_s: float
    n_evals: int                    # fitness evaluations = (G + 1) * N_pop
    n_violation_evals: int
    n_objective_evals: int
    n_unique: int                   # distinct strings ever evaluated
    first_eval: np.ndarray | None   # (2^n,) evaluation index (1-based) of each string's first evaluation, 0 = never
    trace_best_idx: np.ndarray      # (G + 1,) best string of each evaluated generation
    trace_best_in_band: np.ndarray  # (G + 1,)
    trace_best_key: np.ndarray      # (G + 1,) its F(x) (violation) or H_obj (objective, in band) / F (out)
    extra: dict = field(default_factory=dict)

    def top(self, K: int) -> np.ndarray:
        return self.rank_idx[:K]

    def evals_to_hit(self, idx: int) -> int:
        """Evaluation index at which string `idx` was first evaluated (0 = never)."""
        if self.first_eval is None:
            raise ValueError("run with track=True")
        return int(self.first_eval[int(idx)])


def run_ga(problem: Problem, seed: int, params: GAParams = TABLE_4_1, keep: int = 24,
           track: bool = True) -> GAResult:
    """One GA run; returns the top-`keep` distinct strings of the final population, best first."""
    n = problem.n
    Np, G, E, T = params.population, params.generations, params.elitism, params.tournament
    if E >= Np:
        raise ValueError("elitism must be smaller than the population")
    r_m = params.mutation_rate(n)
    n_pairs = (Np - E + 1) // 2
    wts = _weights(n)
    if track and n > MAX_TRACK_N:
        raise ValueError(f"first-evaluation tracking needs n <= {MAX_TRACK_N}")
    first_eval = np.zeros(1 << n, dtype=np.int64) if track else None
    t_track = 0.0
    tr_idx = np.empty(G + 1, np.int64)
    tr_band = np.empty(G + 1, bool)
    tr_key = np.empty(G + 1, np.float64)

    t0 = time.perf_counter()
    c0 = time.process_time()
    rng = np.random.default_rng(int(seed))
    pop = rng.integers(0, 2, size=(Np, n), dtype=np.uint8)       # chromosome layout (C++)

    def evaluate_sort(pop, g):
        nonlocal t_track
        X = problem.chrom_to_x(pop)
        fit = problem.evaluate(X)
        x_idx = X.astype(np.int64) @ wts
        c_idx = pop.astype(np.int64) @ wts
        if track:
            ts = time.perf_counter()
            u, first = np.unique(x_idx, return_index=True)
            new = first_eval[u] == 0
            first_eval[u[new]] = g * Np + first[new] + 1
            t_track += time.perf_counter() - ts
        o = problem.order(fit, c_idx)
        b = o[0]
        tr_idx[g] = x_idx[b]
        tr_band[g] = fit["in_band"][b]
        tr_key[g] = fit["obj"][b] if (problem.rule == "objective" and fit["in_band"][b]) else fit["viol"][b]
        return pop[o], x_idx[o]

    for g in range(G):
        pop, _ = evaluate_sort(pop, g)
        win = tournament_winners(rng, Np, 2 * n_pairs, T)
        do_cx = rng.random(n_pairs) < params.crossover_rate
        point = rng.integers(1, n, size=n_pairs) if n >= 2 else np.ones(n_pairs, np.int64)
        c1, c2 = single_point_crossover(pop[win[0::2]], pop[win[1::2]], do_cx, point)
        kids = np.empty((2 * n_pairs, n), dtype=np.uint8)
        kids[0::2] = c1
        kids[1::2] = c2
        kids = bit_flip(kids, rng.random((2 * n_pairs, n)) < r_m)
        pop = np.concatenate([pop[:E], kids[: Np - E]], axis=0)
    pop, x_sorted = evaluate_sort(pop, G)
    keep_mask = np.ones(x_sorted.size, bool)
    keep_mask[1:] = x_sorted[1:] != x_sorted[:-1]          # identical chromosomes are adjacent
    distinct = x_sorted[keep_mask]
    wall = time.perf_counter() - t0 - t_track
    cpu = time.process_time() - c0

    n_evals = (G + 1) * Np
    return GAResult(
        rule=problem.rule, n=n, seed=int(seed), params=params.as_dict(),
        rank_idx=distinct[:keep].copy(), n_distinct_final=int(distinct.size),
        wall_s=wall, wall_track_s=t_track, cpu_s=cpu, n_evals=n_evals,
        n_violation_evals=n_evals, n_objective_evals=n_evals if problem.rule == "objective" else 0,
        n_unique=int(np.count_nonzero(first_eval)) if track else -1, first_eval=first_eval,
        trace_best_idx=tr_idx, trace_best_in_band=tr_band, trace_best_key=tr_key,
    )


def brute_force(problem: Problem, keep: int = 24, chunk: int = 1 << 16) -> dict:
    """Exhaustive reference: every one of the 2^n strings ranked by the same total order.

    Returns the top-`keep` (classical indices, best first), the full in-band mask (for the band
    cross-check against the rulers) and the wall-clock of the enumeration and ranking.
    """
    n = problem.n
    total = 1 << n
    wts = _weights(n)
    t0 = time.perf_counter()
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    parts = {"cost": [], "viol": [], "in_band": [], "obj": []}
    c_parts = []
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
        X = ((idx[:, None] >> shifts) & 1).astype(np.uint8)
        fit = problem.evaluate(X)
        for k, v in fit.items():
            parts[k].append(v)
        c_parts.append(problem.x_to_chrom(X).astype(np.int64) @ wts)
    fit = {k: np.concatenate(v) for k, v in parts.items() if v}
    order = problem.order(fit, np.concatenate(c_parts))
    top = order[:keep].astype(np.int64)
    wall = time.perf_counter() - t0
    return {"rank_idx": top, "in_band": fit["in_band"], "wall_s": wall, "n_evals": total,
            "viol": fit["viol"], "obj": fit.get("obj")}
