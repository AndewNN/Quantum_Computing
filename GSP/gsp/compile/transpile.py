"""Gate counts of the abstract circuit (PLAN §2.4, D-1): per transition, per mixer layer, per cost
layer, for the start state, and for a whole A1 circuit. No cross-transition gate cancellation is
attempted anywhere (a transition's W† and the next transition's W are counted in full).

Two CNOT counts per transition:
  (ii)  the reported device count, Eq. 4.10 / V18: 2(d - 1) + 6(n - 2) + 2 = 6n + 2d - 12, with the
        n - 1 open controls on n - 2 reusable clean ancillas. Read off `decompose.transition_ii`'s gate
        list (the tests pin it to the formula and reproduce V18's 696 per ring layer).
  (iii) the simulated-equivalent count: W (2(d - 1)) plus Vale et al.'s ancilla-free C^|S|(SU(2)) on the
        hitting set S, read off `decompose.transition_iii`'s gate list.
  (ii_S, information only) the (ii) ancilla route with |S| controls instead of n - 1:
        2(d - 1) + 6(|S| - 1) + 2 (|S| >= 2), 2(d - 1) + 2 (|S| = 1), 2(d - 1) (|S| = 0), with |S| - 1
        ancillas. Eq. 4.10's text anticipates this reduction (m* = min(n - 1, K - 2)); it is not reported.
T companion: (ii) by the V20 formulas (`tcount`), (iii) from the list.
Cost layer: 2 CNOTs per ZZ term, one synthesized rotation per term.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import tcount
from .decompose import Angle, canonical_iii, cnot_count, t_count, t_depth, transition_ii


def ii_formula(n: int, d: int) -> int:
    return 6 * n + 2 * d - 12


def ii_s_formula(d: int, s: int) -> int:
    mc = 0 if s == 0 else 2 if s == 1 else 6 * (s - 1) + 2
    return 2 * (d - 1) + mc


@lru_cache(maxsize=None)
def ii_cnots(n: int, d: int, kind: str = "rx") -> int:
    """(ii) CNOTs of a transition, counted on V18's gate list (canonical layout: pivot 0, D = 0..d-1)."""
    gl = transition_ii(n, (), [(0, k) for k in range(1, d)], 0, kind, Angle(1.0, 0))
    return cnot_count(gl)


@lru_cache(maxsize=None)
def iii_counts(d: int, s: int, kind: str = "rx", ts: int | None = None) -> tuple:
    """(CNOTs, T-count, T-depth) of a transition's (iii) list, which depend on (d, |S|, kind) only."""
    ts = tcount.t_syn() if ts is None else ts
    gl = canonical_iii(d, s, kind)
    return cnot_count(gl), t_count(gl, ts), t_depth(gl, ts)


def transition_counts(tr, ts: int | None = None) -> dict:
    ts = tcount.t_syn() if ts is None else ts
    n, d, s = tr.n, tr.d, len(tr.S)
    c3, t3, td3 = iii_counts(d, s, tr.kind, ts)
    return {
        "d": d, "S": s, "kind": tr.kind,
        "cx_ii": ii_cnots(n, d, tr.kind), "cx_iii": c3, "cx_ii_S": ii_s_formula(d, s),
        "t_ii": tcount.transition_t_ii(n, ts=ts), "tdepth_ii": tcount.transition_tdepth_ii(n, ts=ts),
        "t_iii": t3, "tdepth_iii": td3,
    }


_SUM_KEYS = ("cx_ii", "cx_iii", "cx_ii_S", "t_ii", "tdepth_ii", "t_iii", "tdepth_iii")


def _sum(records) -> dict:
    out = {k: int(sum(r[k] for r in records)) for k in _SUM_KEYS}
    out["n_transitions"] = len(records)
    out["d_mean"] = float(np.mean([r["d"] for r in records])) if records else 0.0
    out["S_mean"] = float(np.mean([r["S"] for r in records])) if records else 0.0
    out["S_max"] = int(max((r["S"] for r in records), default=0))
    return out


def cost_counts(ct, ts: int | None = None) -> dict:
    """Cost layer: 2 CNOTs per ZZ term; T per V20. `ct` is a `gsp.circuits.cost.CostTerms`."""
    ts = tcount.t_syn() if ts is None else ts
    cx = 2 * ct.n_zz
    return {"cx_ii": cx, "cx_iii": cx, "cx_ii_S": cx, "t_ii": tcount.cost_layer_t(ct.n_rot, ts),
            "t_iii": tcount.cost_layer_t(ct.n_rot, ts),
            "tdepth_ii": tcount.cost_layer_tdepth(ct.n, ts, bool(ct.idx_1)),
            "tdepth_iii": tcount.cost_layer_tdepth(ct.n, ts, bool(ct.idx_1)),
            "n_zz": ct.n_zz, "n_rot": ct.n_rot}


def circuit_counts(circ, ct=None, ts: int | None = None) -> dict:
    """Counts of one `PreservingCircuit`: {"mixer": ..., "prep": ..., "cost": ..., "layer": ...}.
    layer = cost + mixer (the A1 layer); without `ct` the cost entry is absent and layer = mixer."""
    ts = tcount.t_syn() if ts is None else ts
    mixer = _sum([transition_counts(tr, ts) for tr in circ.layer])
    prep = _sum([transition_counts(tr, ts) for tr in circ.prep])
    out = {"n": circ.n, "K": circ.K, "connectivity": circ.connectivity, "ring_order": circ.ring_order,
           "symmetrized": circ.symmetrized, "t_syn": ts, "mixer": mixer, "prep": prep}
    if ct is not None:
        cost = cost_counts(ct, ts)
        out["cost"] = cost
        out["layer"] = {k: cost[k] + mixer[k] for k in _SUM_KEYS}
    else:
        out["layer"] = {k: mixer[k] for k in _SUM_KEYS}
    return out


def a1_totals(counts: dict, L: int) -> dict:
    """A whole A1 circuit at depth L: start state + L layers (what §1.3's gate-matched depths use)."""
    return {k: counts["prep"][k] + L * counts["layer"][k] for k in _SUM_KEYS}


def a0_layer_counts(ct, ts: int | None = None) -> dict:
    """One A0 layer (cost + X mixer): 2 n_zz CNOTs; T per V20 (penalty layer)."""
    ts = tcount.t_syn() if ts is None else ts
    c = cost_counts(ct, ts)
    return {"cx": c["cx_ii"], "t": c["t_ii"] + tcount.x_mixer_t(ct.n, ts),
            "tdepth": c["tdepth_ii"] + tcount.x_mixer_tdepth(ts)}


def synthetic_ring_order(n: int, K: int, d: int, seed: int = 0, tries: int = 20000) -> np.ndarray:
    """K distinct n-bit strings in ascending (lexicographic) order whose ring edges (i, i+1 mod K) all
    have Hamming distance d: the synthetic sector of the 696 check (V18: n = 10, K = 12, d = 5)."""
    rng = np.random.default_rng(seed)
    pop = np.array([bin(x).count("1") for x in range(1 << n)])
    for _ in range(tries):
        seq = [int(rng.integers(0, 1 << (n - 2)))]
        ok = True
        for _i in range(K - 1):
            cand = np.flatnonzero((pop[np.arange(1 << n) ^ seq[-1]] == d) & (np.arange(1 << n) > seq[-1]))
            if cand.size == 0:
                ok = False
                break
            seq.append(int(rng.choice(cand[: max(1, cand.size // 3)])))
        if ok and pop[seq[-1] ^ seq[0]] == d:
            return np.array(seq, dtype=np.int64)
    raise RuntimeError("no synthetic sector found")
