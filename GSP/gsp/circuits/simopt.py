"""Exact simulation-only rewrites of gate lists (S4). Counts are always read off the abstract circuit
(`preserving.a1_gates`, `cost.cost_gates`, `xmixer.a0_gates`), never off these lists.

`merge_x`: X gates are delayed until the first gate that does not commute with them, and pairs cancel.
An X on qubit q passes a gate that does not touch q, the TARGET of a cx (X_t commutes with CNOT), and the
target of an rx / mcrx (X commutes with Rx, so with every controlled Rx). Every other gate on q (a control,
rz, ry, h, crz, ...) first receives the pending X. At the end of the list the pending X's are emitted, so the
rewritten list is the same unitary exactly (X is a permutation; no angle changes).

On the compiled preserving mixer it removes the W / W† and open-control X's that meet between consecutive
transitions: 4,398 -> 3,504 gates for A1 at N = 7, K = 12, L = 9 before the cost-layer rewrite.
"""

from __future__ import annotations

from ..compile.decompose import Gate

_PASS_TARGET = ("cx", "rx", "mcrx")


def merge_x(gates) -> list:
    pending: set = set()
    out: list = []

    def flush(qubits) -> None:
        for q in sorted(set(qubits) & pending):
            out.append(Gate("x", (q,)))
            pending.discard(q)

    for g in gates:
        if g.name == "x":
            pending ^= {g.qubits[0]}
            continue
        flush(g.qubits[:-1] if g.name in _PASS_TARGET else g.qubits)
        out.append(g)
    flush(list(pending))
    return out
