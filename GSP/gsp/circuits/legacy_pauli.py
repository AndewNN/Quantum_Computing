"""The completed work's Pauli expansion of the preserving mixer, in numpy (no cudaq): a port of the numbers of
`basis_T_to_pauli_parallel` (Utils/qaoaCUDAQ.py:243-396, called at CUDA/PO_new_ApproxRatio.py:785).

Used for exactly one thing: `legacy_init=True` (PLAN §1.5), whose gamma range was pi / min(|h|, |J|, |c_P|) with
c_P the smallest merged Pauli coefficient of the mixer (`mm_p`, PO_new_ApproxRatio.py:815-817). The new
compiled mixer has no such coefficients (every transition has unit coefficient).

The old recursion (`init_pauli` / `transform_pauli`) builds A = |x><y| + |y><x| over qubit k = character k of
the basis strings (x_0 = MSB of the classical index): 2^(n-1) Pauli words with coefficients +-2^-(n-1). The
per-transition words are merged over the ring edges (i < j, T_ij != 0) of the given order into a dict in
first-insertion order, the sums in float32 (T is float32; numpy 2 keeps float32 * float in float32), as the
old code did. `tests/test_arms_gpu.py` checks words and coefficients against the frozen legacy copy.
"""

from __future__ import annotations

import numpy as np


def _step(A: dict, B: dict, x: str, y: str, first: bool):
    """One qubit of the old recursion on {word: coefficient} dicts (words grow left to right)."""
    if first:
        if x == "0" and y == "0":
            return {"I": 1.0, "Z": 1.0}, {}
        if x == "0" and y == "1":
            return {"X": 1.0}, {"Y": -1.0}
        if x == "1" and y == "0":
            return {"X": 1.0}, {"Y": 1.0}
        return {"I": 1.0, "Z": -1.0}, {}

    def app(D, letter, c):
        return {w + letter: c * v for w, v in D.items()}

    def add(*parts):
        out: dict = {}
        for part in parts:
            for w, v in part.items():
                out[w] = out.get(w, 0.0) + v
        return out

    if x == y:
        s = 1.0 if x == "0" else -1.0
        return (add(app(A, "I", 0.5), app(A, "Z", 0.5 * s)), add(app(B, "I", 0.5), app(B, "Z", 0.5 * s)))
    if x == "0":        # y == "1": A_ = (A X + B Y)/2, B_ = (B X - A Y)/2
        return add(app(A, "X", 0.5), app(B, "Y", 0.5)), add(app(B, "X", 0.5), app(A, "Y", -0.5))
    # x == "1", y == "0": A_ = (A X - B Y)/2, B_ = (B X + A Y)/2
    return add(app(A, "X", 0.5), app(B, "Y", -0.5)), add(app(B, "X", 0.5), app(A, "Y", 0.5))


def transition_terms(x: str, y: str) -> dict:
    """{word: coefficient} of |x><y| + |y><x| (the old `get_pauli`'s A), zero coefficients dropped."""
    A, B = _step({}, {}, x[0], y[0], True)
    for k in range(1, len(x)):
        A, B = _step(A, B, x[k], y[k], False)
    return {w: v for w, v in A.items() if v != 0.0}


def ring_T(K: int) -> np.ndarray:
    """The old ring adjacency (PO_new_ApproxRatio.py:776-780), float32."""
    T = np.zeros((K, K), dtype=np.float32)
    T[:-1, 1:] += np.eye(K - 1, dtype=np.float32)
    T[1:, :-1] += np.eye(K - 1, dtype=np.float32)
    T[0, -1] = T[-1, 0] = 1.0
    return T


def merged_terms(bases, T: np.ndarray):
    """(words, float32 coefficients) exactly as `basis_T_to_pauli_parallel` returns them."""
    summed: dict = {}
    for i in range(T.shape[0]):
        for j in range(i + 1, T.shape[1]):
            if T[i, j] == 0:
                continue
            t_val = T[i, j]
            for w, c in transition_terms(bases[i], bases[j]).items():
                summed[w] = summed.get(w, np.float32(0.0)) + t_val * np.float32(c)
    words = [w for w, c in summed.items() if c != 0]
    return words, np.array([summed[w] for w in words], dtype=np.float32)


def bases_of(order, n: int) -> list:
    """The old basis strings: `bin(i)[2:].zfill(n)` of each classical index (x_0 first)."""
    return [bin(int(i))[2:].zfill(n) for i in order]


def legacy_mm_p(order, n: int) -> np.float32:
    """`mm_p` of the old code for a ring over `order`: np.min(np.abs(mixer_c)) (float32)."""
    _, c = merged_terms(bases_of(order, n), ring_T(len(order)))
    return np.min(np.abs(c)) if len(c) > 0 else 1e9
