"""A small numpy statevector simulator for the gate lists of `decompose.py` (CPU; tests, counts and
the C1 cross-checks). Not a harness simulator: the arms run on CUDA-Q through `gsp.sim.backend`.

States are arrays of shape (2,)*n + (B,): axis k is qubit k = bit x_k (x_0 = MSB, the classical order
of `gsp.instances.bits`), the last axis a batch of B columns, so `apply(gates, cols)` evolves B states
at once and the identity batch gives the full unitary. Gate conventions follow CUDA-Q:
rx(a) = exp(-i a X/2), ry(a) = exp(-i a Y/2), rz(a) = exp(-i a Z/2) = diag(e^{-ia/2}, e^{ia/2}).
"""

from __future__ import annotations

import numpy as np

from .decompose import Gate

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_FIXED = {
    "x": _X,
    "h": _H,
    "s": np.diag([1, 1j]).astype(np.complex128),
    "sdg": np.diag([1, -1j]).astype(np.complex128),
    "t": np.diag([1, np.exp(1j * np.pi / 4)]).astype(np.complex128),
    "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]).astype(np.complex128),
}


def rot(kind: str, a: float) -> np.ndarray:
    c, s = np.cos(a / 2), np.sin(a / 2)
    if kind == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if kind == "ry":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if kind == "rz":
        return np.array([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]], dtype=np.complex128)
    raise ValueError(kind)


def gate_matrix(g: Gate, params=None) -> np.ndarray:
    """The 2 x 2 matrix a gate applies to its target (for cx / mc*: when every control is 1)."""
    if g.name in _FIXED:
        return _FIXED[g.name]
    if g.name == "cx":
        return _X
    if g.name in ("rx", "ry", "rz"):
        return rot(g.name, g.angle.value(params))
    if g.name in ("mcrx", "mcry"):
        return rot(g.name[2:], g.angle.value(params))
    raise ValueError(g.name)


def apply_controlled(psi: np.ndarray, controls, target: int, M: np.ndarray) -> None:
    """In place: M on `target` where every qubit in `controls` is 1. psi has shape (2,)*n + (B,)."""
    n = psi.ndim - 1
    idx0 = [slice(None)] * (n + 1)
    for c in controls:
        idx0[c] = 1
    idx1 = list(idx0)
    idx0[target], idx1[target] = 0, 1
    a0, a1 = psi[tuple(idx0)], psi[tuple(idx1)]
    if M[0, 1] == 0 and M[1, 0] == 0:
        if M[0, 0] != 1:
            a0 *= M[0, 0]
        if M[1, 1] != 1:
            a1 *= M[1, 1]
        return
    if M[0, 0] == 0 and M[1, 1] == 0 and M[0, 1] == 1 and M[1, 0] == 1:
        tmp = a0.copy()
        a0[...] = a1
        a1[...] = tmp
        return
    b0 = M[0, 0] * a0 + M[0, 1] * a1
    b1 = M[1, 0] * a0 + M[1, 1] * a1
    a0[...] = b0
    a1[...] = b1


def apply(gates, psi: np.ndarray, params=None) -> np.ndarray:
    """Apply `gates` (time order) in place to psi of shape (2,)*n + (B,); returns psi."""
    for g in gates:
        M = gate_matrix(g, params)
        if g.name == "cx":
            apply_controlled(psi, (g.qubits[0],), g.qubits[1], M)
        elif g.name in ("mcrx", "mcry"):
            apply_controlled(psi, g.qubits[:-1], g.qubits[-1], M)
        else:
            apply_controlled(psi, (), g.qubits[0], M)
    return psi


def columns(n: int, idx) -> np.ndarray:
    """Basis states |idx_j> (classical indices) as a batch, shape (2,)*n + (len(idx),)."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    psi = np.zeros((1 << n, idx.size), dtype=np.complex128)
    psi[idx, np.arange(idx.size)] = 1.0
    return psi.reshape((2,) * n + (idx.size,))


def flat(psi: np.ndarray) -> np.ndarray:
    """(2,)*n + (B,) -> (2^n, B), rows in classical order."""
    return psi.reshape(-1, psi.shape[-1])


def unitary(gates, n: int, params=None) -> np.ndarray:
    """The 2^n x 2^n unitary of a gate list (rows / columns in classical order)."""
    return flat(apply(gates, columns(n, np.arange(1 << n)), params))


def n_qubits(gates) -> int:
    return 1 + max(q for g in gates for q in g.qubits)


def ideal_controlled(n: int, controls, target: int, M: np.ndarray, open_controls: bool = False) -> np.ndarray:
    """The ideal 2^n x 2^n multi-controlled gate: M on `target` where every control is 1 (0 if
    `open_controls`), the identity elsewhere. Built directly, not from gates (the reference)."""
    x = np.arange(1 << n, dtype=np.int64)
    ok = ((x >> (n - 1 - target)) & 1) == 0
    want = 0 if open_controls else 1
    for c in controls:
        ok &= ((x >> (n - 1 - c)) & 1) == want
    a = x[ok]
    b = a | (1 << (n - 1 - target))
    U = np.eye(1 << n, dtype=np.complex128)
    U[a, a], U[a, b], U[b, a], U[b, b] = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return U
