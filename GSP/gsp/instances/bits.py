"""The one bit-order convention of the harness (PLAN §1.1).

Classical side (enumeration, rulers, QUBO energies, sector files):
    a string x = (x_0, ..., x_{n-1}); x_i is the value of qubit i, i.e. Z_i |x> = (1 - 2 x_i) |x>.
    Its **classical index** is  idx = sum_i x_i 2^(n-1-i)   (x_0 = MSB), exactly as
    `all_state_to_return` of the completed work enumerates strings.

CUDA-Q side:
    `cudaq.get_state` returns amplitudes indexed by  k = sum_i q_i 2^i   (q_0 = LSB).
    `cudaq.sample` returns keys whose character i is qubit i, so a key read as a binary number
    (first character most significant) is already the classical index.

Hence classical index = bit-reverse(CUDA-Q state index): reversing the qubit axes of the state
tensor maps one to the other. `tests/test_bits.py` (CPU) and `tests/test_backend_gpu.py` (GPU)
pin this down.
"""

from __future__ import annotations

import numpy as np


def index_to_bits(idx, n: int) -> np.ndarray:
    """Classical index (scalar or array) -> bits x_0..x_{n-1} (x_0 = MSB), dtype uint8."""
    idx = np.asarray(idx, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    return ((idx[..., None] >> shifts) & 1).astype(np.uint8)


def bits_to_index(bits) -> np.ndarray | int:
    """Bits x_0..x_{n-1} (last axis) -> classical index."""
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.shape[-1]
    weights = np.int64(1) << np.arange(n - 1, -1, -1, dtype=np.int64)
    out = bits @ weights
    return int(out) if out.ndim == 0 else out


def bitstring(idx: int, n: int) -> str:
    """Classical index -> '0101...' with x_0 first (the order of a `cudaq.sample` key)."""
    return format(int(idx), f"0{n}b")


def bitstring_to_index(s: str) -> int:
    """'0101...' (x_0 first; a `cudaq.sample` key) -> classical index."""
    return int(s, 2)


def reverse_index(k, n: int):
    """Bit-reverse an n-bit index (maps CUDA-Q state index <-> classical index)."""
    k = np.asarray(k, dtype=np.int64)
    out = np.zeros_like(k)
    for i in range(n):
        out |= ((k >> i) & 1) << (n - 1 - i)
    return int(out) if out.ndim == 0 else out


def cudaq_to_classical(vec: np.ndarray, n: int) -> np.ndarray:
    """Reorder a CUDA-Q statevector (q_0 = LSB) into classical order (x_0 = MSB).

    Reversing the qubit axes of the 2x...x2 tensor; the map is an involution.
    """
    vec = np.asarray(vec)
    if vec.shape != (1 << n,):
        raise ValueError(f"expected a vector of length 2^{n}, got {vec.shape}")
    return np.ascontiguousarray(vec.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1))).reshape(-1))


classical_to_cudaq = cudaq_to_classical


def bit_matrix(start: int, stop: int, n: int, dtype=np.float32) -> np.ndarray:
    """Rows = strings with classical index start..stop-1, columns = x_0..x_{n-1}.

    Same values as the `l` matrix of the completed work's `all_state_to_return`
    (float32 0/1 by default), built for one chunk.
    """
    idx = np.arange(start, stop, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    return ((idx[:, None] >> shifts) & 1).astype(dtype)
