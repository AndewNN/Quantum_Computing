"""The bit-order convention (PLAN §1.1), CPU side. The GPU side is in test_backend_gpu.py."""

import numpy as np
import pytest

from gsp.instances import bits


@pytest.mark.parametrize("n", [1, 3, 8, 11])
def test_index_bits_roundtrip(n):
    idx = np.arange(1 << n)
    b = bits.index_to_bits(idx, n)
    assert b.shape == (1 << n, n)
    assert np.array_equal(bits.bits_to_index(b), idx)
    # x_0 is the most significant bit
    assert np.array_equal(b[:, 0], (idx >> (n - 1)) & 1)
    for k in [0, 1, (1 << n) - 1, (1 << n) // 3]:
        s = bits.bitstring(k, n)
        assert s == "".join(str(v) for v in b[k])
        assert bits.bitstring_to_index(s) == k


def test_bit_matrix_matches_legacy_enumeration():
    # the `ll` construction of all_state_to_return (Utils/qaoaCUDAQ.py:653-668)
    qb = 7
    ll = np.zeros((qb, 1 << qb), dtype=np.float32)
    idxx = np.arange(1 << qb, dtype=np.int32)
    for i in range(qb):
        ll[i] = np.where(idxx % (1 << (qb - i)) < (1 << (qb - i - 1)), 0.0, 1.0)
    assert np.array_equal(bits.bit_matrix(0, 1 << qb, qb), ll.T)
    assert np.array_equal(bits.bit_matrix(40, 90, qb), ll.T[40:90])


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_cudaq_to_classical_is_bit_reversal(n):
    rng = np.random.default_rng(n)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    c = bits.cudaq_to_classical(v, n)
    k = np.arange(1 << n)
    # amplitude of CUDA-Q index k (q_i = bit i of k) lands at classical index reverse(k)
    assert np.array_equal(c[bits.reverse_index(k, n)], v)
    assert np.array_equal(bits.classical_to_cudaq(c, n), v)


def test_reverse_index_examples():
    # qubit 0 set: CUDA-Q index 1, classical string '100' (x_0 first) = 4
    assert bits.reverse_index(1, 3) == 4
    assert bits.bitstring(bits.reverse_index(1, 3), 3) == "100"
    assert bits.reverse_index(6, 3) == 3
