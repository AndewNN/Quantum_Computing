"""GPU checks (marked `gpu`; skipped when another process holds the GPU).

- the bit-order convention on the real simulator (get_state, sample, observe(Z_i));
- `observe` == diagonal energy to 1e-12 at n = 8 (PLAN §1.1, §5 S1), on a frozen-recipe instance,
  for H_obj, the penalty H(lam) (un-boosted and boosted) and the old SpinOperator;
- the backend's SpinOperator equals the old `qubo_to_ising` operator term by term.
"""

import warnings

import numpy as np
import pytest

from gsp.instances.bits import bitstring, bitstring_to_index, reverse_index
from gsp.instances.draws import make_draw
from gsp.instances.encode import encode, hamiltonian, jh_boost, qubo_energies, qubo_lambda

pytestmark = pytest.mark.gpu
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="module")
def be():
    from gsp.sim import backend
    backend.set_target("nvidia", "fp64")
    return backend


@pytest.fixture(scope="module")
def kernels():
    from gsp.circuits import probe
    return probe


def test_runtime_info(be):
    info = be.runtime_info()
    assert info["cudaq_version"].startswith("CUDA-Q Version 0.15.1")
    assert (info["target"], info["target_option"]) == ("nvidia", "fp64")
    assert info["driver_version"]


def test_bit_order_on_the_simulator(be, kernels):
    n = 6
    bits = [1, 0, 1, 1, 0, 0]                   # x_0 .. x_5
    cls = int("".join(map(str, bits)), 2)       # classical index, x_0 = MSB
    psi = be.get_state(kernels.kernel_basis, n, bits)
    k = int(np.argmax(np.abs(psi)))
    assert abs(psi[k]) == pytest.approx(1.0)
    assert k == sum(b << i for i, b in enumerate(bits))       # CUDA-Q: q_0 = LSB
    assert reverse_index(k, n) == cls
    psi_c = be.get_state_classical(kernels.kernel_basis, n, n, bits)
    assert int(np.argmax(np.abs(psi_c))) == cls
    counts = be.sample(kernels.kernel_basis, n, bits, shots_count=50)
    assert list(counts) == [bitstring(cls, n)] and bitstring_to_index(next(iter(counts))) == cls
    for i in range(n):
        assert be.observe(kernels.kernel_basis, be.z_op(i), n, bits) == pytest.approx(1 - 2 * bits[i], abs=1e-14)


def test_sample_keys_follow_classical_order(be, kernels):
    n = 5
    rng = np.random.default_rng(3)
    th = list(rng.uniform(-np.pi, np.pi, 3 * n))
    p = np.abs(be.get_state_classical(kernels.kernel_probe, n, n, th)) ** 2
    counts = be.sample(kernels.kernel_probe, n, th, shots_count=200_000)
    emp = np.zeros(1 << n)
    for key, c in counts.items():
        emp[bitstring_to_index(key)] += c
    emp /= emp.sum()
    assert np.max(np.abs(emp - p)) < 0.01


@pytest.mark.parametrize("N,e,q,lam", [(4, 4, 1.5, 0.005), (4, 6, 3.0, 0.5)])
def test_observe_equals_diagonal_energy_n8(be, kernels, market, N, e, q, lam):
    d = make_draw(market, N, e)
    E = encode(d.B, d.P, d.ret, d.cov, q)
    n = E.n
    assert n == 8
    rng = np.random.default_rng(100 + e)
    th = list(rng.uniform(-np.pi, np.pi, 3 * n))
    p = np.abs(be.get_state_classical(kernels.kernel_probe, n, n, th)) ** 2
    H_lam = hamiltonian(E.ret_bb, E.cov_bb, E.P_bb, lam, q)
    alpha = jh_boost(H_lam)
    diag_lam = -qubo_energies(qubo_lambda(E.ret_bb, E.cov_bb, E.P_bb, lam, q), lam)
    checks = [
        (E.H_obj, 1.0, -qubo_energies(E.QU_obj, 0.0)),     # objective, un-boosted
        (H_lam, 1.0, diag_lam),                             # H(lam), un-boosted
        (E.Pen, 1.0, -qubo_energies(E.QU_pen, 1.0)),        # penalty (P.x - 1)^2
        (H_lam, alpha, alpha * diag_lam),                   # H(lam) with the Jh boost
    ]
    for H, a, diag in checks:
        val = be.observe(kernels.kernel_probe, be.ising_op(H, a), n, th)
        assert abs(val - float(p @ diag)) <= 1e-12, (a, val, float(p @ diag))
    # basis states: observe == the diagonal entry exactly enough
    for cls in (0, 37, 200, 255):
        bits = [int(c) for c in bitstring(cls, n)]
        val = be.observe(kernels.kernel_basis, be.ising_op(H_lam), n, bits)
        assert abs(val - diag_lam[cls]) <= 1e-12


def test_backend_op_equals_legacy_op(be, kernels, market):
    legacy = pytest.importorskip("tests.legacy.qaoaCUDAQ_instance")
    d = make_draw(market, 4, 4)
    E = encode(d.B, d.P, d.ret, d.cov, 1.5)
    QU = qubo_lambda(E.ret_bb, E.cov_bb, E.P_bb, 0.005, 1.5)
    old = -legacy.qubo_to_ising(QU, 0.005).canonicalize()
    new = be.ising_op(hamiltonian(E.ret_bb, E.cov_bb, E.P_bb, 0.005, 1.5))
    t_old = {t.get_pauli_word(8): t.evaluate_coefficient().real for t in old}
    t_new = {t.get_pauli_word(8): t.evaluate_coefficient().real for t in new.canonicalize()}
    assert t_old == t_new
    th = list(np.random.default_rng(0).uniform(-np.pi, np.pi, 24))
    assert be.observe(kernels.kernel_probe, old, 8, th) == pytest.approx(
        be.observe(kernels.kernel_probe, new, 8, th), abs=1e-15)
