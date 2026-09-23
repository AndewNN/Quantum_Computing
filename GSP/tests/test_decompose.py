"""S3: the explicit decompositions of `gsp.compile.decompose` against ideal gates (numpy), and the
(ii) / (iii) / T counters (PLAN §2.4, §5 S3; V18 and V20 of verify_notes.py)."""

import numpy as np
import pytest

from gsp.compile import decompose as dc
from gsp.compile import npsim, tcount
from gsp.compile import transpile as tp
from gsp.compile import verify as vf
from gsp.compile.decompose import Angle, Gate

TOL = 1e-12


def test_toffoli_exact_and_margolus_up_to_one_sign():
    U = npsim.unitary(dc.toffoli(0, 1, 2), 3)
    assert np.abs(U - npsim.ideal_controlled(3, [0, 1], 2, npsim._X)).max() < TOL
    assert dc.cnot_count(dc.toffoli(0, 1, 2)) == 6 and dc.t_count(dc.toffoli(0, 1, 2), 30) == 7
    R = npsim.unitary(dc.rtof(0, 1, 2), 3)
    D = R @ npsim.ideal_controlled(3, [0, 1], 2, npsim._X).conj().T
    assert np.abs(D - np.diag(np.diag(D))).max() < TOL            # a Toffoli times a diagonal
    assert sorted(np.round(np.diag(D).real).astype(int).tolist()) == [-1] + [1] * 7
    assert dc.cnot_count(dc.rtof(0, 1, 2)) == 3 and dc.t_count(dc.rtof(0, 1, 2), 30) == 4


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_mcx_with_dirty_ancillas(k):
    """Iten et al. Lemma 8: exact C^k(X) with k - 2 dirty ancillas in 8k - 6 CNOTs; the ancillas are
    restored whatever their state (the full unitary is checked)."""
    n = 2 * k - 1
    ctl, t, anc = list(range(k)), k, list(range(k + 1, n))
    g = dc.mcx(ctl, t, anc)
    assert np.abs(npsim.unitary(g, n) - npsim.ideal_controlled(n, ctl, t, npsim._X)).max() < TOL
    assert dc.cnot_count(g) == 8 * k - 6 == dc.mcx_cnots(k)


@pytest.mark.parametrize("kind", ["rx", "ry", "rz"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("open_controls", [False, True])
def test_vale_mc_su2_equals_ideal(kind, k, open_controls):
    r = vf.mc_su2_error(kind, k, angle=0.7351 + 0.1 * k, open_controls=open_controls)
    assert r["err"] < TOL, r
    assert r["cx"] == r["cx_formula"]


def test_vale_large_arity():
    for k in (8, 9):
        r = vf.mc_su2_error("rx", k, angle=-1.234, open_controls=True)
        assert r["err"] < TOL and r["cx"] == 16 * k - 24


def test_vale_cnot_formula():
    assert [dc.vale_cnots(k) for k in range(6)] == [0, 2, 4, 14, 24, 48]
    for k in range(6, 20):
        assert dc.vale_cnots(k) == 16 * k - 24 == 16 * (k + 1) - 40       # the paper's Theorem 3 bound
    for k in range(0, 14):
        assert dc.cnot_count(dc.mc_su2(list(range(k)), k, "rx", Angle(1.0, 0))) == dc.vale_cnots(k)


def test_mc_su2_is_ancilla_free():
    for k in range(1, 10):
        qubits = {q for g in dc.mc_su2(list(range(k)), k, "ry", Angle(0.3)) for q in g.qubits}
        assert qubits == set(range(k + 1))


@pytest.mark.parametrize("s", [1, 2, 3, 4, 5])
def test_transition_iii_equals_native_n6(s):
    """Decomposition == native unitary at n = 6 with open controls, |S| = 1..5 (PLAN §5 S3)."""
    n = 6
    rng = np.random.default_rng(s)
    for case in range(4):
        u, v = (int(x) for x in rng.choice(1 << n, size=2, replace=False))
        D = [k for k in range(n) if ((u ^ v) >> (n - 1 - k)) & 1]
        k0 = D[0]
        others = [k for k in range(n) if k != k0]
        S = tuple(sorted(rng.choice(others, size=s, replace=False).tolist()))
        xm = tuple(k for k in range(n) if (u >> (n - 1 - k)) & 1)
        ladder = [(k0, k) for k in D[1:]]
        for kind in ("rx", "ry"):
            a = Angle(0.9 - 0.3 * case)
            U3 = npsim.unitary(dc.transition_iii(xm, ladder, S, k0, kind, a), n)
            Un = npsim.unitary(dc.transition_native(xm, ladder, S, k0, kind, a), n)
            assert np.abs(U3 - Un).max() < TOL
            assert dc.cnot_count(dc.transition_iii(xm, ladder, S, k0, kind, a)) == 2 * (len(D) - 1) + dc.vale_cnots(s)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_random_transitions_iii_equal_native(n):
    assert vf.transition_errors(n, n_cases=8, seed=10 + n)["max_err"] < TOL


def test_native_transition_is_the_two_level_rotation():
    """W† C^0_S(Rx(2 theta)) W = exp(-i theta (|u><v| + |v><u|)) on span{u, v}, identity on strings hit by S."""
    n, u, v, theta = 5, 0b10110, 0b01010, 0.37
    D = [k for k in range(n) if ((u ^ v) >> (n - 1 - k)) & 1]
    k0 = D[0]
    S = tuple(k for k in range(n) if k != k0)                       # all controls: identity off {u, v}
    xm = tuple(k for k in range(n) if (u >> (n - 1 - k)) & 1)
    U = npsim.unitary(dc.transition_native(xm, [(k0, k) for k in D[1:]], S, k0, "rx", Angle(2 * theta)), n)
    A = np.zeros((32, 32))
    A[u, v] = A[v, u] = 1
    ref = np.eye(32) - (1 - np.cos(theta)) * (A @ A) - 1j * np.sin(theta) * A        # Eq. 4.8 / meth-two-level
    assert np.abs(U - ref).max() < TOL


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_ii_construction_is_the_native_transition(n):
    r = vf.ii_errors(n, n_cases=4, seed=n)
    assert r["block_err"] < TOL and r["leak"] < TOL


def test_ii_counter_equals_eq_4_10():
    for n in range(3, 15):
        for d in range(1, n + 1):
            assert tp.ii_cnots(n, d) == tp.ii_formula(n, d) == 6 * n + 2 * d - 12
            assert tp.ii_cnots(n, d, "ry") == 6 * n + 2 * d - 12
    assert tp.ii_formula(5, 3) == 24 and tp.ii_formula(10, 5) == 58       # V18


def test_v20_t_rules():
    assert (tcount.t_syn(1e-3), tcount.t_syn(1e-5), tcount.t_syn(3e-5)) == (30, 50, 46)
    assert tcount.transition_t_ii(10) == 172 and tcount.transition_t_ii(30) == 452
    assert tcount.transition_tdepth_ii(10) == 76 and tcount.transition_tdepth_ii(30) == 116
    assert 12 * tcount.transition_t_ii(10) == 2064 and 12 * tcount.transition_t_ii(30) == 5424
    assert tcount.cost_layer_t(55) == 1650 and tcount.cost_layer_tdepth(10) == 300
    assert tcount.cost_layer_tdepth(30) == 900 and (tcount.chi_edge(10), tcount.chi_edge(9)) == (9, 9)
    assert tcount.cost_layer_t(55) + tcount.x_mixer_t(10) == 1950                     # V20 penalty layer
    # the explicit V18 list at n = 5: 4 exact-T rotations per relative-phase Toffoli, 2 synthesized
    gl = dc.transition_ii(5, (0, 2, 3), [(0, 1), (0, 2)], 0, "rx", Angle(0.37, 0))
    assert dc.cnot_count(gl) == 24 and dc.t_count(gl, 30) == 4 * 2 * (5 - 2) + 2 * 30


def test_t_cost_classification():
    assert dc.gate_t_cost(Gate("ry", (0,), Angle(np.pi / 2)), 30) == 0       # Clifford
    assert dc.gate_t_cost(Gate("ry", (0,), Angle(-np.pi / 4)), 30) == 1      # one T
    assert dc.gate_t_cost(Gate("rz", (0,), Angle(0.37)), 30) == 30           # synthesized
    assert dc.gate_t_cost(Gate("rx", (0,), Angle(np.pi / 2, 0)), 30) == 30   # a parameter
    assert dc.t_depth([Gate("t", (0,)), Gate("t", (1,)), Gate("cx", (0, 1)), Gate("t", (1,))], 30) == 2
