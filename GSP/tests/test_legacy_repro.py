"""Legacy reproduction (PLAN §5 S1): the new instance code equals the completed work bit for bit.

Two references:
1. the frozen legacy copies in tests/legacy/ (old functions, old draw code) run on today's
   dataset: everything must agree to 0 ulp;
2. the arrays the completed runs actually stored (tests/fixtures/, built from
   CUDA/experiments_approx_Q2_RAND_S1.0_W0.01_Jh): assets, prices, returns and boosts are exact;
   budgets are exact for 52 of 60 draws (N = 5, e = 0 among them) and 1 ulp away for 8; the
   stored covariances differ from today's CSV by <= 6 ulp (the completed runs' own copies
   disagree with each other at that level), so cov-derived numbers are compared to 1e-15
   relative ("0 ulp or 1e-15", PLAN §5 S1).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from gsp.instances.draws import QUBITS_PER_ASSET, draw_seed, make_draw
from gsp.instances.encode import (encode, hamiltonian, jh_boost, qubo_energies, qubo_to_ising,
                                  ret_cov_to_QUBO)
from gsp.instances.rulers import band

warnings.filterwarnings("ignore", category=DeprecationWarning)

legacy_inst = pytest.importorskip("tests.legacy.po_new_approxratio_instance")
legacy_fn = pytest.importorskip("tests.legacy.qaoaCUDAQ_instance")

FIX = Path(__file__).parent / "fixtures"


def spin_terms(op, n):
    """{pauli word: real coefficient} of a cudaq SpinOperator (no raw-data API)."""
    return {t.get_pauli_word(n): t.evaluate_coefficient().real for t in op}


def ising_terms(H):
    out = {"I" * H.n: H.const}
    for i in range(H.n):
        if H.has_h[i]:
            w = ["I"] * H.n
            w[i] = "Z"
            out["".join(w)] = H.h[i]
    for a in range(H.n):
        for b in range(a + 1, H.n):
            if H.has_J[a, b]:
                w = ["I"] * H.n
                w[a] = w[b] = "Z"
                out["".join(w)] = H.J[a, b]
    return out


def assert_same_ising(op, H):
    ref = spin_terms(op, H.n)
    mine = ising_terms(H)
    # the old operator may carry the identity with coefficient 0 on some qubit sets; compare
    # the canonical terms both sides carry
    assert set(ref) <= set(mine) | {k for k, v in ref.items() if v == 0}
    for w, v in ref.items():
        assert mine.get(w, 0.0) == v, (w, mine.get(w), v)      # 0 ulp


# ------------------------------------------------------------------------------------------
# 1. new code == frozen legacy copy on today's data (0 ulp)
# ------------------------------------------------------------------------------------------

CASES = [(5, 0, 1.5, 0.005, "X"), (5, 0, 1.5, 1.0, "Preserving"), (4, 4, 1.0, 0.05, "X"),
         (6, 3, 3.0, 0.5, "X"), (7, 1, 1.5, 0.005, "X"), (8, 9, 1.5, 0.005, "X"),
         (10, 2, 1.5, 0.0005, "X")]


@pytest.mark.parametrize("N,e,q,lam,mode", CASES)
def test_new_equals_legacy_copy(market, N, e, q, lam, mode):
    L = legacy_inst.legacy_instance(N, e, q, lam, mode)
    d = make_draw(market, N, e)
    assert d.seed == draw_seed(N, e) == 911 + 991 * e + 997 * N
    # same assets, same budget, bit for bit
    assert np.array_equal(d.asset_idx, L["asset_idx"])
    assert np.array_equal(d.asset_idx_raw, L["asset_idx_raw"])
    assert np.array_equal(d.names, L["stock_names"].astype(str))
    assert np.array_equal(d.P, L["P"]) and np.array_equal(d.ret, L["ret"])
    assert np.array_equal(d.cov, L["cov"])
    assert d.w == L["weighted"] and d.B_min == L["B_mi"] and d.B_max == L["B_ma"]
    assert d.B == L["B"]
    # same encoding and QUBOs, bit for bit
    E = encode(d.B, d.P, d.ret, d.cov, q)
    assert E.n == L["n_qubit"] == QUBITS_PER_ASSET * N
    for a, b in [(E.P_bb, "P_bb"), (E.ret_bb, "ret_bb"), (E.cov_bb, "cov_bb"), (E.C, "C"),
                 (E.QU_obj, "QU_eval")]:
        assert np.array_equal(a, L[b]), b
    lamb = L["lamb"]
    assert np.array_equal(ret_cov_to_QUBO(E.ret_bb, E.cov_bb, E.P_bb, lamb, q), L["QU"])
    QU_lamb = ret_cov_to_QUBO(np.zeros_like(E.ret_bb), np.zeros_like(E.cov_bb), E.P_bb, lamb, 0.0)
    assert np.array_equal(QU_lamb, L["QU_lamb"])
    # same Hamiltonians (0 ulp per Pauli coefficient) and the same boost
    H_run = hamiltonian(E.ret_bb, E.cov_bb, E.P_bb, lamb, q) if mode == "X" else E.H_obj
    QU_run, lam_run = (L["QU"], lamb) if mode == "X" else (L["QU_eval"], 0.0)
    assert_same_ising(-legacy_fn.qubo_to_ising(QU_run, lam_run).canonicalize(), H_run)
    assert_same_ising(-legacy_fn.qubo_to_ising(L["QU_eval"], 0.0).canonicalize(), E.H_obj)
    assert_same_ising(-legacy_fn.qubo_to_ising(QU_lamb, lamb).canonicalize(), qubo_to_ising(QU_lamb, lamb))
    old_terms = L["ansatz_terms"]
    new_terms = H_run.terms()
    assert old_terms[0] == new_terms[0] and old_terms[2] == new_terms[2] and old_terms[3] == new_terms[3]
    assert np.array_equal(np.asarray(old_terms[1]), np.asarray(new_terms[1]))
    assert np.array_equal(np.asarray(old_terms[4]), np.asarray(new_terms[4]))
    assert jh_boost(H_run) == L["hamiltonian_boost"]
    # same diagonal energies (0 ulp), chunked or not
    for chunk in (1 << 16, 64):
        assert np.array_equal(qubo_energies(E.QU_obj, 0.0, chunk=chunk), L["state_eval"])
        assert np.array_equal(-qubo_energies(QU_run, lam_run, chunk=chunk), L["state_optim"])
        assert np.array_equal(-qubo_energies(QU_lamb, lamb, chunk=chunk), L["state_penalty"])
    # same band as the old idx_feasible at this lambda
    b = band(E.QU_pen, E.P_bb, 0.1)
    assert np.array_equal(b.idx, L["idx_feasible"])


def test_legacy_find_budget_is_the_ported_one():
    from gsp.instances.draws import find_budget
    P = np.array([132.0383606, 190.25999451, 131.78817749, 117.84462738, 207.60145569])
    for tq in (6, 8, 10, 12):
        assert find_budget(tq, P, 108, 216, True) == legacy_fn.find_budget(tq, P, 108, 216, True)
        assert find_budget(tq, P, 108, 216) == legacy_fn.find_budget(tq, P, 108, 216)


# ------------------------------------------------------------------------------------------
# 2. new code == what the completed runs stored
# ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def completed():
    return json.loads((FIX / "legacy_completed_draws.json").read_text())


def unhex(v):
    return np.array([float.fromhex(x) for x in v])


def test_completed_N5_e0_q15_bit_for_bit(market, completed):
    """The S1 headline check: N = 5, e = 0, q = 1.5 of the completed Q2 work."""
    rec = completed["draws"]["N5e0"]
    d = make_draw(market, 5, 0)
    assert list(d.asset_idx_raw) == rec["asset_idx_raw"] == [2, 6, 37, 49, 1]
    assert np.array_equal(d.P, unhex(rec["P"]))
    assert np.array_equal(d.ret, unhex(rec["ret"]))
    assert rec["B_values"] == [d.B.hex()]                       # 470.4053915515262 exactly
    assert d.B == 470.4053915515262
    E = encode(d.B, d.P, d.ret, d.cov, 1.5)
    # boosts of the completed runs: H(lam) for the X runs, H_obj for the Preserving run
    for tag, stored in completed["boost"]["N5e0"].items():
        mode, lam = tag.split("_L")
        H = E.H_obj if mode == "Preserving" else hamiltonian(E.ret_bb, E.cov_bb, E.P_bb, float(lam), 1.5)
        assert jh_boost(H) == stored, tag
    assert completed["boost"]["N5e0"]["X_L0.005"] == 150.3
    # QUBO and diagonal energies from the covariance the run stored: <= 1e-15 relative
    for lam_tag, cov_hex in rec["cov_by_run"].items():
        cov_run = unhex(cov_hex).reshape(5, 5)
        assert np.max(np.abs(cov_run - d.cov) / np.abs(d.cov)) <= 1e-15
        E_run = encode(d.B, d.P, d.ret, cov_run, 1.5)
        assert np.array_equal(E_run.P_bb, E.P_bb) and np.array_equal(E_run.ret_bb, E.ret_bb)
        for lam in (0.0, 0.005):
            Q_new = ret_cov_to_QUBO(E.ret_bb, E.cov_bb, E.P_bb, lam, 1.5)
            Q_run = ret_cov_to_QUBO(E_run.ret_bb, E_run.cov_bb, E_run.P_bb, lam, 1.5)
            assert np.max(np.abs(Q_new - Q_run)) <= 1e-15 * np.max(np.abs(Q_run))
            f_new = qubo_energies(Q_new, lam)
            f_run = qubo_energies(Q_run, lam)
            assert np.max(np.abs(f_new - f_run)) <= 1e-15 * np.max(np.abs(f_run))


# Draws whose stored budget is 1 ulp away from today's recomputation (new code and the frozen
# legacy copy agree with each other exactly on this machine; the completed runs, partly on the
# rented fleet, stored these 1-ulp neighbours). Pinned so that any change is noticed.
BUDGET_1ULP = {"N4e3", "N4e9", "N6e2", "N6e7", "N7e1", "N7e8", "N8e5", "N8e8"}


def ulp_distance(a: float, b: float) -> int:
    return abs(int(np.float64(a).view(np.int64)) - int(np.float64(b).view(np.int64)))


def test_completed_draws_all(market, completed):
    """All 60 stored Q2 draws (N = 3..8, e = 0..9): assets, prices, returns and boosts exact;
    budgets exact except the 8 pinned 1-ulp cases."""
    n_cov_exact = 0
    off = set()
    for key, rec in completed["draws"].items():
        N, e = rec["N"], rec["e"]
        d = make_draw(market, N, e)
        assert list(d.asset_idx_raw) == rec["asset_idx_raw"], key
        assert np.array_equal(d.P, unhex(rec["P"])), key
        assert np.array_equal(d.ret, unhex(rec["ret"])), key
        stored = [float.fromhex(x) for x in rec["B_values"]]
        assert min(ulp_distance(d.B, b) for b in stored) <= 1, key
        if d.B.hex() not in rec["B_values"]:
            off.add(key)
        for cov_hex in rec["cov_by_run"].values():
            cov_run = unhex(cov_hex).reshape(N, N)
            assert np.max(np.abs(cov_run - d.cov) / np.abs(d.cov)) <= 1e-15, key
            n_cov_exact += int(np.array_equal(cov_run, d.cov))
        E = encode(d.B, d.P, d.ret, d.cov, completed["q"])
        for tag, stored in completed["boost"][key].items():
            mode, lam = tag.split("_L")
            H = E.H_obj if mode == "Preserving" else hamiltonian(E.ret_bb, E.cov_bb, E.P_bb, float(lam), 1.5)
            assert jh_boost(H) == stored, (key, tag)
    assert n_cov_exact >= 1
    assert off == BUDGET_1ULP


# ------------------------------------------------------------------------------------------
# 3. the frozen instance on disk (if frozen)
# ------------------------------------------------------------------------------------------

def test_frozen_N05e000q15_matches_legacy():
    from gsp.store.paths import inst_path
    if not inst_path("N05e000q1.5").exists():
        pytest.skip("instances not frozen yet")
    from gsp.instances.instance import load_instance, load_rulers
    inst = load_instance("N05e000q1.5")
    L = legacy_inst.legacy_instance(5, 0, 1.5, 0.005, "X")
    assert list(inst.arrays["asset_idx_raw"]) == [2, 6, 37, 49, 1]
    assert inst.B == L["B"] == 470.4053915515262
    assert np.array_equal(inst.QU_obj, L["QU_eval"])
    assert np.array_equal(inst.qubo(0.005), L["QU"])
    H = inst.hamiltonian(0.005)
    assert jh_boost(H) == L["hamiltonian_boost"] == 150.3
    assert np.array_equal(-qubo_energies(inst.qubo(0.005), 0.005), L["state_optim"])
    rul = load_rulers("N05e000q1.5")
    assert np.array_equal(rul.f_band, -L["state_eval"][rul.band_idx])
    L1 = legacy_inst.legacy_instance(5, 0, 1.5, 1.0, "Preserving")
    assert np.array_equal(rul.band_idx, L1["idx_feasible"])
