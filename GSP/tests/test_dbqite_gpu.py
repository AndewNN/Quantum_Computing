"""S8 on the GPU: the DB-QITE kernel against the example, the reflection formula and npsim; the open-controlled phase;
A4 / A6 end to end through `Arm.run` (store, post-run, aggregate); the self-check against the recorded fixture."""

import json

import numpy as np
import pytest

from gsp.circuits import dbqite as db

pytestmark = pytest.mark.gpu


def _example():
    from gsp.circuits.xmixer import h_start
    levels = np.array([-2.0, -1.2, -0.7, -0.1, 0.3, 0.8, 1.4, 2.1])
    return db.DBCircuit(3, h_start(3), db.DBHam.from_diagonal(levels), kind="penalty", sigma=1.0, units="plan")


EXAMPLE_S = [0.5, 0.5, 0.5, 0.3, 0.5, 0.2, 0.3]                            # example_dbqite.py's printed choices
EXAMPLE_E = [0.0750, -0.6574, -1.2147, -1.5649, -1.7449, -1.8520, -1.9144]    # its printed E (k = 0..6)


def test_gpu_matches_example_dbqite():
    """The GPU circuit (observe on the unrolled 3^k recursion) reproduces the script: the same s at every step, the
    energies against the numpy formula engine to 1e-12 and the printed E to 4 decimals."""
    from gsp.arms.dbqite import NumpyDBEngine, run_dbqite
    C = _example()
    g = run_dbqite(C, np.array(db.GRID), 7)
    r = run_dbqite(NumpyDBEngine(C), np.array(db.GRID), 7)
    assert g.s.tolist() == EXAMPLE_S == r.s.tolist()
    # rounding grows ~3x per step (U_k repeats U_{k-1} three times): 4e-15 at k = 2 ... 2.1e-12 at k = 7 (S8)
    err = np.abs(g.E_loop - r.E_loop)
    assert np.all(err[1:] <= 1e-15 * 3.0 ** np.arange(1, 8) * 3) and err.max() <= 1e-11
    assert np.max(np.abs(g.E_cand[1:] - r.E_cand[1:])) <= 1e-11
    assert [f"{e:+.4f}" for e in g.E_loop[:7]] == [f"{e:+.4f}" for e in EXAMPLE_E]
    psi = C.state(g.s.tolist())
    assert np.max(np.abs(psi - C.formula_state(g.s.tolist()))) <= 1e-11


def test_phase_token_on_basis_states():
    """P(a) = e^{i a |0..0><0..0|}: X-conjugated r1.ctrl on the register, phase on |0...0> only."""
    from gsp.circuits.dbqite_kernel import kernel_dbqite
    from gsp.circuits.program import OPCODES
    from gsp.sim import backend
    n, a = 5, 0.917
    for x in (0, 1, 6, 31):
        code = [OPCODES["x"] + 16 * (k + 0) for k in range(n) if (x >> (n - 1 - k)) & 1]
        psi = backend.get_state_classical(kernel_dbqite, n, n, code or [0], [0.0] * max(1, len(code)), [0],
                                          [0, len(code), len(code), len(code)], [0, 3], [0.0, a])
        want = np.zeros(1 << n, dtype=complex)
        want[x] = np.exp(1j * a) if x == 0 else 1.0
        assert np.max(np.abs(psi - want)) <= 1e-15


@pytest.mark.parametrize("units", ["plan", "normalized"])
def test_a4_a6_gpu_equals_formula_and_gate_list(units):
    from gsp.arms.dbqite import confined_db_circuit, penalty_db_circuit
    from gsp.compile import npsim
    from gsp.instances.instance import load_instance
    inst = load_instance("N04e004q1.5")
    C4, sv = confined_db_circuit(inst, "violation", 12, units=units)
    C6 = penalty_db_circuit(inst, 0.005, units=units)
    for C in (C4, C6):
        s = [x / C.sigma for x in (0.3, 0.05, 0.8)]
        for k in range(4):
            psi = C.state(s[:k])
            assert np.max(np.abs(psi - C.formula_state(s[:k]))) <= 1e-12
            if k <= 2:
                ref = npsim.flat(npsim.apply(C.gates(s[:k]), npsim.columns(C.n, [0])))[:, 0]
                assert np.max(np.abs(psi - ref)) <= 1e-12
            e = C.energy(s[:k])
            assert abs(e - float(np.abs(psi) ** 2 @ C.ham.diagonal())) <= 1e-12 * max(1.0, abs(e))
    psi = C4.state(s)
    mask = np.ones(psi.size, dtype=bool)
    mask[sv.idx] = False
    assert float(np.sum(np.abs(psi[mask]) ** 2)) <= 1e-28                  # never projected, never leaks


def test_arm_run_end_to_end(tmp_path):
    """A4 and A6 through Arm.run into a temporary store: valid records, post-run replay exact, aggregate clean."""
    from gsp.arms.base import make_arm, validate_run_dir
    from gsp.metrics.aggregate import run_row
    from gsp.store.index import build_index
    a4, a6 = make_arm("A4"), make_arm("A6")
    r4 = a4.run("N04e004q1.5", {"connectivity": "adaptive", "rule": "violation", "K": 12}, 2, runs_root=tmp_path)
    r6 = a6.run("N04e004q1.5", None, 2, lam=0.005, runs_root=tmp_path)
    for r in (r4, r6):
        assert r.status == "done" and validate_run_dir(r.path) == []
        pr = json.loads((r.path / "postrun.json").read_text())
        assert pr["status"] == "done" and pr["replay_max_abs"] == 0.0
        tr = r.trajectory
        assert tr["t"].tolist() == [0, 1, 2] and tr["circuits_charged"].tolist() == [0, 7, 14]
        c = r.counts
        assert tr["g2q_ii"][-1] == 7 * (c["series"]["per_circuit"]["cx_ii"][1] + c["series"]["per_circuit"]["cx_ii"][2])
        assert abs(r.record["metric_energy"] - tr["E_loop"][-1]) <= 1e-12
        assert np.all(np.isfinite(tr["s_k"][1:])) and np.isnan(tr["s_k"][0])
    assert r4.record["metric_leak_max"] <= 1e-12
    rows = [run_row(row, tmp_path, None) for row in build_index(tmp_path).to_dict("records")]
    assert [r["anomalies"] for r in rows] == ["", ""]
    assert all(r["chk_conv_g2q"] for r in rows)
    again = a4.run("N04e004q1.5", {"connectivity": "adaptive", "rule": "violation", "K": 12}, 2, runs_root=tmp_path)
    assert again.skipped


def _fixture():
    from gsp.sim.selfcheck import fixture_path
    p = fixture_path("0.15.1")
    if not p.exists():
        pytest.skip("no recorded self-check fixture")
    return p


def test_selfcheck_circuits_of_one_instance():
    from gsp.sim.selfcheck import compare
    res = compare(_fixture(), only_circuits=True, instances=["N04e004q1.5"])
    assert res["n_cases"] == 7 and res["passes"], res["rows"]


@pytest.mark.slow
def test_selfcheck_compare_full():
    from gsp.sim.selfcheck import compare
    res = compare(_fixture())
    assert res["passes"], [r for r in res["rows"] if not r["pass"]]


def test_selfcheck_detects_a_perturbation(tmp_path):
    """compare() fails a case whose stored energy or state is off by more than the tolerance."""
    from gsp.sim.selfcheck import compare
    fx = json.loads(_fixture().read_text())
    fx["circuits"] = [c for c in fx["circuits"] if c["inst_id"] == "N04e004q1.5"][:2]
    fx["circuits"][0]["energy"] += 1e-9
    p = tmp_path / "fx.json"
    p.write_text(json.dumps(fx))
    res = compare(p, only_circuits=True)
    assert [r["pass"] for r in res["rows"]] == [False, True] and not res["passes"]
