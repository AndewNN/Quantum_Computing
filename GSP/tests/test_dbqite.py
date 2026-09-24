"""S8 on the CPU: arms A4 / A6 (DB-QITE) against `example_dbqite.py`, the recursion's tokens and counts, the
multi-controlled phase, the step conventions, the C1 generator check and the arm plumbing (config, planner)."""

import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pytest

from gsp.circuits import dbqite as db
from gsp.circuits.xmixer import h_start
from gsp.compile import decompose as dc
from gsp.compile import npsim
from gsp.compile.decompose import Angle
from gsp.store.paths import GSP_ROOT

LEGACY = GSP_ROOT / "tests" / "legacy" / "example_dbqite.py"
ORIGINAL = Path.home() / "Desktop" / "Quantum_Master_Proposal" / "Lecture_Notes" / "code" / "example_dbqite.py"


def _body(path: Path) -> str:
    s = path.read_text()
    return s.split('"""\n', 1)[1] if path == LEGACY else s


@pytest.fixture(scope="module")
def ex():
    """Run the frozen script; returns (namespace, parsed printed rows)."""
    ns = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(_body(LEGACY), ns)
    rows = []
    for ln in buf.getvalue().splitlines():
        m = re.match(r"k=(\d+)\s+E=([+-][\d.]+)\s+V=([\d.]+)\s+s=([\d.]+)\s+-2sV=\S+\s+ideal step \S+\s+"
                     r"compiled step ([+-][\d.]+)\s+p_ground=([\d.]+)", ln)
        if m:
            rows.append({"k": int(m[1]), "E": float(m[2]), "V": float(m[3]), "s": float(m[4]),
                         "dE": float(m[5]), "p0": float(m[6])})
    return ns, rows


def example_circuit(units="plan"):
    levels = np.array([-2.0, -1.2, -0.7, -0.1, 0.3, 0.8, 1.4, 2.1])
    return db.DBCircuit(3, h_start(3), db.DBHam.from_diagonal(levels), kind="penalty", sigma=1.0, units=units)


def test_frozen_example_equals_source():
    if not ORIGINAL.exists():
        pytest.skip("the original example_dbqite.py is not on this machine")
    assert _body(LEGACY) == ORIGINAL.read_text()


def test_numpy_engine_matches_example_step_by_step(ex):
    """The harness loop (greedy argmin over the grid on the compiled step, sigma unit 1) with the reflection-formula
    engine reproduces the script: every chosen s, every energy (exactly, against its own compiled_step), the printed
    lines to their digits."""
    from gsp.arms.dbqite import NumpyDBEngine, run_dbqite
    ns, rows = ex
    assert len(rows) == 7
    C = example_circuit()
    res = run_dbqite(NumpyDBEngine(C), np.array(ns["grid"]), 7)
    assert [float(v) for v in res.s] == [r["s"] for r in rows]
    psi = np.ones(8, dtype=complex) / np.sqrt(8)
    for k, r in enumerate(rows):
        E = ns["energy"](psi)
        assert abs(E - res.E_loop[k]) <= 1e-14
        new = ns["compiled_step"](psi, r["s"])
        assert f"{E:+.4f}" == f"{r['E']:+.4f}"
        assert f"{ns['energy'](new) - E:+.4f}" == f"{r['dE']:+.4f}"
        assert f"{res.E_loop[k + 1] - res.E_loop[k]:+.4f}" == f"{r['dE']:+.4f}"
        assert abs(ns["energy"](new) - res.E_loop[k + 1]) <= 1e-14
        psi = new
    # the script keeps the leading e^{-i r rho} (a global phase e^{-i r} per step); the harness drops it (PLAN §1.5)
    mine = NumpyDBEngine(C).state(res.s.tolist())
    phase = np.exp(-1j * np.sum(np.sqrt(res.s)))
    assert np.max(np.abs(mine * phase - psi)) <= 1e-14
    # the candidate energies of step 1 = the script's compiled step at every grid value
    psi0 = np.ones(8, dtype=complex) / np.sqrt(8)
    assert np.max(np.abs(res.E_cand[1] - [ns["energy"](ns["compiled_step"](psi0, x)) for x in ns["grid"]])) <= 1e-14


def test_diagonal_expansion_roundtrip():
    rng = np.random.default_rng(3)
    d = rng.normal(size=16)
    H = db.DBHam.from_diagonal(d)
    assert np.max(np.abs(H.diagonal() - d)) <= 1e-14
    assert H.max_body == 4


@pytest.mark.parametrize("units", ["plan", "normalized"])
def test_gate_list_equals_reflection_formula(units):
    """The unrolled simulation gate list (what the kernel applies, merged cost tokens) == the exact formula."""
    C = example_circuit(units)
    C.sigma = 1.7
    s = [0.3, 0.05, 0.8, 0.2]
    for k in range(len(s) + 1):
        psi = npsim.flat(npsim.apply(C.gates(s[:k]), npsim.columns(3, [0])))[:, 0]
        assert np.max(np.abs(psi - C.formula_state(s[:k]))) <= 1e-13


def test_step_units_are_the_same_flow_rescaled():
    """normalized on H == plan on H / sigma with s * sigma: the circuit carries H / sigma, r = sqrt(g)."""
    C = example_circuit("normalized")
    C.sigma = 2.5
    levels = C.ham.diagonal()
    P = db.DBCircuit(3, h_start(3), db.DBHam.from_diagonal(levels / 2.5), kind="penalty", sigma=1.0, units="plan")
    g = [0.2, 0.5, 0.05]
    a = C.formula_state([x / 2.5 for x in g])
    b = P.formula_state(g)
    assert np.max(np.abs(a - b)) <= 1e-13
    for s in (0.01, 0.4, 7.0):
        for u in db.STEP_UNITS:
            rH, rr = db.r_pair(s, u, 2.5)
            assert abs(rH * rr - s) <= 1e-14 * s


def test_tokens_structure_and_merge():
    base = [(db.TOK_U0, 0.0)]
    toks = db.recursion_tokens([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)], merge=False)
    assert len(toks) == (5 * 3 ** 3 - 3) // 2
    kinds = [k for k, _ in toks]
    assert kinds.count(db.TOK_U0) + kinds.count(db.TOK_U0_INV) == 27 and kinds.count(db.TOK_PHASE) == 13
    assert kinds.count(db.TOK_COST) == 26
    one = db.step_tokens(base, 0.1, 0.2, merge=False)
    assert one == [(0, 0.0), (2, 0.1), (1, 0.0), (3, 0.2), (0, 0.0), (2, -0.1)]
    merged = db.recursion_tokens([(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])
    assert len(merged) < len(toks)
    assert all(not (a[0] == db.TOK_COST and b[0] == db.TOK_COST) for a, b in zip(merged, merged[1:]))
    # merging keeps the total cost angle
    assert abs(sum(a for k, a in merged if k == 2) - sum(a for k, a in toks if k == 2)) <= 1e-15


def test_recursion_counts_closed_form_and_tokens():
    c0 = {k: 7 for k in db.COUNT_KEYS}
    cost = {k: 11 for k in db.COUNT_KEYS}
    ph = {k: 5 for k in db.COUNT_KEYS}
    rc = db.recursion_counts(c0, cost, ph, 3, 6)
    for k in range(7):
        assert rc["cx_ii"][k] == 3 ** k * 7 + (3 ** k - 1) // 2 * (2 * 11 + 5)
        assert rc["mc_gates"][k] == 3 ** k * 3 + (3 ** k - 1) // 2
    toks = db.recursion_tokens([(0.1, 0.2)] * 4, merge=False)
    kinds = [k for k, _ in toks]
    n_u0 = kinds.count(0) + kinds.count(1)
    assert rc["cx_ii"][4] == n_u0 * 7 + kinds.count(2) * 11 + kinds.count(3) * 5


def test_charged_series():
    per = {"cx_ii": [10, 40, 130], "cx_iii": [1, 2, 3]}
    ch = db.charged_series(per, 7, 2)
    assert ch["circuits_charged"].tolist() == [0, 7, 14]
    assert ch["g2q_ii"].tolist() == [0, 280, 280 + 910]
    assert ch["g2q_iii"].tolist() == [0, 14, 35]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_mcphase_forms(n):
    a = 0.8123
    ref = np.eye(1 << n, dtype=complex)
    ref[0, 0] = np.exp(1j * a)
    assert np.max(np.abs(npsim.unitary(dc.mcphase_native(n, Angle(a)), n) - ref)) <= 1e-15
    assert np.max(np.abs(npsim.unitary(dc.mcphase_iii(n, Angle(a)), n) - ref)) <= 1e-13
    g2 = dc.mcphase_ii(n, Angle(a))
    m = npsim.n_qubits(g2)
    cols = [x << (m - n) for x in range(1 << n)]
    F = npsim.flat(npsim.apply(g2, npsim.columns(m, cols)))
    assert np.max(np.abs(F[cols] - ref)) <= 1e-14                       # the ancilla-zero block
    assert dc.cnot_count(g2) == 6 * (n - 2) + 2
    assert dc.cnot_count(dc.mcphase_iii(n, Angle(a))) == dc.mcphase_iii_cnots(n)
    pc = db.mcphase_counts(n)
    assert pc["cx_ii"] == 6 * (n - 2) + 2 and pc["cx_iii"] == dc.mcphase_iii_cnots(n)


def test_run_dbqite_logger_is_structurally_separate():
    from gsp.arms.dbqite import NumpyDBEngine, run_dbqite
    C = example_circuit()
    seen = []
    a = run_dbqite(NumpyDBEngine(C), np.array(db.GRID), 5, logger=lambda t, s: seen.append((t, list(s))))
    b = run_dbqite(NumpyDBEngine(C), np.array(db.GRID), 5)
    assert np.array_equal(a.s, b.s) and np.array_equal(a.E_loop, b.E_loop) and np.array_equal(a.E_cand, b.E_cand, True)
    assert [t for t, _ in seen] == list(range(6)) and seen[-1][1] == a.s.tolist()


def test_generator_check():
    from gsp.stats.c1 import a4_generator_check
    rng = np.random.default_rng(0)
    n, S = 6, np.array([3, 9, 17, 40, 55])
    d = rng.normal(size=1 << n)
    psi = np.zeros(1 << n, dtype=complex)
    psi[S] = rng.normal(size=S.size) + 1j * rng.normal(size=S.size)
    psi /= np.linalg.norm(psi)
    r = a4_generator_check(psi, d, S)
    assert r["off_max"] == 0.0 and r["rel"] == 0.0
    assert abs(r["w_norm"] - r["w_norm_numeric"]) <= 1e-13 * r["w_norm"]
    psi2 = psi.copy()
    psi2[0] = 1e-9
    r2 = a4_generator_check(psi2, d, S)
    r3 = a4_generator_check(psi2, d, S, dense=False)
    exp = 1e-9 * np.max(np.abs(psi2) * np.abs(d - d[0]))
    assert abs(r2["off_max"] - exp) <= 1e-22 and abs(r3["off_max"] - exp) <= 1e-22


def test_matched_layers():
    from gsp.stats.c1 import matched_layers
    assert matched_layers(31, 10, 12) == 2 and matched_layers(10, 10, 12) == 1 and matched_layers(2551, 10, 12) == 212


def test_dbqite_kernel_is_generated():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_dbqite_kernel", GSP_ROOT / "scripts" / "gen_dbqite_kernel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (GSP_ROOT / "gsp" / "circuits" / "dbqite_kernel.py").read_text() == mod.source()


# --- the arms (CPU: configs, the planner) ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def n4():
    from gsp.instances.instance import load_instance
    return load_instance("N04e004q1.5")


def test_arm_configs(n4):
    from gsp.arms.base import make_arm, registered_arms
    assert {"A4", "A6"} <= set(registered_arms())
    a4, a6 = make_arm("A4"), make_arm("A6")
    cell = {"connectivity": "adaptive", "rule": "violation", "K": 12}
    c = a4.config(n4, cell, 5, None)
    d = c.to_dict()
    assert d["connectivity"] == "adaptive" and d["effort_kind"] == "steps" and d["step_units"] == "plan"
    assert d["grid"] == "0.02,0.05,0.1,0.2,0.3,0.5,0.8" and d["grid_unit"] == "sigma_H" and d["start"] == "star"
    assert d["ring_order"] == "lex" and d["sector_source"] == "ga" and d["seed_ga"] is not None
    assert a4.config(n4, cell, 5, None, step_units="normalized").run_id != c.run_id
    with pytest.raises(ValueError):
        a4.config(n4, dict(cell, connectivity="ring"), 5, None)
    with pytest.raises(ValueError):
        a4.config(n4, cell, 5, None, restart=1)
    e = a6.config(n4, None, 5, None, lam=0.005).to_dict()
    assert e["start"] == "hadamard" and e["lam"] == 0.005 and e["K"] is None
    with pytest.raises(ValueError):
        a6.config(n4, None, 5, None)


def test_a4_circuit_counts(n4):
    """U_0 = the S3 star prep (its counts); the phase and cost segment counts; sigma_H over the sector."""
    from gsp.arms.dbqite import confined_db_circuit, penalty_db_circuit
    from gsp.compile import transpile as tp
    from gsp.circuits import preserving as pr
    from gsp.arms.qaoa import sector_view
    C, sv = confined_db_circuit(n4, "violation", 12)
    circ = pr.build_circuit(sv, "ring", "lex")
    assert C.start_counts["cx_ii"] == tp.circuit_counts(circ)["prep"]["cx_ii"]
    assert abs(C.sigma - np.std(n4.H_obj.diagonal(sv.idx))) <= 1e-18
    rc = C.counts(3)
    assert rc["cx_ii"][1] == 3 * C.start_counts["cx_ii"] + 2 * 2 * C.cost_counts()["n_zz"] + 6 * (8 - 2) + 2
    C6 = penalty_db_circuit(n4, 0.005)
    assert C6.start_counts["cx_ii"] == 0 and C6.counts(2)["cx_ii"][2] == 8 * 2 * C6.cost_counts()["n_zz"] + 4 * 38
    # the start state of A4 is uniform over the kept strings
    psi0 = db.start_state(C.start, C.n)
    assert np.max(np.abs(np.abs(psi0[sv.idx]) ** 2 - 1 / 12)) <= 1e-14


def test_planner_makes_a4_a6_runnable_with_a_cap():
    from gsp.runner.plan import PlanFilter, PlanParams, expand, load_envelope, resolve
    env = load_envelope()
    params = PlanParams.from_envelope(env)
    params.recursion_cap = {"A4": {4: 3}, "A6": {4: 3}}
    params.lam_star = {4: 0.005}
    specs = expand(env, params, PlanFilter(arms=["A4", "A6"], N=[4], draws=1, q=[1.5]))
    assert specs and not any(s["placeholder"] for s in specs)
    out = resolve(specs)
    a4 = [s for s in out if s["arm"] == "A4"]
    assert len(a4) == 4 and all(s["cell"]["connectivity"] == "adaptive" for s in a4)   # K6, K8, K12, objective K12
    assert all(s["kw"].get("ring_order") == "lex" for s in a4)
    assert len([s for s in out if s["arm"] == "A6"]) == 1 and all(s["effort"] == 3 for s in out)


def test_resources_series():
    from gsp.metrics.resources import resources
    counts = {"charge_model": "series", "effort_unit": "step", "circuits_per_unit": 7,
              "per_circuit": {"cx_ii": 130}, "series": {"cumulative": {"cx_ii": [0, 280, 1190], "cx_iii": [0, 14, 35]}}}
    out = resources(counts, {"conv_t": 1, "conv_circuits": 7, "conv_g2q_ii": 280}, None)
    assert out["conv_exec_cx_ii"] == 280 and out["conv_exec_cx_iii"] == 14 and out["chk_conv_g2q"] is True


# --- the self-check (CPU parts) ------------------------------------------------------------------------------------------
def test_selfcheck_record_refuses_to_overwrite(tmp_path):
    from gsp.sim import selfcheck
    p = tmp_path / "fx.json"
    p.write_text("{}")
    with pytest.raises(FileExistsError):
        selfcheck.record(p)
    assert p.read_text() == "{}"


def test_selfcheck_fixture_is_complete():
    import json
    from gsp.sim import selfcheck
    p = selfcheck.fixture_path("0.15.1")
    if not p.exists():
        pytest.skip("no recorded fixture")
    fx = json.loads(p.read_text())
    assert fx["cudaq_version"] == "0.15.1" and fx["target"] == "nvidia" and fx["target_option"] == "fp64"
    assert {c["arm"] for c in fx["circuits"]} == set(selfcheck.CIRCUIT_ARMS)
    assert {c["inst_id"] for c in fx["circuits"]} == set(selfcheck.INSTANCES) and len(fx["circuits"]) == 21
    for c in fx["circuits"]:
        psi = selfcheck._unb64(c["state_b64"])
        assert psi.size == 1 << c["n"] and abs(np.vdot(psi, psi).real - 1.0) <= 1e-12
    tr = fx["trajectories"]
    assert set(tr) == set(selfcheck.TRAJECTORIES)
    assert tr["A1_adamw"]["n_iter"] == 30 and tr["A3d"]["n_iter"] == 10 and tr["A3_example"]["n_iter"] == 401
    assert len(tr["A4_db"]["series"]["E"]) == 6 and len(tr["A6_db"]["series"]["grid_idx"]) == 5


def test_d1_spec_selects_the_a4_sweep_trajectory():
    """D1 reads A4 on the fixed cell with the plan step convention, and with a4_cap only effort = cap(N)."""
    import pandas as pd
    from gsp.stats.d1 import _match, d1_spec
    spec = d1_spec(a4_cap={4: 9})["A4"]
    assert spec["connectivity"] == "adaptive" and spec["step_units"] == "plan" and spec["ring_order"] == "lex"
    base = {"arm": "A4", "status": "done", "rule": "violation", "K": 12, "connectivity": "adaptive",
            "sector_source": "ga", "ring_order": "lex", "N": 4}
    reg = pd.DataFrame([dict(base, effort=9, step_units="plan", run_id="a"),
                        dict(base, effort=5, step_units="plan", run_id="b"),
                        dict(base, effort=9, step_units="normalized", run_id="c")])
    assert _match(reg, "A4", spec)["run_id"].tolist() == ["a"]
    assert sorted(_match(reg, "A4", d1_spec()["A4"])["run_id"]) == ["a", "b"]
