"""S7 on the CPU: arm A3's McLachlan loop and estimators against `example_varqite.py`, the M1 stencil, C1, the stop
rule, the logger separation, the overlap program, the counts and the A3 / A3d configs (PLAN §1.5-§1.7, D-3).
The GPU circuits are tested in test_varqite_gpu.py."""

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from gsp.circuits.ansatz import penalty_ansatz
from gsp.instances.encode import Ising
from gsp.store.paths import GSP_ROOT, inst_path
from gsp.train.mclachlan import (McLachlanConfig, build_C, kappa_delta, metric_m1, ramp_init, run_mclachlan,
                                 spectrum)
from tests.helpers.np_varqite import NumpyEngine

LEGACY = GSP_ROOT / "tests" / "legacy" / "example_varqite.py"
ORIGINAL = Path.home() / "Desktop" / "Quantum_Master_Proposal" / "Lecture_Notes" / "code" / "example_varqite.py"
DRIVER_COPY = GSP_ROOT / "tests" / "legacy" / "varqite_driver_m1f.py"
DRIVER = GSP_ROOT.parent / "VarQITE" / "experiment" / "driver_4routes_20260916.py"


# --- example_varqite.py ------------------------------------------------------------------------------------------
def _body(path: Path) -> str:
    s = path.read_text()
    return s.split('"""\n', 1)[1] if path == LEGACY else s


@pytest.fixture(scope="module")
def ex():
    src = _body(LEGACY)
    ns = {}
    exec(src[:src.index('print("L=2 in detail:")')], ns)          # the instance and `varqite`, no prints
    return ns


def example_ising(ns) -> Ising:
    n = ns["n"]
    return Ising(n=n, const=0.0, h=np.array(ns["h"], dtype=float), J=np.triu(ns["J"], 1),
                 has_h=np.ones(n, bool), has_J=np.triu(np.ones((n, n), bool), 1))


def example_x0(L: int) -> np.ndarray:
    th = 0.05 * np.random.default_rng(1).normal(size=2 * L)          # (gamma_1, beta_1, gamma_2, ...)
    return np.r_[th[0::2], th[1::2]]                                   # -> [gamma..., beta...]


EX_CFG = dict(dtau=0.01, n_steps=401, tikhonov=1e-6, f_tol=-1.0)   # example: 401 updates, E after 400, eps 1e-6


def test_frozen_example_equals_source():
    if not ORIGINAL.exists():
        pytest.skip("the original example_varqite.py is not on this machine")
    assert _body(LEGACY) == ORIGINAL.read_text()


def test_example_instance(ex):
    H = example_ising(ex)
    assert np.max(np.abs(H.diagonal(np.arange(8)) - np.real(np.diag(ex["H_C"])))) == 0.0


@pytest.mark.parametrize("L", [1, 2, 4, 6])
def test_loop_matches_example_varqite(ex, L):
    """The harness loop with the exact estimators (M6 + exact C, test engine) reproduces the script's final energy."""
    A = penalty_ansatz(example_ising(ex), L, alpha=1.0)
    res = run_mclachlan(NumpyEngine(A), example_x0(L), McLachlanConfig(metric="exact", **EX_CFG))
    assert res.n_steps == 401
    assert abs(res.E_loop[400] - ex["varqite"](L)) <= 1e-10


def test_loop_matches_example_report_lines(ex):
    """L = 2 report: E, V, residual (= R_min), cond(M) at tau = 0, 1, ..., 4 to the printed digits."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ex["varqite"](2, report=True)
    lines = [ln.split() for ln in buf.getvalue().strip().splitlines()]
    A = penalty_ansatz(example_ising(ex), 2, alpha=1.0)
    eng = NumpyEngine(A)
    res = run_mclachlan(eng, example_x0(2), McLachlanConfig(metric="exact", **EX_CFG))
    for row, t in zip(lines, range(0, 401, 100)):
        vals = {kv.split("=")[0]: kv.split("=")[1] for kv in " ".join(row).replace("= ", "=").split()}
        psi = eng.state(res.params_hist[t])
        prob = np.abs(psi) ** 2
        E = prob @ eng.diag_H
        V = prob @ eng.diag_H ** 2 - E ** 2
        R = V - res.step["cooling"][t + 1]
        assert f"{E:+.4f}" == vals["E"] and f"{V:6.3f}".strip() == vals["V"]
        assert f"{R:6.3f}".strip() == vals["residual"]
        assert f"{res.step['cond'][t + 1]:.1e}" == vals["cond(M)"]


# --- the estimators ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ex4(ex):
    A = penalty_ansatz(example_ising(ex), 4, alpha=1.0)
    return A, NumpyEngine(A)


def test_m1_diag_is_the_variance_and_bias_is_linear_in_kappa(ex4):
    """M1's diagonal is Var(G_k) exactly; its off-diagonal stencil bias is O(delta) (forward stencil), so the error
    shrinks ~10x per decade of kappa (the cap is scaled with it)."""
    A, eng = ex4
    x = example_x0(4) + 0.3
    M = eng.exact_metric(x)
    d = eng.var_diag(x)
    assert np.max(np.abs(d - np.diag(M))) <= 1e-12
    errs = []
    for kap in (1e-2, 1e-3):
        M1, n_fid = metric_m1(eng.fidelity, x, kappa_delta(d, eng.gen_scale, kap, 2 * kap), d, 4)
        assert n_fid == 8 + 8 * 7 // 2
        assert np.allclose(M1, M1.T, atol=0)
        errs.append(np.linalg.norm(M1 - M) / np.linalg.norm(M))
    assert errs[0] < 0.02 and 6 < errs[0] / errs[1] < 14


def test_m1_central_stencil_is_second_order(ex4):
    """S9c (O-11 evidence): the central 4-point stencil needs 2p(p-1) overlaps, no F_i; its off-diagonal bias is
    O(delta^2), so it beats the forward stencil and shrinks ~100x per decade of kappa (the legacy driver's formula)."""
    A, eng = ex4
    x = example_x0(4) + 0.3
    M = eng.exact_metric(x)
    d = eng.var_diag(x)
    errs = []
    for kap in (1e-2, 1e-3):
        delta = kappa_delta(d, eng.gen_scale, kap, 2 * kap)
        Mc, n_fid = metric_m1(eng.fidelity, x, delta, d, 4, "central")
        Mf, _ = metric_m1(eng.fidelity, x, delta, d, 4)
        assert n_fid == 2 * 8 * 7
        assert np.allclose(Mc, Mc.T, atol=0) and np.array_equal(np.diag(Mc), d)
        errs.append(np.linalg.norm(Mc - M) / np.linalg.norm(M))
        assert errs[-1] < 0.1 * np.linalg.norm(Mf - M) / np.linalg.norm(M)
    assert errs[1] < errs[0] / 30 or errs[1] < 1e-9
    with pytest.raises(ValueError):
        metric_m1(eng.fidelity, x, delta, d, 4, "backward")
    with pytest.raises(ValueError):
        McLachlanConfig(stencil="backward")


def test_c1_forward_difference(ex4):
    A, eng = ex4
    x = example_x0(4) + 0.3
    C = build_C(eng.energy, x, eng.energy(x), 1e-4)
    Cx = eng.exact_C(x)
    assert np.linalg.norm(C - Cx) / np.linalg.norm(Cx) < 1e-3


def test_kappa_delta_cap():
    d = np.array([1e-4, 1.0, 0.0])
    g = np.array([2.0, 2.0, 1.0])
    np.testing.assert_allclose(kappa_delta(d, g, 0.01, 0.02), [0.005, 0.005, 0.01])
    np.testing.assert_allclose(kappa_delta(np.array([100.0]), np.array([1.0]), 0.01, 0.02), [0.001])


def test_ramp_init_verbatim():
    x = ramp_init(5)
    np.testing.assert_allclose(x[:5], [0.6, 1.2, 1.8, 2.4, 3.0])
    np.testing.assert_allclose(x[5:], [1.5, 1.2, 0.9, 0.6, 0.3])


def test_spectrum():
    A = np.diag([4.0, 2.0, 1e-9, 1e-10])
    s = spectrum(A)
    assert s["rank"] == 4 and s["sv_gap_at"] == 2 and abs(s["sv_gap"] - 2e9) < 1e-3
    assert abs(s["cond"] - 4e10) < 1 and s["eig_min"] == pytest.approx(1e-10)
    s0 = spectrum(np.diag([1.0, 1e-20]))
    assert s0["rank"] == 1


# --- the loop ----------------------------------------------------------------------------------------------------
class ScriptedEngine:
    """p = 2, metric = I (diag route). energy(): E(theta_0), then per step 2 forward calls returning the current value
    (so C = 0) and one post-update call returning the next scripted value."""
    L, n_params, gen_scale = 1, 2, np.ones(2)

    def __init__(self, vals):
        self.vals, self.n, self.k, self.cur = list(vals), 0, 0, vals[0]

    def energy(self, params):
        self.n += 1
        if self.n == 1:
            return self.vals[0]
        if (self.n - 2) % 3 < 2:
            return self.cur
        self.k += 1
        self.cur = self.vals[self.k]
        return self.cur

    def var_diag(self, params):
        return np.ones(2)

    def fidelity(self, a, b, m):
        return 1.0


def test_stop_rule_verbatim():
    """|E_t - E_{t-1}| < f_tol three consecutive times, the first comparison at t = 2 (driver `run_mcLachlan`)."""
    res = run_mclachlan(ScriptedEngine([0.0, 1.0, 2.0, 2.00001, 2.00002, 2.00003, 5.0]), np.zeros(2),
                        McLachlanConfig(metric="diag", n_steps=50))
    assert res.converged and res.n_steps == 5
    assert res.calls == {"variance": 2, "overlap": 0, "energy": 3}
    np.testing.assert_array_equal(res.circuits_charged, [0, 6, 11, 16, 21, 26])
    res = run_mclachlan(ScriptedEngine([0.0] + [float(k) for k in range(1, 10)]), np.zeros(2),
                        McLachlanConfig(metric="diag", n_steps=4))
    assert not res.converged and res.n_steps == 4


def test_logger_does_not_touch_the_loop(ex):
    A = penalty_ansatz(example_ising(ex), 2, alpha=1.0)
    cfg = McLachlanConfig(metric="M1", dtau=0.05, n_steps=15, tikhonov=1e-6)
    seen = []

    def logger(t, p):
        seen.append(t)
        p[:] = 123.0                                                   # a copy: must not reach the loop

    r1 = run_mclachlan(NumpyEngine(A), example_x0(2), cfg, logger=logger)
    r0 = run_mclachlan(NumpyEngine(A), example_x0(2), cfg, logger=None)
    np.testing.assert_array_equal(r1.params_hist, r0.params_hist)
    np.testing.assert_array_equal(r1.E_loop, r0.E_loop)
    assert seen == list(range(r1.n_steps + 1))


# --- the overlap program and the counts ----------------------------------------------------------------------------
def test_overlap_program_numpy(ex):
    """The overlap gate list (forward m layers at pb, inverse at pa, H^n) gives |<psi_m(pa)|psi_m(pb)>|^2 as its
    |0...0> probability, and the inverse layer is exact."""
    from gsp.circuits import overlap as ov
    from gsp.compile import npsim
    A = penalty_ansatz(example_ising(ex), 3, alpha=1.0)
    eng = NumpyEngine(A)
    rng = np.random.default_rng(3)
    lay = A.layer
    U = npsim.unitary(lay + ov.inverse_layer(lay), A.n, [0.7, -0.4])
    assert np.max(np.abs(U - np.eye(1 << A.n))) <= 1e-14
    for m in (1, 2, 3):
        pa, pb = rng.normal(size=6), rng.normal(size=6)
        gl = ov.overlap_gates(A.start, lay, 3, m)
        psi = npsim.flat(npsim.apply(gl, npsim.columns(A.n, [0]), np.r_[pb, pa]))[:, 0]
        assert abs(abs(psi[0]) ** 2 - eng.fidelity(pa, pb, m)) <= 1e-14
    op = ov.encode_overlap(A.start, lay, A.n, 3)
    assert op.seg == [0, 3, 3, 3 + len(lay), 3 + len(lay), 3 + 2 * len(lay), 3 + 2 * len(lay), 6 + 2 * len(lay)]
    with pytest.raises(ValueError):
        op.args(np.zeros(6), np.zeros(6), 4)


def test_a3_counts_closed_form(ex):
    from gsp.arms.varqite import a3_counts, charged_series
    from gsp.compile import tcount
    H = example_ising(ex)
    for L in (1, 3, 5):
        A = penalty_ansatz(H, L, alpha=1.0)
        p = 2 * L
        cx = 2 * A.ct.n_zz                                              # one cost layer (ii) = (iii)
        c = a3_counts(A, "M1")
        cl = c["classes"]
        assert cl["overlap"]["circuits"] == p * (p + 1) // 2 and cl["variance"]["circuits"] == p
        assert cl["energy"]["circuits"] == p + 1 and c["circuits_per_unit"] == p * (p + 1) // 2 + 2 * p + 1
        assert cl["energy"]["cx_ii"] == (p + 1) * L * cx
        assert cl["variance"]["cx_ii"] == L * (L + 1) * cx              # 2 sum_k (k + 1)
        # overlap: F_i at 2(l_i + 1) layers, F_ij at 2(max + 1) layers
        lay = [k % L for k in range(p)]
        want = sum(2 * (lay[i] + 1) for i in range(p)) + sum(2 * (max(lay[i], lay[j]) + 1)
                                                             for i in range(p) for j in range(i + 1, p))
        assert cl["overlap"]["cx_ii"] == want * cx
        assert c["per_unit"]["cx_ii"] == cl["energy"]["cx_ii"] + cl["variance"]["cx_ii"] + cl["overlap"]["cx_ii"]
        ts = tcount.t_syn()
        assert c["per_circuit"]["t_ii"] == L * (A.ct.n_rot + A.n) * ts
        d = a3_counts(A, "diag")
        assert d["classes"]["overlap"]["circuits"] == 0 and d["circuits_per_unit"] == 2 * p + 1
        s = charged_series(c, 3)
        np.testing.assert_array_equal(s["circuits_charged"], [0, c["circuits_per_unit"] + 1,
                                                              2 * c["circuits_per_unit"] + 1,
                                                              3 * c["circuits_per_unit"] + 1])
        assert s["g2q_ii"][1] == c["per_unit"]["cx_ii"] + L * cx


def test_resources_per_unit_charge_model():
    from gsp.metrics.convergence import convergence
    from gsp.metrics.resources import resources
    counts = {"charge_model": "per_unit", "effort_unit": "step", "circuits_per_unit": 10,
              "per_unit": {"cx_ii": 100, "cx_iii": 100, "t_ii": 5, "t_iii": 5, "tdepth_ii": 1, "tdepth_iii": 1},
              "per_circuit": {"cx_ii": 7, "cx_iii": 7}, "first_unit_extra": {"circuits": 1, "cx_ii": 7, "cx_iii": 7}}
    tr = {"t": np.arange(4), "ar_f": np.array([0.1, 0.5, 0.92, 0.95]), "circuits_charged": np.array([0, 11, 21, 31]),
          "g2q_ii": np.array([0, 107, 207, 307]), "g2q_iii": np.array([0, 107, 207, 307])}
    conv = convergence(tr)
    r = resources(counts, conv, tr)
    assert conv["conv_t"] == 2 and r["conv_exec_cx_ii"] == 207 and r["chk_conv_g2q"] is True


# --- configs -----------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not inst_path("N04e004q1.5").exists(), reason="frozen instances absent")
def test_a3_configs_registered():
    from gsp.arms.base import make_arm, registered_arms
    from gsp.instances.instance import load_instance
    assert {"A3", "A3d"} <= set(registered_arms())
    inst = load_instance("N04e004q1.5")
    a3, a3d = make_arm("A3"), make_arm("A3d")
    c = a3.config(inst, None, 5, None, lam=0.005)
    d = a3d.config(inst, None, 5, None, lam=0.005)
    assert c.run_id != d.run_id and c.extra("metric") == "M1" and d.extra("metric") == "diag"
    assert c.extra("n_steps") == 300 and c.extra("dtau") == 0.1 and c.extra("tikhonov") == 1e-6
    assert c.seed == a3.config(inst, None, 5, None, lam=0.005).seed and c.run_id == a3.config(inst, None, 5, None, lam=0.005).run_id
    assert a3.config(inst, None, 5, None, lam=0.005, n_steps=10).run_id != c.run_id
    with pytest.raises(ValueError):
        a3.config(inst, None, 5, None, lam=0.005, restart=1)
    with pytest.raises(ValueError):
        a3.config(inst, None, 5, None)
    mc = a3.mc_config(c)
    assert (mc.metric, mc.dtau, mc.n_steps, mc.tikhonov, mc.fd_shift, mc.kappa, mc.cap_angle, mc.f_tol) == \
        ("M1", 0.1, 300, 1e-6, 1e-4, 0.01, 0.02, 1e-4)


@pytest.mark.skipif(not inst_path("N07e000q1.5").exists(), reason="frozen instances absent")
def test_a3_s9c_flags():
    """S9c: circuit_boosted / stencil hash only when set (every earlier run_id is unchanged: the O-11 sweep-config run
    of S9a queue 3 keeps its id), reach the ansatz / loop / counts, and theta_0 is the same physical state."""
    from gsp.arms.base import make_arm
    from gsp.arms.varqite import a3_counts
    from gsp.instances.instance import load_instance
    inst = load_instance("N07e000q1.5")
    a3 = make_arm("A3")
    c = a3.config(inst, None, 5, None, lam=0.005)
    assert c.run_id == "9ee24c2aaa06283d"                      # S9a queue 3, "O-11 A3 (a) jitter none (sweep config)"
    assert c.run_id == a3.config(inst, None, 5, None, lam=0.005, circuit_boosted=False, stencil="forward").run_id
    b = a3.config(inst, None, 5, None, lam=0.005, circuit_boosted=True)
    cs = a3.config(inst, None, 5, None, lam=0.005, stencil="central", evidence="O-11")
    assert len({c.run_id, b.run_id, cs.run_id}) == 3
    assert b.extra("circuit_boosted") is True and c.extra("circuit_boosted") is None
    assert a3.mc_config(cs).stencil == "central" and a3.mc_config(c).stencil == "forward"
    with pytest.raises(ValueError):
        a3.config(inst, None, 5, None, lam=0.005, stencil="backward")
    with pytest.raises(ValueError):
        make_arm("A3d").config(inst, None, 5, None, lam=0.005, stencil="central")
    Ab, _ = a3.ansatz(b, inst)
    Au, _ = a3.ansatz(c, inst)
    assert Ab.meta["circuit_boosted"] and not Au.meta["circuit_boosted"]
    np.testing.assert_allclose(np.array(Ab.ct.coeff_2), Ab.alpha * np.array(Au.ct.coeff_2), rtol=1e-12)
    x_u, x_b = a3.theta0(c, 5, Au.alpha), a3.theta0(b, 5, Ab.alpha)
    np.testing.assert_array_equal(x_u, ramp_init(5))
    np.testing.assert_allclose(x_b[:5] * Ab.alpha, x_u[:5], rtol=1e-15)
    np.testing.assert_array_equal(x_b[5:], x_u[5:])
    p = 10
    cc = a3_counts(Au, "M1", "central")
    assert cc["classes"]["overlap"]["circuits"] == 2 * p * (p - 1) and cc["stencil"] == "central"
    assert cc["circuits_per_unit"] == 2 * p * (p - 1) + 2 * p + 1 and "stencil" not in a3_counts(Au, "M1")


def test_d1_spec_takes_only_the_sweep_a3():
    from gsp.stats.d1 import d1_spec
    assert d1_spec({4: 0.005})["A3"] == {"_lam": {4: 0.005}, "metric": "M1", "n_steps": 300}


def test_frozen_driver_copy_equals_source():
    if not DRIVER.exists():
        pytest.skip("VarQITE/experiment/driver_4routes_20260916.py is not on this machine")
    src = DRIVER.read_text().splitlines()
    body = DRIVER_COPY.read_text()
    for a, b in ((66, 88), (90, 102), (105, 147), (150, 479)):
        assert "\n".join(src[a - 1:b]) in body


@pytest.mark.skipif(not inst_path("N04e004q1.5").exists(), reason="frozen instances absent")
def test_a3_boosted_circuit_is_the_same_flow():
    """S9c (O-2, §1.5): on the boosted circuit, from theta_0 = the Ramp init with gamma / alpha, the McLachlan flow is
    the same (covariant under gamma -> gamma / alpha): with the exact M and C and a negligible Tikhonov, E agrees to
    rounding until the loop's chaos (O-11) amplifies it (numpy engine, n = 8, L = 2; measured 1e-9 over 3 steps, 0.16
    by step 4). M1's estimators are not covariant: the FD shift 1e-4 and Tikhonov are fixed in parameter units, so on
    the boosted circuit the gamma FD step is alpha x larger physically and Tikhonov no longer dominates the gamma block
    (the O-11 evidence measures what that does)."""
    from gsp.arms.base import make_arm
    from gsp.instances.instance import load_instance
    inst = load_instance("N04e004q1.5")
    a3 = make_arm("A3")
    out = {}
    for boosted in (False, True):
        cfg = a3.config(inst, None, 2, None, lam=0.005, circuit_boosted=boosted, n_steps=3)
        A, _ = a3.ansatz(cfg, inst)
        mc = McLachlanConfig(metric="exact", n_steps=3, f_tol=-1.0, tikhonov=1e-12)
        res = run_mclachlan(NumpyEngine(A), a3.theta0(cfg, 2, A.alpha), mc)
        scale = np.r_[np.full(2, A.alpha if boosted else 1.0), np.ones(2)]
        out[boosted] = (res.E_loop, res.params_hist * scale)
    assert out[True][0][0] == pytest.approx(out[False][0][0], abs=1e-12)
    np.testing.assert_allclose(out[True][0], out[False][0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(out[True][1], out[False][1], rtol=1e-5, atol=1e-8)
