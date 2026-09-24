"""S7 on the GPU (marked `gpu`): the flag-qubit P(0...0) equals the statevector P(0...0) and the old full-depth
"prob" route; the truncated variance circuits; the GPU M1 loop equals the numpy one on example_varqite.py's
instance; A3d reproduces the stored M4C1 trajectory; A3's first M1 steps against the stored M1fC1 run; an A3 / A3d
arm run writes a valid, finalizable record; the logger does not touch the loop."""

import json

import numpy as np
import pytest

from gsp.store.paths import GSP_ROOT, inst_path

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not (inst_path("N04e004q1.5").exists() and inst_path("N07e000q1.5").exists()),
                                 reason="frozen instances absent")]
STORED = GSP_ROOT.parent / "VarQITE" / "experiment" / "exp_Q2_L0.005_q1.5"
KEY = "A7_p5_E0_S0"


@pytest.fixture(scope="module")
def be():
    from gsp.sim import backend
    backend.set_target("nvidia", "fp64")
    return backend


def _ansatz(iid, L, lam=0.005):
    from gsp.arms.qaoa import penalty_arm_ansatz
    from gsp.instances.instance import load_instance
    return penalty_arm_ansatz(load_instance(iid), lam, L)


def _legacy_kernels():
    lk = pytest.importorskip("tests.legacy.varqite_driver_m1f")
    return lk.kernel_qaoa_X_overlap, lk.kernel_qaoa_X_trunc


@pytest.mark.parametrize("iid,L", [("N04e004q1.5", 3), ("N07e000q1.5", 5)])
def test_flag_p0_equals_statevector_and_old_prob_route(be, iid, L):
    from gsp.arms.varqite import CircuitEngine
    from gsp.circuits import xkernel
    from gsp.train.mclachlan import ramp_init
    A = _ansatz(iid, L)
    eng = CircuitEngine(A)
    ovl, _ = _legacy_kernels()
    fixed = xkernel.fixed_args(A.ct, L)
    rng = np.random.default_rng(L)
    err_sv = err_old = 0.0
    for m in range(1, L + 1):
        for scale in (0.02, 0.5):
            pa = ramp_init(L) + rng.normal(size=2 * L)
            pb = pa.copy()
            for k in rng.choice([j for j in range(2 * L) if j % L < m], size=2, replace=False):
                pb[k] += scale * rng.normal()
            f = eng.fidelity(pa, pb, m)
            err_sv = max(err_sv, abs(f - eng.fidelity_statevector(pa, pb, m)))
            amp0 = np.asarray(be.get_state(ovl, list(pa), list(pb), *fixed))[0]     # the old "prob" route, depth L
            err_old = max(err_old, abs(f - abs(amp0) ** 2))
    assert err_sv <= 1e-12 and err_old <= 1e-12


def test_var_diag_equals_numpy_and_old_prob_route(be):
    from gsp.arms.varqite import CircuitEngine
    from gsp.circuits import xkernel
    from gsp.instances.bits import cudaq_to_classical
    from tests.helpers.np_varqite import NumpyEngine
    A = _ansatz("N04e004q1.5", 3)
    eng, npe = CircuitEngine(A), NumpyEngine(A)
    _, trunc = _legacy_kernels()
    fixed = xkernel.fixed_args(A.ct, 3)
    x = np.random.default_rng(0).normal(size=6)
    d = eng.var_diag(x)
    # Var = <G^2> - <G>^2 cancels: the rounding floor is ~eps x <G^2> (<G_beta^2> ~ n^2), not relative to Var
    assert np.max(np.abs(d - npe.var_diag(x))) <= 1e-12 * A.n ** 2
    # the old truncated kernel (n_cost, n_mix) with the old measured-basis values ("prob" route)
    n = A.n
    for k in range(6):
        n_cost, n_mix, xb = (k + 1, k, 0) if k < 3 else (k - 2, k - 2, 1)
        psi = np.asarray(be.get_state(trunc, list(x), *fixed, n_cost, n_mix, xb))
        prob = np.abs(psi) ** 2
        if xb:
            vals = n - 2.0 * np.array([bin(i).count("1") for i in range(1 << n)])
        else:
            vals = A.H.diagonal(np.arange(1 << n))
            prob = np.abs(cudaq_to_classical(psi, n)) ** 2
        m1 = prob @ vals
        assert abs((prob @ vals ** 2 - m1 ** 2) - d[k]) <= 1e-12 * max(1.0, float(prob @ vals ** 2))


@pytest.mark.parametrize("L", [1, 2])
def test_gpu_m1_loop_equals_numpy_on_the_example(be, L):
    """example_varqite.py's instance, the production estimators (M1 + C1), 401 steps: GPU circuits == numpy states,
    final energy == the script's (exact-M flow) to 1e-6. At L = 2 the flow passes an ill-conditioned stretch
    (cond(M) ~ 1e3-1e8) where 1e-15 differences grow to ~1e-4 around step 40 and then contract again (STATUS S7), so
    only the first 10 steps and the end are compared there."""
    from gsp.arms.varqite import CircuitEngine
    from gsp.circuits.ansatz import penalty_ansatz
    from gsp.train.mclachlan import McLachlanConfig, run_mclachlan
    from tests.helpers.np_varqite import NumpyEngine
    from tests.test_varqite import EX_CFG, example_ising, example_x0
    ns = {}
    src = (GSP_ROOT / "tests" / "legacy" / "example_varqite.py").read_text().split('"""\n', 1)[1]
    exec(src[:src.index('print("L=2 in detail:")')], ns)
    A = penalty_ansatz(example_ising(ns), L, alpha=1.0)
    cfg = McLachlanConfig(metric="M1", **EX_CFG)
    rg = run_mclachlan(CircuitEngine(A), example_x0(L), cfg)
    rn = run_mclachlan(NumpyEngine(A), example_x0(L), cfg)
    d = np.abs(rg.E_loop - rn.E_loop)
    if L == 1:
        assert d.max() <= 1e-10
    else:
        assert d[:11].max() <= 1e-9 and d[-1] <= 1e-9
    assert abs(rg.E_loop[400] - ns["varqite"](L)) <= 1e-6


def _stored(route):
    f = STORED / f"expectation_{route}_Ramp_boost_Jh.npz"
    if not f.exists():
        pytest.skip("the stored VarQITE route sweep is not on this machine")
    z = np.load(f)
    return z[f"{KEY}_history"], json.loads(str(z[f"{KEY}_cfg"]))


def _run(route, n_steps, logger=True):
    from gsp.arms.varqite import run_a3
    from gsp.instances.instance import load_instance, load_rulers
    from gsp.metrics.state import metric_context
    from gsp.train.mclachlan import McLachlanConfig
    inst = load_instance("N07e000q1.5")
    A = _ansatz("N07e000q1.5", 5)
    ctx = metric_context(inst, load_rulers("N07e000q1.5"), 0.005)
    return A, run_a3(A, McLachlanConfig(metric="M1" if route == "M1fC1" else "diag", n_steps=n_steps), ctx=ctx,
                     logger=logger)


def test_a3d_reproduces_the_stored_m4c1_trajectory(be):
    """N07e000q1.5, lam 0.005, L = 5: same stop step (33), energy per step to 1e-6 relative (measured 3.4e-9)."""
    hist, cfg = _stored("M4C1")
    assert cfg["A_METHOD"] == "diag" and cfg["PRECISION"] == "fp64" and cfg["TIKHONOV_LAMBDA"] == 1e-6
    A, (res, rows, _, _) = _run("M4C1", 300)
    assert res.n_steps == len(hist) == 33 and res.converged
    rel = np.abs(rows["energy"][1:] - hist[:, 0]) / np.abs(hist[:, 0])
    assert rel.max() <= 1e-6
    rmin = rows["V_tau"][:-1] * A.alpha ** 2 - res.step["cooling"][1:]
    np.testing.assert_allclose(rmin, hist[:, 8], rtol=1e-5, atol=1e-8)
    assert int((rmin < 0).sum()) == int((hist[:, 8] < 0).sum()) == 11


def test_a3_first_m1_steps_against_the_stored_m1fc1(be):
    """M1fC1 is chaotic at this Tikhonov (STATUS S7): the first two steps agree to 1e-6 relative, later ones do not
    (neither does the verbatim old driver on CUDA-Q 0.15.1)."""
    hist, cfg = _stored("M1fC1")
    assert cfg["A_METHOD"] == "fidelity_kappa" and cfg["FID_STENCIL"] == "forward" and cfg["EXACT_MEASURE"] == "prob"
    A, (res, rows, _, _) = _run("M1fC1", 2)
    rel = np.abs(rows["energy"][1:] - hist[:2, 0]) / np.abs(hist[:2, 0])
    assert rel.max() <= 1e-6
    np.testing.assert_allclose(res.params_hist[1:, 0], hist[:2, 6], rtol=1e-5)


def test_logger_on_off_identical_gpu(be):
    _, (r1, _, _, _) = _run("M1fC1", 3, logger=True)
    _, (r0, rows0, _, _) = _run("M1fC1", 3, logger=False)
    assert rows0 is None
    np.testing.assert_array_equal(r1.params_hist, r0.params_hist)
    np.testing.assert_array_equal(r1.E_loop, r0.E_loop)


@pytest.mark.parametrize("arm", ["A3", "A3d"])
def test_arm_run_writes_a_valid_record(be, tmp_path, arm):
    from gsp.arms.base import make_arm, validate_run_dir
    from gsp.metrics.aggregate import run_row
    from gsp.store.index import build_index
    a = make_arm(arm)
    r = a.run("N04e004q1.5", None, 2, lam=0.005, n_steps=3, runs_root=tmp_path)
    assert r.status == "done" and not validate_run_dir(r.path)
    tr = r.trajectory
    T = int(r.record["metric_iterations"])
    assert tr["t"].size == T + 1 and tr["sv"].shape == (T + 1, 4) and np.isnan(tr["cond_M"][0])
    c = r.counts
    assert tr["circuits_charged"][-1] == T * c["circuits_per_unit"] + 1
    assert tr["g2q_ii"][-1] == T * c["per_unit"]["cx_ii"] + c["first_unit_extra"]["cx_ii"]
    assert c["classes"]["overlap"]["circuits"] == (10 if arm == "A3" else 0)
    assert r.record["diag_loop_vs_logger"] <= 1e-12
    post = json.loads((r.path / "postrun.json").read_text())
    assert post["status"] == "done" and post["replay_max_abs"] <= 1e-10
    again = a.run("N04e004q1.5", None, 2, lam=0.005, n_steps=3, runs_root=tmp_path)
    assert again.skipped and again.run_id == r.run_id
    reg = build_index(tmp_path)
    row = run_row(reg.iloc[0].to_dict(), root=tmp_path, inputs_root=None)
    assert row["anomalies"] == "" and row["chk_conv_g2q"] is True
