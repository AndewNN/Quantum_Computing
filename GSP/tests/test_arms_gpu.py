"""S4 on the GPU (marked `gpu`): the layered kernel equals the flat interpreter, numpy and the verbatim
`kernel_qaoa_X`; the legacy equivalence at N = 4, K = 12 (the numbers the report quotes); the trainer runs
identically with the logger on and off and equals the verbatim old loop on a real kernel; an arm run writes a
valid store record and a rerun loads it; the ramp's sign; the stored completed runs replay (un-boosted circuit)."""

import numpy as np
import pytest

from gsp.circuits import preserving as pr, program
from gsp.compile import npsim
from gsp.store.paths import GSP_ROOT, inst_path, sector_path

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not (inst_path("N04e004q1.5").exists()
                                      and sector_path("N04e004", "violation", 12).exists()),
                                 reason="frozen instances / sectors absent")]
IID = "N04e004q1.5"
COMPLETED = GSP_ROOT.parent / "CUDA" / "experiments_approx_Q2_RAND_S1.0_W0.01_Jh"


@pytest.fixture(scope="module")
def be():
    from gsp.sim import backend
    backend.set_target("nvidia", "fp64")
    return backend


@pytest.fixture(scope="module")
def inst():
    from gsp.instances.instance import load_instance
    return load_instance(IID)


def _a1(inst, K=12, conn="ring", order="lex", L=3):
    from gsp.circuits.ansatz import confined_ansatz
    from gsp.sectors.select import load_sector
    return confined_ansatz(inst.H_obj, inst.boost_obj, pr.build_circuit(load_sector(IID, "violation", K), conn, order), L)


@pytest.mark.parametrize("K,conn,order", [(12, "ring", "lex"), (12, "ring", "rank"), (12, "complete", "lex"),
                                          (8, "ring", "lex")])
def test_layered_equals_interp_and_numpy_confined(be, inst, K, conn, order):
    A = _a1(inst, K, conn, order)
    rng = np.random.default_rng(K)
    flat = program.encode(A.abstract_gates(), A.n)
    for _ in range(2):
        th = np.r_[rng.uniform(-1, 1, 3) * 3e4, rng.uniform(-np.pi, np.pi, 3)]
        s = A.state(th)
        s_i = be.get_state_classical(program.kernel(), A.n, *flat.args(th))
        s_np = npsim.flat(npsim.apply(A.abstract_gates(), npsim.columns(A.n, [0]), th))[:, 0]
        assert np.max(np.abs(s - s_i)) <= 1e-12 and np.max(np.abs(s - s_np)) <= 1e-12
        e = A.energy(th)
        assert abs(e - be.observe(program.kernel(), A.op, *flat.args(th))) <= 1e-10
        diag = A.alpha * A.H.diagonal(np.arange(1 << A.n))
        assert abs(e - float(np.abs(s) ** 2 @ diag)) <= 1e-10 * max(1.0, abs(e))
        assert 1 - np.sum(np.abs(s[A.circ.order]) ** 2) <= 1e-12            # confined: no leakage


def test_layered_equals_verbatim_kernel_qaoa_X(be):
    from gsp.circuits import xkernel
    from gsp.circuits.ansatz import penalty_ansatz
    from gsp.instances.instance import load_instance
    H = load_instance("N05e000q1.5").hamiltonian(0.005)
    for boosted in (False, True):
        A = penalty_ansatz(H, 4, circuit_boosted=boosted)
        rng = np.random.default_rng(1)
        for _ in range(2):
            th = np.r_[rng.uniform(-1, 1, 4) * 1.4e4, rng.uniform(-np.pi, np.pi, 4)]
            s = A.state(th)
            s_v = be.get_state_classical(xkernel.kernel_qaoa_X, A.n, list(th), *xkernel.fixed_args(A.ct, 4))
            assert np.max(np.abs(s - s_v)) <= 1e-10
            assert abs(A.energy(th) - be.observe(xkernel.kernel_qaoa_X, A.op, list(th), *xkernel.fixed_args(A.ct, 4))) <= 1e-9


def test_legacy_equivalence_N4_K12(be, inst):
    """PLAN §5 S4: new A1 (rank ring) vs the old Pauli-expansion kernel. Not equivalent (different first-order
    product, opposite sign): the fidelity of old(gamma, beta) and new(gamma, -beta) is 1 - O(beta^4) and drops
    at the Eq. 4.11 beta range. The old kernel itself is the dense product of exp(+i beta c P) in its order."""
    import cudaq
    from tests.legacy import qaoaCUDAQ_instance as LI
    from tests.legacy import qaoaCUDAQ_kernels as LK
    from gsp.arms.qaoa import SectorView, legacy_rank
    from gsp.circuits import legacy_pauli as lp
    from gsp.circuits.ansatz import confined_ansatz
    L, K = 5, 12
    order = legacy_rank(inst, K)
    assert np.array_equal(order, np.load(sector_path("N04e004", "violation", 12))["bf_rank_idx"][:K])
    bases = lp.bases_of(order, inst.n)
    ws, cs = LK.basis_T_to_pauli_serial(bases, lp.ring_T(K), inst.n)
    w2, c2 = lp.merged_terms(bases, lp.ring_T(K))
    assert dict(zip(ws, cs)) == dict(zip(w2, c2))                         # the numpy port of the old expansion
    i1, c1, a2, b2, cc2 = LI.process_ansatz_values(-LI.qubo_to_ising(inst.QU_obj, 0.0).canonicalize())
    fixed = (int(inst.n), L, i1, c1, a2, b2, cc2, [cudaq.pauli_word(w) for w in ws], list(map(float, cs)),
             LK.reversed_str_bases_to_init_state(bases, inst.n))
    sv = SectorView(n=inst.n, idx=np.sort(order), rank_idx=order, source="bf", seed_ga=None)
    A = confined_ansatz(inst.H_obj, inst.boost_obj, pr.build_circuit(sv, "ring", "rank"), L)
    rng = np.random.default_rng(0)
    fid = {}
    for scale in (0.05, np.pi):
        f = []
        for _ in range(6):
            g, b = rng.uniform(-1, 1, L) * 1e3, rng.uniform(-1, 1, L) * scale
            s_old = be.get_state_classical(LK.kernel_qaoa_Preserved, inst.n, list(np.r_[g, b]), *fixed)
            f.append(abs(np.vdot(s_old, A.state(np.r_[g, -b]))) ** 2)
            assert 1 - np.sum(np.abs(s_old[order]) ** 2) <= 1e-11            # the old mixer does not leak either
        fid[scale] = min(f)
    assert fid[0.05] >= 1 - 1e-4
    assert fid[np.pi] < 0.99                                               # not equivalent (reported)


def test_trainer_identical_with_and_without_logger(be, inst):
    from gsp.metrics.state import StateLogger, metric_context
    from gsp.instances.instance import load_rulers
    from gsp.train.adamw import AdamWConfig, train_adamw
    from gsp.train.init import init_params
    A = _a1(inst, L=2)
    x0 = init_params(A, 777)
    cfg = AdamWConfig(max_iter=15)
    a = train_adamw(A.energy, x0, cfg)
    log = StateLogger(A, metric_context(inst, load_rulers(IID), None, sector_idx=A.circ.order))
    b = train_adamw(A.energy, x0, cfg, logger=log)
    assert np.array_equal(a.params_hist, b.params_hist) and np.array_equal(a.f_hist, b.f_hist)
    assert len(log.rows) == b.n_iter + 1
    energies = log.arrays()["energy"]
    assert np.max(np.abs(b.f_hist / A.alpha - energies[:-1])) <= 1e-10    # the loop's f and the logger agree


def test_trainer_equals_verbatim_old_loop_on_kernel_qaoa_X(be, monkeypatch):
    """The verbatim loop copy (tests/legacy) with the real cudaq.observe on the verbatim kernel vs `train_adamw`
    through `backend.observe`: identical to the bit."""
    import torch
    from tests.legacy import po_new_approxratio_train as leg
    from gsp.circuits import xkernel
    from gsp.circuits.ansatz import penalty_ansatz
    from gsp.instances.instance import load_instance
    from gsp.train.adamw import AdamWConfig, train_adamw
    from gsp.train.init import init_params
    inst = load_instance(IID)
    H = inst.hamiltonian(0.005)
    L = 2
    A = penalty_ansatz(H, L)
    fx = xkernel.fixed_args(A.ct, L)
    op = A.op
    lamb_op = be.ising_op(inst.Pen, A.alpha)
    pts, expect, n_iter = leg.legacy_train(xkernel.kernel_qaoa_X, op, op, lamb_op, fx, list(A.ct.coeff_1),
                                           list(A.ct.coeff_2), [], "X", inst.e, inst.N, 0, L, A.alpha, 1.0,
                                           torch.device("cpu"))
    from gsp.arms.base import restart_seed
    assert restart_seed(inst.draw_id, 0) == 4001 + 4099 * inst.e + 4999 * inst.N
    res = train_adamw(lambda p: be.observe(xkernel.kernel_qaoa_X, op, list(p), *fx),
                      init_params(A, restart_seed(inst.draw_id, 0)), AdamWConfig())
    assert res.n_iter == n_iter and np.array_equal(res.params, pts)
    assert np.array_equal(res.f_hist / A.alpha, np.array([r[0] for r in expect]))


def test_arm_run_store_roundtrip(be, tmp_path):
    from gsp.arms.base import validate_run_dir
    from gsp.arms.qaoa import A0, A1
    from gsp.arms.ramp import A2
    cell = {"connectivity": "ring", "rule": "violation", "K": 12}
    r = A1().run(IID, cell, 2, runs_root=tmp_path)
    assert r.status == "done" and not r.skipped and validate_run_dir(r.path) == []
    rec = r.record
    T = rec["metric_iterations"]
    assert r.trajectory["params"].shape == (T + 1, 4)
    assert rec["metric_circuits_charged"] == 5 * T and rec["metric_g2q_ii"] == 5 * T * rec["diag_cx_ii_circuit"]
    assert r.counts["per_unit"]["cx_ii"] == 5 * r.counts["per_circuit"]["cx_ii"]
    assert rec["ring_order"] == "lex" and rec["sector_source"] == "ga" and rec["seed_ga"] is not None
    assert rec["fusion_max_qubits"] == "1" and rec["target_option"] == "fp64"
    assert (r.path / "final_state.npy").exists()
    again = A1().run(IID, cell, 2, runs_root=tmp_path)
    assert again.skipped and again.run_id == r.run_id
    assert np.array_equal(again.trajectory["params"], r.trajectory["params"])
    r0 = A0().run(IID, None, 2, lam=0.005, restart=1, runs_root=tmp_path)
    assert validate_run_dir(r0.path) == [] and r0.record["seed"] != rec["seed"]
    r2 = A2("confined").run(IID, cell, 7, runs_root=tmp_path)
    assert validate_run_dir(r2.path) == [] and r2.record["metric_circuits_charged"] == 1
    assert r2.record["diag_observe_vs_state"] <= 1e-10


@pytest.mark.parametrize("enc", ["penalty", "confined"])
def test_ramp_sign(be, inst, enc):
    """The negated beta anneals down, the positive one up (the memory finding, both encodings)."""
    from gsp.arms.qaoa import confined_arm_ansatz, penalty_arm_ansatz
    from gsp.arms.base import RunConfig
    from gsp.train.schedules import ramp_params
    p = 50
    if enc == "penalty":
        A = penalty_arm_ansatz(inst, 0.005, p)
    else:
        cfg = RunConfig(arm="A2c", encoding="confined", inst_id=IID, effort_kind="ramp_depth", effort=p, K=12,
                        rule="violation", connectivity="ring")
        A, _ = confined_arm_ansatz(inst, cfg, p)
    lo = A.energy(ramp_params(p, 0.2, 3.0, A.alpha, -1))
    hi = A.energy(ramp_params(p, 0.2, 3.0, A.alpha, +1))
    start = A.energy(np.zeros(2 * p))
    assert lo < start < hi


@pytest.mark.skipif(not COMPLETED.exists(), reason="completed-work results absent")
def test_completed_runs_replay_with_unboosted_circuit(be):
    """The stored final parameters of the completed A0 run (N = 5, e = 0, L = 5, lam = 0.005) give the stored AR2
    through the harness's A0 circuit (un-boosted coefficients) to 1e-6, and not with boosted ones."""
    import pandas as pd
    from gsp.circuits.ansatz import penalty_ansatz
    from gsp.instances.encode import jh_boost
    from gsp.instances.instance import load_instance, load_rulers
    from gsp.metrics.state import metric_context
    inst, rul = load_instance("N05e000q1.5"), load_rulers("N05e000q1.5")
    H = inst.hamiltonian(0.005)
    th = np.load(COMPLETED / "exp_L0.005_q1.5" / "expectation_X_boost_Jh.npz")[f"A5_p5_E0_S0_b{jh_boost(H)}_params"]
    rep = pd.read_csv(COMPLETED / "exp_L0.005_q1.5" / "report_X_boost_Jh_AR2.csv")
    ref = float(rep[(rep.Assets == 5) & (rep.Layer == 5) & (rep.Exp == 0) & (rep.Point == 0)].Approximate_ratio.iloc[0])
    ctx = metric_context(inst, rul, 0.005)
    assert abs(ctx.evaluate_state(penalty_ansatz(H, 5).state(th))["ar_f"] - ref) <= 1e-6
    assert abs(ctx.evaluate_state(penalty_ansatz(H, 5, circuit_boosted=True).state(th))["ar_f"] - ref) > 1e-4
