"""S4 on the CPU: the simulation rewrites are exact, the layered encoding is lossless, the trainer is the old loop
bit for bit, the init / ramp / RunConfig conventions, the metric context, the legacy Pauli-coefficient port."""

import importlib.util

import numpy as np
import pytest

from gsp.arms.base import RunConfig, make_extras
from gsp.circuits import preserving as pr, program
from gsp.circuits.cost import CostTerms, cost_gates, cost_gates_sim
from gsp.circuits.simopt import merge_x
from gsp.circuits.xmixer import a0_gates
from gsp.compile import npsim
from gsp.compile.decompose import Angle, Gate
from gsp.store.paths import GSP_ROOT, inst_path, sector_path
from gsp.train.init import mm_i_legacy, random_points
from gsp.train.schedules import SCHEDULES, ramp_params

HAVE_INST = inst_path("N04e004q1.5").exists() and sector_path("N04e004", "violation", 12).exists()
COMPLETED = GSP_ROOT.parent / "CUDA" / "experiments_approx_Q2_RAND_S1.0_W0.01_Jh"


def _state(gates, n, params):
    return npsim.flat(npsim.apply(gates, npsim.columns(n, [0]), params))[:, 0]


def _random_ct(n, rng):
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return CostTerms(n=n, idx_1=tuple(range(n)), coeff_1=tuple(rng.normal(size=n)),
                     idx_2a=tuple(a for a, _ in pairs), idx_2b=tuple(b for _, b in pairs),
                     coeff_2=tuple(rng.normal(size=len(pairs))))


# --- generated kernel file ------------------------------------------------------------------------------
def test_layered_kernel_is_generated():
    spec = importlib.util.spec_from_file_location("gen_layered", GSP_ROOT / "scripts" / "gen_layered_kernel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (GSP_ROOT / "gsp" / "circuits" / "layered.py").read_text() == mod.source()


# --- exact rewrites --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5])
def test_cost_gates_sim_is_the_same_unitary(n):
    rng = np.random.default_rng(n)
    ct = _random_ct(n, rng)
    g = [0.37]
    U = npsim.unitary(cost_gates(ct, 0), n, g)
    V = npsim.unitary(cost_gates_sim(ct, 0), n, g)
    assert np.max(np.abs(U - V)) <= 1e-12
    assert len(cost_gates_sim(ct, 0)) == ct.n_zz + n < len(cost_gates(ct, 0))


def _random_list(n, m, rng):
    out = []
    names = ["x", "x", "x", "cx", "rx", "ry", "rz", "h", "crz", "mcrx", "mcry"]
    for _ in range(m):
        nm = names[rng.integers(len(names))]
        if nm in ("x", "h"):
            out.append(Gate(nm, (int(rng.integers(n)),)))
        elif nm in ("rx", "ry", "rz"):
            out.append(Gate(nm, (int(rng.integers(n)),), Angle(float(rng.normal()), 0)))
        else:
            k = 1 if nm in ("cx", "crz") else int(rng.integers(1, n))
            qs = tuple(int(q) for q in rng.permutation(n)[:k + 1])
            out.append(Gate(nm, qs, None if nm == "cx" else Angle(float(rng.normal()), -1)))
    return out


@pytest.mark.parametrize("seed", range(8))
def test_merge_x_is_the_same_unitary(seed):
    rng = np.random.default_rng(100 + seed)
    n = 4
    gl = _random_list(n, 40, rng)
    U = npsim.unitary(gl, n, [0.7])
    V = npsim.unitary(merge_x(gl), n, [0.7])
    assert np.max(np.abs(U - V)) <= 1e-12
    assert sum(g.name == "x" for g in merge_x(gl)) <= sum(g.name == "x" for g in gl)


@pytest.mark.skipif(not HAVE_INST, reason="frozen instances / sectors absent")
@pytest.mark.parametrize("conn,K,order", [("ring", 12, "lex"), ("ring", 12, "rank"), ("complete", 12, "lex"),
                                          ("ring", 6, "lex")])
def test_confined_simulation_program_equals_abstract_circuit(conn, K, order):
    from gsp.circuits.ansatz import confined_ansatz
    from gsp.instances.instance import load_instance
    from gsp.sectors.select import load_sector
    inst = load_instance("N04e004q1.5")
    circ = pr.build_circuit(load_sector("N04e004q1.5", "violation", K), conn, order)
    for boosted in (False, True):
        A = confined_ansatz(inst.H_obj, inst.boost_obj, circ, 3, circuit_boosted=boosted)
        th = np.random.default_rng(K).uniform(-2, 2, 6) * np.r_[np.full(3, 50.0), np.ones(3)]
        a, b = _state(A.abstract_gates(), A.n, th), _state(A.unrolled(), A.n, th)
        assert np.max(np.abs(a - b)) <= 1e-12
        assert A.prog.n_gates < len(A.abstract_gates())


@pytest.mark.skipif(not HAVE_INST, reason="frozen instances absent")
def test_penalty_simulation_program_equals_kernel_qaoa_X_list():
    from gsp.circuits.ansatz import penalty_ansatz
    from gsp.instances.instance import load_instance
    H = load_instance("N04e004q1.5").hamiltonian(0.005)
    A = penalty_ansatz(H, 3)
    th = np.r_[np.random.default_rng(3).uniform(-300, 300, 3), np.random.default_rng(4).uniform(-3, 3, 3)]
    a, b = _state(a0_gates(A.ct, 3), A.n, th), _state(A.unrolled(), A.n, th)
    assert np.max(np.abs(a - b)) <= 1e-12
    # the abstract list is kernel_qaoa_X gate for gate: H^n, then per layer rz per field, cx-rz-cx per pair, rx
    gl = a0_gates(A.ct, 1)
    assert [g.name for g in gl[:A.n]] == ["h"] * A.n and [g.name for g in gl[-A.n:]] == ["rx"] * A.n
    assert sum(g.name == "cx" for g in gl) == 2 * A.ct.n_zz == A.counts["layer"]["cx_ii"]
    assert A.counts["per_circuit"]["cx_ii"] == 3 * A.counts["layer"]["cx_ii"]


def test_encode_layered_packs_losslessly():
    rng = np.random.default_rng(5)
    n = 7
    start = [Gate("x", (1,)), Gate("h", (2,)), Gate("mcry", (0, 3, 4), Angle(0.3, -1))]
    layer = [Gate("crz", (0, 6), Angle(-0.5, 0)), Gate("rz", (6,), Angle(0.25, 0)), Gate("cx", (2, 5)),
             Gate("mcrx", (1, 2, 3, 6), Angle(2.0, 1)), Gate("rx", (4,), Angle(2.0, 1)), Gate("ry", (0,), Angle(0.1, -1))]
    lp = program.encode_layered(start, layer, n, 3)
    assert lp.seg == [0, 3, 3, 9] and lp.n_gates == 3 + 3 * 6
    for g, code, coef in zip(start + layer, lp.code, lp.coef):
        o, rest = code % 16, code // 16
        t, rest = rest % 32, rest // 32
        k, rest = rest % 32, rest // 32
        sl, c = rest % 4, rest // 4
        assert o == program.OPCODES[g.name] and t == g.qubits[-1] and k == len(g.qubits) - 1
        assert lp.ctl[c:c + k] == list(g.qubits[:-1]) or k == 0
        assert sl == (0 if g.angle is None or g.angle.pidx < 0 else g.angle.pidx + 1)
        assert coef == (0.0 if g.angle is None else g.angle.coef)
    assert len(lp.args(rng.normal(size=6))) == 7
    with pytest.raises(ValueError):
        program.encode_layered([Gate("rx", (0,), Angle(1.0, 0))], [], n, 2)      # start gates must be constant
    with pytest.raises(ValueError):
        program.encode_layered([], [Gate("rx", (0,), Angle(1.0, 2))], n, 2)      # pidx must be 0 or 1
    with pytest.raises(ValueError):
        lp.args([0.0] * 5)
    flat = program.unroll_layered(start, layer, 3)
    assert [g.angle.pidx for g in flat if g.angle is not None and g.angle.pidx >= 0] == [0, 0, 3, 3, 1, 1, 4, 4, 2, 2, 5, 5]


# --- conventions ---------------------------------------------------------------------------------------------
def test_ramp_params_convention():
    p, db, dg, a = 5, 0.2, 3.0, 150.3
    th = ramp_params(p, db, dg, a)
    for i in range(p):
        assert th[i] == dg * a * (i + 1) / p
        assert th[p + i] == -db * (1 - i / p)
    assert np.array_equal(ramp_params(p, db, dg, a, sign=+1)[p:], -th[p:])
    assert SCHEDULES == {"primary": (0.2, 3.0), "secondary": (1.5, 3.0)}
    with pytest.raises(ValueError):
        ramp_params(3, db, dg, a, sign=0)


def test_random_init_is_the_old_code():
    """Lines 839-845 of PO_new_ApproxRatio.py with the GLOBAL np.random, replayed."""
    for seed, L, mm in [(28996, 5, 116.5), (4001, 9, np.float32(1608.5))]:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            points = np.random.uniform(-1, 1, (2 * L))
            points[:L] *= mm
            points[L:] *= np.pi
        finally:
            np.random.set_state(state)
        assert np.array_equal(random_points(seed, L, mm), points)


def test_mm_i_legacy_keeps_the_old_float32():
    c1, c2 = [0.5, -0.25], [0.125]
    assert mm_i_legacy(c1, c2) == np.pi / 0.125
    v = mm_i_legacy(c1, c2, np.float32(2.0 ** -9))
    assert isinstance(v, np.float32) and v == np.float32(np.pi / np.float32(2.0 ** -9))


@pytest.mark.skipif(not (COMPLETED.exists() and HAVE_INST), reason="completed-work results absent")
def test_init_reproduces_the_stored_first_iterates():
    """theta_1 = theta_0 (1 - lr wd) - lr g / (|g| + 1e-8): the stored first iterate is within lr = 0.01 of
    theta_0 (1 - 1e-4) only if the init (seed table, un-boosted kappa_min, legacy mm_p) is the old one."""
    from gsp.arms.base import restart_seed
    from gsp.arms.qaoa import legacy_rank
    from gsp.circuits.ansatz import confined_ansatz, penalty_ansatz
    from gsp.instances.encode import jh_boost
    from gsp.instances.instance import load_instance
    from gsp.train.init import init_params
    zx = np.load(COMPLETED / "exp_L0.005_q1.5" / "expectation_X_boost_Jh.npz")
    zp = np.load(COMPLETED / "exp_L1_q1.5" / "expectation_Preserving12_boost_Jh.npz")

    def b(x):
        return str(int(x)) if float(x).is_integer() else repr(float(x))

    for e in (0, 1, 3):
        inst = load_instance(f"N05e{e:03d}q1.5")
        seed = restart_seed(f"N05e{e:03d}", 0)
        H = inst.hamiltonian(0.005)
        A0 = penalty_ansatz(H, 5)
        tr = zx[f"A5_p5_E{e}_S0_b{b(jh_boost(H))}"]
        assert np.abs(tr[0, 3:5] - init_params(A0, seed)[:2] * (1 - 1e-4)).max() <= 0.01 + 1e-6
        order = legacy_rank(inst, 12)
        sv = type("S", (), {"n": inst.n, "idx": np.sort(order), "rank_idx": order})
        A1 = confined_ansatz(inst.H_obj, inst.boost_obj, pr.build_circuit(sv, "ring", "rank"), 5)
        trp = zp[f"A5_p5_E{e}_S0_b{b(inst.boost_obj)}"]
        assert np.abs(trp[0, 3:5] - init_params(A1, seed, legacy=True)[:2] * (1 - 1e-4)).max() <= 0.01 + 1e-6
        # the boosted-circuit convention would put theta_0 alpha times closer to 0: far outside lr
        Ab = penalty_ansatz(H, 5, circuit_boosted=True)
        assert np.abs(tr[0, 3:5] - init_params(Ab, seed)[:2] * (1 - 1e-4)).max() > 10


def test_runconfig_hash_and_extras():
    base = dict(arm="A1", encoding="confined", inst_id="N04e004q1.5", effort_kind="depth", effort=5, K=12,
                rule="violation", connectivity="ring", restart=0, seed=24193)
    a = RunConfig(**base, extras=make_extras(ring_order="lex", init="random", grad="fd_forward"))
    b = RunConfig(**base, extras=make_extras(grad="fd_forward", init="random", ring_order="lex"))
    c = RunConfig(**base, extras=make_extras(ring_order="rank", init="random", grad="fd_forward"))
    assert a.run_id == b.run_id != c.run_id
    d = a.to_dict()
    assert d["ring_order"] == "lex" and d["harness_version"] and d["lam"] is None
    with pytest.raises(KeyError):
        RunConfig(**base, extras=(("arm", "A0"),)).to_dict()
    assert make_extras(x=None, y=1) == (("y", 1),)


# --- the trainer is the old loop ----------------------------------------------------------------------------
class _FakeObs:
    def __init__(self, v):
        self.v = v

    def expectation(self):
        return self.v


def _toy_energy(p):
    p = np.asarray(p, dtype=np.float64)
    return float(np.sum(np.sin(p) * np.arange(1, p.size + 1)) + 0.3 * np.sum(np.cos(2 * p[::2])))


def test_trainer_is_the_old_loop_bit_for_bit(monkeypatch):
    """The verbatim copy of PO_new_ApproxRatio.py:812-926 (tests/legacy) and `train_adamw` + `ForwardFD` on the
    same energy give identical parameters, energies and stopping iteration."""
    import torch
    from tests.legacy import po_new_approxratio_train as leg
    from gsp.train.adamw import train_adamw
    monkeypatch.setattr(leg.cudaq, "observe",
                        lambda kernel, H, params, *a: _FakeObs(1.0 if H == "lamb" else _toy_energy(params)))
    L = 3
    c1, c2 = [0.02, -0.7], [0.05, 0.9]
    for e in (0, 1):
        pts, expect, n_iter = leg.legacy_train(None, "ansatz", "eval", "lamb", (), c1, c2, [], "X", e, 4, 0, L, 1.0,
                                               1.0, torch.device("cpu"))
        seed = 4001 + 4099 * e + 4999 * 4
        res = train_adamw(_toy_energy, random_points(seed, L, mm_i_legacy(c1, c2)))
        assert res.n_iter == n_iter
        assert np.array_equal(res.params, pts)
        assert np.array_equal(res.f_hist, np.array([r[0] for r in expect]))
        assert np.array_equal(res.params_hist[1:, :2], np.array([r[3:5] for r in expect]))


def test_logger_never_reaches_the_update():
    from gsp.train.adamw import AdamWConfig, train_adamw
    seen = []

    def nasty_logger(t, p):          # mutates what it gets and burns time: must change nothing
        seen.append(t)
        p[:] = 1e9

    x0 = np.linspace(-1, 1, 6)
    a = train_adamw(_toy_energy, x0, AdamWConfig(max_iter=40))
    b = train_adamw(_toy_energy, x0, AdamWConfig(max_iter=40), logger=nasty_logger)
    assert np.array_equal(a.params_hist, b.params_hist) and np.array_equal(a.f_hist, b.f_hist)
    assert seen == list(range(b.n_iter + 1))
    assert np.array_equal(b.circuits_charged, np.arange(b.n_iter + 1) * 7)


# --- metrics -------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_INST, reason="frozen instances absent")
def test_metric_context_on_simple_states():
    from gsp.instances.instance import load_instance, load_rulers
    from gsp.metrics.state import metric_context
    inst, rul = load_instance("N04e004q1.5"), load_rulers("N04e004q1.5")
    ctx = metric_context(inst, rul, None)
    prob = np.zeros(1 << inst.n)
    prob[rul.xstar_idx] = 1.0 / rul.xstar_idx.size
    m = ctx.evaluate(prob)
    assert abs(m["ar_f"] - 1) < 1e-12 and abs(m["p_feas"] - 1) < 1e-12 and abs(m["p_opt"] - 1) < 1e-12
    assert abs(m["energy"] - rul.E_min) < 1e-9 * max(1.0, abs(rul.E_min)) and m["p_top10"] == pytest.approx(1)
    prob[:] = 0
    prob[rul.band_idx] = 1.0 / rul.F_size
    m = ctx.evaluate(prob)
    assert m["ar_f"] == pytest.approx(np.mean((rul.E_max - rul.f_band) / (rul.E_max - rul.E_min)), abs=1e-12)
    assert m["eps_tilde"] <= inst.eps + 1e-12
    out = np.setdiff1d(np.arange(1 << inst.n), rul.band_idx)[0]
    prob[:] = 0
    prob[out] = 1
    m = ctx.evaluate(prob)
    assert m["p_feas"] == 0 and m["ar_f"] == 0 and m["eps_tilde"] > inst.eps
    # the penalty context uses H(lam)'s diagonal: <H(lam)> = <H_obj> + lam <Pen> on a basis state
    ctx_l = metric_context(inst, rul, 0.5)
    assert ctx_l.evaluate(prob)["energy"] == pytest.approx(m["energy"] + 0.5 * ctx.delta2[out], abs=1e-9)


# --- the legacy Pauli-coefficient port ---------------------------------------------------------------------
def test_legacy_pauli_transition_terms_are_the_dense_operator():
    from gsp.circuits.legacy_pauli import merged_terms, ring_T, transition_terms
    n = 4
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    P = {"I": np.eye(2), "X": X, "Y": Y, "Z": Z}
    for u, v in [(0b0101, 0b0011), (0b1111, 0b0000), (0b1000, 0b1001)]:
        x, y = format(u, "04b"), format(v, "04b")
        terms = transition_terms(x, y)
        assert len(terms) == 2 ** (n - 1) and set(np.abs(list(terms.values()))) == {2.0 ** -(n - 1)}
        M = sum(c * np.kron(np.kron(np.kron(P[w[0]], P[w[1]]), P[w[2]]), P[w[3]]) for w, c in terms.items())
        D = np.zeros((16, 16))
        D[u, v] = D[v, u] = 1
        assert np.max(np.abs(M - D)) < 1e-15
    words, c = merged_terms([format(i, "04b") for i in (1, 2, 4, 8)], ring_T(4))
    assert c.dtype == np.float32 and len(words) == len(set(words))
