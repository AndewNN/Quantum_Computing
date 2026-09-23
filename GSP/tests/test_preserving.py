"""S3: the compiled preserving mixer on the CPU (numpy): W and the hitting sets, the ordered product
against the dense K x K reference on real S2 sectors (C1 operator level), the star start state, the
Trotter-order sanity check, the (ii) 696 check and the counters (PLAN §2, §5 S3)."""

import numpy as np
import pytest

from gsp.circuits import preserving as pr
from gsp.circuits import program
from gsp.compile import decompose as dc
from gsp.compile.decompose import Angle, Gate
from gsp.compile import npsim
from gsp.compile import transpile as tp
from gsp.stats import c1
from gsp.store.paths import sector_path

needs_sectors = pytest.mark.skipif(not sector_path("N04e004", "violation", 12).exists(),
                                   reason="sector files absent (run gsp sectors build)")

# (inst_id, rule, K, connectivity): every cell type at n <= 12, including K = 24 and the objective rule
REAL = [("N04e004q1.5", "violation", 6, "ring"), ("N04e004q1.5", "violation", 12, "complete"),
        ("N05e000q1.5", "violation", 8, "ring"), ("N05e000q1.5", "violation", 24, "ring"),
        ("N05e000q1.0", "objective", 12, "ring"), ("N06e000q1.5", "violation", 12, "ring"),
        ("N06e000q1.5", "violation", 24, "ring"), ("N06e000q1.5", "violation", 12, "complete"),
        ("N06e001q3.0", "objective", 12, "ring")]


def _sector(iid, rule, K):
    from gsp.sectors.select import load_sector
    return load_sector(iid, rule, K)


def test_w_maps_u_to_zero_and_v_to_pivot():
    rng = np.random.default_rng(0)
    for n in (4, 8, 12):
        for _ in range(20):
            u, v = (int(x) for x in rng.choice(1 << n, size=2, replace=False))
            D = pr.diff_set(u, v, n)
            y = pr.w_image([u, v], u, v, n)
            assert y[0] == 0 and y[1] == 1 << (n - 1 - D[0])
            allx = np.arange(1 << n)
            assert np.unique(pr.w_image(allx, u, v, n)).size == 1 << n           # a permutation
            # the vectorised image equals the gate list's action
            psi = npsim.columns(n, [int(x) for x in rng.choice(1 << n, 3)])
            cols = np.flatnonzero(npsim.flat(psi).sum(axis=1))
            xm = tuple(k for k in range(n) if (u >> (n - 1 - k)) & 1)
            npsim.apply([dc.Gate("x", (k,)) for k in xm] + [dc.Gate("cx", (D[0], k)) for k in D[1:]], psi)
            assert sorted(np.flatnonzero(npsim.flat(psi).sum(axis=1)).tolist()) == sorted(
                pr.w_image(cols, u, v, n).tolist())


def test_hitting_set_greedy_and_deterministic():
    n = 5
    # images 01100, 01010, 00011 (pivot 0): qubit 1 hits two, then qubit 3 hits the last (tie 3/4 -> 3)
    imgs = [0b01100, 0b01010, 0b00011]
    assert pr.hitting_set(imgs, 0, n) == (1, 3)
    assert pr.hitting_set([], 0, n) == ()
    with pytest.raises(ValueError):
        pr.hitting_set([0b10000], 0, n)                  # e_k0 itself cannot be hit


@needs_sectors
@pytest.mark.parametrize("case", REAL)
@pytest.mark.parametrize("ring_order", ["lex", "rank"])
def test_hitting_sets_on_real_sectors(case, ring_order):
    iid, rule, K, conn = case
    sec = _sector(iid, rule, K)
    circ = pr.build_circuit(sec, conn, ring_order)
    n, order = circ.n, circ.order
    assert (order == (sec.idx if ring_order == "lex" else sec.rank_idx)).all()
    assert len(circ.layer) == (K if conn == "ring" else K * (K - 1) // 2)
    for tr in circ.layer:
        i, j = tr.edge
        others = np.array([order[p] for p in range(K) if p not in (i, j)])
        img = pr.w_image(others, tr.u, tr.v, n)
        assert tr.k0 not in tr.S and len(tr.S) <= min(n - 1, K - 2)
        assert all(any((y >> (n - 1 - s)) & 1 for s in tr.S) for y in img)
    for j, tr in enumerate(circ.prep, start=1):
        assert tr.edge == (0, j) and tr.kind == "ry" and not tr.param
        carried = pr.w_image(order[1:j], tr.u, tr.v, n)
        assert all(any((y >> (n - 1 - s)) & 1 for s in tr.S) for y in carried)
    assert circ.prep[0].S == ()                      # step 2: only u_1 carries amplitude


def test_edge_orders():
    assert pr.layer_edges(5, "ring") == [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 0, 1.0)]
    assert [e[:2] for e in pr.layer_edges(4, "complete")] == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    sym = pr.layer_edges(3, "ring", symmetrized=True)
    assert [e[:2] for e in sym] == [(0, 1), (1, 2), (2, 0), (2, 0), (1, 2), (0, 1)] and {e[2] for e in sym} == {0.5}
    with pytest.raises(ValueError):
        pr.sector_order(type("S", (), {"idx": [1, 2], "rank_idx": [2, 1]})(), "random")


@needs_sectors
@pytest.mark.parametrize("case", REAL)
@pytest.mark.parametrize("ring_order", ["lex", "rank"])
@pytest.mark.parametrize("symmetrized", [False, True])
def test_c1_operator_level_numpy(case, ring_order, symmetrized):
    """Circuit action on the sector == the dense ordered product in K x K; leakage <= 1e-13."""
    iid, rule, K, conn = case
    circ = pr.build_circuit(_sector(iid, rule, K), conn, ring_order, symmetrized)
    for r in c1.a1_operator_level(circ, [0.613, -2.1, 1e-3], engine="numpy"):
        assert r["leakage"] <= 1e-13 and r["block_err"] <= 1e-12, r


@needs_sectors
@pytest.mark.parametrize("case", [c for c in REAL if c[0][:3] in ("N04", "N05")])
def test_decomposed_layer_equals_native_numpy(case):
    """simulate_decomposed (the explicit (iii) circuit) == native at n <= 10, on the CPU."""
    iid, rule, K, conn = case
    circ = pr.build_circuit(_sector(iid, rule, K), conn)
    rn = c1.a1_operator_level(circ, 0.77, engine="numpy")
    rd = c1.a1_operator_level(circ, 0.77, engine="numpy", decomposed=True)
    assert rd["leakage"] <= 1e-13 and rd["block_err"] <= 1e-12 and rn["block_err"] <= 1e-12


@needs_sectors
@pytest.mark.parametrize("case", REAL)
@pytest.mark.parametrize("ring_order", ["lex", "rank"])
def test_star_prep_uniform(case, ring_order):
    iid, rule, K, conn = case
    circ = pr.build_circuit(_sector(iid, rule, K), conn, ring_order)
    for dec in ((False, True) if circ.n <= 10 else (False,)):
        r = c1.star_prep_fidelity(circ, engine="numpy", decomposed=dec)
        assert r["infidelity"] <= 1e-12 and r["leakage"] <= 1e-13, r
        assert r["max_imag"] <= 1e-13 and abs(r["min_amp"] - 1 / np.sqrt(K)) <= 1e-12   # every phase +1


def test_star_phis():
    for K in (2, 6, 12, 24):
        c = np.ones(1)
        amps = []
        for phi in pr.star_phis(K):
            amps.append(c[0] * np.sin(phi))
            c = c * np.cos(phi)
        assert np.allclose(amps + [c[0]], 1 / np.sqrt(K), atol=1e-15)


def test_ring_uniform_state_is_an_eigenvector_up_to_trotter_order():
    """H_M |s> = 2 |s> on the ring, so U_M(beta)|s> = e^{-2 i beta}|s> + O(beta^2) for the first-order
    product and + O(beta^3) for the symmetrized one (dense and compiled agree)."""
    for K in (6, 12, 24):
        s = pr.uniform_state(K)
        H = pr.ring_hamiltonian(K)
        assert np.allclose(H @ s, 2 * s)
        for sym, order in ((False, 2), (True, 3)):
            errs = []
            for beta in (0.04, 0.02, 0.01):
                U = pr.dense_layer(K, "ring", beta, sym)
                errs.append(np.linalg.norm(U @ s - np.exp(-2j * beta) * s))
            ratios = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
            assert np.all(np.abs(ratios - order) < 0.15), (K, sym, ratios)
    # the compiled circuit on a synthetic sector gives the same vector as the dense product
    o = tp.synthetic_ring_order(8, 6, 3)
    circ = pr.circuit_from_order(o, 8, "ring")
    psi = npsim.columns(8, [0])
    npsim.apply(pr.prep_gates(circ) + pr.layer_gates(circ, 0), psi, [0.03])
    v = npsim.flat(psi)[:, 0][circ.order]
    assert np.abs(v - pr.dense_layer(6, "ring", 0.03) @ pr.uniform_state(6)).max() < 1e-13


def test_696_reproduced_by_the_ii_counter():
    """V18: n = 10, K = 12, every ring edge at d = 5: 58 CNOTs per transition, 696 per ring layer."""
    o = tp.synthetic_ring_order(10, 12, 5)
    assert np.all(np.diff(o) > 0) and len(set(o.tolist())) == 12
    circ = pr.circuit_from_order(o, 10, "ring")
    assert [tr.d for tr in circ.layer] == [5] * 12
    per = [tp.transition_counts(tr)["cx_ii"] for tr in circ.layer]
    assert per == [58] * 12
    cc = tp.circuit_counts(circ)
    assert cc["mixer"]["cx_ii"] == 696
    assert cc["mixer"]["t_ii"] == 2064 and cc["mixer"]["tdepth_ii"] == 912         # V20
    # the complete graph on the same sector: 66 transitions (V18 quotes 3,828 at uniform d = 5)
    cm = tp.circuit_counts(pr.circuit_from_order(o, 10, "complete"))
    assert cm["mixer"]["n_transitions"] == 66


@needs_sectors
def test_counts_on_a_real_cell():
    from gsp.circuits.cost import cost_terms
    from gsp.instances.instance import load_instance

    inst = load_instance("N07e000q1.5")
    circ = pr.build_circuit(_sector("N07e000q1.5", "violation", 12), "ring")
    ct = cost_terms(inst.H_obj, inst.boost_obj)
    cc = tp.circuit_counts(circ, ct)
    n = 14
    assert cc["cost"]["cx_ii"] == 2 * ct.n_zz == n * (n - 1)                   # dense QUBO
    assert cc["mixer"]["cx_ii"] == sum(6 * n + 2 * tr.d - 12 for tr in circ.layer)
    assert cc["mixer"]["cx_iii"] == sum(2 * (tr.d - 1) + dc.vale_cnots(len(tr.S)) for tr in circ.layer)
    assert cc["prep"]["n_transitions"] == 11
    assert cc["layer"]["cx_ii"] == cc["cost"]["cx_ii"] + cc["mixer"]["cx_ii"]
    tot = tp.a1_totals(cc, 5)
    assert tot["cx_ii"] == cc["prep"]["cx_ii"] + 5 * cc["layer"]["cx_ii"]


@needs_sectors
def test_cell_instances():
    from gsp.sectors.select import cell_instances
    assert len(cell_instances({"N": 4, "K": 12, "rule": "violation", "connectivity": "ring"})) == 90
    k24 = cell_instances({"N": 5, "K": 24, "rule": "violation", "connectivity": "ring", "draws": "k24_eligible"})
    assert len(k24) == 30 and k24[0].startswith("N05e000")


@needs_sectors
@pytest.mark.slow
def test_c1_numpy_every_real_sector():
    """The full sweep of `gsp mixer c1` (numpy): every sector file at n <= 12, both connectivities."""
    from gsp.compile.report import c1_sweep
    d = c1_sweep(engine="numpy", decomposed_n_max=10)
    assert len(d) > 4000 and d.leakage.max() <= 1e-13 and d.block_err.max() <= 1e-12


def test_program_encoding_round_trip():
    gl = [Gate("x", (1,)), Gate("cx", (0, 2)), Gate("mcrx", (0, 1, 3), Angle(2.0, 1)), Gate("rz", (2,), Angle(0.5))]
    pg = program.encode(gl, 4)
    assert pg.op == [0, 6, 10, 9] and pg.tgt == [1, 2, 3, 2] and pg.nc == [0, 1, 2, 0]
    assert pg.ctl == [0, 0, 1] and pg.cs == [0, 0, 1, 3] and pg.pidx == [-1, -1, 1, -1] and pg.n_params == 2
    with pytest.raises(ValueError):
        pg.args([0.1])


def test_interp_kernel_is_generated():
    """gsp/circuits/interp.py is the output of scripts/gen_interp_kernel.py (never hand-edited)."""
    import importlib.util
    from gsp.store.paths import GSP_ROOT
    spec = importlib.util.spec_from_file_location("gen_interp", GSP_ROOT / "scripts" / "gen_interp_kernel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (GSP_ROOT / "gsp" / "circuits" / "interp.py").read_text() == mod.source()
    assert program.MAX_CONTROLS == mod.MAXC
