"""S3 on the GPU (marked `gpu`): the CUDA-Q kernels of the compiled preserving mixer.

- C1 operator level on real S2 sectors at n <= 12, ring and complete, both ring orders: the kernel's
  action on every kept string == the dense ordered product in K x K, leakage <= 1e-13;
- the two engines (interpreter kernel, PLAN §2.4's builder kernel) agree;
- the star start state has fidelity >= 1 - 1e-12 on the simulator;
- simulate_decomposed (the explicit (iii) circuit) equals the native circuit at n <= 10, and a whole A1
  circuit (prep + cost + mixer layers) on the GPU equals the numpy simulation of the same gate list;
- every arity branch (1..19 controls) of the interpreter kernel.
"""

import numpy as np
import pytest

from gsp.circuits import preserving as pr
from gsp.circuits import program
from gsp.compile import npsim
from gsp.compile.decompose import Angle, Gate
from gsp.stats import c1
from gsp.store.paths import sector_path

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not sector_path("N04e004", "violation", 12).exists(),
                                 reason="sector files absent (run gsp sectors build)")]

REAL = [("N04e004q1.5", "violation", 6, "ring"), ("N04e004q1.5", "violation", 12, "complete"),
        ("N05e000q1.5", "violation", 24, "ring"), ("N05e000q1.0", "objective", 12, "ring"),
        ("N06e000q1.5", "violation", 12, "ring"), ("N06e000q1.5", "violation", 24, "ring"),
        ("N06e000q1.5", "violation", 12, "complete"), ("N06e001q3.0", "objective", 12, "ring")]


@pytest.fixture(scope="module")
def be():
    from gsp.sim import backend
    backend.set_target("nvidia", "fp64")
    return backend


def _sector(iid, rule, K):
    from gsp.sectors.select import load_sector
    return load_sector(iid, rule, K)


@pytest.mark.parametrize("case", REAL)
@pytest.mark.parametrize("ring_order", ["lex", "rank"])
def test_c1_operator_level_gpu(be, case, ring_order):
    iid, rule, K, conn = case
    circ = pr.build_circuit(_sector(iid, rule, K), conn, ring_order)
    for r in c1.a1_operator_level(circ, [0.613, -2.1], engine="cudaq"):
        assert r["leakage"] <= 1e-13 and r["block_err"] <= 1e-12, r


def test_c1_symmetrized_gpu(be):
    circ = pr.build_circuit(_sector("N06e000q1.5", "violation", 12), "complete", "lex", symmetrized=True)
    r = c1.a1_operator_level(circ, 0.9, engine="cudaq")
    assert r["leakage"] <= 1e-13 and r["block_err"] <= 1e-12, r


@pytest.mark.parametrize("case", [("N04e004q1.5", "violation", 6, "ring"), ("N05e000q1.0", "objective", 12, "ring")])
def test_builder_engine_agrees(be, case):
    """PLAN §2.4's builder-API kernel (crx / cry with runtime control lists) passes the same check."""
    iid, rule, K, conn = case
    circ = pr.build_circuit(_sector(iid, rule, K), conn)
    r = c1.a1_operator_level(circ, 0.613, engine="builder")
    assert r["leakage"] <= 1e-13 and r["block_err"] <= 1e-12, r
    f = c1.star_prep_fidelity(circ, engine="builder")
    assert f["infidelity"] <= 1e-12


@pytest.mark.parametrize("case", REAL)
def test_star_prep_fidelity_gpu(be, case):
    iid, rule, K, conn = case
    circ = pr.build_circuit(_sector(iid, rule, K), conn)
    r = c1.star_prep_fidelity(circ, engine="cudaq")
    assert r["infidelity"] <= 1e-12 and r["leakage"] <= 1e-13 and r["max_imag"] <= 1e-13, r


@pytest.mark.parametrize("case", [("N04e004q1.5", "violation", 12, "ring"), ("N04e004q1.5", "violation", 12, "complete"),
                                  ("N05e000q1.5", "violation", 24, "ring"), ("N05e000q1.0", "objective", 12, "ring")])
def test_simulate_decomposed_equals_native_gpu(be, case):
    """A whole A1 circuit (prep + 2 x (cost + mixer)) at n <= 10: the explicit (iii) circuit == native
    on the GPU, and both == the numpy simulation of the native gate list."""
    from gsp.circuits.cost import cost_terms
    from gsp.instances.instance import load_instance

    iid, rule, K, conn = case
    inst = load_instance(iid)
    ct = cost_terms(inst.H_obj, inst.boost_obj)
    circ = pr.build_circuit(_sector(iid, rule, K), conn)
    n = circ.n
    assert n <= 10
    params = [0.31, -0.72, 0.45, 1.13]                   # gamma_1, gamma_2, beta_1, beta_2
    native = pr.a1_gates(circ, 2, cost=ct)
    decomp = pr.a1_gates(circ, 2, cost=ct, decomposed=True)
    kern = program.kernel()
    v_nat = be.get_state_classical(kern, n, *program.encode(native, n).args(params))
    v_dec = be.get_state_classical(kern, n, *program.encode(decomp, n).args(params))
    v_np = npsim.flat(npsim.apply(native, npsim.columns(n, [0]), params))[:, 0]
    assert np.abs(v_dec - v_nat).max() <= 1e-12
    assert np.abs(v_nat - v_np).max() <= 1e-12
    mask = np.ones(1 << n, dtype=bool)
    mask[circ.order] = False
    assert np.abs(v_nat[mask]).max() <= 1e-13                        # stays in the sector
    # observe of the boosted H_obj == the diagonal expectation of that state
    E = be.observe(kern, be.ising_op(inst.H_obj, inst.boost_obj), *program.encode(native, n).args(params))
    diag = inst.boost_obj * inst.H_obj.diagonal(np.arange(1 << n))
    assert abs(E - float(np.abs(v_nat) ** 2 @ diag)) <= 1e-12


def test_builder_decomposed_equals_interp(be):
    circ = pr.build_circuit(_sector("N04e004q1.5", "violation", 8), "ring")
    bk = pr.build_kernel(circ, L=1, simulate_decomposed=True)
    v_b = be.get_state_classical(bk.kernel, circ.n, bk.params(betas=[0.4]))
    pg = program.encode(pr.a1_gates(circ, 1), circ.n)
    v_i = be.get_state_classical(program.kernel(), circ.n, *pg.args([0.4]))
    assert np.abs(v_b - v_i).max() <= 1e-12


def test_interpreter_every_arity(be):
    """Each mc-rx / mc-ry branch (1..19 controls, closed and open via X) against numpy."""
    rng = np.random.default_rng(5)
    kern = program.kernel()
    for k in range(1, program.MAX_CONTROLS + 1):
        n = k + 1
        perm = rng.permutation(n)
        ctl, tgt = [int(c) for c in perm[:k]], int(perm[k])
        prep = [Gate("ry", (q,), Angle(float(a))) for q, a in enumerate(rng.uniform(0.2, 1.2, n))]
        for kind in ("mcrx", "mcry"):
            flips = [Gate("x", (c,)) for c in ctl[: k // 2]]
            gates = prep + flips + [Gate(kind, tuple(ctl) + (tgt,), Angle(1.7, 0))] + flips
            v = be.get_state_classical(kern, n, *program.encode(gates, n).args([0.8]))
            ref = npsim.flat(npsim.apply(gates, npsim.columns(n, [0]), [0.8]))[:, 0]
            assert np.abs(v - ref).max() <= 1e-12, (k, kind)


@pytest.mark.slow
def test_timing_runs(be):
    from gsp.compile import timing
    out = timing.time_cells(depths=(5,), n_inst=1, repeats=2, builder=False)
    assert len(out["rows"]) == 3 and all(r["observe_median_s"] > 0 for r in out["rows"])
