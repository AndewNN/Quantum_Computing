"""S2: the numpy GA, the brute-force reference, sector files, the Altafini check (PLAN §5 S2).

Tests that read the frozen instances skip when `results/instances/` is absent; the results-level
test skips when `gsp sectors build` has not run.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from gsp.instances.bits import bits_to_index
from gsp.sectors import control
from gsp.sectors.ga import (TABLE_4_1, GAParams, Problem, bit_flip, brute_force, chromosome_perm,
                            problem_from_instance, run_ga, single_point_crossover, tournament_winners)
from gsp.store.paths import inst_path, sector_jobs_dir, sector_path
from tests.legacy.ga_callsite import cpp_total_cost, feasible_reversed_basis

needs_instances = pytest.mark.skipif(not inst_path("N04e004q1.5").exists(),
                                     reason="frozen instances absent (run gsp instances freeze)")
PYBIND_DIR = Path(__file__).resolve().parents[2] / "MyLib" / "Genetic"


def _inst(iid):
    from gsp.instances.instance import load_instance
    return load_instance(iid)


# --- chromosome layout ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 5, 7])
def test_chromosome_perm_matches_old_call_site(N):
    rng = np.random.default_rng(N)
    chroms = rng.integers(0, 2, size=(20, 2 * N)).astype(bool).tolist()
    strings = feasible_reversed_basis(chroms, N, 2)
    perm = chromosome_perm([2] * N)
    for c, s in zip(chroms, strings):
        x = np.empty(2 * N, np.int64)
        x[perm] = np.asarray(c, np.int64)
        assert bits_to_index(x) == int(s, 2)


def test_chromosome_perm_mixed_lengths():
    # asset blocks of length 3, 1, 2: position s + b holds qubit s + L - 1 - b
    assert chromosome_perm([3, 1, 2]).tolist() == [2, 1, 0, 3, 5, 4]


@needs_instances
@pytest.mark.parametrize("iid", ["N04e004q1.5", "N06e000q3.0"])
def test_violation_matches_cpp_fitness(iid):
    inst = _inst(iid)
    pr = problem_from_instance(inst, "violation")
    rng = np.random.default_rng(1)
    C = rng.integers(0, 2, size=(64, inst.n)).astype(np.uint8)
    fit = pr.evaluate(pr.chrom_to_x(C))
    for row, cost, viol in zip(C, fit["cost"], fit["viol"]):
        tc, f = cpp_total_cost(row.tolist(), [2] * inst.N, inst.P.tolist(), inst.B)
        assert cost == pytest.approx(tc / inst.B, rel=1e-13)
        assert viol == pytest.approx(f / inst.B ** 2, rel=1e-9, abs=1e-15)


# --- fitness vs the frozen rulers ----------------------------------------------------------------
@needs_instances
@pytest.mark.parametrize("iid", ["N04e004q1.0", "N05e000q1.5", "N07e003q3.0"])
def test_brute_force_matches_rulers(iid):
    from gsp.instances.instance import load_rulers
    inst, rul = _inst(iid), load_rulers(iid)
    band = np.zeros(1 << inst.n, bool)
    band[rul.band_idx] = True
    v = brute_force(problem_from_instance(inst, "violation"))
    o = brute_force(problem_from_instance(inst, "objective"))
    assert np.array_equal(v["in_band"], band) and np.array_equal(o["in_band"], band)
    f = o["obj"][rul.band_idx]
    assert np.max(np.abs(f - rul.f_band)) <= 1e-12 * (rul.E_max - rul.E_min)
    assert np.array_equal(o["rank_idx"][:10], rul.top10_idx)          # |X*| = 1, top-10 set of size 10
    K = min(24, rul.F_size)
    ruler_rank = rul.band_idx[np.lexsort((rul.band_idx, rul.band_pen))]
    assert set(v["rank_idx"][:K].tolist()) == set(ruler_rank[:K].tolist())
    assert np.all(band[v["rank_idx"][:K]])


# --- operators -----------------------------------------------------------------------------------
def test_single_point_crossover():
    p1 = np.zeros((3, 6), np.uint8)
    p2 = np.ones((3, 6), np.uint8)
    c1, c2 = single_point_crossover(p1, p2, np.array([True, True, False]), np.array([1, 4, 3]))
    assert c1.tolist() == [[0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0] * 6]
    assert c2.tolist() == [[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1] * 6]


def test_bit_flip_and_tournament():
    C = np.array([[0, 1, 0], [1, 1, 1]], np.uint8)
    F = np.array([[True, False, True], [False, False, True]])
    assert bit_flip(C, F).tolist() == [[1, 1, 1], [1, 1, 0]]
    rng = np.random.default_rng(0)
    w = tournament_winners(rng, 2000, 20000, 5)
    assert w.min() >= 0 and w.max() < 2000
    # P(winner position >= k) = (1 - k/N)^5: the median winner sits near N (1 - 2^(-1/5)) = 259
    assert 230 < np.median(w) < 290


# --- the GA ----------------------------------------------------------------------------------------
@needs_instances
@pytest.mark.parametrize("rule", ["violation", "objective"])
def test_ga_counters_determinism_elitism(rule):
    inst = _inst("N06e000q1.5")
    pr = problem_from_instance(inst, rule)
    small = GAParams(population=300, generations=8)
    a = run_ga(pr, 11, small)
    b = run_ga(pr, 11, small)
    assert np.array_equal(a.rank_idx, b.rank_idx) and np.array_equal(a.first_eval, b.first_eval)
    assert a.n_evals == 9 * 300 and a.n_violation_evals == 9 * 300
    assert a.n_objective_evals == (9 * 300 if rule == "objective" else 0)
    assert a.n_unique == np.count_nonzero(a.first_eval) <= 9 * 300
    assert a.first_eval.max() <= 9 * 300
    # every returned string was evaluated, the ranking is distinct
    assert np.all(a.first_eval[a.rank_idx] > 0) and len(set(a.rank_idx.tolist())) == a.rank_idx.size
    # elitism: the best of a generation never gets worse
    band = a.trace_best_in_band
    assert not np.any(band[:-1] & ~band[1:])
    same = band[:-1] == band[1:]
    assert np.all(a.trace_best_key[1:][same] <= a.trace_best_key[:-1][same])
    assert a.trace_best_idx[-1] == a.rank_idx[0]


@needs_instances
@pytest.mark.parametrize("rule", ["violation", "objective"])
def test_ga_table41_equals_bf_at_n8(rule):
    inst = _inst("N04e004q1.5")
    pr = problem_from_instance(inst, rule)
    ga = run_ga(pr, 6007, TABLE_4_1)
    bf = brute_force(pr)
    for K in (6, 8, 12):
        assert set(ga.top(K).tolist()) == set(bf["rank_idx"][:K].tolist())


def test_problem_rejects_bad_input():
    with pytest.raises(ValueError):
        Problem("objective", 2, np.ones(2), 0.1, None, np.array([0, 1]))
    with pytest.raises(ValueError):
        Problem("violation", 2, np.ones(2), 0.1, None, np.array([0, 0]))


# --- controllability -------------------------------------------------------------------------------
def test_altafini_holds_and_matches_closure():
    rng = np.random.default_rng(3)
    for K in (4, 5, 6):
        E = rng.normal(size=K)
        for conn in control.CONNECTIVITIES:
            r = control.altafini(E, control.edges_for(conn, K))
            assert r.holds and r.d_eff == K * K - 1
            assert control.closure_dim(E, control.edges_for(conn, K)) == K * K - 1


def test_altafini_flags():
    E = np.arange(6.0)                                    # equal ring gaps (and one wrap gap of 5)
    r = control.altafini(E, control.ring_edges(6))
    assert r.connected and r.gaps_nonzero and not r.gaps_distinct and not r.holds and r.d_eff == -1
    E = np.array([0.0, 1.0, 1.0, 3.0])                    # a zero gap on the ring
    r = control.altafini(E, control.ring_edges(4))
    assert not r.gaps_nonzero and not r.holds
    E = np.array([0.0, 1.0, 3.5, 7.9])
    r = control.altafini(E, [(0, 1), (2, 3)])             # disconnected
    assert not r.connected and not r.holds
    assert control.closure_dim(np.ones(5), control.ring_edges(5)) == 1
    assert control.closure_dim(np.array([1.0, 0, 0, 0, 0, 0]), control.ring_edges(6)) < 35


def test_edges():
    assert control.ring_edges(4) == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert len(control.complete_edges(12)) == 66
    assert control.is_connected(12, control.ring_edges(12))
    assert not control.is_connected(3, [(0, 1)])


# --- plan, files, cache ----------------------------------------------------------------------------
@needs_instances
def test_plan_matches_cells():
    from gsp.sectors.select import plan_jobs
    jobs = plan_jobs(extension=False)
    files = [(j.scope_id, j.rule, K) for j in jobs for K in j.file_Ks]
    assert len(files) == len(set(files)) == 751
    vio = [f for f in files if f[1] == "violation"]
    assert len(vio) == 391 and sum(K == 24 for _, _, K in vio) == 31
    assert sum(f[1] == "objective" for f in files) == 360
    assert all(K == 12 for _, r, K in files if r == "objective")
    assert len(jobs) == 480 and all(j.N in (4, 5, 6, 7) for j in jobs)
    ext = plan_jobs(extension=True)
    assert len(ext) == 840 and sum(j.in_cell for j in ext) == 480
    assert all(not j.file_Ks for j in ext if not j.in_cell)
    k24 = [j for j in jobs if 24 in j.file_Ks]
    assert sorted({j.N for j in k24}) == [5, 6] and all(j.F_size >= 24 for j in k24)


@needs_instances
def test_job_files_roundtrip_and_cache(tmp_path):
    from gsp.instances.instance import load_instance
    from gsp.instances.rulers import objective_on
    from gsp.sectors import select
    it = select.load_instances_table()
    sha = dict(zip(it["inst_id"], it["inst_sha256"]))
    jobs = [j for j in select.plan_jobs(extension=False, N_values=[4]) if j.e == 4]
    assert {j.rule for j in jobs} == {"violation", "objective"} and len(jobs) == 4
    for j in jobs:
        cfg = select.job_config(j, TABLE_4_1, sha)
        assert not select.job_is_current(j, cfg, tmp_path)
        out = select.run_job(j, TABLE_4_1, None, tmp_path)
        rec = select.write_job(j, out, cfg, tmp_path)
        assert select.job_is_current(j, cfg, tmp_path)
        assert json.loads((sector_jobs_dir(tmp_path) / f"{j.job_id}.json").read_text())["config_hash"] \
            == rec["config_hash"]
        for K in j.file_Ks:
            assert sector_path(j.scope_id, j.rule, K, tmp_path).exists()
    s = select.load_sector("N04e004q3.0", "violation", 12, tmp_path)
    assert s.K == 12 and s.idx.size == 12 and np.all(np.diff(s.idx) > 0)
    assert sorted(s.rank_idx.tolist()) == s.idx.tolist()
    assert [int(b, 2) for b in s.bitstrings] == s.idx.tolist()
    assert np.array_equal(s.E, objective_on(load_instance("N04e004q3.0").QU_obj, s.idx))
    assert s.identical_to_bf
    o = select.load_sector("N04e004q1.0", "objective", 12, tmp_path)
    assert o.arrays["inst_ids"].tolist() == ["N04e004q1.0"]
    with pytest.raises(FileNotFoundError):
        select.load_sector("N04e004q1.5", "objective", 8, tmp_path)
    t = select.build_tables(tmp_path)
    assert len(t["runs"]) == 4 and len(t["sectors"]) == 3 * 3 + 3


@pytest.mark.skipif(not (PYBIND_DIR / "ga_solver.cpython-311-x86_64-linux-gnu.so").exists(),
                    reason="the completed work's ga_solver build is absent")
@needs_instances
def test_pybind_violation_top12_equals_bf():
    """D-5: the completed work's C++ GA (budget mode) at N = 4 agrees with brute force and the numpy GA.
    Runs in a subprocess so the external module never enters this process."""
    import subprocess
    import sys
    code = (
        "import sys, numpy as np\n"
        "from gsp.sectors.pybind_check import import_ga_solver, cpp_run\n"
        "from gsp.sectors.ga import brute_force, problem_from_instance, run_ga\n"
        "from gsp.instances.instance import load_instance\n"
        f"mod = import_ga_solver({str(PYBIND_DIR)!r})\n"
        "if mod is None: print('SKIP'); sys.exit(0)\n"
        "inst = load_instance('N04e004q1.5'); pr = problem_from_instance(inst, 'violation')\n"
        "b = set(brute_force(pr)['rank_idx'][:12].tolist())\n"
        "c, _ = cpp_run(mod, inst, 'violation', None)\n"
        "g = run_ga(pr, 6007).rank_idx\n"
        "print(set(c[:12].tolist()) == b, set(g[:12].tolist()) == b)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    if out.stdout.strip() == "SKIP":
        pytest.skip("ga_solver does not import in this env")
    assert out.stdout.split() == ["True", "True"]


@pytest.mark.skipif(not sector_jobs_dir().exists(), reason="gsp sectors build has not run")
def test_production_sector_files_complete():
    """Every (cell, instance) of PLAN §1.2 has a current production sector file."""
    from gsp.sectors import select
    it = select.load_instances_table()
    sha = dict(zip(it["inst_id"], it["inst_sha256"]))
    jobs = select.plan_jobs(extension=False)
    stale = [j.job_id for j in jobs if not select.job_is_current(j, select.job_config(j, TABLE_4_1, sha))]
    assert not stale, stale[:10]
    for c in select.load_cells():
        for row in it[it["N"] == c["N"]].itertuples():
            if c.get("draws") == "k24_eligible" and not row.k24_eligible:
                continue
            s = select.load_sector(row.inst_id, c["rule"], c["K"])
            assert s.idx.size == c["K"]


@pytest.mark.skipif(not sector_jobs_dir().exists(), reason="gsp sectors build has not run")
def test_standalone_solver_reproduces_the_build():
    """`baselines.ga_solver.solve` with the seed-table seed reproduces the S2 record of that instance."""
    from gsp.baselines.ga_solver import solve
    for iid in ("N05e000q1.5", "N07e003q3.0"):
        rec = json.loads((sector_jobs_dir() / f"{iid}_objective.json").read_text())
        out = solve(iid)
        assert out["seed"] == rec["seed"] and out["n_evals"] == 36 * 2000
        for k, v in out.items():
            if k.startswith("solver_"):
                assert v == rec[k], (iid, k, v, rec[k])
        assert out["solver_hit_xstar"] and out["solver_ar_best"] == 1.0
