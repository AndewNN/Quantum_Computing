"""S5: the per-run metric library (PLAN §1.7) -- example_metrics.py reproduced to 1e-12, AR_best_S exact against
enumeration, convergence, resources, the preprocessing join, simulation difficulty (numpy vs quimb), the
aggregate over copies of stored runs, and (gpu) the post-run step."""

import contextlib
import io
import itertools
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

from gsp.metrics import quality as ql
from gsp.metrics import simdiff as sd
from gsp.metrics.convergence import convergence
from gsp.metrics.resources import resources
from gsp.metrics.state import MetricContext
from gsp.store.paths import GSP_ROOT, inst_path, runs_dir, sector_path

LEGACY = GSP_ROOT / "tests" / "legacy" / "example_metrics.py"
ORIGINAL = Path.home() / "Desktop" / "Quantum_Master_Proposal" / "Lecture_Notes" / "code" / "example_metrics.py"
HAVE_INST = inst_path("N04e004q1.5").exists() and sector_path("N04e004", "violation", 12).exists()
HAVE_RUNS = runs_dir().exists() and any(runs_dir().glob("A1/*/final_state.npy"))


def _run_example():
    with contextlib.redirect_stdout(io.StringIO()) as out:
        g = runpy.run_path(str(LEGACY))
    return g, out.getvalue()


# --- example_metrics.py ------------------------------------------------------------------------------------
def test_legacy_copy_is_the_source():
    if not ORIGINAL.exists():
        pytest.skip("the proposal repository is not on this machine")
    body = LEGACY.read_text().split('"""\n', 1)[1]
    assert body == ORIGINAL.read_text()


def test_example_metrics_reproduced_to_1e12():
    g, printed = _run_example()
    feas, prob, f, Delta, quality = g["feasible"], g["prob"], g["f"], g["Delta"], g["quality"]
    band = np.nonzero(feas)[0]
    E_min, E_max = f[feas].min(), f[feas].max()
    xstar = np.nonzero(feas & np.isclose(f, E_min))[0]
    ctx = MetricContext(n=4, band_idx=band, ar_weight=ql.normalized_quality(f[band], E_min, E_max),
                        delta2=Delta ** 2, diag=f, xstar_idx=xstar, top10_idx=band)
    m = ctx.evaluate(prob)
    assert abs(m["p_feas"] - g["p_feas"]) <= 1e-12
    assert abs(m["eps_tilde"] - g["eps_tilde"]) <= 1e-12
    assert abs(m["ar_f"] - g["AR_F"]) <= 1e-12
    assert abs(m["p_opt"] - g["p_opt"]) <= 1e-12
    assert np.max(np.abs(ctx.ar_weight - quality[band])) <= 1e-12
    # the Monte-Carlo AR_best_S of the script, on the script's own shots
    S = g["S"]
    assert abs(ql.ar_best_from_shots(g["shots"], quality, feas) - np.mean(g["best"])) <= 1e-12
    assert abs(ql.p_seen(m["p_opt"], S) - (1 - (1 - g["p_opt"]) ** S)) <= 1e-12
    # the exact AR_best_S agrees with the script's estimate within its sampling error
    exact = ql.ar_best_exact(prob[band], ctx.ar_weight, S)
    se = np.std(g["best"], ddof=1) / np.sqrt(len(g["best"]))
    assert abs(exact - np.mean(g["best"])) <= 5 * se
    assert "AR_best_S (S=20) = 0.821" in printed and f"{exact:.2f}" == "0.82"


# --- AR_best_S -------------------------------------------------------------------------------------------------
def _brute_ar_best(prob, quality, feas, S):
    num = den = 0.0
    for tup in itertools.product(range(prob.size), repeat=S):
        w = float(np.prod(prob[list(tup)]))
        fe = [i for i in tup if feas[i]]
        if fe:
            num += w * max(quality[i] for i in fe)
            den += w
    return num / den


@pytest.mark.parametrize("S", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ar_best_exact_equals_enumeration(S, seed):
    rng = np.random.default_rng(seed)
    n = 9
    prob = rng.random(n) ** 2
    prob /= prob.sum()
    feas = rng.random(n) < 0.6
    feas[0] = True
    quality = np.round(rng.random(n), 1)          # ties on purpose
    exact = ql.ar_best_exact(prob[feas], quality[feas], S)
    assert abs(exact - _brute_ar_best(prob, quality, feas, S)) <= 1e-12


def test_ar_best_edge_cases():
    assert np.isnan(ql.ar_best_exact(np.zeros(3), np.array([1.0, 0.5, 0.0]), 10))
    assert ql.ar_best_exact(np.array([0.3]), np.array([0.7]), 1000) == pytest.approx(0.7, abs=1e-15)
    # S large: the best string with any support
    assert ql.ar_best_exact(np.array([1e-2, 0.5, 0.0]), np.array([1.0, 0.2, 0.9]), 100000) == pytest.approx(1.0)
    vals, pm = ql.best_quality_distribution(np.array([0.1, 0.2]), np.array([1.0, 0.0]), 5)
    assert np.allclose(vals, [1.0, 0.0]) and pm.sum() == pytest.approx(1.0)
    assert ql.sample_tail_probability(1.0, np.array([0.5, 0.5]), np.array([1.0, 0.0]), 1000) == pytest.approx(1.0)
    assert ql.sample_tail_probability(0.0, np.array([0.5, 0.5]), np.array([1.0, 0.0]), 1000) < 1e-250


def test_best_of_shots_and_band_positions():
    band = np.array([1, 4, 7, 9])
    q = np.array([0.1, 1.0, 0.5, 0.0])
    ok, pos = ql.band_positions(band, [0, 4, 9, 12])
    assert ok.tolist() == [False, True, True, False] and pos.tolist() == [-1, 1, 3, -1]
    assert ql.best_of_shots([0, 7, 9], band, q) == 0.5
    assert np.isnan(ql.best_of_shots([0, 2], band, q))


# --- convergence and resources ----------------------------------------------------------------------------------
def test_convergence_first_crossing_and_charges():
    t = np.arange(6)
    tr = {"t": t, "ar_f": np.array([0.1, 0.5, 0.9, 0.95, 0.97, 1.0]), "circuits_charged": 11 * t,
          "g2q_ii": 11 * t * 7, "g2q_iii": 11 * t * 3}
    c = convergence(tr)
    assert c["conv_t"] == 3 and c["conv_circuits"] == 33 and c["conv_g2q_ii"] == 231 and c["ar_f_final"] == 1.0
    one = {"t": np.array([0]), "ar_f": np.array([0.4]), "circuits_charged": np.array([1]),
           "g2q_ii": np.array([500]), "g2q_iii": np.array([90])}
    assert convergence(one)["conv_t"] == 0 and convergence(one)["conv_g2q_ii"] == 500
    nan = dict(tr, ar_f=np.r_[np.full(5, np.nan), 1.0])
    assert convergence(nan)["conv_t"] is None


def test_resources_from_counts():
    counts = {"effort_unit": "iteration", "circuits_per_unit": 11,
              "per_circuit": {"cx_ii": 10, "cx_iii": 4, "t_ii": 70, "t_iii": 30, "tdepth_ii": 20, "tdepth_iii": 9},
              "per_unit": {"cx_ii": 110, "cx_iii": 44, "t_ii": 770, "t_iii": 330, "tdepth_ii": 220,
                           "tdepth_iii": 99},
              "layer": {}, "start": {}}
    t = np.arange(4)
    tr = {"t": t, "ar_f": np.array([0.2, 0.5, 0.96, 1.0]), "circuits_charged": 11 * t, "g2q_ii": 110 * t,
          "g2q_iii": 44 * t}
    r = resources(counts, convergence(tr), tr)
    assert r["per_unit_cx_ii"] == 110 and r["conv_exec_cx_ii"] == 220 and r["chk_conv_g2q"] is True
    assert r["conv_exec_t_ii"] == 22 * 70 and r["total_g2q_ii"] == 330


@pytest.mark.skipif(not HAVE_INST, reason="frozen instances / sectors absent")
def test_preprocessing_join():
    from gsp.metrics.preprocessing import preprocessing
    rec = {"encoding": "confined", "rule": "violation", "K": 12, "inst_id": "N04e004q1.5", "sector_source": "ga"}
    p = preprocessing(rec)
    assert p["pre_available"] and p["pre_scope"] == "draw" and p["pre_ga_n_evals"] == 72000
    assert p["pre_file"] == "sectors_N04e004_violation_K12.npz" and p["pre_ga_wall_s"] > 0
    obj = preprocessing(dict(rec, rule="objective"))
    assert obj["pre_scope"] == "instance" and obj["pre_file"] == "sectors_N04e004q1.5_objective_K12.npz"
    assert preprocessing({"encoding": "penalty", "inst_id": "N04e004q1.5"}) == {}


# --- simulation difficulty ----------------------------------------------------------------------------------------
def _rand(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return v / np.linalg.norm(v)


def test_simdiff_known_states():
    n = 6
    prod = np.zeros(1 << n, complex)
    prod[5] = 1.0
    r = sd.simdiff(prod, engine="numpy")
    assert r["chi_star"] == 1 and r["S_half"] == pytest.approx(0.0, abs=1e-12) and r["F_chi1"] == pytest.approx(1.0)
    ghz = np.zeros(1 << n, complex)
    ghz[0] = ghz[-1] = 2 ** -0.5
    r = sd.simdiff(ghz, engine="numpy")
    assert r["F_chi1"] == pytest.approx(0.5, abs=1e-12) and r["chi_star"] == 2
    assert r["S_half"] == pytest.approx(1.0, abs=1e-12)
    assert r["mem_repr_bytes"] == 32 * n * 4 and r["mem_dense_bytes"] == 16 * 64


@pytest.mark.parametrize("K", [6, 12, 24])
def test_simdiff_sector_states_obey_the_bounds(K):
    n = 10
    rng = np.random.default_rng(K)
    psi = np.zeros(1 << n, complex)
    idx = rng.choice(1 << n, K, replace=False)
    psi[idx] = rng.normal(size=K) + 1j * rng.normal(size=K)
    r = sd.simdiff(psi, K=K, engine="numpy")
    assert r["chi_exact_max"] <= K and r["chi_star"] <= K
    assert r["S_half"] <= np.log2(K) + 1e-12 and 0 <= r["S_over_log2K"] <= 1 + 1e-12
    assert r["S_half"] == pytest.approx(sd.half_cut_entropy(psi), abs=1e-12)


@pytest.mark.parametrize("n", [5, 8, 10])
def test_numpy_and_quimb_agree_to_1e10(n):
    for seed in range(3):
        psi = _rand(n, seed)
        assert sd.cross_check(psi) <= 1e-10
        a = sd.simdiff(psi, engine="numpy")
        b = sd.simdiff(psi, engine="quimb")
        for k, v in a.items():
            if k.startswith("F_") or k in ("chi_star", "S_half", "mem_mps_bytes"):
                assert abs(v - b[k]) <= 1e-10, k


def test_chi_star_bisection_equals_scan_and_bounds_hold():
    rng = np.random.default_rng(7)
    for n in (6, 8, 9):
        for seed in range(3):
            # partly entangled: a random MPS-like state plus noise, so chi_star is inside the range
            psi = _rand(n, seed) * 0.05
            psi[rng.integers(0, 1 << n, 4)] += 1.0
            psi /= np.linalg.norm(psi)
            r = sd.simdiff(psi, engine="numpy")
            assert r["chi_star"] == sd.chi_star_scan(psi)
            for chi in sd.chi_grid(n):
                assert r[f"F_bound_chi{chi}"] <= r[f"F_chi{chi}"] + 1e-12
            assert r["chi_bracket_lo"] <= r["chi_star"] <= r["chi_bracket_hi"] <= r["chi_star_bound"] + 0


def test_peak_rss_reports_bytes():
    with sd.peak_rss() as m:
        x = np.ones(4_000_000)
        x += 1
    if m["peak_rss_bytes"] is None:
        pytest.skip("/proc/self/clear_refs not writable here")
    assert m["work_bytes"] >= 20_000_000


# --- the post-run digest and the aggregate (CPU) -------------------------------------------------------------------
def test_sample_digest_and_counts():
    from gsp.metrics.postrun import counts_to_arrays, sample_digest
    idx, cnt = counts_to_arrays({"0101": 3, "0001": 5, "1111": 2}, 4)
    assert idx.tolist() == [1, 5, 15] and cnt.tolist() == [5, 3, 2]
    ctx = MetricContext(n=4, band_idx=np.array([1, 5]), ar_weight=np.array([0.0, 1.0]), delta2=np.zeros(16),
                        diag=np.zeros(16), xstar_idx=np.array([5]), top10_idx=np.array([1, 5]),
                        sector_idx=np.array([1, 5, 15]))
    prob = np.zeros(16)
    prob[[1, 5, 15]] = [0.5, 0.3, 0.2]
    d = sample_digest(idx, cnt, ctx, prob=prob)
    assert d["ar_best_S_sampled"] == 1.0 and d["p_feas_sampled"] == 0.8 and d["p_sector_sampled"] == 1.0
    assert d["sample_tail_p"] > 0.5


def _copy_store(tmp_path, n_per_arm=2):
    src = GSP_ROOT / "results"
    root = tmp_path / "results"
    (root / "runs").mkdir(parents=True)
    for sub in ("instances", "sectors"):
        (root / sub).symlink_to(src / sub)
    picked = []
    for arm in ("A0", "A1", "A2c", "A2p"):
        for d in sorted((src / "runs" / arm).iterdir())[:n_per_arm]:
            dst = root / "runs" / arm / d.name
            shutil.copytree(d, dst, ignore=shutil.ignore_patterns("samples.npz", "postrun.json"))
            picked.append(dst)
    return root, picked


@pytest.mark.skipif(not (HAVE_INST and HAVE_RUNS), reason="stored runs absent")
def test_aggregate_on_copies_of_stored_runs(tmp_path):
    from gsp.metrics.aggregate import aggregate
    from gsp.store import load_metrics
    root, picked = _copy_store(tmp_path)
    df = aggregate(root)
    assert len(df) == len(picked) and (df["anomalies"] == "").all(), df["anomalies"].tolist()
    assert (df["postrun_source"] == "final_state").all()
    assert df["chk_state_vs_record"].max() <= 1e-10 and df["chk_traj_vs_record"].max() <= 1e-12
    conf = df[df["encoding"] == "confined"]
    assert conf["pre_available"].all() and (conf["sd_chi_star"] <= conf["K"]).all()
    assert (df["ar_best_S"] >= df["st_ar_f"] - 1e-12).all()
    back = load_metrics(root)
    assert len(back) == len(df)
    again = aggregate(root)                         # incremental: every row reused, same content
    assert again["agg_key"].tolist() == back["agg_key"].tolist()
    assert np.array_equal(again["ar_best_S"].to_numpy(float), back["ar_best_S"].to_numpy(float))
    assert np.array_equal(again["sd_wall_total_s"].to_numpy(float), back["sd_wall_total_s"].to_numpy(float))


# --- gpu: the post-run step --------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not (HAVE_INST and HAVE_RUNS), reason="stored runs absent")
def test_finalize_run_replays_and_samples(tmp_path):
    from gsp.metrics.postrun import POSTRUN_FILE, SAMPLES_FILE, finalize_run
    from gsp.store.io import load_npz
    root, picked = _copy_store(tmp_path, n_per_arm=1)
    for d in picked:
        out = finalize_run(d, root=root)
        assert out["status"] == "done" and out["replay_max_abs"] <= 1e-10
        s = load_npz(d / SAMPLES_FILE)
        assert int(s["counts"].sum()) == 1000 and int(s["shots"]) == 1000
        assert out["sample_tail_p"] >= 1e-4 and abs(out["sample_feas_z"]) <= 5
        rec = json.loads((d / "run.json").read_text())
        assert abs(out["ar_f"] - rec["metric_ar_f"]) <= 1e-10
    d = picked[1]
    first = load_npz(d / SAMPLES_FILE)
    finalize_run(d, root=root, force=True)
    again = load_npz(d / SAMPLES_FILE)
    assert np.array_equal(first["idx"], again["idx"]) and np.array_equal(first["counts"], again["counts"])
    assert json.loads((d / POSTRUN_FILE).read_text())["sample_seed"] == int(first["seed"])


@pytest.mark.gpu
@pytest.mark.skipif(not HAVE_INST, reason="frozen instances absent")
def test_arm_run_writes_samples_and_postrun(tmp_path):
    from gsp.arms.ramp import A2
    cell = {"connectivity": "ring", "rule": "violation", "K": 12}
    r = A2("confined").run("N04e004q1.5", cell, 5, runs_root=tmp_path)
    assert (r.path / "samples.npz").exists()
    pr = json.loads((r.path / "postrun.json").read_text())
    assert pr["status"] == "done" and pr["replay_max_abs"] == 0.0 and pr["sd_chi_star"] <= 12
    assert pr["p_sector"] == pytest.approx(1.0, abs=1e-12)
