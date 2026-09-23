"""S5: Rules D1, D2, C1 (state level, growth fit) and C2 on synthetic data with a known answer (PLAN §5 S5
"Done when"), plus the exact signed-rank test and Holm."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from gsp.stats import c1, c2, d1, d2
from gsp.stats import synthetic as syn
from gsp.stats.ranktests import holm, signed_rank_exact
from gsp.store.paths import inst_path, sector_path

HAVE_INST = inst_path("N04e004q1.5").exists() and sector_path("N04e004", "violation", 12).exists()


# --- the exact signed-rank test and Holm ------------------------------------------------------------------
@pytest.mark.parametrize("m", [5, 9, 16, 30])
def test_signed_rank_equals_scipy_exact_without_ties(m):
    rng = np.random.default_rng(m)
    for shift in (0.0, 0.3, 1.0):
        d = rng.normal(size=m) + shift
        ours = signed_rank_exact(d)
        ref = st.wilcoxon(d, method="exact", alternative="two-sided")
        assert abs(ours.p - ref.pvalue) <= 1e-12
        assert min(ours.t_plus, ours.t_minus) == ref.statistic


def _brute_p(d, zero_tol=1e-12):
    nz = d[np.abs(d) > zero_tol]
    r = st.rankdata(np.abs(nz))
    t = r[nz > 0].sum()
    sums = np.array([sum(r[i] for i in range(nz.size) if s[i]) for s in itertools.product([0, 1], repeat=nz.size)])
    lo, hi = np.mean(sums <= t + 1e-9), np.mean(sums >= t - 1e-9)
    return min(1.0, 2 * min(lo, hi))


@pytest.mark.parametrize("seed", range(4))
def test_signed_rank_exact_with_ties_and_zeros_equals_enumeration(seed):
    rng = np.random.default_rng(seed)
    d = np.round(rng.normal(0.4, 1.0, 12), 1)          # ties
    d[:2] = 0.0                                       # zeros (dropped)
    assert abs(signed_rank_exact(d).p - _brute_p(d)) <= 1e-12
    nz = np.abs(d[np.abs(d) > 1e-12])
    r = signed_rank_exact(d)
    assert r.m == nz.size and r.ties == (np.unique(nz).size < nz.size)


def test_signed_rank_limits():
    assert signed_rank_exact(np.ones(30) * 0.1 + np.arange(30) * 1e-3).p == pytest.approx(2 / 2 ** 30, rel=1e-12)
    assert signed_rank_exact(np.zeros(10)).p == 1.0
    assert signed_rank_exact(np.array([1e-13, -1e-13])).m == 0


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    q = holm(np.array([0.01, np.nan, 0.02, np.nan]), m=4)
    assert np.isnan(q[1]) and np.isnan(q[3]) and np.allclose(q[[0, 2]], [0.04, 0.06])
    with pytest.raises(ValueError):
        holm(np.array([0.1, 0.2]), m=1)


# --- Rule D1 -------------------------------------------------------------------------------------------------
CASES = {
    "ordering": {"A0": -0.15},
    "below one rung": {"A0": lambda r: -0.006 + 0.001 * r.standard_normal(30)},
    "indistinguishable": {"A0": lambda r: 0.05 * r.standard_normal(30)},
}
GBAR = {(5, "A1"): 0.01, (5, "A0"): 0.004}


@pytest.mark.parametrize("expected", list(CASES))
@pytest.mark.parametrize("pairing", ["median", "best"])
def test_d1_returns_each_outcome(expected, pairing):
    res = d1.run_d1(syn.d1_curves(CASES[expected], seed=1), gbar=GBAR, pairings=(pairing,))
    p = res["pairs"]
    row = p[(p.family == "primary") & (p.arm_a == "A1") & (p.arm_b == "A0") & (p.k == 4)].iloc[0]
    assert row["outcome"] == expected and row["n_draws"] == 30
    assert row["delta_star"] == 0.02
    others = p[(p.family == "primary") & (p.arm_b != "A0")]
    assert (others["outcome"] == "no data").all()
    if expected == "ordering":
        assert row["direction"] == "A1>A0" and row["p_adj"] == pytest.approx(4 * 2 / 2 ** 30)


def test_d1_gap_threshold_uses_the_larger_gbar():
    res = d1.run_d1(syn.d1_curves({"A0": -0.03}, seed=2), gbar={(5, "A1"): 0.05, (5, "A0"): 0.01},
                    pairings=("median",))
    row = res["pairs"].query("family == 'primary' and arm_b == 'A0' and k == 4").iloc[0]
    assert row["delta_star"] == 0.05 and row["outcome"] == "below one rung"


def test_d1_truncation_and_best_of_R():
    g, a = np.array([0, 10, 20, 30]), np.array([0.1, 0.2, 0.3, 0.4])
    assert d1.value_at(g, a, 25) == 0.3 and d1.value_at(g, a, 30) == 0.4 and np.isnan(d1.value_at(g + 5, a, 4))
    cur = syn.d1_curves({}, seed=0, n_draws=4, configs={"A1": [("L5", 100, 0.0, 10.0)]})
    B = [1000.0]
    med = d1.instance_values(cur, B, "median")
    best = d1.instance_values(cur, B, "best")
    one = cur[(cur.inst_id == cur.inst_id.iloc[0])]
    vals_B = [d1.value_at(gq, ar, 1000.0) for gq, ar in zip(one.g2q, one.ar)]
    vals_BR = [d1.value_at(gq, ar, 200.0) for gq, ar in zip(one.g2q, one.ar)]
    assert med[1000.0].iloc[0] == pytest.approx(np.median(vals_B), abs=0)
    assert best[1000.0].iloc[0] == pytest.approx(max(vals_BR), abs=0)
    # a missing restart removes the configuration on that instance
    drop = cur.drop(index=one.index[-1])
    assert np.isnan(d1.instance_values(drop, B, "median")[1000.0].iloc[0])


def test_d1_q_average_needs_all_three_q():
    cur = syn.d1_curves({}, seed=0, n_draws=3)
    iv = d1.instance_values(cur, [1e4], "median")
    dv = d1.draw_values(iv, [1e4])
    g = iv[iv.draw_id == "N05e000"]
    assert dv[dv.draw_id == "N05e000"][1e4].iloc[0] == pytest.approx(g[1e4].mean(), abs=1e-15)
    part = d1.draw_values(iv[iv.q != 3.0], [1e4])
    assert part[1e4].isna().all()


def test_d1_cstar_depends_on_the_budget():
    # A0: a cheap config that saturates low and an expensive one that saturates high
    cfgs = {"A0": [("L5", 10, -0.2, 5.0), ("L9", 1000, 0.0, 5.0)]}
    res = d1.run_d1(syn.d1_curves({"A0": 0.0}, seed=3, configs=cfgs), gbar=GBAR, pairings=("median",))
    ch = res["choices"]
    chosen = ch[(ch.arm == "A0") & ch["chosen"].fillna(False).astype(bool)].set_index("k")["cfg_label"]
    assert chosen.loc[2] == "L5" and chosen.loc[5] == "L9"


def test_d1_exploratory_pairs_are_uncorrected_and_a6_is_in_no_pair():
    cur = syn.d1_curves({"A0": -0.1, "A2c": -0.05, "A6": 0.2}, seed=4,
                        configs={"A2c": [("p5", 500, 0.0, 0.0)], "A6": [("s1", 100, 0.0, 3.0)]})
    res = d1.run_d1(cur, gbar=GBAR, pairings=("median",))
    p = res["pairs"]
    ex = p[(p.family == "exploratory") & (p.arm_a == "A0") & (p.arm_b == "A2c")]
    assert len(ex) and np.allclose(ex["p_adj"], ex["p_raw"], equal_nan=True)
    assert not ((p.arm_a == "A6") | (p.arm_b == "A6")).any()
    assert set(map(tuple, p[p.family == "primary"][["arm_a", "arm_b"]].drop_duplicates().values)) == \
        set(d1.PRIMARY_PAIRS)
    # A2c has data only from its single circuit's cost on (B >= 500)
    a2 = p[(p.family == "primary") & (p.arm_b == "A2c")].set_index("k")["outcome"]
    assert a2.loc[2] == "no data" and a2.loc[3] != "no data"


@pytest.mark.skipif(not HAVE_INST, reason="frozen instances / sectors absent")
def test_gbar_from_the_frozen_set():
    t = d1.gbar_table(N_values=(4,), arms=("A1", "A0"))
    a1 = t[t.arm == "A1"].iloc[0]
    a0 = t[t.arm == "A0"].iloc[0]
    assert a1["kind"] == "sector" and a0["kind"] == "band" and a1["n_inst"] == 90
    assert 0 < a0["gbar"] < a1["gbar"] < 1        # the 12-string sector ladder is coarser than the band's


# --- Rule D2 -------------------------------------------------------------------------------------------------
def test_d2_recovers_a_planted_exponent():
    r = d2.d2_diagnostic(syn.d2_data("deff", exponent=-1.0, seed=0))          # the rule's 2000 resamples
    assert r.separable and r.preferred and r.ratio <= 0.8 and r.ratio_ci[1] < 1
    assert r.exponent_ci[0] <= -1.0 <= r.exponent_ci[1] and abs(r.exponent + 1.0) < 0.02


def test_d2_n_truth_is_not_preferred():
    r = d2.d2_diagnostic(syn.d2_data("n", exponent=-0.4, seed=1), n_boot=500)
    assert r.separable and not r.preferred and r.ratio > 1


def test_d2_collinear_predictors_are_not_separable():
    data = syn.d2_data("collinear", exponent=-1.0, seed=2)
    r = d2.d2_diagnostic(data, n_boot=200)
    assert not r.separable and abs(r.precond["spearman_rho"]) > 0.8 and r.precond["design_ok"]
    out = d2.run_d2({"V_A1": data, "drop_A4": syn.d2_data("deff", seed=3)}, n_boot=200)
    assert out["verdict"]["verdict"] == "not separable"
    few = syn.d2_data("deff", seed=4, K_values=(6, 8))
    assert not d2.d2_diagnostic(few, n_boot=100).precond["design_ok"]


def test_d2_verdicts():
    same = {"V_A1": syn.d2_data("deff", -1.0, seed=5), "drop_A4": syn.d2_data("deff", -1.0, seed=6)}
    assert d2.run_d2(same, n_boot=500)["verdict"]["verdict"] == "supported"
    diff = {"V_A1": syn.d2_data("deff", -1.0, seed=7), "drop_A4": syn.d2_data("deff", -0.5, seed=8)}
    assert d2.run_d2(diff, n_boot=500)["verdict"]["verdict"] == "partially supported"
    one = {"V_A1": syn.d2_data("deff", -1.0, seed=9), "drop_A4": syn.d2_data("n", -0.4, seed=10)}
    assert d2.run_d2(one, n_boot=500)["verdict"]["verdict"] == "refuted"


# --- Rule C1 -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(3))
def test_c1_growth_fit_separates_sqrtD_from_linear(seed):
    D, e = syn.leakage_series("sqrt", seed=seed)
    g = c1.growth_exponent(D, e)
    assert g["class"] == "sqrt(D)" and abs(g["exponent"] - 0.5) < 0.2
    D, e = syn.leakage_series("linear", seed=seed)
    g = c1.growth_exponent(D, e)
    assert g["class"] == "linear" and abs(g["exponent"] - 1.0) < 0.15


def test_c1_state_level_rule():
    eps = np.array([1e-16, 2e-16, 3e-16])
    assert c1.state_level([5e-16, 1e-15, 2e-15], eps)["passes"]
    bad = c1.state_level([5e-16, 1e-15, 4e-15], eps)
    assert not bad["passes"] and bad["first_fail_k"] == 2 and bad["max_ratio"] > 1
    lk = c1.leakage(np.array([0.6, 0.8, 0.0, 0.0]), [0, 1])
    assert abs(lk["leak"]) <= 1e-15 and lk["out_mass"] == 0.0


@pytest.mark.skipif(not HAVE_INST, reason="frozen instances / sectors absent")
def test_c1_eps_num_curve_numpy():
    from gsp.circuits import preserving as pr
    from gsp.instances.instance import load_instance
    from gsp.sectors.select import load_sector
    inst = load_instance("N04e004q1.5")
    circ = pr.build_circuit(load_sector("N04e004q1.5", "violation", 12), "ring", "lex")
    out = c1.eps_num_curve(inst, circ, [1, 2, 4], seed=12345, n_circuits=3)
    assert [r["mc_gates"] for r in out] == [10 + 12 * L for L in (1, 2, 4)]
    assert all(r["eps_num"] < 1e-13 and r["max_out_mass"] == 0.0 for r in out)   # native gates: exact zeros


# --- Rule C2 -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [2, 12, 144, 576])
def test_c2_haar_bins_are_the_exact_integrals(d):
    lp = c2.haar_bin_log_probs(d)
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-12)
    e = np.linspace(0, 1, c2.N_BINS + 1)
    i = 3
    from scipy.integrate import quad
    ref = quad(lambda F: (d - 1) * (1 - F) ** (d - 2), e[i], e[i + 1], epsabs=0, epsrel=1e-13)[0]
    assert np.exp(lp[i]) == pytest.approx(ref, rel=1e-9)


def test_c2_kl_of_haar_states_is_small_and_a_fixed_state_is_not():
    rng = np.random.default_rng(0)
    d = 12
    a, b = c2.haar_states(d, 5000, rng), c2.haar_states(d, 5000, rng)
    fh = c2.pair_fidelities(a, b)
    kh = c2.kl_bootstrap(fh, d, n_boot=200)
    assert kh["kl"] < 0.02 and 0 <= kh["lo"] < kh["hi"] < 0.03     # (the fixed-bin estimator is biased upward)
    near = np.clip(1 - np.abs(rng.normal(0, 0.01, 5000)), 0, 1)       # an ensemble that barely moves
    assert c2.kl_to_haar(near, d) > 3.0


def test_c2_completion():
    rows = [{"arm": "A1", "connectivity": c, "level": L, "kl": 0.1, "lo": 0.05, "hi": 0.2, "d": 144}
            for c in ("ring", "complete") for L in (1, 3, 5, 7)]
    assert c2.completion(pd.DataFrame(rows))["complete"]
    assert not c2.completion(pd.DataFrame(rows[:-1]))["complete"]


def test_d1_config_key_ignores_instance_identity():
    base = {"arm": "A1", "encoding": "confined", "inst_id": "N04e000q1.0", "K": 12, "rule": "violation",
            "connectivity": "ring", "effort_kind": "depth", "effort": 5, "restart": 0, "lam": None,
            "schedule": None, "seed": 1, "seed_ga": 2, "harness_version": "0.1.0", "ring_order": "lex"}
    other = dict(base, inst_id="N04e003q3.0", restart=4, seed=9, seed_ga=7, N=4, q=3.0, draw_id="N04e003",
                 run_dir="A1/x")
    assert d1.config_key(base) == d1.config_key(other)
    assert d1.config_key(base) != d1.config_key(dict(base, effort=7))
