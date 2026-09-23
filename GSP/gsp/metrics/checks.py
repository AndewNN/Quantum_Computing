"""`gsp metrics checks`: the S5 evidence (PLAN §5 S5 "Done when") -> results/tables/s5_checks.json.

  example    tests/legacy/example_metrics.py (the frozen copy of Lecture_Notes/code/example_metrics.py) run as is;
             the library's p_feas / eps_tilde / AR_F / p_opt / quality / Monte-Carlo AR_best_S / P(opt seen) against
             the script's own values (max abs error), and the exact AR_best_S against its 20000-row estimate.
  simdiff    numpy vs quimb over every stored final state (max |F_numpy - F_quimb| over the chi grid, entropy and
             chi_star agreement) and the timing of `simdiff` on random states at n = 16, 18, 20 (worst case).
  d1         the three outcomes on synthetic paired data, both R pairings.
  d2         the planted exponent (d_eff truth), the n truth, collinear predictors, and the three verdicts.
  c1         the growth-exponent classes on synthetic sqrt(D) and linear leakage (5 seeds each).
  c2         KL of Haar pairs vs a concentrated ensemble.
"""

from __future__ import annotations

import contextlib
import io
import json
import runpy
import time

import numpy as np

from ..store.paths import GSP_ROOT, runs_dir, tables_dir


def example_check() -> dict:
    from . import quality as ql
    from .state import MetricContext
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(str(GSP_ROOT / "tests" / "legacy" / "example_metrics.py"))
    feas, prob, f, Delta, quality = g["feasible"], g["prob"], g["f"], g["Delta"], g["quality"]
    band = np.nonzero(feas)[0]
    E_min, E_max = f[feas].min(), f[feas].max()
    ctx = MetricContext(n=4, band_idx=band, ar_weight=ql.normalized_quality(f[band], E_min, E_max),
                        delta2=Delta ** 2, diag=f, xstar_idx=np.nonzero(feas & np.isclose(f, E_min))[0],
                        top10_idx=band)
    m = ctx.evaluate(prob)
    S = g["S"]
    errs = {"p_feas": abs(m["p_feas"] - g["p_feas"]), "eps_tilde": abs(m["eps_tilde"] - g["eps_tilde"]),
            "AR_F": abs(m["ar_f"] - g["AR_F"]), "p_opt": abs(m["p_opt"] - g["p_opt"]),
            "quality": float(np.max(np.abs(ctx.ar_weight - quality[band]))),
            "AR_best_S_MC": abs(ql.ar_best_from_shots(g["shots"], quality, feas) - float(np.mean(g["best"]))),
            "P_opt_seen": abs(ql.p_seen(m["p_opt"], S) - (1 - (1 - g["p_opt"]) ** S))}
    exact = ql.ar_best_exact(prob[band], ctx.ar_weight, S)
    se = float(np.std(g["best"], ddof=1) / np.sqrt(len(g["best"])))
    return {"errors": errs, "max_error": max(errs.values()), "values": {k: float(m[k]) for k in m},
            "AR_best_S_script": float(np.mean(g["best"])), "AR_best_S_exact": exact, "AR_best_S_se": se,
            "exact_vs_script_in_se": abs(exact - float(np.mean(g["best"]))) / se, "S": int(S)}


def simdiff_check(root=None, timing_n=(16, 18, 20)) -> dict:
    from . import simdiff as sd
    files = sorted(runs_dir(root).glob("*/*/final_state.npy"))
    worst_F = worst_S = 0.0
    chi_mismatch = 0
    for p in files:
        psi = np.load(p)
        a = sd.simdiff(psi, engine="numpy", measure_rss=False)
        b = sd.simdiff(psi, engine="quimb", measure_rss=False)
        worst_F = max(worst_F, max(abs(a[k] - b[k]) for k in a if k.startswith("F_chi")))
        worst_S = max(worst_S, abs(a["S_half"] - b["S_half"]))
        chi_mismatch += int(a["chi_star"] != b["chi_star"])
    timing = []
    rng = np.random.default_rng(0)
    for n in timing_n:
        v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        v /= np.linalg.norm(v)
        for eng in sd.ENGINES:
            t = time.perf_counter()
            r = sd.simdiff(v, engine=eng)
            timing.append({"n": n, "engine": eng, "wall_s": time.perf_counter() - t, "chi_star": r["chi_star"],
                           "n_sweeps": r["n_sweeps"], "work_bytes": r["work_bytes"]})
    return {"n_states": len(files), "max_F_diff": worst_F, "max_S_diff": worst_S, "chi_star_mismatch": chi_mismatch,
            "default_engine": sd.DEFAULT_ENGINE, "timing_random": timing}


def d1_check() -> dict:
    from ..stats import d1, synthetic as syn
    cases = {"ordering": {"A0": -0.15},
             "below one rung": {"A0": lambda r: -0.006 + 0.001 * r.standard_normal(30)},
             "indistinguishable": {"A0": lambda r: 0.05 * r.standard_normal(30)}}
    out = []
    for name, sh in cases.items():
        res = d1.run_d1(syn.d1_curves(sh, seed=1), gbar={(5, "A1"): 0.01, (5, "A0"): 0.004})
        p = res["pairs"]
        for _, r in p[(p.family == "primary") & (p.arm_b == "A0") & (p.k == 4)].iterrows():
            out.append({"planted": name, "pairing": r["pairing"], "median_d": r["median_d"], "p_raw": r["p_raw"],
                        "p_holm": r["p_adj"], "delta_star": r["delta_star"], "outcome": r["outcome"],
                        "ok": r["outcome"] == name})
    return {"rows": out, "all_ok": all(r["ok"] for r in out)}


def d2_check() -> dict:
    from ..stats import d2, synthetic as syn
    rows = []
    for kind, e, seed in (("deff", -1.0, 0), ("deff", -0.5, 1), ("n", -0.4, 2), ("collinear", -1.0, 3)):
        r = d2.d2_diagnostic(syn.d2_data(kind, exponent=e, seed=seed))
        rows.append({"truth": kind, "planted": e, "separable": r.separable, "rho": r.precond["spearman_rho"],
                     "ratio": r.ratio, "ratio_ci": list(r.ratio_ci), "preferred": r.preferred,
                     "exponent": r.exponent, "exponent_ci": list(r.exponent_ci)})
    ver = {}
    for name, (a, b) in {"supported": (("deff", -1.0), ("deff", -1.0)),
                         "partially supported": (("deff", -1.0), ("deff", -0.5)),
                         "refuted": (("deff", -1.0), ("n", -0.4)),
                         "not separable": (("collinear", -1.0), ("deff", -1.0))}.items():
        out = d2.run_d2({"V_A1": syn.d2_data(a[0], a[1], seed=11), "drop_A4": syn.d2_data(b[0], b[1], seed=12)})
        ver[name] = out["verdict"]["verdict"]
    return {"diagnostics": rows, "verdicts": ver, "verdicts_ok": all(k == v for k, v in ver.items())}


def c1_check() -> dict:
    from ..stats import c1, synthetic as syn
    rows = []
    for kind in ("sqrt", "linear"):
        for seed in range(5):
            D, e = syn.leakage_series(kind, seed=seed)
            g = c1.growth_exponent(D, e)
            rows.append({"truth": kind, "seed": seed, "exponent": g["exponent"], "ci": [g["lo"], g["hi"]],
                         "class": g["class"], "ok": g["class"] == ("sqrt(D)" if kind == "sqrt" else "linear")})
    return {"rows": rows, "all_ok": all(r["ok"] for r in rows)}


def c2_check() -> dict:
    from ..stats import c2
    rng = np.random.default_rng(0)
    out = {}
    for d in (12, 144):
        f = c2.pair_fidelities(c2.haar_states(d, c2.N_PAIRS, rng), c2.haar_states(d, c2.N_PAIRS, rng))
        out[f"haar_d{d}"] = c2.kl_bootstrap(f, d)
    near = np.clip(1 - np.abs(rng.normal(0, 0.01, c2.N_PAIRS)), 0, 1)
    out["concentrated_d12"] = c2.kl_bootstrap(near, 12)
    return out


def run_all(root=None, write: bool = True, log=None) -> dict:
    out = {}
    for name, fn in (("example", example_check), ("d1", d1_check), ("d2", d2_check), ("c1", c1_check),
                     ("c2", c2_check), ("simdiff", lambda: simdiff_check(root))):
        t = time.perf_counter()
        out[name] = fn()
        if log:
            log(f"{name}: {time.perf_counter() - t:.1f} s")
    if write:
        p = tables_dir(root) / "s5_checks.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return out
