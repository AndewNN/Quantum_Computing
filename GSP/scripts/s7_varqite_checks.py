"""S7 checks of arms A3 / A3d (PLAN §5 S7 "Done when"): writes results/tables/s7_varqite.json (one section per
subcommand, merged) and, with `report`, reports/varqite.md.

    ~/anaconda3/envs/gsp/bin/python scripts/s7_varqite_checks.py example|flag|repro|ensemble|smoke|time|report

One GPU process (refuses to start if another holds the GPU). Uses the frozen copies in tests/legacy (example_varqite.py;
the verbatim old driver `varqite_driver_m1f.py`), the numpy reference engine in tests/helpers, and the stored route
sweep in VarQITE/experiment/exp_Q2_L0.005_q1.5 (read-only). Sections:
  example    example_varqite.py's 3-qubit instance, L = 1, 2, 4, 6: the harness loop with the exact estimators (M6,
             numpy) and with the production estimators (M1 + C1) on the GPU and in numpy, vs the script
  flag       flag-qubit P(0...0) vs the statevector P(0...0) and the old full-depth "prob" route, N = 4 / 7 / 10
  repro      N07e000q1.5, lam 0.005, L = 5: A3d vs the stored M4C1 run; A3 vs the stored M1fC1 run; the verbatim old
             driver on CUDA-Q 0.15.1 vs the same stored run; the step-1 decomposition (M, C, thetadot) harness vs driver
  ensemble   A3 from the Ramp init jittered by N(0, 1e-9) rad (5 seeds) + the unjittered run + the driver replica:
             the rounding-noise spread of the final energy / stop step, against the stored M1fC1 final
  smoke      A3 at N = 4 and N = 10 (depth 5, 10 steps) and A3d at N = 4, stored in the production store (lam 0.005)
  time       A3 and A3d at N = 4 / 7 / 10, depth 5 and 9, 4 steps each, stored: s / step, circuits and g2q per step
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GSP))

import gsp._threads  # noqa: E402,F401
import numpy as np  # noqa: E402

STORED = GSP.parent / "VarQITE" / "experiment" / "exp_Q2_L0.005_q1.5"
KEY = "A7_p5_E0_S0"
REPRO_INST = "N07e000q1.5"
LAM = 0.005
SMOKE = {4: "N04e004q1.5", 7: "N07e000q1.5", 10: "N10e000q1.5"}
OUT = GSP / "results" / "tables" / "s7_varqite.json"


def log(m):
    print(m, file=sys.stderr, flush=True)


def load_out() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_section(name: str, data: dict) -> None:
    out = load_out()
    data["written"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out[name] = data
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"wrote section {name} -> {OUT}")


def stored(route):
    z = np.load(STORED / f"expectation_{route}_Ramp_boost_Jh.npz")
    return z[f"{KEY}_history"], json.loads(str(z[f"{KEY}_cfg"])), z[f"{KEY}_params"]


def a3_run(route, n_steps=300, x0=None, logger=True):
    from gsp.arms.qaoa import penalty_arm_ansatz
    from gsp.arms.varqite import run_a3
    from gsp.instances.instance import load_instance, load_rulers
    from gsp.metrics.state import metric_context
    from gsp.train.mclachlan import McLachlanConfig
    inst = load_instance(REPRO_INST)
    A = penalty_arm_ansatz(inst, LAM, 5)
    ctx = metric_context(inst, load_rulers(REPRO_INST), LAM)
    res, rows, psi, eng = run_a3(A, McLachlanConfig(metric="M1" if route == "M1fC1" else "diag", n_steps=n_steps),
                                 ctx=ctx, x0=x0, logger=logger)
    return A, res, rows


def rel_curve(E, Es):
    k = min(len(E), len(Es))
    return np.abs(E[:k] - Es[:k]) / np.abs(Es[:k])


def first_above(rel, thr):
    i = np.flatnonzero(rel > thr)
    return int(i[0]) + 1 if i.size else None


# --- example ------------------------------------------------------------------------------------------------------
def cmd_example(args):
    from gsp.arms.varqite import CircuitEngine
    from gsp.circuits.ansatz import penalty_ansatz
    from gsp.train.mclachlan import McLachlanConfig, run_mclachlan
    from tests.helpers.np_varqite import NumpyEngine
    from tests.test_varqite import EX_CFG, example_ising, example_x0
    src = (GSP / "tests" / "legacy" / "example_varqite.py").read_text().split('"""\n', 1)[1]
    ns = {}
    exec(src[:src.index('print("L=2 in detail:")')], ns)
    H = example_ising(ns)
    rows = []
    for L in (1, 2, 4, 6):
        A = penalty_ansatz(H, L, alpha=1.0)
        ex = float(ns["varqite"](L))
        r_ex = run_mclachlan(NumpyEngine(A), example_x0(L), McLachlanConfig(metric="exact", **EX_CFG))
        cfg = McLachlanConfig(metric="M1", **EX_CFG)
        t = time.perf_counter()
        r_g = run_mclachlan(CircuitEngine(A), example_x0(L), cfg)
        tg = time.perf_counter() - t
        r_n = run_mclachlan(NumpyEngine(A), example_x0(L), cfg)
        d = np.abs(r_g.E_loop - r_n.E_loop)
        rows.append({"L": L, "example_E": ex, "exact_loop_E": float(r_ex.E_loop[400]),
                     "exact_vs_example": abs(float(r_ex.E_loop[400]) - ex),
                     "m1_gpu_E": float(r_g.E_loop[400]), "m1_gpu_vs_example": float(r_g.E_loop[400]) - ex,
                     "m1_numpy_E": float(r_n.E_loop[400]), "gpu_vs_numpy_max": float(d.max()),
                     "gpu_vs_numpy_first10": float(d[:11].max()), "gpu_vs_numpy_final": float(d[-1]),
                     "cond_M_step1": float(r_g.step["cond"][1]), "cond_M_max": float(np.nanmax(r_g.step["cond"])),
                     "gpu_s_per_step": tg / 401, "exact_ground": float(np.min(np.real(np.diag(ns["H_C"]))))})
        log(json.dumps(rows[-1]))
    save_section("example", {"rows": rows, "cfg": EX_CFG})


# --- flag ---------------------------------------------------------------------------------------------------------
def cmd_flag(args):
    from gsp.arms.qaoa import penalty_arm_ansatz
    from gsp.arms.varqite import CircuitEngine
    from gsp.circuits import xkernel
    from gsp.instances.instance import load_instance
    from gsp.sim import backend
    from gsp.train.mclachlan import ramp_init
    from tests.legacy.varqite_driver_m1f import kernel_qaoa_X_overlap
    backend.ensure_target()
    rows = []
    for N, iid in SMOKE.items():
        for L in (5, 9):
            A = penalty_arm_ansatz(load_instance(iid), LAM, L)
            eng = CircuitEngine(A)
            fixed = xkernel.fixed_args(A.ct, L)
            rng = np.random.default_rng(100 * N + L)
            e_sv, e_old, n = 0.0, 0.0, 0
            for m in sorted({1, (L + 1) // 2, L}):
                for scale in (0.01, 0.3):
                    pa = ramp_init(L) + rng.normal(size=2 * L)
                    pb = pa.copy()
                    ks = rng.choice([j for j in range(2 * L) if j % L < m], size=2, replace=False)
                    pb[ks] += scale * rng.normal(size=2)
                    f = eng.fidelity(pa, pb, m)
                    e_sv = max(e_sv, abs(f - eng.fidelity_statevector(pa, pb, m)))
                    amp0 = backend.get_state(kernel_qaoa_X_overlap, list(pa), list(pb), *fixed)[0]
                    e_old = max(e_old, abs(f - abs(amp0) ** 2))
                    n += 1
            rows.append({"N": N, "n": A.n, "L": L, "pairs": n, "flag_vs_statevector": e_sv, "flag_vs_old_prob": e_old})
            log(json.dumps(rows[-1]))
    save_section("flag", {"rows": rows, "max_flag_vs_statevector": max(r["flag_vs_statevector"] for r in rows),
                          "max_flag_vs_old_prob": max(r["flag_vs_old_prob"] for r in rows)})


# --- repro --------------------------------------------------------------------------------------------------------
def replica(route, n_steps, x0=None):
    from tests.legacy import varqite_driver_m1f as D
    hist, scfg, _ = stored(route)
    cfg = dict(scfg)
    cfg["N_STEPS"] = n_steps
    s = D.VarQITE(cfg)
    x0 = s.make_points_init() if x0 is None else x0
    t = time.perf_counter()
    params, h, it, n = s.run_mcLachlan(x0, pbar=False)
    return s, params, h, n, time.perf_counter() - t


def cmd_repro(args):
    out = {}
    for route in ("M4C1", "M1fC1"):
        hist, scfg, sparams = stored(route)
        t = time.perf_counter()
        A, res, rows = a3_run(route)
        wall = time.perf_counter() - t
        E = rows["energy"][1:]
        rel = rel_curve(E, hist[:, 0])
        rmin = rows["V_tau"][:-1] * A.alpha ** 2 - res.step["cooling"][1:]
        k = min(len(rmin), len(hist))
        sec = {"stored_cfg": {kk: scfg[kk] for kk in ("A_METHOD", "FID_STENCIL", "KAPPA", "CAP_ANGLE", "EXACT_MEASURE",
                                                     "PRECISION", "TARGET", "TIKHONOV_LAMBDA", "DTAU", "FD_SHIFT")},
               "T_harness": res.n_steps, "T_stored": len(hist), "converged": bool(res.converged),
               "rel_E_steps_1_10": [float(v) for v in rel[:10]],
               "rel_E_max_first50": float(rel[:50].max()), "rel_E_max_all_common": float(rel.max()),
               "first_step_rel_gt_1e-6": first_above(rel, 1e-6),
               "final_E_harness": float(E[-1]), "final_E_stored": float(hist[-1, 0]),
               "final_rel": abs(float(E[-1]) - float(hist[-1, 0])) / abs(float(hist[-1, 0])),
               "final_P_ground_stored": float(hist[-1, 3]),
               "R_min_neg_harness": int((rmin < 0).sum()), "R_min_neg_stored": int((hist[:, 8] < 0).sum()),
               "R_min_max_abs_diff_common": float(np.max(np.abs(rmin[:k] - hist[:k, 8]))),
               "harness_s_per_step": float(res.wall_hist[-1] / res.n_steps), "harness_wall_s": wall,
               "ar_f_final": float(rows["ar_f"][-1]), "p_feas_final": float(rows["p_feas"][-1])}
        # the verbatim old driver on CUDA-Q 0.15.1 (fp64, EXACT_MEASURE "prob", default fusion of this process)
        s, rp, rh, rn, rw = replica(route, 300 if args.full_replica else 60)
        rrel = rel_curve(rh[:, 0], hist[:, 0])
        hr = rel_curve(E, rh[:, 0])
        import os
        sec.update({"replica_T": int(rn), "replica_steps_run": 300 if args.full_replica else 60,
                    "replica_fusion_max_qubits": os.environ.get("CUDAQ_FUSION_MAX_QUBITS"),
                    "replica_rel_E_steps_1_10": [float(v) for v in rrel[:10]],
                    "replica_first_step_rel_gt_1e-6": first_above(rrel, 1e-6),
                    "replica_rel_E_max_first50": float(rrel[:50].max()),
                    "replica_final_E": float(rh[-1, 0]), "replica_s_per_step": rw / rn,
                    "harness_vs_replica_rel_steps_1_10": [float(v) for v in hr[:10]]})
        if route == "M1fC1":
            sec["step1"] = step1_decomposition(s)
        out[route] = sec
        log(json.dumps({route: {k2: v for k2, v in sec.items() if not isinstance(v, (list, dict))}}))
    save_section("repro", out)


def step1_decomposition(s) -> dict:
    """M, C, thetadot at theta_0: the harness engine (flag route, observe) vs the old driver ("prob" route)."""
    from gsp.arms.qaoa import penalty_arm_ansatz
    from gsp.arms.varqite import CircuitEngine
    from gsp.instances.instance import load_instance
    from gsp.train.mclachlan import build_C, kappa_delta, metric_m1, ramp_init
    A = penalty_arm_ansatz(load_instance(REPRO_INST), LAM, 5)
    eng = CircuitEngine(A)
    x = ramp_init(5)
    x_old = s.make_points_init()
    d = eng.var_diag(x)
    delta = kappa_delta(d, eng.gen_scale, 0.01, 0.02)
    M, _ = metric_m1(eng.fidelity, x, delta, d, 5)
    C = build_C(eng.energy, x, eng.energy(x), 1e-4)
    td = np.linalg.solve(M + 1e-6 * np.eye(10), C)
    M_old = s.build_A(x_old.copy())
    C_old = s.build_C(x_old.copy())
    td_old = s.solve_theta_dot(M_old, C_old)
    w = np.linalg.eigvalsh(M)
    return {"theta0_equal": bool(np.array_equal(x, x_old)),
            "M_max_abs_diff": float(np.max(np.abs(M - M_old))), "M_max_abs": float(np.max(np.abs(M))),
            "M_diag_max_abs_diff": float(np.max(np.abs(np.diag(M) - np.diag(M_old)))),
            "C_max_rel_diff": float(np.max(np.abs(C - C_old)) / np.max(np.abs(C))),
            "thetadot_max_rel_diff": float(np.max(np.abs(td - td_old)) / np.max(np.abs(td))),
            "thetadot_max_abs": float(np.max(np.abs(td))),
            "cond_M": float(np.linalg.cond(M)), "eig_M": [float(v) for v in w],
            "kappa_delta": [float(v) for v in delta], "amplification_bound_1_over_tikhonov": 1e6}


def cmd_ensemble(args):
    hist, _, _ = stored("M1fC1")
    from gsp.train.mclachlan import ramp_init
    runs = []
    for seed in range(args.members):
        x0 = ramp_init(5) + np.random.default_rng(seed).normal(scale=1e-9, size=10)
        A, res, rows = a3_run("M1fC1", x0=x0)
        E = rows["energy"][1:]
        runs.append({"seed": seed, "T": res.n_steps, "final_E": float(E[-1]), "ar_f": float(rows["ar_f"][-1]),
                     "p_feas": float(rows["p_feas"][-1]), "E_step5": float(E[4]),
                     "rel_vs_unjittered_step5": None})
        log(json.dumps(runs[-1]))
    A, res, rows = a3_run("M1fC1")
    base = {"seed": None, "T": res.n_steps, "final_E": float(rows["energy"][-1]), "ar_f": float(rows["ar_f"][-1]),
            "p_feas": float(rows["p_feas"][-1])}
    rep = load_out().get("repro", {}).get("M1fC1", {})
    finals = [r["final_E"] for r in runs] + [base["final_E"]]
    if rep.get("replica_steps_run") == 300:
        finals.append(rep["replica_final_E"])
    Es = float(hist[-1, 0])
    save_section("ensemble", {
        "members": runs, "unjittered": base, "jitter_rad": 1e-9,
        "final_E_all": finals, "final_E_min": min(finals), "final_E_max": max(finals),
        "final_E_median": float(np.median(finals)), "stored_final_E": Es, "stored_T": len(hist),
        "stored_within_range": bool(min(finals) <= Es <= max(finals)),
        "stored_rank_among": int(np.sum(np.array(finals) < Es)), "T_all": [r["T"] for r in runs] + [base["T"]],
        "E0_ground": float(np.min(A.H.diagonal(np.arange(1 << A.n))))})


# --- smoke / time -------------------------------------------------------------------------------------------------
def _stored_run(arm, iid, L, n_steps):
    from gsp.arms.base import make_arm, validate_run_dir
    a = make_arm(arm)
    t = time.perf_counter()
    r = a.run(iid, None, L, lam=LAM, n_steps=n_steps)
    wall = time.perf_counter() - t
    rec, tr, c = r.record, r.trajectory, r.counts
    per_step = np.diff(tr["wall"])
    return {"arm": arm, "inst_id": iid, "n": int(rec["diag_n"]), "L": L, "steps": int(rec["metric_iterations"]),
            "run_id": r.run_id, "status": rec["status"], "valid": not validate_run_dir(r.path),
            "problems": validate_run_dir(r.path), "skipped": r.skipped,
            "s_per_step_mean": float(tr["wall"][-1] / len(per_step)),
            "s_per_step_median": float(np.median(per_step)),
            "s_per_step_after_first": float(np.mean(per_step[1:])) if len(per_step) > 1 else None,
            "logger_s_per_step": float(rec["time_logger_s"]) / len(per_step),
            "setup_s": float(rec["time_setup_s"]), "run_wall_s": float(rec["wall_s"]), "call_wall_s": wall,
            "circuits_per_step": int(c["circuits_per_unit"]), "overlap_per_step": int(c["classes"]["overlap"]["circuits"]),
            "variance_per_step": int(c["classes"]["variance"]["circuits"]),
            "energy_per_step": int(c["classes"]["energy"]["circuits"]),
            "g2q_ii_per_step": int(c["per_unit"]["cx_ii"]), "g2q_ii_energy_circuit": int(c["per_circuit"]["cx_ii"]),
            "R_min_neg": rec.get("metric_R_min_neg"), "cond_M_max": rec.get("metric_cond_M_max"),
            "ar_f_final": rec.get("metric_ar_f"), "energy_final": rec.get("metric_energy")}


def cmd_smoke(args):
    rows = []
    for arm, N in (("A3", 4), ("A3", 10), ("A3d", 4)):
        rows.append(_stored_run(arm, SMOKE[N], 5, 10))
        log(json.dumps(rows[-1]))
    save_section("smoke", {"rows": rows, "all_valid": all(r["valid"] for r in rows)})


def cmd_time(args):
    rows = []
    for arm in ("A3", "A3d"):
        for N in (4, 7, 10):
            for L in (5, 9):
                rows.append(_stored_run(arm, SMOKE[N], L, 4))
                log(json.dumps({k: rows[-1][k] for k in ("arm", "n", "L", "s_per_step_median", "circuits_per_step")}))
    save_section("time", {"rows": rows})


# --- report -------------------------------------------------------------------------------------------------------
def cmd_report(args):
    from gsp.arms.varqite_report import write
    print(f"written to {write(load_out())}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["example", "flag", "repro", "ensemble", "smoke", "time", "report"])
    ap.add_argument("--full-replica", action="store_true", help="repro: run the old driver to its stop (<= 300 steps)")
    ap.add_argument("--members", type=int, default=5, help="ensemble: jittered runs")
    args = ap.parse_args(argv)
    if args.cmd != "report":
        from gsp.sim import backend
        busy = backend.gpu_compute_pids()
        if busy:
            log(f"GPU busy (PIDs {busy}); one GPU process at a time")
            return 2
    {"example": cmd_example, "flag": cmd_flag, "repro": cmd_repro, "ensemble": cmd_ensemble, "smoke": cmd_smoke,
     "time": cmd_time, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
