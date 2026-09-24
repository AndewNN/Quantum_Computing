"""S9a (PLAN §5 S9): build and check the WP2 calibration queues -> results/queues/s9_q{1,2,3}_*.jsonl,
configs/matched_depths.yaml, reports/matched_depths.md, results/tables/s9a_calibration.json.

    ~/anaconda3/envs/gsp/bin/python scripts/s9_calibration.py depths | bench | queues | smoke-setup | smoke-check

  depths       (CPU) the proposed gate-matched depths L0(N, L1) of PLAN §1.3 from the S3 (ii) counters
               (`gsp.runner.calibration.write_matched_depths`).
  bench        (GPU, one process) the observe micro-benchmark: seconds per observe of every distinct loop circuit of the
               three queues (keyed by `calibration.bench_key`), for the estimates where S4 / S7 / S8 measured no rate;
               plus the extreme-effort circuit checks (A0 at L0(5, 9) and L0(7, 9); A4 / A6 at k = 10 on N = 4: one
               observe each, observe energy == state energy).
  queues       (CPU) queue 1 (timing), queue 2 (the lambda pilot = `gsp plan --pilot`), queue 3 (evidence O-2 / O-11)
               with run_ids, est_s per run, sidecars; run counts and GPU-hours per queue; queue 3 is trimmed first if
               the total exceeds --budget-h (12).
  smoke-setup  (CPU) a scratch results root (--root) whose instances / sectors are symlinks to the production store,
               and two tiny queues with one entry of every new spec kind (evidence flags, A3 overrides, A4 / A6, the
               K = 24 / complete cells, a pilot lam). Run them with the launcher itself:
                   GSP_RESULTS=ROOT S9_QUEUES="ROOT/queues/s9_smoke_a.jsonl ROOT/queues/s9_smoke_b.jsonl" \
                       bash scripts/s9_calibration.sh
  smoke-check  (CPU) validate the scratch runs (done, finalized, valid; the flags reached the circuits; D1 drops the
               evidence runs) -> results/tables/s9a_calibration.json "smoke".
One GPU process at a time: `bench` refuses to start while another process holds the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

GSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GSP))
OUT = GSP / "results" / "tables" / "s9a_calibration.json"
BUDGET_H = 12.0


def log(m):
    print(m, file=sys.stderr, flush=True)


def load_out() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_section(name: str, data) -> None:
    out = load_out()
    out[name] = data
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# --- depths -----------------------------------------------------------------------------------------------------------
def cmd_depths(args):
    from gsp.runner import calibration as C
    r = C.write_matched_depths(ring_order="lex")
    det = r["detail"]
    log(f"wrote {r['yaml']} and {r['md']}; check vs S3: {r['check']}")
    print("L0(N, L1)   L1=5  L1=7  L1=9")
    for N, m in sorted(det.items()):
        print(f"N={N}        {m[5]['L0']:>4}  {m[7]['L0']:>4}  {m[9]['L0']:>4}")
    save_section("matched_depths", {"L0": {N: {L1: d["L0"] for L1, d in m.items()} for N, m in det.items()},
                                    "detail": det, "check": r["check"]})


# --- queues -----------------------------------------------------------------------------------------------------------
def _all_specs():
    from gsp.runner import calibration as C
    L0 = C.load_matched_depths()
    q1 = C.resolve_strict(C.timing_specs(L0))
    q2, meta2 = C.pilot_queue_specs()
    q3 = C.resolve_strict(C.evidence_specs())
    return L0, q1, (q2, meta2), q3


def _bench():
    return (load_out().get("bench") or {}).get("per_observe_s") or {}


def cmd_queues(args):
    from gsp.runner import calibration as C
    from gsp.runner import plan as P
    L0, q1, (q2, meta2), q3 = _all_specs()
    bench = _bench()
    est = {1: C.estimate_specs(q1, bench), 2: C.estimate_specs(q2, bench), 3: C.estimate_specs(q3, bench)}
    total = sum(e["est_h"] for e in est.values())
    trimmed = None
    if total > args.budget_h:                      # trim queue 3 first (the orchestrator's rule), O-11 before O-2
        keep = []
        over = total - args.budget_h
        cut_h = 0.0
        for s in reversed(q3):
            if cut_h < over and s.get("est_s"):
                cut_h += s["est_s"] / 3600
                continue
            keep.append(s)
        trimmed = {"removed_runs": len(q3) - len(keep), "removed_h": cut_h}
        q3 = list(reversed(keep))
        est[3] = C.estimate_specs(q3, bench)
        total = sum(e["est_h"] for e in est.values())
    qdir = GSP / "results" / "queues"
    paths = {k: qdir / f"{v}.jsonl" for k, v in C.QUEUE_NAMES.items()}
    common = {"session": "S9a", "matched_depths": {str(N): {str(a): b for a, b in m.items()} for N, m in L0.items()},
              "rate_sources": C.RATE_SOURCES, "bench_keys_used": len(bench)}
    C.write_queue(q1, paths[1], dict(common, queue="timing (priority 1)", estimate=est[1]))
    C.write_queue(q2, paths[2], dict(common, **meta2, queue="lambda pilot (priority 2)", estimate=est[2]))
    C.write_queue(q3, paths[3], dict(common, queue="evidence O-2 / O-11 (priority 3)", estimate=est[3],
                                     trimmed=trimmed))
    rows = []
    for k, q in ((1, q1), (2, q2), (3, q3)):
        for s in q:
            rows.append({"queue": k, "arm": s["arm"], "label": s.get("label"), "run_id": s["run_id"],
                         "inst_id": s["inst_id"], "effort": s["effort"], "est_s": s.get("est_s"),
                         "source": s.get("est_source")})
    from gsp.runner.queue import run_state
    done = {k: sum(1 for s in q if run_state(s["arm"], s["run_id"])[0] == "done") for k, q in ((1, q1), (2, q2), (3, q3))}
    summary = {str(k): {"file": str(paths[k].relative_to(GSP)), "runs": len(q), "already_done": done[k],
                        "est_h": round(est[k]["est_h"], 3), "per_arm": est[k]["per_arm"], "unknown": est[k]["unknown"]}
               for k, q in ((1, q1), (2, q2), (3, q3))}
    save_section("queues", {"summary": summary, "total_est_h": round(total, 3), "budget_h": args.budget_h,
                            "trimmed_queue3": trimmed, "runs": rows})
    for k in (1, 2, 3):
        s = summary[str(k)]
        print(f"queue {k}: {s['file']}: {s['runs']} runs ({s['already_done']} already done), est {s['est_h']:.2f} GPU-h"
              f"; per arm {s['per_arm']}" + (f"; NO RATE: {s['unknown']}" if s["unknown"] else ""))
    print(f"total est {total:.2f} GPU-h (budget {args.budget_h} h)" + (f"; queue 3 trimmed: {trimmed}" if trimmed else ""))
    if args.list:
        for r in rows:
            print(f"  q{r['queue']} {r['arm']:<4} {r['run_id']} {r['inst_id']} eff={r['effort']:<3} "
                  f"est {r['est_s']} s  {r['label']}  [{r['source']}]")


# --- bench (GPU) ------------------------------------------------------------------------------------------------------
def _gpu_free_or_die():
    from gsp.sim import backend
    busy = backend.gpu_compute_pids()
    if busy:
        raise SystemExit(f"refused: GPU busy (PIDs {busy}); one GPU process at a time")


def _time_calls(f, reps=3):
    f()                                           # warm-up (kernel compile, op build)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def cmd_bench(args):
    _gpu_free_or_die()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from gsp.arms.base import make_arm
    from gsp.instances.adhoc import load_any
    from gsp.runner import calibration as C
    from gsp.sim import backend
    backend.ensure_target()
    L0, q1, (q2, _), q3 = _all_specs()
    todo = {}
    for s in q1 + q2 + q3:
        if s["arm"] in ("A0", "A1", "A2p", "A2c"):
            todo.setdefault(C.bench_key(s), s)
    per_obs, per_state, gates = {}, {}, {}
    arms = {}
    for i, (key, s) in enumerate(sorted(todo.items())):
        arm = arms.setdefault(s["arm"], make_arm(s["arm"]))
        inst, _, adhoc = load_any(s["inst_id"])
        cfg = arm.config(inst, s["cell"], s["effort"], None, adhoc=adhoc, **s["kw"])
        A, _ = arm.ansatz(cfg, inst)
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, A.n_params)
        per_obs[key] = _time_calls(lambda: A.energy(x))
        per_state[key] = _time_calls(lambda: A.state(x), reps=1)
        gates[key] = int(A.prog.n_gates)
        log(f"[{i + 1}/{len(todo)}] {key}: observe {per_obs[key] * 1e3:.2f} ms, state {per_state[key] * 1e3:.2f} ms, "
            f"{gates[key]} sim gates")
    # extreme-effort circuit checks: observe energy == state energy
    checks = []
    from gsp.instances.instance import load_rulers
    from gsp.metrics.state import metric_context
    for N in (5, 7):
        s = next(x for x in q1 if x["arm"] == "A0" and x.get("matched_L1") == 9 and x["N"] == N)
        arm = arms.setdefault("A0", make_arm("A0"))
        inst, rul, _ = load_any(s["inst_id"])
        cfg = arm.config(inst, None, s["effort"], None, **s["kw"])
        A, _ = arm.ansatz(cfg, inst)
        from gsp.train.init import init_params
        x0 = init_params(A, cfg.seed)
        t0 = time.perf_counter()
        f = A.energy(x0)
        t_obs = time.perf_counter() - t0
        psi = A.state(x0)
        e_state = float(np.abs(psi) ** 2 @ A.H.diagonal(np.arange(1 << A.n)))
        checks.append({"what": f"A0 N{N} L0={s['effort']} (Eq. 4.11 init)", "observe_over_alpha": f / A.alpha,
                       "state_energy": e_state, "abs_diff": abs(f / A.alpha - e_state), "observe_s": t_obs,
                       "norm_err": float(abs(np.vdot(psi, psi) - 1))})
        log(json.dumps(checks[-1]))
    from gsp.arms.dbqite import confined_db_circuit, penalty_db_circuit
    inst, _, _ = load_any("N04e004q1.5")
    for name, C_ in (("A6 N4 k10", penalty_db_circuit(inst, 0.005, "normalized")),
                     ("A4 N4 K12 k10", confined_db_circuit(inst, "violation", 12, "ga", "lex", "normalized")[0])):
        s_list = [0.02 / C_.sigma] * 10            # g = 0.02 at every step (params are s = g / sigma_H)
        t0 = time.perf_counter()
        try:
            e = C_.energy(s_list)
            t_obs = time.perf_counter() - t0
            psi = C_.state(s_list)
            e_state = float(np.abs(psi) ** 2 @ C_.ham.diagonal())
            chk = {"what": name, "observe_s": t_obs, "energy": float(e), "state_energy": e_state,
                   "abs_diff": abs(float(e) - e_state), "norm_err": float(abs(np.vdot(psi, psi) - 1)),
                   "tokens": len(C_.tokens(s_list))}
        except Exception as exc:                   # record, never hide
            chk = {"what": name, "error": f"{type(exc).__name__}: {exc}"}
        checks.append(chk)
        log(json.dumps(chk, default=str))
    save_section("bench", {"per_observe_s": per_obs, "per_state_s": per_state, "sim_gates": gates,
                           "extreme_checks": checks, "gpu": backend.runtime_info().get("gpu_name"),
                           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


# --- smoke ------------------------------------------------------------------------------------------------------------
def _smoke_specs():
    from gsp.runner import calibration as C
    sp = C.spec
    i4, i5, i6 = "N04e004q1.5", "N05e000q1.5", "N06e000q1.5"
    ring12 = {"connectivity": "ring", "rule": "violation", "K": 12}
    a = [sp("A0", i4, 2, kw={"restart": 0, "lam": 0.005, "circuit_boosted": True, "evidence": "O-2"}, lam=0.005,
            tag="smoke", label="A0 boosted evidence"),
         sp("A0", i4, 2, kw={"restart": 0, "lam": 0.005}, lam=0.005, tag="smoke", label="A0 un-boosted twin"),
         sp("A1", i4, 2, cell=ring12, kw={"restart": 0, "ring_order": "lex", "circuit_boosted": True,
                                           "evidence": "O-2"}, tag="smoke", label="A1 boosted evidence"),
         sp("A3", i4, 2, kw={"lam": 0.005, "n_steps": 3, "psd_project": True, "evidence": "O-11"}, lam=0.005,
            tag="smoke", label="A3 psd evidence"),
         sp("A3", i4, 2, kw={"lam": 0.005, "n_steps": 3, "tikhonov": 1e-4, "theta0_jitter": 1e-9,
                             "theta0_jitter_seed": 0, "evidence": "O-11"}, lam=0.005, tag="smoke",
            label="A3 tikhonov 1e-4 + jitter 0"),
         sp("A3", i4, 2, kw={"lam": 0.005, "n_steps": 3, "theta0_jitter": 1e-9, "theta0_jitter_seed": 1,
                             "evidence": "O-11"}, lam=0.005, tag="smoke", label="A3 default + jitter 1"),
         sp("A3", i4, 2, kw={"lam": 0.005, "n_steps": 3}, lam=0.005, tag="smoke", label="A3 timing kind")]
    b = [sp("A3d", i4, 2, kw={"lam": 0.005, "n_steps": 3}, lam=0.005, tag="smoke", label="A3d timing kind"),
         sp("A4", i4, 2, cell={"connectivity": "adaptive", "rule": "violation", "K": 12},
            kw={"ring_order": "lex", "step_units": "normalized"}, tag="smoke", label="A4 k2"),
         sp("A6", i4, 2, kw={"lam": 0.005, "step_units": "normalized"}, lam=0.005, tag="smoke", label="A6 k2"),
         sp("A2p", i4, 5, kw={"lam": 0.005, "schedule_tag": "primary"}, lam=0.005, schedule="primary", tag="smoke",
            label="A2p p5"),
         sp("A2c", i4, 5, cell=ring12, kw={"schedule_tag": "primary", "ring_order": "lex"}, schedule="primary",
            tag="smoke", label="A2c p5"),
         sp("A1", i5, 1, cell={"connectivity": "ring", "rule": "violation", "K": 24},
            kw={"restart": 0, "ring_order": "lex"}, tag="smoke", label="A1 K24 N5 L1"),
         sp("A1", i6, 1, cell={"connectivity": "complete", "rule": "violation", "K": 12},
            kw={"restart": 0, "ring_order": "lex"}, tag="smoke", label="A1 complete N6 L1"),
         sp("A0", i4, 1, kw={"restart": 0, "lam": 0.0005}, lam=0.0005, tag="smoke", label="A0 pilot lam 0.0005 L1"),
         sp("A3", "N07e000q1.5", 5, kw={"lam": 0.005, "n_steps": 5, "theta0_jitter": 1e-9, "theta0_jitter_seed": 0,
                                        "evidence": "O-11"}, lam=0.005, tag="smoke",
            label="A3 N7 L5 jitter 0, 5 steps (vs S7 ensemble member 0)")]
    return a, b


def cmd_smoke_setup(args):
    from gsp.runner import calibration as C
    root = Path(args.root).resolve()
    prod = GSP / "results"
    if root == prod.resolve():
        raise SystemExit("refused: the smoke root must not be the production store")
    root.mkdir(parents=True, exist_ok=True)
    for d in ("instances", "sectors"):
        link = root / d
        if not link.exists():
            link.symlink_to(prod / d, target_is_directory=True)
    a, b = _smoke_specs()
    qa, qb = C.resolve_strict(a), C.resolve_strict(b)
    (root / "queues").mkdir(exist_ok=True)
    for name, q in (("s9_smoke_a", qa), ("s9_smoke_b", qb)):
        C.write_queue(q, root / "queues" / f"{name}.jsonl", {"session": "S9a smoke"})
        log(f"wrote {root / 'queues' / (name + '.jsonl')}: {len(q)} specs")


def cmd_smoke_check(args):
    from gsp.arms.base import is_evidence, load_run, validate_run_dir
    from gsp.runner.plan import read_queue
    from gsp.runner.queue import postrun_state, run_state
    from gsp.store.paths import run_dir
    from gsp.train.mclachlan import ramp_init
    root = Path(args.root).resolve()
    res = []
    by_label = {}
    for name in ("s9_smoke_a", "s9_smoke_b"):
        for s in read_queue(root / "queues" / f"{name}.jsonl"):
            st, rec = run_state(s["arm"], s["run_id"], root)
            d = run_dir(s["arm"], s["run_id"], root)
            row = {"queue": name, "label": s.get("label"), "arm": s["arm"], "run_id": s["run_id"], "status": st,
                   "postrun": postrun_state(d) if st == "done" else None,
                   "problems": validate_run_dir(d) if st == "done" else ["not done"],
                   "evidence": (rec or {}).get("evidence"), "wall_s": (rec or {}).get("wall_s"),
                   "ar_f": (rec or {}).get("metric_ar_f")}
            if st == "done":
                rr = load_run(s["arm"], s["run_id"], root)
                if s["arm"] in ("A0", "A1"):
                    row["gamma_range"] = rr.record.get("diag_gamma_range")
                    row["circuit_boosted"] = rr.record.get("circuit_boosted")
                if s["arm"] == "A3":
                    p0 = rr.trajectory["params"][0]
                    row["theta0_minus_ramp_max"] = float(np.max(np.abs(p0 - ramp_init(int(s["effort"])))))
                    row["psd_clip"] = ("psd_clip_sum" in rr.trajectory)
                    row["tikhonov"] = rr.record.get("tikhonov")
                    row["psd_clip_sum_max"] = rr.record.get("metric_psd_clip_sum_max")
            by_label[s.get("label")] = row
            res.append(row)
    ok = all(r["status"] == "done" and r["postrun"] == "finalized" and not r["problems"] for r in res)
    checks = {"all_done_finalized_valid": ok}
    b, u = by_label.get("A0 boosted evidence"), by_label.get("A0 un-boosted twin")
    if b and u and b.get("gamma_range") and u.get("gamma_range"):
        checks["A0_boost_changes_gamma_range"] = bool(b["gamma_range"] < u["gamma_range"])
        checks["A0_gamma_range_boosted_vs_unboosted"] = [b["gamma_range"], u["gamma_range"]]
    j0, j1 = by_label.get("A3 tikhonov 1e-4 + jitter 0"), by_label.get("A3 default + jitter 1")
    if j0 and j1:
        checks["A3_jitter_applied"] = all(0 < r.get("theta0_minus_ramp_max", 0) < 1e-8 for r in (j0, j1))
    pz = by_label.get("A3 timing kind")
    if pz:
        checks["A3_no_jitter_default"] = pz.get("theta0_minus_ramp_max") == 0.0
    ps = by_label.get("A3 psd evidence")
    if ps:
        checks["A3_psd_diagnostics"] = bool(ps.get("psd_clip"))
    tk = by_label.get("A3 tikhonov 1e-4 + jitter 0")
    if tk:
        checks["A3_tikhonov_1e-4"] = tk.get("tikhonov") == 1e-4
    s7 = by_label.get("A3 N7 L5 jitter 0, 5 steps (vs S7 ensemble member 0)")
    if s7 and s7["status"] == "done":
        tr = load_run("A3", s7["run_id"], root).trajectory
        ref = json.loads((GSP / "results" / "tables" / "s7_varqite.json").read_text())["ensemble"]["members"][0]
        checks["A3_jitter_reproduces_S7_member0_step5"] = {"harness": float(tr["energy"][5]),
                                                           "s7": float(ref["E_step5"]),
                                                           "abs_diff": abs(float(tr["energy"][5]) - ref["E_step5"])}
        checks["A3_jitter_S7_match"] = abs(float(tr["energy"][5]) - ref["E_step5"]) <= 1e-12
    # D1 never reads an evidence run
    from gsp.stats import d1
    from gsp.store.index import build_index
    reg = build_index(root)
    checks["registry_rows"] = int(len(reg))
    ev_ids = {r["run_id"] for r in res if r.get("evidence")}
    spec = d1.d1_spec()
    seen = set()
    for arm in ("A0", "A1", "A3"):
        seen |= set(d1._match(reg, arm, spec.get(arm, {}))["run_id"]) if arm in spec else set()
    checks["d1_excludes_evidence"] = not (ev_ids & seen)
    checks["evidence_runs"] = len(ev_ids)
    checks["evidence_flag_rows_in_registry"] = int(sum(is_evidence(r) for r in reg.to_dict("records")))
    for r in res:
        print(f"{r['queue']} {r['arm']:<4} {r['run_id']} {r['status']:<6} postrun={r['postrun']} "
              f"problems={r['problems'] or '-'} wall={r['wall_s'] and round(r['wall_s'], 2)} {r['label']}")
    print(json.dumps(checks, indent=1, default=str))
    save_section("smoke", {"root": str(root), "runs": res, "checks": checks})
    return 0 if ok and all(v for k, v in checks.items() if isinstance(v, bool)) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["depths", "bench", "queues", "smoke-setup", "smoke-check"])
    ap.add_argument("--budget-h", type=float, default=BUDGET_H, help="queues: trim queue 3 above this total")
    ap.add_argument("--list", action="store_true", help="queues: one line per run")
    ap.add_argument("--root", default=None, help="smoke-setup / smoke-check: the scratch results root")
    args = ap.parse_args(argv)
    if args.cmd.startswith("smoke") and not args.root:
        ap.error("--root is required")
    fn = {"depths": cmd_depths, "bench": cmd_bench, "queues": cmd_queues, "smoke-setup": cmd_smoke_setup,
          "smoke-check": cmd_smoke_check}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
