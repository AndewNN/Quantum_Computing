"""S8 evidence for arms A4 / A6 (DB-QITE) and Rule C1 -> results/tables/s8_dbqite.json, reports/dbqite.md.

    ~/anaconda3/envs/gsp/bin/python scripts/s8_dbqite_checks.py example | c1op | c1state | smoke | time | report

  example  example_dbqite.py's 3-qubit instance, 7 steps: the GPU circuit and the numpy reflection-formula engine vs the
           script (its own compiled_step replayed with its chosen s): energies and the chosen s per step.
  c1op     Rule C1 operator level over every ring-row cell with n <= 12 (N = 4, 5, 6: 14 cells; the 3 complete cells
           share A4's runs with the baseline ring cells, PLAN §1.2), the first 3 accepted draws at q = 1.5 (K = 24: the
           first 3 k24-eligible draws): A4 for 5 steps on the GPU, the dense generator [rho_k, H] of every psi_k
           (k = 0..5); plus the compiled step's operator on the kept strings (npsim) at k = 0, 1, 2 on the first draw.
  c1state  Rule C1 state level over all 18 ring-row cells (first draw, q = 1.5): leak_k = 1 - p_sector(psi_k) along A4
           (5 steps) against 10 x eps_num(n, D_k) of 10 random A1 ring circuits matched in multi-controlled gates, and
           the growth exponents.
  smoke    stored runs (production store): A4 N04e004q1.5 / N07e000q1.5 (ring-row K12 cell, adaptive) and A6
           N04e004q1.5 / N10e000q1.5 (lam 0.005), 5 steps, step_units plan (PLAN §1.5) and normalized (O-13).
  time     stored runs A4 N07e000q1.5 and A6 N10e000q1.5, plan units, 8 steps: per-step wall clock for the 3^k model.
One GPU process at a time: every command that touches the GPU refuses to start while another holds it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

GSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GSP))
OUT = GSP / "results" / "tables" / "s8_dbqite.json"
LAM = 0.005
SMOKE = {("A4", 4): "N04e004q1.5", ("A4", 7): "N07e000q1.5", ("A6", 4): "N04e004q1.5", ("A6", 10): "N10e000q1.5"}
ADAPTIVE_K12 = {"connectivity": "adaptive", "rule": "violation", "K": 12}


def log(m):
    print(m, file=sys.stderr, flush=True)


def load_out() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_section(name: str, data: dict) -> None:
    out = load_out()
    out[name] = data
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def ring_cells() -> list[dict]:
    import yaml
    env = yaml.safe_load((GSP / "configs" / "envelope.yaml").read_text())
    return [c for c in env["confined_cells"] if c["connectivity"] == "ring"]


def cell_insts(cell: dict, k: int) -> list[str]:
    from gsp.sectors.select import cell_instances
    ids = [i for i in cell_instances(cell) if i.endswith("q1.5")]
    return ids[:k]


def cell_tag(c: dict) -> str:
    return f"{c['rule']}|K{c['K']}|N{c['N']}"


# --- example --------------------------------------------------------------------------------------------------------
def cmd_example(args):
    from gsp.arms.dbqite import NumpyDBEngine, run_dbqite
    from gsp.circuits import dbqite as db
    from gsp.circuits.xmixer import h_start
    src = (GSP / "tests" / "legacy" / "example_dbqite.py").read_text().split('"""\n', 1)[1]
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(src, ns)
    printed = buf.getvalue()
    levels = np.asarray(ns["levels"], dtype=float)
    C = db.DBCircuit(3, h_start(3), db.DBHam.from_diagonal(levels), kind="penalty", sigma=1.0, units="plan")
    t = time.perf_counter()
    g = run_dbqite(C, np.array(ns["grid"]), 7)
    tg = time.perf_counter() - t
    r = run_dbqite(NumpyDBEngine(C), np.array(ns["grid"]), 7)
    # the script's own trajectory, replayed with its functions (its greedy choice, its compiled_step)
    psi = np.ones(8, dtype=complex) / np.sqrt(8)
    E_s, s_s = [ns["energy"](psi)], []
    for _ in range(7):
        cand = [ns["energy"](ns["compiled_step"](psi, x)) for x in ns["grid"]]
        s = ns["grid"][int(np.argmin(cand))]
        psi = ns["compiled_step"](psi, s)
        s_s.append(float(s))
        E_s.append(ns["energy"](psi))
    E_s = np.array(E_s)
    rows = [{"k": k, "s_script": s_s[k], "s_gpu": float(g.s[k]), "s_numpy": float(r.s[k]),
             "E_script": float(E_s[k + 1]), "E_gpu": float(g.E_loop[k + 1]), "E_numpy": float(r.E_loop[k + 1]),
             "err_gpu": float(abs(g.E_loop[k + 1] - E_s[k + 1])), "err_numpy": float(abs(r.E_loop[k + 1] - E_s[k + 1])),
             "cand_err_gpu": float(np.max(np.abs(g.E_cand[k + 1] - r.E_cand[k + 1]))),
             "sim_gates": C.sim_gates(g.s[:k + 1].tolist()), "tokens": len(C.tokens(g.s[:k + 1].tolist()))}
            for k in range(7)]
    psi_g = C.state(g.s.tolist())
    phase = np.exp(-1j * np.sum(np.sqrt(g.s)))                  # the script keeps the leading e^{-i r rho}
    out = {"rows": rows, "s_agree": all(r_["s_script"] == r_["s_gpu"] == r_["s_numpy"] for r_ in rows),
           "max_err_gpu": max(r_["err_gpu"] for r_ in rows), "max_err_numpy": max(r_["err_numpy"] for r_ in rows),
           "E0": float(E_s[0]), "E0_gpu": float(g.E_loop[0]),
           "state_err_gpu_vs_script": float(np.max(np.abs(psi_g * phase - psi))),
           "p_ground_final": float(abs(psi_g[0]) ** 2), "gpu_s": tg, "printed": printed}
    log(json.dumps({k: v for k, v in out.items() if k not in ("rows", "printed")}))
    save_section("example", out)


# --- C1 operator level --------------------------------------------------------------------------------------------------
def a4_trajectory(iid: str, cell: dict, steps: int, units: str = "plan"):
    """(DBCircuit, sector view, instance, DBResult, states psi_0..psi_T) of A4 on the GPU (not stored)."""
    from gsp.arms.dbqite import confined_db_circuit, run_dbqite
    from gsp.circuits import dbqite as db
    from gsp.instances.instance import load_instance
    inst = load_instance(iid)
    C, sv = confined_db_circuit(inst, cell["rule"], cell["K"], units=units)
    states = []
    res = run_dbqite(C, np.asarray(db.GRID) / C.sigma, steps, logger=lambda t, s: states.append(C.state(s)))
    return C, sv, inst, res, states


def cmd_c1op(args):
    from gsp.stats.c1 import OPERATOR_TOL, a4_generator_check, a4_step_operator
    rows, steps_rows = [], []
    for cell in ring_cells():
        if 2 * int(cell["N"]) > 12:
            continue
        for j, iid in enumerate(cell_insts(cell, args.n_inst)):
            t = time.perf_counter()
            C, sv, inst, res, states = a4_trajectory(iid, cell, args.steps)
            d = C.ham.diagonal()
            for k, psi in enumerate(states):
                g = a4_generator_check(psi, d, sv.idx, dense=True)
                rows.append({"cell": cell_tag(cell), "axis": cell["axis"], "N": int(cell["N"]), "n": C.n,
                             "K": int(cell["K"]), "inst_id": iid, "k": k, **{x: g[x] for x in (
                                 "off_max", "w_norm", "rel", "w_max", "out_amp_max", "norm_err", "variance")}})
            if j == 0:
                for k in range(min(3, args.steps)):
                    so = a4_step_operator(C, res.s[:k].tolist(), float(res.s[k]), sv.idx)
                    steps_rows.append({"cell": cell_tag(cell), "n": C.n, "K": int(cell["K"]), "inst_id": iid, **so})
            log(f"{cell_tag(cell)} {iid}: max rel {max(r['rel'] for r in rows if r['inst_id'] == iid and r['cell'] == cell_tag(cell)):.2e} "
                f"({time.perf_counter() - t:.1f} s)")
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = (df.groupby(["cell", "axis", "N", "n", "K"], sort=False)
           .agg(instances=("inst_id", "nunique"), states=("k", "size"), rel_max=("rel", "max"),
                off_max=("off_max", "max"), out_amp_max=("out_amp_max", "max"), w_norm_min=("w_norm", "min"))
           .reset_index())
    tab["passes"] = tab["rel_max"] <= OPERATOR_TOL
    so = pd.DataFrame(steps_rows)
    out = {"table": tab.to_dict("records"), "rows": rows, "step_operator": steps_rows,
           "max_rel": float(df["rel"].max()), "all_pass": bool(tab["passes"].all()), "tol": OPERATOR_TOL,
           "step_operator_max_leak": float(so["leakage"].max()) if len(so) else None,
           "step_operator_max_block_err": float(so["block_err"].max()) if len(so) else None,
           "n_inst": args.n_inst, "steps": args.steps}
    log(tab.to_string(index=False))
    log(f"max rel {out['max_rel']:.2e}; step operator leak {out['step_operator_max_leak']:.2e}, block err "
        f"{out['step_operator_max_block_err']:.2e}")
    save_section("c1op", out)


# --- C1 state level ---------------------------------------------------------------------------------------------------
def cmd_c1state(args):
    from gsp.arms.base import draw_of, restart_seed
    from gsp.circuits import preserving as pr
    from gsp.stats.c1 import a4_state_level, leakage
    rows = []
    for cell in ring_cells():
        iid = cell_insts(cell, 1)[0]
        t = time.perf_counter()
        C, sv, inst, res, states = a4_trajectory(iid, cell, args.steps)
        lk = [leakage(psi, sv.idx) for psi in states]
        leak = [r["leak"] for r in lk[1:]]
        mc = C.counts(args.steps)["mc_gates"][1:]
        circ = pr.build_circuit(sv, "ring", "lex")
        st = a4_state_level(inst, circ, leak, mc, restart_seed(draw_of(iid), 0))
        st.update({"cell": cell_tag(cell), "axis": cell["axis"], "N": int(cell["N"]), "n": C.n, "K": int(cell["K"]),
                   "inst_id": iid, "out_mass_k": [r["out_mass"] for r in lk[1:]],
                   "norm_err_k": [r["norm_err"] for r in lk[1:]], "wall_s": time.perf_counter() - t})
        rows.append(st)
        log(f"{cell_tag(cell)} {iid}: leak {['%.1e' % v for v in leak]} eps {['%.1e' % v for v in st['eps_num_k']]} "
            f"pass {st['passes']} ratio {st['max_ratio']:.2f} exp A4 {st['growth_a4']['exponent']:.2f} "
            f"({st['growth_a4']['class']}) A1 {st['growth_a1']['exponent']:.2f} ({st['growth_a1']['class']}) "
            f"{st['wall_s']:.0f} s")
    # the pooled growth exponents (every cell's points)
    from gsp.stats.c1 import growth_exponent
    D4 = [d for r in rows for d in r["mc_k"]]
    L4 = [v for r in rows for v in r["leak_k"]]
    D1 = [d for r in rows for d in r["mc_a1"]]
    E1 = [v for r in rows for v in r["eps_num_k"]]
    out = {"rows": rows, "all_pass": all(r["passes"] for r in rows), "max_ratio": max(r["max_ratio"] for r in rows),
           "pooled_growth_a4": growth_exponent(D4, L4), "pooled_growth_a1": growth_exponent(D1, E1),
           "steps": args.steps}
    log(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    save_section("c1state", out)


# --- smoke / time ---------------------------------------------------------------------------------------------------------
def _stored_db_run(arm, iid, steps, units):
    from gsp.arms.base import make_arm, validate_run_dir
    a = make_arm(arm)
    kw = {"lam": LAM} if arm == "A6" else {}
    cell = ADAPTIVE_K12 if arm == "A4" else None
    t = time.perf_counter()
    r = a.run(iid, cell, steps, step_units=units, **kw)
    call = time.perf_counter() - t
    rec, tr, c = r.record, r.trajectory, r.counts
    pr = json.loads((r.path / "postrun.json").read_text()) if (r.path / "postrun.json").exists() else {}
    return {"arm": arm, "inst_id": iid, "n": int(rec["diag_n"]), "steps": int(rec["metric_steps"]), "units": units,
            "run_id": r.run_id, "status": rec["status"], "valid": not validate_run_dir(r.path), "skipped": r.skipped,
            "sigma_H": rec["diag_sigma_H"], "step_wall_s": np.diff(tr["wall"]).tolist(), "e0_s": float(tr["wall"][0]),
            "train_s": float(rec["time_train_s"]), "logger_s": float(rec["time_logger_s"]),
            "setup_s": float(rec["time_setup_s"]), "run_wall_s": float(rec["wall_s"]), "call_wall_s": call,
            "postrun_s": pr.get("postrun_s"),
            "cx_ii_circuit": tr["cx_ii_circuit"].tolist(), "cx_iii_circuit": tr["cx_iii_circuit"].tolist(),
            "g2q_ii_cum": tr["g2q_ii"].tolist(), "g2q_iii_cum": tr["g2q_iii"].tolist(),
            "sim_gates": tr["sim_gates"].tolist(), "tokens": tr["tokens"].tolist(), "mc_gates": tr["mc_gates"].tolist(),
            "s_k": tr["s_k"].tolist(), "g_k": tr["g_k"].tolist(), "grid_idx": tr["grid_idx"].tolist(),
            "r_rho": tr["r_rho"].tolist(), "r_H": tr["r_H"].tolist(),
            "E_loop": tr["E_loop"].tolist(), "energy_drop": tr["energy_drop"].tolist(),
            "variance": tr["variance"].tolist(), "ar_f": tr["ar_f"].tolist(), "p_feas": tr["p_feas"].tolist(),
            "p_opt": tr["p_opt"].tolist(), "leak_max": rec.get("metric_leak_max"),
            "out_mass_max": rec.get("metric_out_mass_max"), "replay": pr.get("replay_max_abs"),
            "postrun_status": pr.get("status")}


def cmd_smoke(args):
    rows = []
    for units in ("plan", "normalized"):
        for (arm, N), iid in SMOKE.items():
            rows.append(_stored_db_run(arm, iid, 5, units))
            r = rows[-1]
            log(f"{arm} N{N} {units}: {r['run_id']} valid {r['valid']} step s {['%.3f' % v for v in r['step_wall_s']]} "
                f"AR_F {r['ar_f'][0]:.4f} -> {r['ar_f'][-1]:.4f}")
    save_section("smoke", {"rows": rows, "all_valid": all(r["valid"] for r in rows)})


def cmd_time(args):
    rows = []
    for (arm, N) in (("A4", 7), ("A6", 10)):
        rows.append(_stored_db_run(arm, SMOKE[(arm, N)], args.steps, "plan"))
        r = rows[-1]
        log(f"{arm} N{N}: step s {['%.2f' % v for v in r['step_wall_s']]}, logger {r['logger_s']:.1f} s, "
            f"run {r['run_wall_s']:.1f} s, postrun {r['postrun_s']}")
    save_section("time", {"rows": rows})


def cmd_report(args):
    from gsp.arms.dbqite_report import write
    print(f"written to {write(load_out())}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["example", "c1op", "c1state", "smoke", "time", "report"])
    ap.add_argument("--steps", type=int, default=None, help="A4 steps (c1op / c1state: 5; time: 8)")
    ap.add_argument("--n-inst", type=int, default=3, help="c1op: instances per cell")
    args = ap.parse_args(argv)
    if args.steps is None:
        args.steps = 8 if args.cmd == "time" else 5
    if args.cmd != "report":
        from gsp.sim import backend
        busy = backend.gpu_compute_pids()
        if busy:
            log(f"GPU busy (PIDs {busy}); one GPU process at a time")
            return 2
    {"example": cmd_example, "c1op": cmd_c1op, "c1state": cmd_c1state, "smoke": cmd_smoke, "time": cmd_time,
     "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
