"""S4 wall-clock per training iteration, before / after the simulation speedups (STATUS S4).

Cases (PLAN §5 S4 hand-off): A0 at N = 10, L = 9 (n = 20, lam = 0.005) and A1 at N = 7, K = 12, L = 9 (n = 14, ring
violation lex, GA sector), q = 1.5, first `n_inst` accepted draws. For each case and engine, a few AdamW iterations
(forward FD: 2L + 1 = 19 observes each, no logger) are timed and the median per iteration is kept, plus the cost of
one logger call (get_state + metrics).

Engines:
  A0  verbatim   `xkernel.kernel_qaoa_X` (the old kernel copied verbatim; cx-rz-cx cost layer)
      interp     the flat gate list of `xmixer.a0_gates` on the S3 interpreter kernel
      layered    the harness route: layered kernel, `cost_gates_sim` (crz + merged rz)
  A1  interp     S3's route: the flat `preserving.a1_gates` list on the interpreter kernel
      layered    the harness route: layered kernel, `cost_gates_sim` + `merge_x`
The gate-fusion limit is a per-process setting (CUDAQ_FUSION_MAX_QUBITS, read when the target is set), so
`run_all` runs one subprocess per setting, sequentially (one GPU process at a time): 4 = the CUDA-Q default
("before"), 1 = the harness default ("after").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

from ..store.paths import tables_dir

A0_CASE = {"N": 10, "L": 9, "lam": 0.005}
A1_CASE = {"N": 7, "K": 12, "L": 9}


def _time_train(energy, x0, iters: int) -> float:
    from ..train.adamw import AdamWConfig, train_adamw
    res = train_adamw(energy, x0, AdamWConfig(max_iter=iters))
    return float(np.median(np.diff(res.wall_hist)))


def measure(n_inst: int = 3, iters: int = 4) -> dict:
    from ..circuits import ansatz as az, preserving as pr, program, xkernel
    from ..instances.instance import load_instance, load_instances_table, load_rulers
    from ..metrics.state import StateLogger, metric_context
    from ..sectors.select import load_sector
    from ..sim import backend
    from ..train.init import init_params

    backend.set_target()
    it = load_instances_table()
    rows = []

    def ids(N):
        return it[(it["N"] == N) & (it["q"] == 1.5)].sort_values("accept_rank")["inst_id"].tolist()[:n_inst]

    for iid in ids(A0_CASE["N"]):
        inst, rul = load_instance(iid), load_rulers(iid)
        H = inst.hamiltonian(A0_CASE["lam"])
        A = az.penalty_ansatz(H, A0_CASE["L"])
        x0 = init_params(A, 1234)
        kv = xkernel.kernel_qaoa_X
        fx = xkernel.fixed_args(A.ct, A.L)
        flat = program.encode(A.abstract_gates(), A.n)
        engines = {
            "verbatim": lambda p: backend.observe(kv, A.op, list(p), *fx),
            "interp": lambda p: backend.observe(program.kernel(), A.op, *flat.args(p)),
            "layered": A.energy,
        }
        for name, fn in engines.items():
            fn(x0)                                          # JIT / warm-up
            rows.append({"case": "A0 N10 L9", "inst_id": iid, "engine": name, "n": A.n,
                         "gates": {"verbatim": len(A.abstract_gates()), "interp": flat.n_gates,
                                   "layered": A.prog.n_gates}[name],
                         "s_per_iter": _time_train(fn, x0, iters)})
        log = StateLogger(A, metric_context(inst, rul, A0_CASE["lam"]))
        t = time.perf_counter()
        for _ in range(3):
            log(0, x0)
        rows.append({"case": "A0 N10 L9", "inst_id": iid, "engine": "logger (layered)", "n": A.n,
                     "gates": A.prog.n_gates, "s_per_iter": (time.perf_counter() - t) / 3})
    for iid in ids(A1_CASE["N"]):
        inst, rul = load_instance(iid), load_rulers(iid)
        sec = load_sector(iid, "violation", A1_CASE["K"])
        circ = pr.build_circuit(sec, "ring", "lex")
        A = az.confined_ansatz(inst.H_obj, inst.boost_obj, circ, A1_CASE["L"])
        x0 = init_params(A, 1234)
        flat = program.encode(A.abstract_gates(), A.n)
        engines = {"interp": lambda p: backend.observe(program.kernel(), A.op, *flat.args(p)), "layered": A.energy}
        for name, fn in engines.items():
            fn(x0)
            rows.append({"case": "A1 N7 K12 L9", "inst_id": iid, "engine": name, "n": A.n,
                         "gates": {"interp": flat.n_gates, "layered": A.prog.n_gates}[name],
                         "s_per_iter": _time_train(fn, x0, iters)})
        log = StateLogger(A, metric_context(inst, rul, None, sector_idx=sec.idx))
        t = time.perf_counter()
        for _ in range(3):
            log(0, x0)
        rows.append({"case": "A1 N7 K12 L9", "inst_id": iid, "engine": "logger (layered)", "n": A.n,
                     "gates": A.prog.n_gates, "s_per_iter": (time.perf_counter() - t) / 3})
    return {"fusion_max_qubits": os.environ.get("CUDAQ_FUSION_MAX_QUBITS"), **backend.runtime_info(), "rows": rows}


def run_all(fusions=("4", "1"), n_inst: int = 3, iters: int = 4, log=print) -> dict:
    """One subprocess per fusion setting, sequentially; merged into results/tables/s4_timing.json."""
    from ..sim.backend import gpu_compute_pids
    out = {"cases": {"A0": A0_CASE, "A1": A1_CASE}, "runs": []}
    for fu in fusions:
        busy = gpu_compute_pids()
        if busy:
            raise RuntimeError(f"GPU busy (compute PIDs {busy}); one GPU process at a time")
        env = dict(os.environ, CUDAQ_FUSION_MAX_QUBITS=str(fu))
        code = ("import json, sys; from gsp.arms import timing; "
                f"print('@@' + json.dumps(timing.measure({n_inst}, {iters})))")
        t = time.perf_counter()
        cp = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        line = [x for x in cp.stdout.splitlines() if x.startswith("@@")][-1]
        res = json.loads(line[2:])
        out["runs"].append(res)
        if log:
            log(f"fusion {fu}: {time.perf_counter() - t:.0f} s")
            for r in res["rows"]:
                log(f"  {r['case']:<13} {r['inst_id']} {r['engine']:<17} gates={r['gates']:>6} "
                    f"{r['s_per_iter']:.3f} s/iter")
    d = tables_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "s4_timing.json").write_text(json.dumps(out, indent=1))
    return out
