"""GPU timing of the compiled A1 circuit (PLAN §5 S3 "Timing"): build + one `observe`, sequentially.

Cells: N = 7 K = 12 ring; N = 6 K = 24 ring; N = 6 K = 12 complete (violation rule, lex order), on the
first `n_inst` instances of each cell at q = 1.5. For each depth L and each engine:
  interp   the interpreter kernel (default): encode the gate list (CPU), the first `observe` (the
           process-wide JIT happens on the very first call only), `repeats` more `observe` calls with
           fresh random angles (steady state), and `get_state` calls (what the S4 logger will pay);
  builder  PLAN §2.4's builder kernel: construction, first `observe` (JIT of this circuit), steady
           `observe`, and one `get_state` (which re-compiles the builder kernel on every call).
Writes results/tables/mixer_timing.json.
One GPU process only: refuses to start if another process holds a CUDA context.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from ..circuits import preserving as pr
from ..circuits.cost import cost_terms
from ..instances.instance import load_instance
from ..sectors.select import cell_instances, load_sector
from ..store.io import atomic_write_bytes
from ..store.paths import tables_dir
from . import transpile as tp

CELLS = ({"connectivity": "ring", "rule": "violation", "K": 12, "N": 7},
         {"connectivity": "ring", "rule": "violation", "K": 24, "N": 6, "draws": "k24_eligible"},
         {"connectivity": "complete", "rule": "violation", "K": 12, "N": 6})


def time_cells(root=None, depths=(1, 5, 7, 9), n_inst: int = 3, repeats: int = 10, seed: int = 0,
               builder: bool = True) -> dict:
    from ..circuits import program
    from ..sim import backend

    busy = backend.gpu_compute_pids()
    if busy:
        raise RuntimeError(f"GPU busy (compute PIDs {busy}); one GPU process at a time")
    backend.set_target()
    rng = np.random.default_rng(seed)
    rows = []
    for cell in CELLS:
        ids = [i for i in cell_instances(cell, root) if i.endswith("q1.5")][:n_inst]
        label = f"{cell['connectivity']} · K{cell['K']} · N{cell['N']}"
        for iid in ids:
            inst = load_instance(iid, root)
            sec = load_sector(iid, cell["rule"], cell["K"], root)
            ct = cost_terms(inst.H_obj, inst.boost_obj)
            op = backend.ising_op(inst.H_obj, inst.boost_obj)
            t = time.perf_counter()
            circ = pr.build_circuit(sec, cell["connectivity"], "lex")
            desc_s = time.perf_counter() - t
            counts = tp.circuit_counts(circ, ct)
            for L in depths:
                gates = pr.a1_gates(circ, L, cost=ct)
                base = {"cell": label, "inst_id": iid, "n": circ.n, "K": circ.K, "L": L, "n_gates": len(gates),
                        "cx_ii": tp.a1_totals(counts, L)["cx_ii"], "cx_iii": tp.a1_totals(counts, L)["cx_iii"],
                        "descriptors_s": desc_s, "repeats": repeats}

                def angles():
                    return list(rng.uniform(-1, 1, 2 * L))

                # interpreter kernel
                t = time.perf_counter()
                prog = program.encode(gates, circ.n)
                enc_s = time.perf_counter() - t
                kern = program.kernel()
                a_first = angles()
                t = time.perf_counter()
                e0 = backend.observe(kern, op, *prog.args(a_first))
                first_s = time.perf_counter() - t
                obs = []
                for _ in range(repeats):
                    a = angles()
                    t = time.perf_counter()
                    backend.observe(kern, op, *prog.args(a))
                    obs.append(time.perf_counter() - t)
                st = []
                for _ in range(3):
                    a = angles()
                    t = time.perf_counter()
                    backend.get_state(kern, *prog.args(a))
                    st.append(time.perf_counter() - t)
                rows.append({**base, "engine": "interp", "build_s": enc_s, "first_observe_s": first_s,
                             "observe_median_s": float(np.median(obs)), "observe_min_s": float(np.min(obs)),
                             "get_state_median_s": float(np.median(st)), "energy_first": e0})
                # builder kernel (PLAN §2.4's route)
                if builder:
                    t = time.perf_counter()
                    bk = pr.build_kernel(circ, L, cost=ct)
                    build_s = time.perf_counter() - t
                    t = time.perf_counter()
                    e_b = backend.observe(bk.kernel, op, a_first)
                    first_s = time.perf_counter() - t
                    obs = []
                    for _ in range(repeats):
                        a = angles()
                        t = time.perf_counter()
                        backend.observe(bk.kernel, op, a)
                        obs.append(time.perf_counter() - t)
                    t = time.perf_counter()
                    backend.get_state(bk.kernel, angles())
                    gs = time.perf_counter() - t
                    rows.append({**base, "engine": "builder", "build_s": build_s, "first_observe_s": first_s,
                                 "observe_median_s": float(np.median(obs)), "observe_min_s": float(np.min(obs)),
                                 "get_state_median_s": gs, "energy_first": e_b,
                                 "energy_diff_vs_interp": abs(e_b - e0)})
    info = backend.runtime_info()
    out = {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info, "rows": rows}
    return out


def write(root=None, **kw) -> dict:
    out = time_cells(root, **kw)
    d = tables_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(d / "mixer_timing.json", json.dumps(out, indent=1).encode())
    return out
