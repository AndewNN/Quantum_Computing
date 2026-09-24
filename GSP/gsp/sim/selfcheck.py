"""`gsp selfcheck --record | --compare`: the CUDA-Q migration's acceptance test (PLAN §3.5, built in S8).

`record()` runs a fixed set of circuits and short trajectories on the current CUDA-Q and writes
tests/fixtures/selfcheck_cudaq-<version>.json; `compare()` runs the same set again (with the fixture's own inputs:
parameters, step sizes, initial points) and compares. Tolerances (PLAN §3.5, step 3): states and energies to 1e-12,
trajectories to 1e-9 (relative to the largest |value| of the series).

The set:
  circuits      on 3 fixed frozen instances (N04e004q1.5, N05e000q1.5, N06e000q1.5), cell ring / violation / K12,
                lam 0.005 for the penalty arms: the observe energy and the final state (get_state, classical order,
                stored as base64 little-endian complex128) of
                  A0   A0's circuit at depth 2, Eq. 4.11 random parameters (the draw's restart seed r = 0);
                  A1   A1's circuit at depth 2, same rule;
                  A2p  the ramp at p = 5, primary schedule (A0's circuit);
                  A2c  the ramp at p = 5, primary schedule (A1's circuit);
                  A3   A0's circuit at depth 2 at A3's ramp init (varqite_routes.py);
                  A4   the DB-QITE circuit U_2 (star prep, H_obj) at s = (0.2, 0.5) / sigma_H;
                  A6   the DB-QITE circuit U_2 (H^n, H(lam)) at s = (0.2, 0.5) / sigma_H;
  trajectories  A1_adamw   30 AdamW iterations of A1 at N04e004q1.5, depth 5 (f per iteration, final parameters);
                A3_example A3's production estimators (M1 forward stencil + C1) on example_varqite.py's 3-qubit instance
                           at L = 1, 401 steps of dtau 0.01 (S7: well conditioned, GPU vs numpy 3e-12; the N = 7 M1 loop
                           is chaotic, O-11, so it is not used here);
                A3d        10 steps of A3d (M4C1) at N07e000q1.5, depth 5, lam 0.005 (S7: 3e-9 against the stored 0.13
                           run);
                A4 / A6    5 greedy DB-QITE steps at N04e004q1.5 (chosen grid index and loop energy per step).
The fixture is never overwritten by `record` unless force=True: it is the reference the next CUDA-Q is judged by.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np

SCHEMA = 1
INSTANCES = ("N04e004q1.5", "N05e000q1.5", "N06e000q1.5")
CELL = {"connectivity": "ring", "rule": "violation", "K": 12}
LAM = 0.005
DEPTH = 2
RAMP_P = 5
DB_G = (0.2, 0.5)                      # grid values (units of 1 / sigma_H) of the two fixed DB-QITE steps
TRAJ_INST = "N04e004q1.5"
A3D_INST = "N07e000q1.5"
TOL_STATE = 1e-12
TOL_ENERGY = 1e-12
TOL_TRAJ = 1e-9
CIRCUIT_ARMS = ("A0", "A1", "A2p", "A2c", "A3", "A4", "A6")


def fixture_path(version: str | None = None) -> Path:
    """tests/fixtures/selfcheck_cudaq-<version>.json (default: the running CUDA-Q's version)."""
    from ..store.paths import GSP_ROOT
    return GSP_ROOT / "tests" / "fixtures" / f"selfcheck_cudaq-{cudaq_version() if version is None else version}.json"


def cudaq_version() -> str:
    import cudaq
    v = str(cudaq.__version__)
    for tok in v.replace("(", " ").split():
        if tok[:1].isdigit():
            return tok.split("+")[0]
    return v


def _b64(psi: np.ndarray) -> str:
    return base64.b64encode(np.asarray(psi, dtype="<c16").tobytes()).decode()


def _unb64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<c16").copy()


# --- the circuits ------------------------------------------------------------------------------------------------
def _circuit(arm: str, inst, root=None):
    """(object with energy / state, params, energy scale) of one fixed circuit (module doc)."""
    from ..arms.base import draw_of, make_arm, restart_seed
    from ..arms.dbqite import confined_db_circuit, penalty_db_circuit
    from ..train.init import init_params
    from ..train.mclachlan import ramp_init
    from ..train.schedules import ramp_params, schedule
    seed = restart_seed(draw_of(inst.inst_id), 0, root)
    if arm in ("A0", "A3"):
        A, _ = make_arm("A0").ansatz(make_arm("A0").config(inst, None, DEPTH, None, lam=LAM, root=root), inst, root)
        x = init_params(A, seed) if arm == "A0" else ramp_init(DEPTH)
        return A, np.asarray(x, dtype=np.float64), A.alpha
    if arm == "A1":
        a = make_arm("A1")
        A, _ = a.ansatz(a.config(inst, CELL, DEPTH, None, root=root), inst, root)
        return A, np.asarray(init_params(A, seed), dtype=np.float64), A.alpha
    if arm in ("A2p", "A2c"):
        a = make_arm(arm)
        kw = {"lam": LAM} if arm == "A2p" else {}
        cfg = a.config(inst, None if arm == "A2p" else CELL, RAMP_P, None, schedule_tag="primary", root=root, **kw)
        A, _ = a.ansatz(cfg, inst, root)
        db_, dg = schedule("primary")
        return A, np.asarray(ramp_params(RAMP_P, db_, dg, A.alpha, cfg.extra("ramp_sign")), dtype=np.float64), A.alpha
    if arm == "A4":
        C, _ = confined_db_circuit(inst, CELL["rule"], CELL["K"], root=root)
        return C, np.asarray(DB_G, dtype=np.float64) / C.sigma, 1.0
    if arm == "A6":
        C = penalty_db_circuit(inst, LAM)
        return C, np.asarray(DB_G, dtype=np.float64) / C.sigma, 1.0
    raise ValueError(arm)


def _run_circuit(obj, params) -> tuple:
    p = list(params) if not isinstance(params, np.ndarray) else params.tolist()
    return float(obj.energy(p)), np.asarray(obj.state(p), dtype=np.complex128)


def circuit_cases(root=None, log=None) -> list[dict]:
    from ..instances.instance import load_instance
    out = []
    for iid in INSTANCES:
        inst = load_instance(iid, root)
        for arm in CIRCUIT_ARMS:
            obj, params, alpha = _circuit(arm, inst, root)
            t = time.perf_counter()
            e, psi = _run_circuit(obj, params)
            out.append({"id": f"{arm}/{iid}", "arm": arm, "inst_id": iid, "n": int(inst.n), "params": params.tolist(),
                        "energy": e, "energy_note": "observe of alpha H (A0-A3) / un-boosted H (A4, A6)",
                        "state_b64": _b64(psi), "state_dim": int(psi.size), "wall_s": time.perf_counter() - t})
            if log:
                log(f"circuit {arm}/{iid}: E = {e:.15g}")
    return out


def _recompute_circuit(case: dict, root=None) -> tuple:
    from ..instances.instance import load_instance
    inst = load_instance(case["inst_id"], root)
    obj, _, _ = _circuit(case["arm"], inst, root)
    return _run_circuit(obj, np.asarray(case["params"], dtype=np.float64))


# --- the trajectories -----------------------------------------------------------------------------------------------
def example_varqite_ising():
    """example_varqite.py's instance (Lecture_Notes/code, frozen in tests/legacy): n = 3, J and h from
    default_rng(11), H_C = sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i (qubit 0 = the leftmost kron factor = x_0)."""
    from ..instances.encode import Ising
    rng = np.random.default_rng(11)
    J = rng.normal(size=(3, 3))
    h = rng.normal(size=3)
    return Ising(n=3, const=0.0, h=np.array(h, dtype=float), J=np.triu(J, 1), has_h=np.ones(3, bool),
                 has_J=np.triu(np.ones((3, 3), bool), 1))


def _traj_a1(x0=None, root=None) -> dict:
    from ..arms.base import draw_of, make_arm, restart_seed
    from ..instances.instance import load_instance
    from ..train.adamw import AdamWConfig, train_adamw
    from ..train.gradients import ForwardFD
    from ..train.init import init_params
    inst = load_instance(TRAJ_INST, root)
    a = make_arm("A1")
    A, _ = a.ansatz(a.config(inst, CELL, 5, None, root=root), inst, root)
    if x0 is None:
        x0 = init_params(A, restart_seed(draw_of(TRAJ_INST), 0, root))
    res = train_adamw(A.energy, np.asarray(x0, dtype=np.float64), AdamWConfig(max_iter=30), ForwardFD(1e-4))
    return {"x0": np.asarray(x0).tolist(), "series": {"f": res.f_hist.tolist()},
            "final_params": res.params.tolist(), "n_iter": int(res.n_iter)}


def _traj_a3_example(x0=None) -> dict:
    from ..arms.varqite import CircuitEngine
    from ..circuits.ansatz import penalty_ansatz
    from ..train.mclachlan import McLachlanConfig, run_mclachlan
    A = penalty_ansatz(example_varqite_ising(), 1, alpha=1.0)
    if x0 is None:
        th = 0.05 * np.random.default_rng(1).normal(size=2)     # example_varqite.py's theta_0 at L = 1
        x0 = np.r_[th[0::2], th[1::2]]
    cfg = McLachlanConfig(metric="M1", dtau=0.01, n_steps=401, tikhonov=1e-6, f_tol=-1.0)
    res = run_mclachlan(CircuitEngine(A), np.asarray(x0, dtype=np.float64), cfg)
    return {"x0": np.asarray(x0).tolist(), "series": {"E": res.E_loop.tolist()}, "final_params": res.params.tolist(),
            "n_iter": int(res.n_steps)}


def _traj_a3d(x0=None, root=None) -> dict:
    from ..arms.qaoa import penalty_arm_ansatz
    from ..arms.varqite import run_a3
    from ..instances.instance import load_instance
    from ..train.mclachlan import McLachlanConfig, ramp_init
    inst = load_instance(A3D_INST, root)
    A = penalty_arm_ansatz(inst, LAM, 5)
    x0 = ramp_init(5) if x0 is None else np.asarray(x0, dtype=np.float64)
    res, _, _, _ = run_a3(A, McLachlanConfig(metric="diag", n_steps=10), x0=x0, logger=False)
    return {"x0": np.asarray(x0).tolist(), "series": {"E": res.E_loop.tolist()}, "final_params": res.params.tolist(),
            "n_iter": int(res.n_steps)}


def _traj_db(arm: str, root=None) -> dict:
    from ..arms.dbqite import confined_db_circuit, penalty_db_circuit, run_dbqite
    from ..circuits.dbqite import GRID
    from ..instances.instance import load_instance
    inst = load_instance(TRAJ_INST, root)
    C = (confined_db_circuit(inst, CELL["rule"], CELL["K"], root=root)[0] if arm == "A4"
         else penalty_db_circuit(inst, LAM))
    res = run_dbqite(C, np.asarray(GRID) / C.sigma, 5)
    return {"series": {"E": res.E_loop.tolist(), "grid_idx": res.grid_idx.astype(float).tolist()},
            "final_params": res.s.tolist(), "n_iter": int(res.n_steps), "sigma_H": C.sigma}


TRAJECTORIES = ("A1_adamw", "A3_example", "A3d", "A4_db", "A6_db")


def _trajectory(name: str, x0=None, root=None) -> dict:
    if name == "A1_adamw":
        return _traj_a1(x0, root)
    if name == "A3_example":
        return _traj_a3_example(x0)
    if name == "A3d":
        return _traj_a3d(x0, root)
    if name in ("A4_db", "A6_db"):
        return _traj_db(name[:2], root)
    raise ValueError(name)


# --- record / compare ------------------------------------------------------------------------------------------------
def _header() -> dict:
    from .._version import HARNESS_VERSION
    from . import backend
    backend.ensure_target()
    info = backend.runtime_info()
    try:
        import subprocess
        from ..store.paths import GSP_ROOT
        git = subprocess.run(["git", "-C", str(GSP_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
    except Exception:
        git = None
    return {"schema": SCHEMA, "cudaq_version": cudaq_version(), "cudaq_version_full": info.get("cudaq_version"),
            "target": info.get("target"), "target_option": info.get("target_option"),
            "fusion_max_qubits": info.get("fusion_max_qubits"), "gpu_name": info.get("gpu_name"),
            "driver_version": info.get("driver_version"), "harness_version": HARNESS_VERSION, "git": git,
            "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "tolerances": {"state": TOL_STATE, "energy": TOL_ENERGY, "trajectory": TOL_TRAJ},
            "instances": list(INSTANCES), "cell": CELL, "lam": LAM, "depth": DEPTH, "ramp_p": RAMP_P, "db_g": list(DB_G)}


def record(path=None, force: bool = False, root=None, log=None) -> Path:
    path = fixture_path(cudaq_version()) if path is None else Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists: the self-check fixture is a migration reference (force=True replaces it)")
    out = _header()
    t0 = time.perf_counter()
    out["circuits"] = circuit_cases(root, log)
    out["trajectories"] = {}
    for name in TRAJECTORIES:
        t = time.perf_counter()
        out["trajectories"][name] = _trajectory(name, root=root) | {"wall_s": time.perf_counter() - t}
        if log:
            log(f"trajectory {name}: {out['trajectories'][name]['n_iter']} units")
    out["wall_s"] = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return path


def _series_err(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    scale = max(float(np.max(np.abs(b))) if b.size else 0.0, 1e-300)
    return float(np.max(np.abs(a - b)) / scale) if a.size else 0.0


def compare(path=None, root=None, log=None, only_circuits: bool = False, arms=None, instances=None) -> dict:
    """Recompute the fixture's set; per case the error and pass / fail. `arms` / `instances` restrict the circuits
    (tests); only_circuits skips the trajectories."""
    path = fixture_path("0.15.1") if path is None else Path(path)
    fx = json.loads(Path(path).read_text())
    from . import backend
    backend.ensure_target()
    rows = []
    for case in fx["circuits"]:
        if (arms and case["arm"] not in arms) or (instances and case["inst_id"] not in instances):
            continue
        e, psi = _recompute_circuit(case, root)
        ref = _unb64(case["state_b64"])
        de = abs(e - case["energy"])
        ds = float(np.max(np.abs(psi - ref))) if psi.shape == ref.shape else float("inf")
        ok = de <= TOL_ENERGY * max(1.0, abs(case["energy"])) and ds <= TOL_STATE
        rows.append({"id": case["id"], "kind": "circuit", "energy_err": de, "state_err": ds, "pass": bool(ok)})
        if log:
            log(f"{case['id']}: |dE| {de:.2e}, max|dpsi| {ds:.2e} {'ok' if ok else 'FAIL'}")
    if not only_circuits:
        for name, ref in fx["trajectories"].items():
            x0 = ref.get("x0")
            got = _trajectory(name, x0=x0, root=root)
            errs = {k: _series_err(got["series"].get(k, []), v) for k, v in ref["series"].items()}
            errs["final_params"] = _series_err(got["final_params"], ref["final_params"])
            worst = max(errs.values()) if errs else 0.0
            ok = worst <= TOL_TRAJ and got["n_iter"] == ref["n_iter"]
            rows.append({"id": name, "kind": "trajectory", "errors": errs, "worst": worst, "n_iter": got["n_iter"],
                         "n_iter_ref": ref["n_iter"], "pass": bool(ok)})
            if log:
                log(f"{name}: worst relative error {worst:.2e} ({got['n_iter']} vs {ref['n_iter']} units) "
                    f"{'ok' if ok else 'FAIL'}")
    return {"fixture": str(path), "fixture_version": fx.get("cudaq_version"), "running_version": cudaq_version(),
            "rows": rows, "n_cases": len(rows), "n_fail": sum(not r["pass"] for r in rows),
            "passes": all(r["pass"] for r in rows)}
