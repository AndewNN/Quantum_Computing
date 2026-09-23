"""The post-run step of one stored run (S5): the metrics that need the final STATE, and the 1000-shot sample.

`finalize_run(run_dir)` (GPU; get_state / sample after the run, never in an update loop, PLAN §1.6, §3.3):
  1. rebuild the run's circuit from its run.json (`runconfig_from_record`, the arm's `ansatz`) and its final
     parameters (trajectory params[-1]); replay the final state (get_state) and compare it with final_state.npy
     where stored (n <= 14): `replay_max_abs`;
  2. seed CUDA-Q's sampler with the run's seed from the seed table (the restart seed; A2, which has none, uses its
     draw's restart seed r = 0), draw S = 1000 shots through the backend and write samples.npz
     (idx = classical indices of the observed strings, counts, shots, seed);
  3. write postrun.json: the state metrics (`state_metrics`: AR_F, p_feas, eps_tilde, p_opt, p_top10, p_sector,
     energy recomputed; AR_best_S exact at S = 1000; P(any feasible), P(optimum seen); the simulation difficulty of
     `simdiff`) and the sample digest (`sample_digest`: the best feasible quality among the 1000 shots, the
     sampled feasible fraction, and their consistency with the state: the exact tail probability of the observed
     best and the binomial z of the feasible count).
`Arm.run` calls it at the end of every stored run (finalize=True); `gsp metrics finalize` backfills the runs that
lack it. Existing files are never overwritten unless force=True; run.json / trajectory.npz / counts.json /
final_state.npy are never touched. `state_metrics` alone (CPU) is what `aggregate` falls back to when a run has a
final_state.npy but no postrun.json.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from ..store.io import atomic_write_bytes, load_npz, save_npz
from .quality import (AR_BEST_SHOTS, ar_best_exact, band_positions, best_of_shots, p_any_feasible, p_seen,
                      sample_tail_probability)
from .simdiff import DEFAULT_ENGINE, simdiff

POSTRUN_VERSION = 1
SAMPLES_FILE = "samples.npz"
POSTRUN_FILE = "postrun.json"


def state_metrics(psi: np.ndarray, ctx, K: int | None = None, S: int = AR_BEST_SHOTS,
                  engine: str = DEFAULT_ENGINE, with_simdiff: bool = True) -> dict:
    """Every per-run metric of PLAN §1.7 that is a function of the final state (flat dict)."""
    psi = np.asarray(psi, dtype=np.complex128)
    prob = np.abs(psi) ** 2
    m = ctx.evaluate(prob)
    pb = prob[ctx.band_idx]
    m["ar_best_S"] = ar_best_exact(pb, ctx.ar_weight, S)
    m["ar_best_shots"] = int(S)
    m["p_any_feasible_S"] = p_any_feasible(m["p_feas"], S)
    m["p_opt_seen_S"] = p_seen(m["p_opt"], S)
    m["norm"] = float(prob.sum())
    if with_simdiff:
        m.update({f"sd_{k}": v for k, v in simdiff(psi, ctx.n, K=K, engine=engine).items()})
    return m


def counts_to_arrays(counts: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """backend.sample keys (x_0 first) -> (classical indices ascending, counts)."""
    from ..instances.bits import bitstring_to_index
    items = sorted((bitstring_to_index(k), int(v)) for k, v in counts.items())
    for k in counts:
        if len(k) != n:
            raise ValueError(f"sample key {k!r} is not {n} bits")
    idx = np.array([i for i, _ in items], dtype=np.int64)
    cnt = np.array([c for _, c in items], dtype=np.int64)
    return idx, cnt


def sample_digest(idx, cnt, ctx, prob=None, S_exact: int | None = None) -> dict:
    """The one-sample check of AR_best_S. prob (optional, all 2^n) gives the consistency statistics."""
    idx = np.asarray(idx, dtype=np.int64)
    cnt = np.asarray(cnt, dtype=np.int64)
    shots = int(cnt.sum())
    ok, _ = band_positions(ctx.band_idx, idx)
    k_feas = int(cnt[ok].sum())
    out = {"sample_shots": shots, "ar_best_S_sampled": best_of_shots(idx, ctx.band_idx, ctx.ar_weight),
           "p_feas_sampled": k_feas / shots if shots else float("nan"), "sample_n_distinct": int(idx.size)}
    if ctx.sector_idx is not None:
        oks, _ = band_positions(np.sort(ctx.sector_idx), idx)
        out["p_sector_sampled"] = int(cnt[oks].sum()) / shots if shots else float("nan")
    if prob is not None:
        prob = np.asarray(prob, dtype=np.float64)
        pb = prob[ctx.band_idx]
        pf = float(pb.sum())
        S = shots if S_exact is None else S_exact
        if np.isfinite(out["ar_best_S_sampled"]):
            out["sample_tail_p"] = sample_tail_probability(out["ar_best_S_sampled"], pb, ctx.ar_weight, S)
        else:                                   # no feasible shot: the probability of that event
            out["sample_tail_p"] = 1.0 - p_any_feasible(pf, shots)
        var = shots * pf * (1.0 - pf)
        out["sample_feas_z"] = (k_feas - shots * pf) / math.sqrt(var) if var > 0 else (
            0.0 if k_feas == round(shots * pf) else float("inf"))
    return out


def rebuild(rec: dict, root=None):
    """(cfg, instance, rulers, ansatz, sector_idx, MetricContext) of a stored run."""
    from ..arms.base import make_arm, runconfig_from_record
    from ..instances.adhoc import load_any
    from .state import metric_context
    cfg = runconfig_from_record(rec)
    arm = make_arm(rec["arm"])
    inst, rul, _ = load_any(rec["inst_id"], root)
    A, sector_idx = arm.ansatz(cfg, inst, root)
    ctx = metric_context(inst, rul, cfg.lam if A.kind == "penalty" else None, sector_idx=sector_idx)
    return cfg, inst, rul, A, sector_idx, ctx


def sample_seed(rec: dict, root=None) -> int:
    if rec.get("seed") is not None:
        return int(rec["seed"])
    from ..arms.base import draw_of, restart_seed
    return restart_seed(draw_of(rec["inst_id"]), 0, root)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o).__name__)


def write_json(path: Path, d: dict) -> None:
    atomic_write_bytes(path, json.dumps(d, indent=1, sort_keys=True, default=_json_default,
                                        allow_nan=True).encode())


def read_postrun(run_dir) -> dict | None:
    p = Path(run_dir) / POSTRUN_FILE
    return json.loads(p.read_text()) if p.exists() else None


def read_samples(run_dir) -> dict | None:
    p = Path(run_dir) / SAMPLES_FILE
    return load_npz(p) if p.exists() else None


def finalize_run(run_dir, root=None, shots: int = AR_BEST_SHOTS, final_state=None, force: bool = False,
                 engine: str = DEFAULT_ENGINE, catch: bool = False) -> dict:
    """See the module doc. Returns the postrun dict (existing one if present and not force)."""
    from ..sim import backend
    from ..store.records import read_record
    run_dir = Path(run_dir)
    if not force and (run_dir / POSTRUN_FILE).exists() and (run_dir / SAMPLES_FILE).exists():
        return read_postrun(run_dir)
    t0 = time.perf_counter()
    try:
        rec = read_record(run_dir / "run.json")
        if rec.get("status") != "done":
            raise ValueError(f"run {rec.get('run_id')} is {rec.get('status')}, not done")
        tr = load_npz(run_dir / "trajectory.npz")
        cfg, inst, rul, A, sector_idx, ctx = rebuild(rec, root)
        params = np.asarray(tr["params"][-1], dtype=np.float64)
        psi = A.state(params)
        stored = final_state
        if stored is None and (run_dir / "final_state.npy").exists():
            stored = np.load(run_dir / "final_state.npy")
        replay = float(np.max(np.abs(psi - np.asarray(stored)))) if stored is not None else None
        seed = sample_seed(rec, root)
        backend.set_random_seed(seed)
        t1 = time.perf_counter()
        counts = A.sample(params, shots)
        sample_s = time.perf_counter() - t1
        idx, cnt = counts_to_arrays(counts, A.n)
        info = backend.runtime_info()
        if force or not (run_dir / SAMPLES_FILE).exists():
            save_npz(run_dir / SAMPLES_FILE, {
                "idx": idx, "counts": cnt, "shots": np.int64(shots), "seed": np.int64(seed), "n": np.int64(A.n),
                "run_id": np.array(rec["run_id"]), "cudaq_version": np.array(str(info.get("cudaq_version"))),
                "target": np.array(f"{info.get('target')}:{info.get('target_option')}")})
        K = int(cfg.K) if cfg.K is not None else None
        out = {"postrun_version": POSTRUN_VERSION, "status": "done", "run_id": rec["run_id"], "arm": rec["arm"],
               "replay_max_abs": replay, "sample_seed": seed, "sample_s": sample_s}
        out.update(state_metrics(psi, ctx, K=K, S=AR_BEST_SHOTS, engine=engine))
        out.update(sample_digest(idx, cnt, ctx, prob=np.abs(psi) ** 2))
        out["postrun_s"] = time.perf_counter() - t0
    except Exception as exc:
        if not catch:
            raise
        out = {"postrun_version": POSTRUN_VERSION, "status": "failed", "error": "".join(traceback.format_exception(exc))}
    write_json(run_dir / POSTRUN_FILE, out)
    return out


def finalize_all(root=None, arms=None, limit: int | None = None, force: bool = False, log=None) -> dict:
    """Backfill finalize_run over every done run of the registry lacking postrun.json / samples.npz."""
    from ..store.index import load_registry
    from ..store.paths import runs_dir
    reg = load_registry(root, rebuild=True)
    done = reg[reg["status"] == "done"] if not reg.empty else reg
    if arms:
        done = done[done["arm"].isin(list(arms))]
    base = runs_dir(root)
    n_new = n_skip = n_fail = 0
    for _, r in done.iterrows():
        d = base / r["run_dir"]
        if not force and (d / POSTRUN_FILE).exists() and (d / SAMPLES_FILE).exists():
            n_skip += 1
            continue
        if limit is not None and n_new + n_fail >= limit:
            break
        out = finalize_run(d, root=root, force=force, catch=True)
        if out.get("status") == "done":
            n_new += 1
        else:
            n_fail += 1
        if log:
            log(f"{r['arm']} {r['run_id']} {out.get('status')} replay={out.get('replay_max_abs')} "
                f"ar_best={out.get('ar_best_S')} sampled={out.get('ar_best_S_sampled')}")
    return {"finalized": n_new, "skipped": n_skip, "failed": n_fail}
