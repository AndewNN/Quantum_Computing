"""Validation of the numpy GA against the completed work's pybind11 `ga_solver` (D-5; PLAN §5 S2).

The external module is **not** a dependency: this file imports it from a directory given at run
time (`--module-dir`) and is used only by `gsp sectors pybind` and a test that skips when the module
does not import. Two builds exist:
  prebuilt  `MyLib/Genetic/ga_solver.cpython-311-x86_64-linux-gnu.so` (Feb 2026): budget (violation)
            mode only, seeded from std::random_device (not reproducible);
  source    `MyLib/Genetic/genetic_solver.cpp` at commit 8634ca8 ("Obj focus", 2026-09-24) adds the
            objective-aware mode (`--GA_OBJ`) and a `seed` argument. S2 compiled it into the session
            scratchpad (never installed anywhere) to validate the objective rule.
Only one build can be loaded per process (both register the pybind11 type `ga_solver.*`), so each
validation runs in its own process.

Chromosome -> harness string: the old call site (`CUDA/PO_new_ApproxRatio.py:516-526`) reversed
each asset block; `chromosome_perm` is the same map (tested against a verbatim copy).
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..instances.instance import load_instance
from .ga import TABLE_4_1, GAParams, brute_force, chromosome_perm, problem_from_instance, run_ga


def import_ga_solver(module_dir):
    """Import `ga_solver` from `module_dir` only (None if it does not import)."""
    module_dir = str(Path(module_dir).resolve())
    if "ga_solver" in sys.modules:
        mod = sys.modules["ga_solver"]
        if str(Path(mod.__file__).resolve().parent) != module_dir:
            raise RuntimeError("another ga_solver build is already loaded in this process")
        return mod
    sys.path.insert(0, module_dir)
    try:
        return importlib.import_module("ga_solver")
    except Exception:
        return None
    finally:
        sys.path.remove(module_dir)


def capabilities(mod) -> dict:
    doc = mod.GeneticAlgorithm.__init__.__doc__ or ""
    return {"objective": "returns" in doc, "seed": "seed" in doc}


def chromosome_to_index(chrom, n_max) -> int:
    perm = chromosome_perm(n_max)
    x = np.empty(len(chrom), dtype=np.int64)
    x[perm] = np.asarray(chrom, dtype=np.int64)
    return int(x @ (np.int64(1) << np.arange(x.size - 1, -1, -1, dtype=np.int64)))


def cpp_run(mod, inst, rule: str, seed: int | None, params: GAParams = TABLE_4_1, keep: int = 24):
    """One C++ GA run with the old call site's arguments; returns (top indices best first, wall_s)."""
    n = int(inst.n)
    n_max = [int(v) for v in np.asarray(inst.arrays["n_max"]).ravel()]
    kw = dict(prices=[float(v) for v in inst.P], asset_bit_lengths=n_max, budget=float(inst.B),
              population_size=params.population, mutation_rate=params.mutation_rate(n),
              crossover_rate=params.crossover_rate, elitism_count=params.elitism,
              tournament_size=params.tournament)
    cap = capabilities(mod)
    if rule == "objective":
        if not cap["objective"]:
            raise RuntimeError("this ga_solver build has no objective-aware mode")
        kw.update(returns=[float(v) for v in inst.ret], covariance=np.asarray(inst.cov, float).tolist(),
                  q=float(inst.q), band=float(inst.eps))
    if cap["seed"] and seed is not None:
        kw["seed"] = int(seed)
    t0 = time.perf_counter()
    ga = mod.GeneticAlgorithm(**kw)
    ga.run(params.generations, verbose=False)
    top = ga.get_top_n_individuals(keep, False)
    wall = time.perf_counter() - t0
    return np.array([chromosome_to_index(t.chromosome, n_max) for t in top], np.int64), wall


def validate(module_dir, label: str, rules=("violation", "objective"), N_values=(4, 5, 6, 7),
             Ks=(6, 8, 12, 24), root=None, log=print) -> pd.DataFrame:
    """Every §1.2 GA job at N_values: C++ top-K vs brute force and vs the numpy GA (same seed)."""
    from .select import plan_jobs

    mod = import_ga_solver(module_dir)
    if mod is None:
        if log:
            log(f"{label}: ga_solver does not import from {module_dir}; skipped")
        return pd.DataFrame()
    cap = capabilities(mod)
    rows = []
    for job in plan_jobs(root, extension=False, N_values=N_values):
        if job.rule not in rules or (job.rule == "objective" and not cap["objective"]):
            continue
        inst = load_instance(job.inst_ids[0], root)
        pr = problem_from_instance(inst, job.rule)
        bf = brute_force(pr, keep=max(Ks))["rank_idx"]
        npy = run_ga(pr, job.seed, keep=max(Ks), track=False).rank_idx
        cpp, wall = cpp_run(mod, inst, job.rule, job.seed, keep=max(Ks))
        for K in Ks:
            if K > job.F_size:
                continue
            b, c, g = set(bf[:K].tolist()), set(cpp[:K].tolist()), set(npy[:K].tolist())
            rows.append({"module": label, "rule": job.rule, "scope_id": job.scope_id, "N": job.N,
                         "n": job.n, "K": K, "in_file": K in job.file_Ks, "seeded": cap["seed"],
                         "cpp_eq_bf": c == b, "cpp_missed": len(b - c), "numpy_eq_bf": g == b,
                         "numpy_missed": len(b - g), "cpp_eq_numpy": c == g, "cpp_wall_s": wall})
    df = pd.DataFrame(rows)
    if log and not df.empty:
        log(df.groupby(["rule", "N", "K"])[["cpp_eq_bf", "numpy_eq_bf", "cpp_eq_numpy"]].mean().to_string())
    return df


def seed_study(module_dir, draw_inst_ids, rule: str, K: int, n_seeds: int = 200, root=None,
               params: GAParams = TABLE_4_1) -> pd.DataFrame:
    """P(top-K == BF) of the C++ and numpy GAs over many seeds on the same instances (distributional
    equivalence of the two implementations)."""
    mod = import_ga_solver(module_dir)
    if mod is None:
        return pd.DataFrame()
    rows = []
    for iid in draw_inst_ids:
        inst = load_instance(iid, root)
        pr = problem_from_instance(inst, rule)
        b = set(brute_force(pr, keep=K)["rank_idx"].tolist())
        for s in range(n_seeds):
            cpp, _ = cpp_run(mod, inst, rule, 10_000 + s, params, keep=K)
            npy = run_ga(pr, 10_000 + s, params, keep=K, track=False).rank_idx
            rows.append({"inst_id": iid, "rule": rule, "K": K, "seed": s,
                         "cpp_eq_bf": set(cpp.tolist()) == b, "numpy_eq_bf": set(npy.tolist()) == b,
                         "cpp_missed": len(b - set(cpp.tolist())), "numpy_missed": len(b - set(npy.tolist()))})
    return pd.DataFrame(rows)
