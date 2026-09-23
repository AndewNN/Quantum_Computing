"""The objective-aware GA as a standalone classical solver (WP5; PLAN §5 S2 and S13).

The same numpy GA as the sector selection (`gsp.sectors.ga`, objective rule, Table 4.1), read as a
solver: its answer is the best string of the final population. For S13's break-even curves it
reports the answer's quality against the frozen rulers and the cost to reach it in objective
evaluations:
  evals_to_best    evaluation index (1-based, population order) at which the answer was first evaluated
  evals_to_xstar   the same for the band optimum X* (0 = never evaluated)
  n_evals          the run's total budget, (G + 1) * N_pop objective evaluations
The S2 sector build records these fields for every objective-rule run (`ga_runs.parquet`,
columns `solver_*`); `solve` runs one GA directly (S13 can pass another budget via `params`).
"""

from __future__ import annotations

import numpy as np

from ..instances.rulers import objective_on
from ..sectors.ga import TABLE_4_1, GAParams, GAResult, problem_from_instance, run_ga


def solver_record(ga: GAResult, inst, rul) -> dict:
    """Solver fields of an objective-rule GA run on instance `inst` with rulers `rul`."""
    if ga.rule != "objective":
        raise ValueError("the standalone solver is the objective-aware GA")
    band = np.zeros(1 << int(inst.n), bool)
    band[rul.band_idx] = True
    best = int(ga.rank_idx[0])
    best_in_band = bool(band[best])
    xstar = [int(x) for x in rul.xstar_idx]
    f_best = float(objective_on(inst.QU_obj, np.array([best]))[0])
    e2x = [ga.evals_to_hit(x) for x in xstar]
    return {
        "solver_best_idx": best, "solver_best_in_band": best_in_band, "solver_best_f": f_best,
        "solver_ar_best": (rul.E_max - f_best) / (rul.E_max - rul.E_min) if best_in_band else None,
        "solver_hit_xstar": best in xstar, "solver_evals_to_best": ga.evals_to_hit(best),
        "solver_evals_to_xstar": min([v for v in e2x if v > 0], default=0),
        "solver_gen_to_best": int(np.argmax(ga.trace_best_idx == best)),
    }


def solve(inst_id: str, seed: int | None = None, params: GAParams = TABLE_4_1, root=None) -> dict:
    """Run the objective-aware GA on one frozen instance. `seed=None` takes `ga_seed_objective`
    from the seed table (the seed the S2 sector build used, so the result reproduces it)."""
    from ..instances.instance import load_instance, load_rulers, load_seed_table

    inst = load_instance(inst_id, root)
    rul = load_rulers(inst_id, root)
    if seed is None:
        st = load_seed_table(root)
        seed = int(st.loc[st["draw_id"] == inst.draw_id, "ga_seed_objective"].iloc[0])
    ga = run_ga(problem_from_instance(inst, "objective"), seed, params, keep=1, track=True)
    return {"inst_id": inst_id, "seed": int(seed), "n_evals": ga.n_evals,
            "n_objective_evals": ga.n_objective_evals, "n_unique": ga.n_unique,
            "wall_s": ga.wall_s, **solver_record(ga, inst, rul)}
