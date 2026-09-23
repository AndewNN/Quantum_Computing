"""Resource metrics of one run (PLAN §1.7) from its counts.json (S4 layout: per_circuit / per_unit / layer / start
counts of the ABSTRACT circuit: cx_ii, cx_iii, t_ii, t_iii, tdepth_ii, tdepth_iii; circuits_per_unit; effort_unit)
and the convergence effort:
  per unit       two-qubit gates (ii) / (iii), T-count and T-depth (ii) / (iii) per effort unit;
  per circuit    the same per charged circuit (start circuit included);
  to convergence executions (circuits) to convergence, and their product with the per-circuit counts: total
                 two-qubit gate executions / T to convergence ((ii) and (iii)); the (ii) product equals the
                 trajectory's g2q_ii at t_conv (checked: `chk_conv_g2q`).
  totals         the run's final cumulative circuits and g2q_ii / g2q_iii.
The restart factor (R x for best-of-R, 1 for median-of-R) is applied by the reader (Rule D1 does it on the
budget axis).
"""

from __future__ import annotations

KEYS = ("cx_ii", "cx_iii", "t_ii", "t_iii", "tdepth_ii", "tdepth_iii")


def resources(counts: dict, conv: dict | None = None, traj: dict | None = None) -> dict:
    out = {"effort_unit": counts.get("effort_unit"), "circuits_per_unit": counts.get("circuits_per_unit")}
    for grp in ("per_unit", "per_circuit", "layer", "start"):
        for k in KEYS:
            v = counts.get(grp, {}).get(k)
            out[f"{grp}_{k}"] = None if v is None else int(v)
    if "mixer" in counts:
        for k in ("cx_ii", "cx_iii", "S_mean", "S_max", "d_mean"):
            out[f"mixer_{k}"] = counts["mixer"].get(k)
    if "cx_ii_S_per_circuit" in counts:
        out["per_circuit_cx_ii_S"] = int(counts["cx_ii_S_per_circuit"])
    if conv is not None and conv.get("conv_circuits") is not None:
        c = int(conv["conv_circuits"])
        pc = counts.get("per_circuit", {})
        for k in KEYS:
            if k in pc:
                out[f"conv_exec_{k}"] = c * int(pc[k])
        out["chk_conv_g2q"] = (conv.get("conv_g2q_ii") == out.get("conv_exec_cx_ii"))
    if traj is not None and len(traj.get("t", [])):
        out["total_circuits"] = int(traj["circuits_charged"][-1])
        out["total_g2q_ii"] = int(traj["g2q_ii"][-1])
        out["total_g2q_iii"] = int(traj["g2q_iii"][-1])
    return out
