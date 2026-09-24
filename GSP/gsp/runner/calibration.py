"""S9a (PLAN §5 S9): the WP2 calibration queues and the proposed gate-matched A0 depths.

  matched depths  L0(N, L1) = ceil(median over the N's accepted draws of CNOT_ii(A1 baseline circuit at depth L1,
                  start circuit included) / CNOT(one A0 layer)), PLAN §1.3. Pure counting with the S3 (ii) counters
                  (`compile.transpile`) on the production GA sectors, ring_order lex (O-1 default), Eq. 4.10 (ii)
                  (O-5 default). One A0 layer = 2 x (ZZ terms of H(lam)); it is checked lam-independent over the
                  pilot's five lam. Proposed here (configs/matched_depths.yaml, reports/matched_depths.md); S9b
                  finalizes them in the envelope.
  queue 1         the timing set (`timing_specs`): one instance per cell, the first accepted draw at q = 1.5 (the
                  K = 24 cells: the first k24-eligible draw), lam = 0.005 where a lam is needed (a placeholder: wall
                  clock does not depend on lam). Every timing run is a normal config (no tag).
  queue 2         the lambda pilot, `gsp plan --pilot` (PLAN §1.4; `plan.pilot_specs`).
  queue 3         evidence for Sensei's open decisions (`evidence_specs`), each run tagged with the hashed extra
                  `evidence` ("O-2" / "O-11"), so no rule reads it (`stats.d1._match`). A run whose config is exactly
                  a sweep config (the un-boosted A1 counterparts of O-2, O-11's default variant without jitter) is
                  queued untagged: it is the same computation the sweep would run, and it is reused by run_id.
                    O-2   A0 at N in {5, 7}: the first 10 pilot draws, q 1.5, lam in {0.005, 0.05}, depth 7, r 0,
                          circuit_boosted (the un-boosted counterparts are pilot runs, queue 2);
                          A1 ring . violation . K12 at N in {5, 7}: the same 10 draws, L 5, r 0, both conventions.
                    O-11  A3 at N 7, L 5, lam 0.005, q 1.5, the first 5 accepted draws, 300 steps (normal stop rule);
                          variants (a) default, (b) psd_project, (c) tikhonov 1e-4; each at theta_0 = the Ramp init and
                          with two 1e-9 rad jitters (theta0_jitter_seed 0 and 1: S7's ensemble members 0 and 1).
  estimates       seconds per run from the rates measured in S4 / S5 / S7 / S8 (STATUS; `RATES`), completed by the
                  observe micro-benchmark of the S9a smoke where a rate was not measured (`estimate_specs`).

Queue files are JSONL (plan.QUEUE_KEYS + `label` + `est_s`), with a `.plan.json` sidecar, as `gsp plan` writes them;
`scripts/queue.sh` runs them unchanged.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .._version import HARNESS_VERSION
from ..store.io import atomic_write_bytes
from . import plan as P

TIMING_LAM = 0.005                   # the S4 placeholder (PLAN §5 S9: wall clock does not depend on lam)
PILOT_LAMS = (0.0005, 0.005, 0.05, 0.5, 5.0)
MATCHED_N = (4, 5, 6, 7)
MATCHED_L1 = (5, 7, 9)
BASELINE = {"connectivity": "ring", "rule": "violation", "K": 12}
QUEUE_NAMES = {1: "s9_q1_timing", 2: "s9_q2_pilot", 3: "s9_q3_evidence"}
EXTRA_KEYS = ("label", "est_s")


# --- A. gate-matched depths ------------------------------------------------------------------------------------------
def matched_depth_rows(root=None, ring_order: str = "lex", Ns=MATCHED_N, L1s=MATCHED_L1, lams=PILOT_LAMS
                       ) -> pd.DataFrame:
    """One row per accepted instance (N in Ns, all three q): the A1 baseline circuit's (ii) CNOTs at every L1 (start
    circuit included), its parts, the informational counts (cx_ii_S = O-5's restricted count; the rank ring order,
    O-1), and one A0 layer's CNOTs for every pilot lam."""
    from ..circuits import preserving as pr
    from ..circuits.cost import cost_terms
    from ..compile import transpile as tp
    from ..instances.instance import load_instance, load_instances_table
    from ..sectors.select import load_sector, sector_scope

    it = load_instances_table(root).sort_values(["N", "accept_rank", "q"], kind="stable")
    other = "rank" if ring_order == "lex" else "lex"
    circ_cache: dict = {}
    rows = []
    for N in Ns:
        for r in it[it["N"] == int(N)].itertuples(index=False):
            sec = load_sector(r.inst_id, BASELINE["rule"], BASELINE["K"], root)
            inst = load_instance(r.inst_id, root)
            ct = cost_terms(inst.H_obj, 1.0)
            cc = {}
            for ro in (ring_order, other):
                key = (sector_scope(r.inst_id, BASELINE["rule"]), ro)
                if key not in circ_cache:
                    circ_cache[key] = pr.build_circuit(sec, BASELINE["connectivity"], ro)
                cc[ro] = tp.circuit_counts(circ_cache[key], ct)
            a0 = {float(lam): int(tp.a0_layer_counts(cost_terms(inst.hamiltonian(float(lam)), 1.0))["cx"])
                  for lam in lams}
            row = {"N": int(N), "n": int(inst.n), "draw_id": r.draw_id, "inst_id": r.inst_id, "q": float(r.q),
                   "accept_rank": int(r.accept_rank), "ring_order": ring_order,
                   "prep_cx_ii": int(cc[ring_order]["prep"]["cx_ii"]),
                   "layer_cx_ii": int(cc[ring_order]["layer"]["cx_ii"]),
                   "mixer_cx_ii": int(cc[ring_order]["mixer"]["cx_ii"]),
                   "cost_cx_ii": int(cc[ring_order]["cost"]["cx_ii"]),
                   "n_zz_obj": int(cc[ring_order]["cost"]["n_zz"]),
                   "a0_layer_cx": a0[float(lams[0])], "a0_layer_cx_distinct": len(set(a0.values())),
                   "a0_layer_cx_min": min(a0.values()), "a0_layer_cx_max": max(a0.values())}
            for L1 in L1s:
                row[f"a1_L{L1}_cx_ii"] = int(tp.a1_totals(cc[ring_order], L1)["cx_ii"])
                row[f"a1_L{L1}_cx_ii_S"] = int(tp.a1_totals(cc[ring_order], L1)["cx_ii_S"])
                row[f"a1_L{L1}_cx_ii_{other}"] = int(tp.a1_totals(cc[other], L1)["cx_ii"])
            rows.append(row)
    return pd.DataFrame(rows)


def _ceil_ratio(x: float, a0: int) -> int:
    return int(math.ceil(float(x) / float(a0) - 1e-12))


def matched_depths(rows: pd.DataFrame, L1s=MATCHED_L1) -> dict:
    """{N: {L1: detail}} with detail L0 (PLAN §1.3), the median over draws of the A1 count, the A0 layer, the match
    error of L0, the per-draw range of ceil(count / layer), and the informational L0 under cx_ii_S (O-5) and the
    other ring order (O-1). Per-draw value = the draw's count; the three q of a draw must agree (checked)."""
    other = [c for c in rows.columns if c.startswith(f"a1_L{L1s[0]}_cx_ii_") and c != f"a1_L{L1s[0]}_cx_ii_S"]
    other_ro = other[0].rsplit("_", 1)[1] if other else None
    out: dict = {}
    for N, g in rows.groupby("N"):
        a0 = g["a0_layer_cx"].unique()
        if len(a0) != 1 or int(g["a0_layer_cx_distinct"].max()) != 1:
            raise ValueError(f"N={N}: the A0 layer count is not unique over instances / lam: {sorted(a0)}")
        a0 = int(a0[0])
        n = int(g["n"].iloc[0])
        det = {}
        for L1 in L1s:
            col = f"a1_L{L1}_cx_ii"
            per_q = g.groupby("draw_id")[col].nunique()
            if int(per_q.max()) != 1:
                raise ValueError(f"N={N} L1={L1}: the A1 count differs between the q of a draw")
            per_draw = g.groupby("draw_id")[col].first()
            med = float(np.median(per_draw.to_numpy(dtype=float)))
            L0 = _ceil_ratio(med, a0)
            per_draw_L0 = [_ceil_ratio(v, a0) for v in per_draw]
            d = {"L0": L0, "draws": int(per_draw.size), "a0_layer_cx": a0, "a0_layer_cx_formula": n * (n - 1),
                 "a1_cx_ii_median": med, "a1_cx_ii_min": int(per_draw.min()), "a1_cx_ii_max": int(per_draw.max()),
                 "ratio_median": med / a0, "a0_cx_at_L0": L0 * a0, "match_rel_error": L0 * a0 / med - 1.0,
                 "L0_per_draw_min": int(min(per_draw_L0)), "L0_per_draw_max": int(max(per_draw_L0)),
                 "params_at_L0": 2 * L0}
            s_med = float(np.median(g.groupby("draw_id")[f"a1_L{L1}_cx_ii_S"].first().to_numpy(dtype=float)))
            d["info_L0_cx_ii_S"] = _ceil_ratio(s_med, a0)
            if other_ro:
                o_med = float(np.median(g.groupby("draw_id")[f"a1_L{L1}_cx_ii_{other_ro}"].first()
                                        .to_numpy(dtype=float)))
                d[f"info_L0_ring_{other_ro}"] = _ceil_ratio(o_med, a0)
            det[int(L1)] = d
        out[int(N)] = det
    return out


def check_against_s3(rows: pd.DataFrame, root=None) -> dict:
    """Cross-check with S3's results/tables/mixer_counts.parquet (baseline cell, same ring order): max |diff|."""
    from ..store.paths import tables_dir
    p = tables_dir(root) / "mixer_counts.parquet"
    if not p.exists():
        return {"available": False}
    m = pd.read_parquet(p)
    ro = rows["ring_order"].iloc[0]
    m = m[(m["axis"] == "baseline") & (m["ring_order"] == ro)].set_index("inst_id")
    j = rows.set_index("inst_id").join(m, rsuffix="_s3", how="left")
    out = {"available": True, "rows": int(len(j)), "missing_in_s3": int(j["a1_L9_cx_ii_s3"].isna().sum())}
    for L1 in MATCHED_L1:
        out[f"max_abs_diff_L{L1}"] = float((j[f"a1_L{L1}_cx_ii"] - j[f"a1_L{L1}_cx_ii_s3"]).abs().max())
    out["max_abs_diff_prep"] = float((j["prep_cx_ii"] - j["prep_cx_ii_s3"]).abs().max())
    out["max_abs_diff_a0_layer_vs_obj"] = float((j["a0_layer_cx"] - j["a0_layer_cx_s3"]).abs().max())
    return out


def matched_depths_yaml(det: dict, check: dict, ring_order: str = "lex") -> str:
    import yaml
    today = _dt.date.today().isoformat()
    doc = {
        "status": "proposed",
        "written_by": f"S9a ({today}); S9b finalizes these into envelope.yaml gate_matched_depths",
        "rule": "PLAN §1.3: L0(N, L1) = ceil(median over the N's accepted draws of CNOT_ii(A1 ring.violation.K12 at "
                "depth L1, start circuit included) / CNOT(one A0 layer)); one A0 layer = 2 x ZZ terms of H(lam)",
        "count": "ii (Eq. 4.10, all n - 1 controls; O-5 default)",
        "ring_order": ring_order,
        "sector_source": "ga",
        "harness_version": HARNESS_VERSION,
        "gate_matched_depths": {int(N): {int(L1): int(d["L0"]) for L1, d in m.items()} for N, m in det.items()},
        "detail": {int(N): {int(L1): {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                                      for k, v in d.items()} for L1, d in m.items()} for N, m in det.items()},
        "check_vs_s3_mixer_counts": check,
    }
    return ("# Proposed gate-matched A0 depths (PLAN §1.3), from `scripts/s9_calibration.py depths`.\n"
            "# Proposed values only: S9b finalizes them into configs/envelope.yaml (gate_matched_depths).\n"
            + yaml.safe_dump(doc, sort_keys=False, width=120))


def matched_depths_markdown(det: dict, rows: pd.DataFrame, check: dict, ring_order: str = "lex") -> str:
    other = "rank" if ring_order == "lex" else "lex"
    L = ["# Gate-matched A0 depths L0(N, L1): proposed (S9a)", "",
         f"Written by `scripts/s9_calibration.py depths` on {_dt.date.today().isoformat()} "
         f"(harness {HARNESS_VERSION}). **Proposed values**; S9b finalizes them in `configs/envelope.yaml`.", "",
         "## Rule (PLAN §1.3)", "",
         "`L0(N, L1) = ceil( median over the N's accepted draws of CNOT_ii(A1 baseline circuit at depth L1, start "
         "circuit included) / CNOT(one A0 layer) )`", "",
         "- A1 baseline = ring · violation · K = 12, production GA sector, "
         f"ring order **{ring_order}** (O-1 default), (ii) = Eq. 4.10 with all n − 1 controls (O-5 default), "
         "counted by S3's transpiler on the abstract circuit (star prep + L1 × (cost + mixer)).",
         "- One A0 layer = 2 × (ZZ terms of H(λ)) CNOTs; the X mixer is free. Checked identical for all five pilot λ "
         "and equal to n(n − 1) (dense QUBO) on every instance.",
         "- Per draw: the violation sector depends only on the draw; the three q give the same count (checked). "
         "Median over the 30 accepted draws.", "",
         "## Proposed L0(N, L1)", "",
         "| N | n | A0 layer CX | L1 = 5 | L1 = 7 | L1 = 9 | params 2·L0 at L1 = 9 |",
         "|---|---|---|---|---|---|---|"]
    for N, m in sorted(det.items()):
        n = int(rows[rows["N"] == N]["n"].iloc[0])
        L.append(f"| {N} | {n} | {m[5]['a0_layer_cx']} | **{m[5]['L0']}** | **{m[7]['L0']}** | **{m[9]['L0']}** | "
                 f"{m[9]['params_at_L0']} |")
    L += ["", "## Detail", "",
          "| N | L1 | median CX_ii(A1) | min–max over draws | ratio | L0 | A0 CX at L0 | match error | "
          "per-draw L0 range |", "|---|---|---|---|---|---|---|---|---|"]
    for N, m in sorted(det.items()):
        for L1, d in sorted(m.items()):
            L.append(f"| {N} | {L1} | {d['a1_cx_ii_median']:.1f} | {d['a1_cx_ii_min']}–{d['a1_cx_ii_max']} | "
                     f"{d['ratio_median']:.2f} | {d['L0']} | {d['a0_cx_at_L0']} | {100 * d['match_rel_error']:+.2f} % | "
                     f"{d['L0_per_draw_min']}–{d['L0_per_draw_max']} |")
    L += ["", "## Information only (not proposed): the open decisions that move L0", "",
          f"| N | L1 | L0 (proposed) | L0 with O-5's restricted count `cx_ii_S` | L0 with the {other} ring (O-1) |",
          "|---|---|---|---|---|"]
    for N, m in sorted(det.items()):
        for L1, d in sorted(m.items()):
            L.append(f"| {N} | {L1} | {d['L0']} | {d['info_L0_cx_ii_S']} | {d.get(f'info_L0_ring_{other}', '—')} |")
    L += ["", "## Cross-check", ""]
    if check.get("available"):
        L.append(f"- Against S3's `results/tables/mixer_counts.parquet` (baseline rows, {ring_order}): "
                 f"{check['rows']} instances, {check['missing_in_s3']} missing there; max |Δ| of the A1 totals "
                 f"L1 = 5 / 7 / 9: {check['max_abs_diff_L5']:.0f} / {check['max_abs_diff_L7']:.0f} / "
                 f"{check['max_abs_diff_L9']:.0f}; start circuit {check['max_abs_diff_prep']:.0f}; A0 layer of "
                 f"H(λ) vs S3's layer of H_obj: {check['max_abs_diff_a0_layer_vs_obj']:.0f}.")
    else:
        L.append("- S3's mixer_counts.parquet is not available in this results root.")
    L += ["", "## Cost note for S9b", "",
          "At L0 an A0 iteration charges 2·L0 + 1 circuits of L0 layers each, so an iteration costs ~(L0/9)² of a "
          "depth-9 iteration. The timing set (queue 1) runs A0 at L0(5, 9) and L0(7, 9) full length; §1.8's cut 2 "
          "(gate-matched A0 R = 5 → 3) acts on these runs.", ""]
    return "\n".join(L)


def write_matched_depths(root=None, ring_order: str = "lex", yaml_path=None, md_path=None) -> dict:
    from ..store.paths import configs_dir, reports_dir
    rows = matched_depth_rows(root, ring_order)
    det = matched_depths(rows)
    check = check_against_s3(rows, root)
    yp = Path(yaml_path) if yaml_path else configs_dir() / "matched_depths.yaml"
    mp = Path(md_path) if md_path else reports_dir() / "matched_depths.md"
    atomic_write_bytes(yp, matched_depths_yaml(det, check, ring_order).encode())
    atomic_write_bytes(mp, matched_depths_markdown(det, rows, check, ring_order).encode())
    return {"detail": det, "check": check, "yaml": str(yp), "md": str(mp), "rows": rows}


def load_matched_depths(path=None) -> dict:
    """{N: {L1: L0}} from configs/matched_depths.yaml."""
    import yaml
    from ..store.paths import configs_dir
    p = Path(path) if path else configs_dir() / "matched_depths.yaml"
    doc = yaml.safe_load(p.read_text())
    return {int(N): {int(a): int(b) for a, b in m.items()} for N, m in doc["gate_matched_depths"].items()}


# --- B-D. specs ------------------------------------------------------------------------------------------------------
def _instances(root=None) -> pd.DataFrame:
    from ..instances.instance import load_instances_table
    return load_instances_table(root).sort_values(["N", "accept_rank", "q"], kind="stable").reset_index(drop=True)


def first_draws(N: int, k: int = 1, q: float = 1.5, k24: bool = False, it: pd.DataFrame | None = None,
                root=None) -> list[str]:
    """inst_ids of the first k accepted draws of N at q (accept order); k24: of the k24-eligible subset."""
    it = _instances(root) if it is None else it
    g = it[(it["N"] == int(N)) & (it["q"] == float(q))]
    if k24:
        g = g[g["k24_eligible"].astype(bool)]
    ids = list(g["inst_id"][:int(k)])
    if len(ids) < k:
        raise ValueError(f"N={N}: only {len(ids)} accepted draws (k24={k24})")
    return ids


def spec(arm: str, inst_id: str, effort: int, *, cell: dict | None = None, kw: dict | None = None, tag: str,
         label: str, axis: str | None = None, lam: float | None = None, schedule: str | None = None,
         matched_L1: int | None = None, tier: str = "calibration") -> dict:
    """A queue spec in `gsp plan`'s format (runnable; run_id filled by `resolve_strict`)."""
    from ..store.ids import parse_inst_id
    pi = parse_inst_id(inst_id)
    N = int(pi["N"])
    kw = dict(kw or {})
    enc = "penalty" if cell is None else "confined"
    if cell is None:
        row = lab = P.PENALTY_ROW.format(N=N)
        axis = axis or "penalty"
    else:
        base_conn = "ring" if cell["connectivity"] == "adaptive" else cell["connectivity"]
        row = P.cell_label(base_conn, cell["rule"], cell["K"], N)
        lab = P.cell_label(cell["connectivity"], cell["rule"], cell["K"], N)
    return {"arm": arm, "encoding": enc, "tier": tier, "secondary": arm == "A6", "N": N,
            "draw_id": inst_id.split("q")[0], "e": int(pi["e"]), "inst_id": inst_id, "q": float(pi["q"]),
            "accept_rank": None, "row": row, "cell_label": lab, "axis": axis, "cell": cell, "effort_kind": None,
            "effort": int(effort), "schedule": schedule, "restart": int(kw.get("restart", 0)),
            "lam": lam if lam is not None else kw.get("lam"), "ring_order": kw.get("ring_order"), "tag": tag,
            "matched_L1": matched_L1, "kw": kw, "run_id": None, "placeholder": "", "label": label, "est_s": None}


def resolve_strict(specs: list[dict], root=None) -> list[dict]:
    """run_id of every spec through the registered arm's `config`, refusing any kwarg the config does not accept
    (a silently dropped flag would make an evidence run hash as a plain run). Specs that hash to one run_id are merged
    (labels joined); two with different kwargs and one run_id are an error."""
    from ..arms.base import make_arm
    from ..instances.adhoc import load_any
    arms: dict = {}
    seen: dict = {}
    out = []
    with P._cached_sectors():
        for s in specs:
            a = s["arm"]
            if a not in arms:
                arms[a] = make_arm(a)
            arm = arms[a]
            names = set(inspect.signature(type(arm).config).parameters)
            bad = set(s["kw"]) - names
            if bad:
                raise P.PlanError(f"{a}.config does not accept {sorted(bad)} ({s['label']})")
            inst, _, adhoc = load_any(s["inst_id"], root)
            cfg = arm.config(inst, s["cell"], s["effort"], None, adhoc=adhoc, root=root, **s["kw"])
            s = dict(s, run_id=cfg.run_id, effort_kind=cfg.effort_kind)
            prev = seen.get(s["run_id"])
            if prev is not None:
                if prev["kw"] != s["kw"]:
                    raise P.PlanError(f"{a}: {prev['kw']} and {s['kw']} hash to one run_id {s['run_id']}")
                prev["label"] = f"{prev['label']} + {s['label']}"
                continue
            seen[s["run_id"]] = s
            out.append(s)
    return out


def timing_specs(L0: dict, root=None) -> list[dict]:
    """Queue 1 (the timing set of PLAN §5 S9 as the orchestrator specified it), shortest runs first."""
    it = _instances(root)
    f = lambda N, k24=False: first_draws(N, 1, 1.5, k24, it)[0]          # noqa: E731
    lam = TIMING_LAM
    ring = lambda K: {"connectivity": "ring", "rule": "violation", "K": K}           # noqa: E731
    comp12 = {"connectivity": "complete", "rule": "violation", "K": 12}
    ada12 = {"connectivity": "adaptive", "rule": "violation", "K": 12}
    S = []
    # A2 (one circuit per p): primary schedule at p = 300, plus p = 50 for the scaling fit
    for p in (50, 300):
        for N in (5, 10):
            S.append(spec("A2p", f(N), p, kw={"lam": lam, "schedule_tag": "primary"}, lam=lam, schedule="primary",
                          tag="timing", label=f"A2p N{N} p{p} primary"))
        for N in (5, 7):
            S.append(spec("A2c", f(N), p, cell=ring(12), axis="baseline",
                          kw={"schedule_tag": "primary", "ring_order": "lex"}, schedule="primary", tag="timing",
                          label=f"A2c N{N} K12 p{p} primary"))
    # A3 / A3d: 20 steps (the n_steps override for timing runs)
    S.append(spec("A3d", f(7), 9, kw={"lam": lam, "n_steps": 20}, lam=lam, tag="timing", label="A3d N7 L9 20 steps"))
    for N in (5, 7, 10):
        S.append(spec("A3", f(N), 9, kw={"lam": lam, "n_steps": 20}, lam=lam, tag="timing",
                      label=f"A3 N{N} L9 20 steps"))
    # A0 at depth 9, A1 at depth 9 (full length, r = 0)
    for N in (5, 10):
        S.append(spec("A0", f(N), 9, kw={"restart": 0, "lam": lam}, lam=lam, tag="timing", label=f"A0 N{N} L9"))
    for N in (5, 7):
        S.append(spec("A1", f(N), 9, cell=ring(12), axis="baseline", kw={"restart": 0, "ring_order": "lex"},
                      tag="timing", label=f"A1 ring N{N} K12 L9"))
    for N in (5, 6):
        S.append(spec("A1", f(N, k24=True), 9, cell=ring(24), axis="K", kw={"restart": 0, "ring_order": "lex"},
                      tag="timing", label=f"A1 ring N{N} K24 L9"))
    for N in (5, 6):
        S.append(spec("A1", f(N), 9, cell=comp12, axis="connectivity", kw={"restart": 0, "ring_order": "lex"},
                      tag="timing", label=f"A1 complete N{N} K12 L9"))
    # A0 at the largest proposed matched depth L0(N, 9)
    for N in (5, 7):
        S.append(spec("A0", f(N), int(L0[N][9]), kw={"restart": 0, "lam": lam}, lam=lam, tag="timing",
                      matched_L1=9, label=f"A0 N{N} L0(N,9)={L0[N][9]}"))
    # A4 / A6 to k = 10 (normalized), for the recursion-cap rule of §1.8
    S.append(spec("A4", f(7), 10, cell=ada12, axis="baseline",
                  kw={"ring_order": "lex", "step_units": "normalized"}, tag="timing", label="A4 N7 K12 k10"))
    S.append(spec("A6", f(10), 10, kw={"lam": lam, "step_units": "normalized"}, lam=lam, tag="timing",
                  label="A6 N10 k10"))
    return S


O11_VARIANTS = (("a", {}), ("b", {"psd_project": True}), ("c", {"tikhonov": 1e-4}))
O11_JITTER = 1e-9
O11_JITTER_SEEDS = (None, 0, 1)


def evidence_specs(root=None, o2_draws: int = 10, o11_draws: int = 5) -> list[dict]:
    """Queue 3 (module doc): O-2 then O-11."""
    it = _instances(root)
    S = []
    ring12 = {"connectivity": "ring", "rule": "violation", "K": 12}
    for N in (5, 7):
        draws = first_draws(N, o2_draws, 1.5, False, it)
        for lam in (0.005, 0.05):
            for iid in draws:
                S.append(spec("A0", iid, 7, kw={"restart": 0, "lam": lam, "circuit_boosted": True, "evidence": "O-2"},
                              lam=lam, tag="evidence:O-2", label=f"O-2 A0 boosted N{N} lam{lam:g} L7"))
        for iid in draws:
            S.append(spec("A1", iid, 5, cell=ring12, axis="baseline", kw={"restart": 0, "ring_order": "lex"},
                          tag="evidence:O-2", label=f"O-2 A1 un-boosted (sweep config) N{N} L5"))
            S.append(spec("A1", iid, 5, cell=ring12, axis="baseline",
                          kw={"restart": 0, "ring_order": "lex", "circuit_boosted": True, "evidence": "O-2"},
                          tag="evidence:O-2", label=f"O-2 A1 boosted N{N} L5"))
    for iid in first_draws(7, o11_draws, 1.5, False, it):
        for var, extra in O11_VARIANTS:
            for js in O11_JITTER_SEEDS:
                kw = {"lam": 0.005, **extra}
                if js is not None:
                    kw.update(theta0_jitter=O11_JITTER, theta0_jitter_seed=js)
                sweep_cfg = not extra and js is None
                if not sweep_cfg:
                    kw["evidence"] = "O-11"
                S.append(spec("A3", iid, 5, kw=kw, lam=0.005, tag="evidence:O-11",
                              label=f"O-11 A3 ({var}) jitter {'none' if js is None else js}"
                                    + (" (sweep config)" if sweep_cfg else "")))
    return S


def pilot_queue_specs(root=None) -> tuple[list[dict], dict]:
    """Queue 2: exactly what `gsp plan --pilot` writes (plan.build_plan(pilot=True)), plus its sidecar meta."""
    env = P.load_envelope()
    specs = P.build_plan(env, pilot=True, root=root)
    for s in specs:
        s["label"] = f"pilot A0 N{s['N']} lam{s['lam']:g} L7"
        s["est_s"] = None
    meta = {"envelope": env["_path"], "envelope_sha256": env["_sha256"], "envelope_status": env.get("status"),
            "pilot": True, "order": "draw", "counts": P.counts(specs),
            "equivalent_command": "gsp plan --pilot --name s9_q2_pilot"}
    return specs, meta


# --- S9c (PLAN §5 S9c, §4.2): the queues of the boosted default ------------------------------------------------------
# The boosted circuit is the default for A0 / A1 / A3 (O-2); its configs hash with circuit_boosted = True (absent =
# the legacy un-boosted convention, which every stored record without the key is). So the runs below, queued with the
# flag, carry exactly the run_ids of the boosted default: untagged runs are the sweep's own computations.
S9C_QUEUE_NAMES = {4: "s9c_q4_pilot_boosted", 5: "s9c_q5_retiming", 6: "s9c_q6_o11"}


def boosted_pilot_specs(root=None) -> tuple[list[dict], dict]:
    """Queue 4: the lambda pilot of PLAN §1.4 (queue 2's specs) on the boosted circuit; lambda*(N) comes from it."""
    specs, meta = pilot_queue_specs(root)
    out = [dict(s, kw=dict(s["kw"], circuit_boosted=True), run_id=None,
                label=f"pilot-boosted A0 N{s['N']} lam{s['lam']:g} L7") for s in specs]
    meta = dict(meta, circuit_boosted=True, equivalent_command="gsp plan --pilot, with circuit_boosted (S9c)")
    return resolve_strict(out, root), meta


def retiming_specs(root=None) -> list[dict]:
    """Queue 5: queue 1's A0 / A1 depth-9 timing runs on the boosted circuit (S9c: confirm the time models)."""
    it = _instances(root)
    f = lambda N: first_draws(N, 1, 1.5, False, it)[0]                      # noqa: E731
    lam = TIMING_LAM
    S = [spec("A0", f(N), 9, kw={"restart": 0, "lam": lam, "circuit_boosted": True}, lam=lam, tag="timing",
              label=f"A0 N{N} L9 boosted") for N in (5, 10)]
    S += [spec("A1", f(N), 9, cell=dict(BASELINE), axis="baseline",
               kw={"restart": 0, "ring_order": "lex", "circuit_boosted": True}, tag="timing",
               label=f"A1 ring N{N} K12 L9 boosted") for N in (5, 7)]
    return S


O11_S9C_VARIANTS = O11_VARIANTS + (("d", {"stencil": "central"}),)


def o11_round_specs(root=None, o11_draws: int = 5) -> list[dict]:
    """Queue 6, the O-11 round of §4.2: {boosted} x {(a) default, (b) psd_project, (c) tikhonov 1e-4, (d) central
    stencil}, plus the old convention's (d) (queue 3 ran the old (a)-(c)); A3 at N 7, L 5, lam 0.005, q 1.5, the first
    5 accepted draws, each at the Ramp init and the two 1e-9 rad jitters (seeds 0, 1). The boosted default without
    jitter is the sweep config of the boosted default (untagged); every other run is tagged "O-11"."""
    it = _instances(root)
    S = []
    for iid in first_draws(7, o11_draws, 1.5, False, it):
        for boosted in (True, False):
            for var, extra in O11_S9C_VARIANTS:
                if not boosted and var != "d":
                    continue
                for js in O11_JITTER_SEEDS:
                    kw = {"lam": 0.005, **extra}
                    if boosted:
                        kw["circuit_boosted"] = True
                    if js is not None:
                        kw.update(theta0_jitter=O11_JITTER, theta0_jitter_seed=js)
                    sweep_cfg = boosted and not extra and js is None
                    if not sweep_cfg:
                        kw["evidence"] = "O-11"
                    conv = "boosted" if boosted else "old"
                    S.append(spec("A3", iid, 5, kw=kw, lam=0.005, tag="evidence:O-11",
                                  label=f"O-11 A3 {conv} ({var}) jitter {'none' if js is None else js}"
                                        + (" (sweep config)" if sweep_cfg else "")))
    return S


# --- F. estimates ----------------------------------------------------------------------------------------------------
# Measured rates (RTX 4080, fp64, fusion 1), with their sources. Seconds.
RATES = {
    # S4 (reports/arms.md §6; registry): s per FD iteration (2L + 1 observes), logger excluded
    ("A0", 8, 5): 0.0233, ("A0", 8, 7): 0.0361, ("A0", 8, 9): 0.0513, ("A0", 10, 5): 0.0313, ("A0", 20, 9): 0.3719,
    ("A1", 14, 9, "ring", 12): 0.2569, ("A1", 8, 9, "ring", 12): 0.1389, ("A1", 8, 9, "complete", 12): 0.4655,
    ("A1", 10, 5, "ring", 12): 0.0662,
    # S7 (STATUS S7 table): s per McLachlan step, logger excluded
    ("A3", 8, 5): 0.158, ("A3", 14, 5): 0.458, ("A3", 20, 5): 1.642,
    ("A3", 8, 9): 0.568, ("A3", 14, 9): 1.903, ("A3", 20, 9): 5.631,
    ("A3d", 8, 9): 0.127, ("A3d", 14, 9): 0.335, ("A3d", 20, 9): 1.670,
}
RATE_SOURCES = {"A0/A1": "S4 reports/arms.md §6 + registry per_iter_s", "A3/A3d": "STATUS S7 per-step table",
                "A4/A6": "STATUS S8 (k = 10 extrapolation incl. post-run)", "post-run": "STATUS S5 simdiff worst case",
                "logger": "S4 / S7 per-call logger cost"}
# S8: a single run to k = 10 incl. the post-run step (minutes -> s)
DB_K10_S = {("A4", 14): 13.6 * 60, ("A6", 20): 12.7 * 60}
MAX_ITER = 300          # AdamW T_max / stop rule upper bound (PLAN §1.6): the budget uses the bound
A3_MAX_STEPS = 300


def postrun_s(n: int) -> float:
    """The S5 post-run step: ~0.15 s at n <= 14 (+ the 1000-shot sample), simdiff worst case 0.25 / 1.7 / 11.4 s at
    n = 16 / 18 / 20 (+ state and sample)."""
    return {16: 1.0, 18: 3.0, 20: 12.0}.get(int(n), 0.5) if n > 14 else 0.5


def logger_s(n: int) -> float:
    """One post-update logger call (get_state + metrics): 14 ms (n = 14) / 34 ms (n = 20) in S4."""
    return 0.005 if n <= 10 else (0.015 if n <= 14 else (0.02 if n <= 16 else (0.025 if n <= 18 else 0.035)))


def _interp_n(table: dict, n: int) -> float | None:
    """Log-linear interpolation in n between measured sizes (None outside [min, max])."""
    ns = sorted(table)
    if n in table:
        return table[n]
    lo = [x for x in ns if x < n]
    hi = [x for x in ns if x > n]
    if not lo or not hi:
        return None
    a, b = lo[-1], hi[0]
    t = (n - a) / (b - a)
    return float(math.exp((1 - t) * math.log(table[a]) + t * math.log(table[b])))


def estimate_spec(s: dict, bench: dict | None = None) -> dict:
    """{'est_s', 'source', 'units', 's_per_unit'} of one spec (a conservative bound: 300 iterations / steps where the
    stop rule may end earlier). `bench`: {bench_key(spec): seconds per observe} from the S9a smoke."""
    from ..store.ids import parse_inst_id
    arm, eff = s["arm"], int(s["effort"])
    n = 2 * int(parse_inst_id(s["inst_id"])["N"])
    kw = s.get("kw") or {}
    pr = postrun_s(n)
    bkey = bench_key(s)
    per_obs = (bench or {}).get(bkey)
    if arm in ("A0", "A1"):
        L = eff
        circ = 2 * L + 1
        cell = s.get("cell") or {}
        key = ("A0", n, L) if arm == "A0" else ("A1", n, L, cell.get("connectivity"), int(cell.get("K", 0)))
        if key in RATES:
            spi, src = RATES[key], "measured (S4)"
        elif per_obs is not None:
            spi, src = circ * per_obs, "smoke micro-benchmark x (2L + 1)"
        else:
            spi, src = None, "no rate"
        if spi is None:
            return {"est_s": None, "source": src, "units": MAX_ITER, "s_per_unit": None}
        return {"est_s": MAX_ITER * (spi + logger_s(n)) + pr + 1.0, "source": src, "units": MAX_ITER,
                "s_per_unit": spi}
    if arm in ("A2p", "A2c"):
        if per_obs is None:
            per_obs = 0.02 + eff * (1e-4 if n <= 14 else 1.5e-3)     # rough fallback
            src = "fallback"
        else:
            src = "smoke micro-benchmark"
        return {"est_s": 2 * per_obs + pr + 1.0, "source": src, "units": 1, "s_per_unit": per_obs}
    if arm in ("A3", "A3d"):
        L = eff
        steps = int(kw.get("n_steps", A3_MAX_STEPS))
        tab = {k[1]: v for k, v in RATES.items() if k[0] == arm and len(k) == 3 and k[2] == L}
        sps = _interp_n(tab, n)
        src = "measured (S7)" if n in tab else "S7 log-interpolated in n"
        if sps is None:
            return {"est_s": None, "source": "no rate", "units": steps, "s_per_unit": None}
        return {"est_s": steps * (sps + logger_s(n)) + pr + 2.0, "source": src, "units": steps, "s_per_unit": sps}
    if arm in ("A4", "A6"):
        v = DB_K10_S.get((arm, n))
        if v is not None and eff == 10:
            return {"est_s": v, "source": "S8 extrapolation (incl. post-run)", "units": eff, "s_per_unit": None}
        return {"est_s": None, "source": "no rate", "units": eff, "s_per_unit": None}
    return {"est_s": None, "source": "unknown arm", "units": None, "s_per_unit": None}


def bench_key(s: dict) -> str:
    """The circuit shape a spec's loop observes (what the S9a micro-benchmark times): arm family, N, depth, cell.
    The instance, lam and the circuit convention change angles, not gates (penalty QUBOs are dense; a sector changes
    a confined layer's gate count by a few %), so they are not part of the key."""
    from ..store.ids import parse_inst_id
    cell = s.get("cell") or {}
    fam = {"A0": "pen", "A2p": "pen", "A3": "pen", "A3d": "pen", "A1": "conf", "A2c": "conf"}.get(s["arm"], s["arm"])
    return "|".join(str(x) for x in (fam, parse_inst_id(s["inst_id"])["N"], int(s["effort"]),
                                     cell.get("connectivity"), cell.get("K")))


def estimate_specs(specs: list[dict], bench: dict | None = None) -> dict:
    tot, unknown = 0.0, []
    per_arm: dict = {}
    for s in specs:
        e = estimate_spec(s, bench)
        s["est_s"] = None if e["est_s"] is None else round(float(e["est_s"]), 1)
        s["est_source"] = e["source"]
        if e["est_s"] is None:
            unknown.append(s["label"])
            continue
        tot += e["est_s"]
        a = per_arm.setdefault(s["arm"], {"runs": 0, "est_h": 0.0})
        a["runs"] += 1
        a["est_h"] += e["est_s"] / 3600
    return {"runs": len(specs), "est_h": tot / 3600, "unknown": unknown,
            "per_arm": {a: {"runs": v["runs"], "est_h": round(v["est_h"], 3)} for a, v in per_arm.items()}}


# --- queue files -----------------------------------------------------------------------------------------------------
def spec_line(s: dict) -> str:
    return json.dumps({k: s.get(k) for k in P.QUEUE_KEYS + EXTRA_KEYS}, separators=(",", ":"), default=str)


def write_queue(specs: list[dict], path, meta: dict | None = None) -> dict:
    """JSONL (plan.QUEUE_KEYS + label + est_s) and the `.plan.json` sidecar, like `plan.write_queue`."""
    path = Path(path)
    if any(s.get("run_id") is None for s in specs):
        raise P.PlanError("unresolved spec (call resolve_strict first)")
    atomic_write_bytes(path, ("\n".join(spec_line(s) for s in specs) + ("\n" if specs else "")).encode())
    from ..store.records import git_info
    side = dict(meta or {})
    side.update({"plan_schema": P.PLAN_SCHEMA, "queue": str(path), "n_runs": len(specs), "n_placeholders": 0,
                 "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "harness_version": HARNESS_VERSION, "writer": "gsp.runner.calibration (S9a)"})
    side["git_sha"], side["git_dirty"] = git_info()
    atomic_write_bytes(P.sidecar_path(path), json.dumps(side, indent=1, sort_keys=True, default=str).encode())
    return side
