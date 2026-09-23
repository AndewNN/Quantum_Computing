"""S3 tables and `reports/mixer_counts.md` (PLAN §5 S3).

  count_rows()   gate counts (ii)/(iii)/T of every (confined cell, instance, ring order):
                 results/tables/mixer_counts.parquet
  c1_sweep()     the C1 operator-level check on every real sector with n <= 12 (both connectivities):
                 results/tables/c1_operator_{engine}.parquet
  markdown()     the report: per-cell medians over instances, the 696 check, the (iii) construction,
                 verification numbers, C1 maxima, D-9 (lex vs rank), and the GPU timing if measured
                 (`gsp.compile.timing`, results/tables/mixer_timing.json).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..circuits import preserving as pr
from ..circuits.cost import cost_terms
from ..instances.instance import load_instance
from ..sectors.select import cell_instances, load_cells, load_sector, sector_scope
from ..store.io import atomic_write_bytes
from ..store.paths import reports_dir, tables_dir
from . import decompose as dc
from . import tcount
from . import transpile as tp
from . import verify as vf

DEPTHS = (5, 7, 9)
PARTS = ("mixer", "prep", "cost", "layer")
KEYS = ("cx_ii", "cx_iii", "cx_ii_S", "t_ii", "tdepth_ii", "t_iii", "tdepth_iii")


# --- counts -------------------------------------------------------------------------------------------
def count_rows(root=None, ring_orders=("lex", "rank")) -> pd.DataFrame:
    rows = []
    circ_cache: dict = {}
    for cell in load_cells():
        N, K, rule, conn = int(cell["N"]), int(cell["K"]), cell["rule"], cell["connectivity"]
        for iid in cell_instances(cell, root):
            sec = load_sector(iid, rule, K, root)
            inst = load_instance(iid, root)
            ct = cost_terms(inst.H_obj, inst.boost_obj)
            for ro in ring_orders:
                key = (sector_scope(iid, rule), rule, K, conn, ro)
                if key not in circ_cache:
                    circ_cache[key] = pr.build_circuit(sec, conn, ro)
                cc = tp.circuit_counts(circ_cache[key], ct)
                row = {"axis": cell["axis"], "connectivity": conn, "rule": rule, "K": K, "N": N,
                       "n": int(sec.n), "inst_id": iid, "draw_id": iid.split("q")[0], "ring_order": ro,
                       "d_mean": cc["mixer"]["d_mean"], "S_mean": cc["mixer"]["S_mean"],
                       "S_max": cc["mixer"]["S_max"], "prep_S_mean": cc["prep"]["S_mean"],
                       "n_zz": cc["cost"]["n_zz"], "n_rot": cc["cost"]["n_rot"],
                       "n_transitions": cc["mixer"]["n_transitions"]}
                for part in PARTS:
                    for k in KEYS:
                        row[f"{part}_{k}"] = cc[part][k]
                for L in DEPTHS:
                    tot = tp.a1_totals(cc, L)
                    row[f"a1_L{L}_cx_ii"], row[f"a1_L{L}_cx_iii"] = tot["cx_ii"], tot["cx_iii"]
                    row[f"a1_L{L}_t_ii"] = tot["t_ii"]
                row["a0_layer_cx"] = tp.a0_layer_counts(ct)["cx"]
                rows.append(row)
    return pd.DataFrame(rows)


def write_counts(root=None) -> pd.DataFrame:
    df = count_rows(root)
    d = tables_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "mixer_counts.parquet", index=False)
    return df


# --- C1 operator level over the real sectors -------------------------------------------------------------
def sector_jobs(root=None, n_max: int = 12) -> list[dict]:
    """One job per distinct (sector file, connectivity) at the cells with n <= n_max."""
    seen, jobs = set(), []
    for cell in load_cells():
        N, K, rule, conn = int(cell["N"]), int(cell["K"]), cell["rule"], cell["connectivity"]
        if 2 * N > n_max:
            continue
        for iid in cell_instances(cell, root):
            key = (sector_scope(iid, rule), rule, K, conn)
            if key in seen:
                continue
            seen.add(key)
            jobs.append({"inst_id": iid, "scope": key[0], "rule": rule, "K": K, "connectivity": conn, "N": N})
    return jobs


def c1_sweep(root=None, engine: str = "numpy", ring_orders=("lex", "rank"), n_max: int = 12,
             betas=(0.613, -2.1), symmetrized=(False, True), decomposed_n_max: int = 0,
             limit: int | None = None, log=None) -> pd.DataFrame:
    """C1 operator level on every job of `sector_jobs`. With engine="cudaq" the kernels run through
    `gsp.sim.backend` (one GPU process; check nvidia-smi first). decomposed_n_max > 0 adds the
    explicit (iii) circuit (simulate_decomposed) at n <= that size (lex, plain)."""
    from ..stats.c1 import a1_operator_level

    jobs = sector_jobs(root, n_max)
    if limit:
        jobs = jobs[:limit]
    rows = []
    t0 = time.time()
    for jn, job in enumerate(jobs):
        sec = load_sector(job["inst_id"], job["rule"], job["K"], root)
        variants = [(ro, sym, False) for ro in ring_orders for sym in symmetrized if not (sym and ro != "lex")]
        if decomposed_n_max and sec.n <= decomposed_n_max:
            variants.append(("lex", False, True))
        for ro, sym, dec in variants:
            circ = pr.build_circuit(sec, job["connectivity"], ro, sym)
            t = time.time()
            res = a1_operator_level(circ, list(betas), engine=engine, decomposed=dec)
            wall = time.time() - t
            for r in res:
                r.update({k: job[k] for k in ("inst_id", "scope", "rule", "N")}, wall_s=wall / len(res))
                rows.append(r)
        if log and (jn + 1) % 50 == 0:
            log(f"c1 {engine}: {jn + 1}/{len(jobs)} sectors, {time.time() - t0:.0f} s")
    return pd.DataFrame(rows)


def write_c1(root=None, engine: str = "numpy", **kw) -> pd.DataFrame:
    df = c1_sweep(root, engine=engine, **kw)
    d = tables_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"c1_operator_{engine}.parquet", index=False)
    return df


# --- the report ---------------------------------------------------------------------------------------
def _med(x):
    return float(np.median(x))


def _fmt(x, nd=0):
    if isinstance(x, (int, np.integer)) or (nd == 0 and float(x).is_integer()):
        return f"{int(round(float(x))):,}"
    return f"{float(x):,.{nd}f}"


def _cell_table(df: pd.DataFrame, ro: str = "lex") -> str:
    d = df[df["ring_order"] == ro]
    cols = ["axis", "connectivity", "rule", "K", "N"]
    order = {"baseline": 0, "K": 1, "rule": 2, "connectivity": 3}
    head = ("| cell | n | inst | d̄ | \\|S\\|̄ | mixer (ii) | mixer (iii) | cost CX | **layer (ii)** | "
            "**layer (iii)** | prep (ii) | prep (iii) | T layer (ii) | T-depth layer (ii) | T layer (iii) | "
            "T-depth layer (iii) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    groups = sorted(d.groupby(cols), key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2], kv[0][3], kv[0][4]))
    for (axis, conn, rule, K, N), g in groups:
        label = f"{conn} · {rule} · K{K} · N{N}"
        lines.append(
            f"| {label} | {2 * N} | {len(g)} | {_med(g.d_mean):.2f} | {_med(g.S_mean):.2f} | "
            f"{_fmt(_med(g.mixer_cx_ii))} | {_fmt(_med(g.mixer_cx_iii))} | {_fmt(_med(g.cost_cx_ii))} | "
            f"**{_fmt(_med(g.layer_cx_ii))}** | **{_fmt(_med(g.layer_cx_iii))}** | {_fmt(_med(g.prep_cx_ii))} | "
            f"{_fmt(_med(g.prep_cx_iii))} | {_fmt(_med(g.layer_t_ii))} | {_fmt(_med(g.layer_tdepth_ii))} | "
            f"{_fmt(_med(g.layer_t_iii))} | {_fmt(_med(g.layer_tdepth_iii))} |")
    return "\n".join(lines)


def _range_table(df: pd.DataFrame, ro: str = "lex") -> str:
    d = df[df["ring_order"] == ro]
    cols = ["connectivity", "rule", "K", "N"]
    lines = ["| cell | mixer (ii) min–max | mixer (iii) min–max | max \\|S\\| | A1 CX (ii) L = 5 / 7 / 9 | "
             "A1 CX (iii) L = 5 / 7 / 9 | A0 layer CX | layer (ii) / A0 layer |",
             "|---|---|---|---|---|---|---|---|"]
    for (conn, rule, K, N), g in sorted(d.groupby(cols), key=lambda kv: (kv[0][0] != "ring", kv[0][1], kv[0][2], kv[0][3])):
        a1ii = " / ".join(_fmt(_med(g[f"a1_L{L}_cx_ii"])) for L in DEPTHS)
        a1iii = " / ".join(_fmt(_med(g[f"a1_L{L}_cx_iii"])) for L in DEPTHS)
        lines.append(f"| {conn} · {rule} · K{K} · N{N} | {_fmt(g.mixer_cx_ii.min())}–{_fmt(g.mixer_cx_ii.max())} | "
                     f"{_fmt(g.mixer_cx_iii.min())}–{_fmt(g.mixer_cx_iii.max())} | {int(g.S_max.max())} | {a1ii} | "
                     f"{a1iii} | {_fmt(_med(g.a0_layer_cx))} | {_med(g.layer_cx_ii / g.a0_layer_cx):.1f}× |")
    return "\n".join(lines)


def _d9_table(df: pd.DataFrame) -> str:
    lines = ["| cell | mixer (ii) lex | mixer (ii) rank | mixer (iii) lex | mixer (iii) rank | d̄ lex | d̄ rank | "
             "\\|S\\|̄ lex | \\|S\\|̄ rank |", "|---|---|---|---|---|---|---|---|---|"]
    cols = ["connectivity", "rule", "K", "N"]
    for (conn, rule, K, N), g in sorted(df.groupby(cols), key=lambda kv: (kv[0][0] != "ring", kv[0][1], kv[0][2], kv[0][3])):
        a, b = g[g.ring_order == "lex"], g[g.ring_order == "rank"]
        lines.append(f"| {conn} · {rule} · K{K} · N{N} | {_fmt(_med(a.mixer_cx_ii))} | {_fmt(_med(b.mixer_cx_ii))} | "
                     f"{_fmt(_med(a.mixer_cx_iii))} | {_fmt(_med(b.mixer_cx_iii))} | {_med(a.d_mean):.2f} | "
                     f"{_med(b.d_mean):.2f} | {_med(a.S_mean):.2f} | {_med(b.S_mean):.2f} |")
    return "\n".join(lines)


def _vale_table(ts: int) -> str:
    lines = ["| \\|S\\| = k | k1 / k2 | CNOTs (list) | 16(k+1) − 40 (Vale bound) | T-count (list) | T-depth (list) |",
             "|---|---|---|---|---|---|"]
    for k in range(0, 14):
        gl = dc.mc_su2(list(range(k)), k, "rx", dc.Angle(1.0, 0))
        k1, k2 = (k + 1) // 2, k // 2
        bound = 16 * (k + 1) - 40 if k >= 2 else "—"
        lines.append(f"| {k} | {k1} / {k2} | {dc.cnot_count(gl)} | {bound} | {dc.t_count(gl, ts)} | {dc.t_depth(gl, ts)} |")
    return "\n".join(lines)


def _c1_summary(root=None) -> str:
    out = []
    for engine in ("numpy", "cudaq"):
        p = tables_dir(root) / f"c1_operator_{engine}.parquet"
        if not p.exists():
            out.append(f"- `{engine}`: not run (`gsp mixer c1 --engine {engine}`).")
            continue
        d = pd.read_parquet(p)
        n_sec = len(d.drop_duplicates(["scope", "rule", "K", "connectivity"]))
        out.append(f"- **`{engine}`** ({p.name}): {n_sec} distinct (sector file, connectivity) × variants × 2 β "
                   f"= {len(d)} checks.")
        for keys, g in d.groupby(["connectivity", "ring_order", "symmetrized", "decomposed"]):
            conn, ro, sym, dec = keys
            tag = f"{conn}, {ro}" + (", symmetrized" if sym else "") + (", decomposed (iii)" if dec else "")
            out.append(f"  - {tag}: {len(g)} checks, n = {g.n.min()}–{g.n.max()}, K = {g.K.min()}–{g.K.max()}: "
                       f"max leakage **{g.leakage.max():.1e}**, max block error **{g.block_err.max():.1e}**, "
                       f"max column leak norm {g.leak_norm.max():.1e}")
        ok = bool((d.leakage <= 1e-13).all() and (d.block_err <= 1e-12).all())
        out.append(f"  - all within C1 (leakage ≤ 1e-13, block error ≤ 1e-12): **{ok}**")
    return "\n".join(out)


def _timing(root=None) -> str:
    p = tables_dir(root) / "mixer_timing.json"
    if not p.exists():
        return "Not measured yet (`gsp mixer time`, GPU)."
    t = json.loads(p.read_text())
    rows = pd.DataFrame(t["rows"])
    lines = [f"Measured {t.get('when', '?')} on {t.get('gpu_name', '?')} (driver {t.get('driver_version', '?')}), "
             f"{t.get('cudaq_version', '?')}, target {t.get('target', '?')}/{t.get('target_option', '?')}; one GPU "
             "process. A1 circuit = star prep + L × (cost layer + mixer layer), native multi-controlled gates, "
             "`observe` of the boosted H_obj. Medians over the instances timed per cell "
             f"({rows.groupby('cell').inst_id.nunique().max()} per cell, q = 1.5, lex order).",
             "",
             "- **interp** = the interpreter kernel (`gsp.circuits.program`; compiled once per process, the "
             "*build* column is the CPU encoding of the gate list); **builder** = PLAN §2.4's `cudaq.make_kernel` "
             "route (*build* = kernel construction, *first observe* includes this circuit's JIT; `get_state` "
             "re-compiles a builder kernel on every call).",
             "",
             "| cell | n | K | L | gates | CNOT (ii) | engine | build (s) | first observe (s) | observe (ms) | "
             "get_state (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    g = rows.groupby(["cell", "n", "K", "L", "engine"], sort=False).agg(
        gates=("n_gates", "median"), cx=("cx_ii", "median"), build=("build_s", "median"),
        first=("first_observe_s", "median"), obs=("observe_median_s", "median"),
        gs=("get_state_median_s", "median")).reset_index()
    for r in g.itertuples():
        lines.append(f"| {r.cell} | {r.n} | {r.K} | {r.L} | {_fmt(r.gates)} | {_fmt(r.cx)} | {r.engine} | "
                     f"{r.build:.3f} | {r.first:.3f} | {1e3 * r.obs:.2f} | {1e3 * r.gs:.1f} |")
    top = g[g.L == g.L.max()]
    proj = []
    for r in top.itertuples():
        it = (2 * r.L + 1) * r.obs
        proj.append(f"{r.cell} {r.engine}: {it:.2f} s per FD iteration, {300 * it / 60:.1f} min per 300 iterations"
                    f" (+ {1e3 * r.gs:.0f} ms per logger `get_state`)")
    lines += ["", f"Projection at L = {int(g.L.max())} (forward FD = 2L + 1 `observe` calls per iteration, §1.5), "
              "before any logger or optimizer overhead:", ""] + [f"- {x}" for x in proj]
    if "energy_diff_vs_interp" in rows:
        dmax = rows["energy_diff_vs_interp"].dropna().max()
        lines += ["", f"The two engines' first `observe` energies agree to max |ΔE| = {dmax:.1e} (same angles)."]
    return "\n".join(lines)


def markdown(root=None) -> str:
    from ..store.paths import tables_dir as _td
    p = _td(root) / "mixer_counts.parquet"
    df = pd.read_parquet(p) if p.exists() else write_counts(root)
    ts = tcount.t_syn()
    # the 696 check
    o = tp.synthetic_ring_order(10, 12, 5)
    c696 = tp.circuit_counts(pr.circuit_from_order(o, 10, "ring"))
    # verification numbers (CPU, seconds)
    vale = [vf.mc_su2_error(kind, k, open_controls=True) for kind in ("rx", "ry") for k in range(1, 10)]
    trans = [vf.transition_errors(n) for n in range(2, 8)]
    ii = [vf.ii_errors(n) for n in (3, 4, 5, 6)]
    lines = [
        "# Mixer counts: the compiled preserving mixer, (ii) / (iii) / T per layer (S3)",
        "",
        "Generated by `gsp mixer report` (`gsp/compile/report.py`) from `results/tables/mixer_counts.parquet` "
        "(`gsp mixer counts`), the C1 tables (`gsp mixer c1`) and the GPU timing (`gsp mixer time`). "
        "PLAN §2, §5 S3; D-1; D-9 open (lex is the default order, rank is shown for comparison).",
        "",
        "## What is counted",
        "",
        "- **Transition** (§2.1): W (X on the ones of u, CNOT ladder from the pivot k0 = min D), a multi-controlled "
        "R_x (mixer) or R_y (start state) on k0 with **open** controls on a greedy minimum hitting set S, then W†.",
        "- **(ii) device count, reported** (Eq. 4.10, V18): `2(d−1) + 6(n−2) + 2 = 6n + 2d − 12` CNOTs per transition, "
        "with all n − 1 non-pivot qubits as controls and n − 2 reusable clean ancillas (relative-phase Toffoli chain + "
        "controlled rotation). Counted on the explicit V18 gate list (`decompose.transition_ii`), which equals the "
        "formula for every n ≤ 14, d ≤ n (tested).",
        "- **(iii) simulated-equivalent count**: W plus the **ancilla-free** multi-controlled SU(2) of Vale et al. 2024 "
        "(arXiv:2302.06377, Theorem 3 / Corollary 2) on the |S| controls the simulation actually uses. Counted on the "
        "explicit gate list (`decompose.transition_iii`).",
        "- **Cost layer**: 2 CNOTs per ZZ term (dense QUBO: n(n−1)/2 terms); **layer = cost + mixer**. The A0 layer "
        "(cost + X mixer) costs the same 2·n_ZZ CNOTs.",
        "- **Start state** (§2.3): K − 1 R_y transitions (the star), counted like mixer transitions; the X gates "
        "preparing u_1 are free.",
        f"- **T companion** (V20): T_syn = ⌈3 log₂(1/ε)⌉ = **{ts}** at ε = 1e-3 per arbitrary rotation. (ii): "
        "`2(n−2)·c_Tof + 2·T_syn` per transition with c_Tof = 7, T-depth `2(n−2)·d_Tof + 2·T_syn` with d_Tof = 1; "
        "cost layer n_rot·T_syn, T-depth (χ'(K_n) + 1)·T_syn; a layer's transitions serialize. (iii): T-count and "
        "ASAP T-depth read off the explicit list (T/T† and R_y(±π/4) = 1, Clifford = 0, any other rotation = T_syn).",
        "- **No cross-transition gate cancellation is attempted** (e.g. W† of one transition against W of the next, "
        "or the X-conjugations of consecutive open controls). Every count here is an upper bound in that sense.",
        "",
        "## The (iii) construction: Vale et al. 2024, as implemented",
        "",
        "C^k(R) for R ∈ {R_x, R_y, R_z} (SU(2) with a real-valued diagonal) on the k + 1 qubits only:",
        "",
        "- split the controls into k1 = ⌈k/2⌉ and k2 = ⌊k/2⌋; time order `MCX(k1) · A · MCX(k2) · A† · MCX(k1) · A · "
        "MCX(k2) · A†` on the target, so the all-ones action is (A† X A X)² = R and every other control pattern gives "
        "the identity (the paper's Eqs. 3, 4, 6);",
        "- A = R_z(−θ/4) for R_z(θ), A = R_y(−θ/4) for R_y(θ); R_x(θ) = H R_z(θ) H (Lemma 2);",
        "- each MCX borrows the other group as **dirty** ancillas (Iten et al. 2016, Lemma 8): k_i ≥ 3 controls cost "
        "8k_i − 6 CNOTs (Barenco's Toffoli chain; the two Toffolis on the target exact, 6 CNOTs; every ancilla-targeting "
        "Toffoli the 3-CNOT Margolus gate, consecutive pairs sharing their outer halves, 4 CNOTs per pair); k_i = 2 is "
        "one exact Toffoli (6), k_i = 1 a CNOT, k_i = 0 an unconditional X (so C¹(R) costs 2 CNOTs).",
        "",
        "**CNOT formula:** C(k) = 2c(⌈k/2⌉) + 2c(⌊k/2⌋), c(0) = 0, c(1) = 1, c(2) = 6, c(m ≥ 3) = 8m − 6. "
        "It equals the paper's Theorem 3 bound 16(k+1) − 40 = 16k − 24 for k ≥ 6 and is below it for k ≤ 5. "
        "A transition's (iii) count is `2(d − 1) + C(|S|)`.",
        "",
        "This is the paper's construction reproduced from its text (Fig. 7 structure, Eqs. 6 and 16, the 8k − 6 MCX of "
        "[Iten, Lemma 8] as the paper describes it). The general-SU(2) variant (20n − 38) is not needed: every gate "
        "here is an R_x or R_y. The paper's printed Eq. 16 for R_x reads H(R_z(−θ/4) σx R_z(θ/4) σx)² H, which "
        "multiplies out to R_x(−θ) in the standard convention; the sign used here is the one that gives R_x(+θ) "
        "(verified numerically).",
        "",
        _vale_table(ts),
        "",
        "## Verification (numpy, `gsp.compile.verify`)",
        "",
        f"- Vale C^k(R_x / R_y), open controls, against the ideal multi-controlled gate, k = 1…9: max |ΔU| = "
        f"**{max(r['err'] for r in vale):.1e}**; the CNOT count of every list equals the formula.",
        f"- Full transition (iii) against the native transition (full 2^n unitaries, random u, v, S, R_x and R_y), "
        f"n = 2…7, 10 cases each: max |ΔU| = **{max(r['max_err'] for r in trans):.1e}**.",
        f"- (ii) V18 construction against the native transition with all n − 1 open controls, n = 3…6: ancilla-zero "
        f"block error **{max(r['block_err'] for r in ii):.1e}**, leakage out of the ancilla-zero subspace "
        f"**{max(r['leak'] for r in ii):.1e}**.",
        "",
        "## The 696 check (V18)",
        "",
        f"Synthetic sector, n = 10, K = 12, ring (lex order) with every edge at Hamming distance d = 5 "
        f"(`transpile.synthetic_ring_order(10, 12, 5)` = {o.tolist()}): the (ii) counter gives "
        f"**{c696['mixer']['cx_ii']} CNOTs per ring layer** = 12 × 58 (V18: 696). Its T companion "
        f"{c696['mixer']['t_ii']:,} T and T-depth {c696['mixer']['tdepth_ii']:,} = V20's 12 × 172 = 2,064 and 12 × 76 = 912. "
        f"(iii) on the same sector: {c696['mixer']['cx_iii']} CNOTs (|S|̄ = {c696['mixer']['S_mean']:.2f}).",
        "",
        "## Gates per layer at every confined cell (median over the cell's instances; lex order)",
        "",
        "Instances: every accepted draw × 3 q (the K = 24 cells: the k24_eligible draws). Violation-rule sectors are "
        "shared by the three q, so the mixer counts repeat per draw; the cost layer is per instance.",
        "",
        _cell_table(df, "lex"),
        "",
        "Spread, whole-circuit totals and the ratio to an A0 layer (lex):",
        "",
        _range_table(df, "lex"),
        "",
        "A1 CX = start state + L layers (the input of §1.3's gate-matched A0 depths, fixed in S9).",
        "",
        "## D-9 (open): ring over the lexicographic list vs the GA ranking",
        "",
        "The order changes which pairs are coupled, so d, S and every count change. Medians over instances:",
        "",
        _d9_table(df),
        "",
        "## C1 operator level on the real S2 sectors (n ≤ 12)",
        "",
        "Every sector file of the N = 4–6 cells (ring: K = 6, 8, 12, 24 violation and K = 12 objective; complete: "
        "K = 12 violation); one mixer layer at β ∈ {0.613, −2.1}; every kept basis string as input. "
        "Leakage = max |U[x, u_j]| over non-kept x; block error = max |U[u_i, u_j] − U_dense[i, j]| against the dense "
        "ordered product in K × K. Engines: `numpy` = the gate list on `gsp.compile.npsim`; `cudaq` = the same list on "
        "the GPU interpreter kernel, the input |u_j> made by exact X gates (the builder engine, tested separately, "
        "uses ry(π·bit), whose cos(π/2) = 6.1e-17 is its leakage floor). Variants: lex and rank order, the "
        "symmetrized layer (lex), and the explicit (iii) circuit (`simulate_decomposed`, lex) at n ≤ 10.",
        "",
        _c1_summary(root),
        "",
        "## Timing (GPU, sequential)",
        "",
        _timing(root),
        "",
    ]
    return "\n".join(lines)


def write(root=None):
    p = reports_dir() / "mixer_counts.md"
    atomic_write_bytes(p, markdown(root).encode())
    return p
