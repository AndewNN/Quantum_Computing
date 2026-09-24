"""Rule D1, the headline comparison (PLAN §1.7; deck: the statistic is AR_F, budgets are powers of ten of
cumulative charged two-qubit gate executions g2q_ii).

At each N and budget B = 10^k:
1. Value of configuration c on instance i: AR_F at the last t with g2q_ii(t) <= B (no such t: no data).
   * median-of-R (primary): every restart truncated at B, then the median over the R restarts;
   * best-of-R: every restart truncated at B / R, then the max over the R restarts.
   A configuration needs all R restarts of the arm (R_EXPECTED) on an instance, else it has no data there.
   The per-draw value is the mean over the draw's three q (all three required).
2. Arm value at (N, B): the configuration c*(B) = argmax_c median over draws, among the arm's configurations with
   data at B on every draw; ties go to the first configuration key in sorted order. c*(B) is reported.
3. Primary pairs: A1 against A0, A2c, A3, A4. Exact two-sided Wilcoxon signed-rank test on the per-draw
   differences (`ranktests.signed_rank_exact`), Holm-corrected over the 4 pairs (family size fixed at 4; a pair
   without data counts as never rejected).
4. Gap threshold delta* = max{0.02, gbar_A, gbar_B}; gbar_arm(N) = median over the N's instances of the median
   adjacent spacing of the sorted distinct normalized objective values the arm can return (the D1 sector for the
   confined arms A1 / A2c / A4, the band for the penalty arms A0 / A3).
5. Outcomes: test and gap (|median d| >= delta*) -> "ordering" (direction = sign of the median); test only ->
   "below one rung"; otherwise "indistinguishable" (a gap without the test is flagged `gap_only`, still
   indistinguishable). A pair without data at B: "no data".
6. The 6 other pairs among {A0, A2c, A3, A4} are reported uncorrected and labelled exploratory. A6 (and A2p) are
   in no pair. Both R pairings are reported.

Input: a *curves* table, one row per run: arm, N, draw_id, inst_id, q, cfg (configuration key), cfg_label,
restart, g2q (array, non-decreasing), ar (array). `curves_from_store` builds it from the registry and the
trajectories with the D1 filters (`d1_spec`); tests build it synthetically.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .ranktests import holm, signed_rank_exact

PRIMARY_PAIRS = (("A1", "A0"), ("A1", "A2c"), ("A1", "A3"), ("A1", "A4"))
EXPLORATORY_PAIRS = (("A0", "A2c"), ("A0", "A3"), ("A0", "A4"), ("A2c", "A3"), ("A2c", "A4"), ("A3", "A4"))
D1_ARMS = ("A0", "A1", "A2c", "A3", "A4")
R_EXPECTED = {"A0": 5, "A1": 5, "A2c": 1, "A2p": 1, "A3": 1, "A4": 1, "A6": 1}
CONFINED_ARMS = ("A1", "A2c", "A4")
Q_VALUES = (1.0, 1.5, 3.0)
GAP_FLOOR = 0.02
ALPHA = 0.05
PAIRINGS = ("median", "best")
OUTCOMES = ("ordering", "below one rung", "indistinguishable", "no data")
D1_CELL = {"connectivity": "ring", "rule": "violation", "K": 12}   # the fixed A1 version (deck)


# --- 1. values ----------------------------------------------------------------------------------------------
def value_at(g2q, ar, B: float) -> float:
    """AR_F at the last t with g2q(t) <= B; NaN if none."""
    g2q = np.asarray(g2q)
    i = int(np.searchsorted(g2q, B, side="right")) - 1
    return float(ar[i]) if i >= 0 else float("nan")


def values_at(g2q, ar, budgets) -> np.ndarray:
    g2q = np.asarray(g2q)
    ar = np.asarray(ar, dtype=np.float64)
    i = np.searchsorted(g2q, np.asarray(budgets, dtype=np.float64), side="right") - 1
    out = np.full(i.shape, np.nan)
    ok = i >= 0
    out[ok] = ar[i[ok]]
    return out


def budget_grid(curves: pd.DataFrame) -> list[int]:
    """Exponents k of B = 10^k spanning the data: floor(log10(smallest positive g2q)) .. ceil(log10(largest))."""
    pos = [float(g[g > 0].min()) for g in curves["g2q"] if np.any(np.asarray(g) > 0)]
    if not pos:
        return [0]
    hi = max(float(np.max(g)) for g in curves["g2q"])
    return list(range(int(math.floor(math.log10(min(pos)))), int(math.ceil(math.log10(hi))) + 1))


def instance_values(curves: pd.DataFrame, budgets, pairing: str) -> pd.DataFrame:
    """One row per (arm, N, draw_id, inst_id, q, cfg) with the configuration's value at every budget (columns =
    budgets) under `pairing`."""
    if pairing not in PAIRINGS:
        raise ValueError(pairing)
    budgets = np.asarray(budgets, dtype=np.float64)
    keys = ["arm", "N", "draw_id", "inst_id", "q", "cfg"]
    rows = []
    for key, g in curves.groupby(keys, sort=True):
        arm = key[0]
        R = R_EXPECTED.get(arm, 1)
        if sorted(int(r) for r in g["restart"]) != list(range(R)):
            vals = np.full(budgets.size, np.nan)
        else:
            B_eff = budgets if pairing == "median" else budgets / R
            V = np.stack([values_at(gq, a, B_eff) for gq, a in zip(g["g2q"], g["ar"])])   # (R, nB)
            bad = np.any(~np.isfinite(V), axis=0)
            vals = np.median(V, axis=0) if pairing == "median" else np.max(V, axis=0)
            vals[bad] = np.nan
        rows.append(list(key) + [g["cfg_label"].iloc[0]] + list(vals))
    return pd.DataFrame(rows, columns=keys + ["cfg_label"] + list(budgets))


def draw_values(inst_vals: pd.DataFrame, budgets, q_values=Q_VALUES) -> pd.DataFrame:
    """Mean over the draw's q values (all of q_values required; None = whatever is present)."""
    budgets = list(np.asarray(budgets, dtype=np.float64))
    rows = []
    for key, g in inst_vals.groupby(["arm", "N", "draw_id", "cfg"], sort=True):
        V = g[budgets].to_numpy(dtype=np.float64)
        if q_values is not None and sorted(float(x) for x in g["q"]) != sorted(float(x) for x in q_values):
            v = np.full(len(budgets), np.nan)
        else:
            v = V.mean(axis=0)                 # NaN if any q lacks data
        rows.append(list(key) + [g["cfg_label"].iloc[0]] + list(v))
    return pd.DataFrame(rows, columns=["arm", "N", "draw_id", "cfg", "cfg_label"] + budgets)


# --- 2. the arm value ---------------------------------------------------------------------------------------
def arm_choice(dv: pd.DataFrame, arm: str, N: int, B: float, draws) -> tuple:
    """(c*, cfg_label, per-draw values as a Series indexed by draw_id, table of candidate medians)."""
    sub = dv[(dv["arm"] == arm) & (dv["N"] == N)]
    if sub.empty:
        return None, None, None, pd.DataFrame()
    piv = sub.pivot_table(index="cfg", columns="draw_id", values=B, aggfunc="first").reindex(columns=list(draws))
    ok = piv.notna().all(axis=1)
    med = piv.median(axis=1)
    tab = pd.DataFrame({"cfg": piv.index, "median": med.values, "has_all_draws": ok.values})
    labels = sub.drop_duplicates("cfg").set_index("cfg")["cfg_label"]
    tab["cfg_label"] = tab["cfg"].map(labels)
    cand = tab[tab["has_all_draws"]].sort_values("cfg", kind="stable")
    if cand.empty:
        return None, None, None, tab
    best = cand.loc[cand["median"].idxmax()] if cand["median"].notna().any() else None
    if best is None:
        return None, None, None, tab
    c = best["cfg"]
    tab["chosen"] = tab["cfg"] == c
    return c, labels[c], piv.loc[c], tab


# --- 3-5. pairs ---------------------------------------------------------------------------------------------
def delta_star(gbar_a: float, gbar_b: float, floor: float = GAP_FLOOR) -> float:
    vals = [floor] + [g for g in (gbar_a, gbar_b) if g is not None and np.isfinite(g)]
    return float(max(vals))


def outcome(test_holds: bool, gap_holds: bool) -> str:
    if test_holds and gap_holds:
        return "ordering"
    if test_holds:
        return "below one rung"
    return "indistinguishable"


def compare(va: pd.Series | None, vb: pd.Series | None) -> dict:
    """Paired statistics of two per-draw value vectors (draw-aligned)."""
    if va is None or vb is None:
        return {"n_draws": 0}
    d = (va - vb).dropna().to_numpy(dtype=np.float64)
    if d.size == 0:
        return {"n_draws": 0}
    sr = signed_rank_exact(d)
    q1, q3 = np.percentile(d, [25, 75])
    return {"n_draws": int(d.size), "n_nonzero": sr.m, "median_d": float(np.median(d)), "mean_d": float(d.mean()),
            "iqr_lo": float(q1), "iqr_hi": float(q3), "t_plus": sr.t_plus, "t_minus": sr.t_minus,
            "ties": sr.ties, "p_raw": sr.p}


def run_d1(curves: pd.DataFrame, gbar, budgets_k=None, pairings=PAIRINGS, q_values=Q_VALUES,
           primary=PRIMARY_PAIRS, exploratory=EXPLORATORY_PAIRS, alpha: float = ALPHA) -> dict:
    """Rule D1 over every N of `curves`. gbar: callable (N, arm) -> gbar, or a dict {(N, arm): gbar}.
    Returns {"pairs": DataFrame (one row per N, B, pairing, pair), "choices": DataFrame (candidate
    configurations and c*)}."""
    gb = gbar if callable(gbar) else (lambda N, arm: gbar.get((N, arm), float("nan")))
    pair_rows, choice_rows = [], []
    for N, cN in curves.groupby("N", sort=True):
        ks = budget_grid(cN) if budgets_k is None else list(budgets_k)
        budgets = [float(10.0 ** k) for k in ks]
        draws = sorted(cN["draw_id"].unique())
        for pairing in pairings:
            dv = draw_values(instance_values(cN, budgets, pairing), budgets, q_values)
            for k, B in zip(ks, budgets):
                arms = {}
                for arm in sorted(set(a for p in tuple(primary) + tuple(exploratory) for a in p)):
                    c, lab, v, tab = arm_choice(dv, arm, int(N), B, draws)
                    arms[arm] = (c, lab, v)
                    if not tab.empty:
                        t = tab.copy()
                        t.insert(0, "arm", arm)
                        t.insert(0, "pairing", pairing)
                        t.insert(0, "B", B)
                        t.insert(0, "k", k)
                        t.insert(0, "N", int(N))
                        choice_rows.append(t)
                for family, pairs in (("primary", primary), ("exploratory", exploratory)):
                    block = []
                    for a, b in pairs:
                        ca, la, va = arms.get(a, (None, None, None))
                        cb, lb, vb = arms.get(b, (None, None, None))
                        st = compare(va, vb)
                        ga, gbb = gb(int(N), a), gb(int(N), b)
                        row = {"N": int(N), "k": k, "B": B, "pairing": pairing, "family": family, "arm_a": a,
                               "arm_b": b, "cfg_a": la, "cfg_b": lb, "gbar_a": ga, "gbar_b": gbb,
                               "delta_star": delta_star(ga, gbb),
                               "value_a": float(va.median()) if va is not None else np.nan,
                               "value_b": float(vb.median()) if vb is not None else np.nan}
                        row.update(st)
                        block.append(row)
                    p = np.array([r.get("p_raw", np.nan) if r["n_draws"] else np.nan for r in block])
                    adj = holm(p, m=len(pairs)) if family == "primary" else p
                    for r, pa in zip(block, adj):
                        r["p_adj"] = float(pa)
                        if not r["n_draws"]:
                            r.update(test_holds=False, gap_holds=False, gap_only=False, outcome="no data",
                                     direction="")
                        else:
                            th = bool(pa < alpha)
                            gh = bool(abs(r["median_d"]) >= r["delta_star"])
                            r.update(test_holds=th, gap_holds=gh, gap_only=bool(gh and not th),
                                     outcome=outcome(th, gh),
                                     direction=("" if r["median_d"] == 0 else
                                                (f"{r['arm_a']}>{r['arm_b']}" if r["median_d"] > 0
                                                 else f"{r['arm_b']}>{r['arm_a']}")))
                        pair_rows.append(r)
    pairs_df = pd.DataFrame(pair_rows)
    choices_df = pd.concat(choice_rows, ignore_index=True) if choice_rows else pd.DataFrame()
    return {"pairs": pairs_df, "choices": choices_df}


# --- gbar ---------------------------------------------------------------------------------------------------
def gbar_instance(inst_id: str, arm: str, root=None, cell=D1_CELL) -> float:
    """The ladder spacing of what one arm can return on one instance: the D1 sector for the confined arms, the
    band for the penalty arms (`rulers.ladder_gap`, the S1 definition)."""
    from ..instances.instance import load_rulers
    from ..instances.rulers import ladder_gap
    rul = load_rulers(inst_id, root)
    if arm in CONFINED_ARMS:
        from ..sectors.select import load_sector
        sec = load_sector(inst_id, cell["rule"], int(cell["K"]), root)
        return ladder_gap(np.asarray(sec.E, dtype=np.float64), rul.E_min, rul.E_max)[0]
    return float(rul.gap_band)


def gbar_table(N_values=(4, 5, 6, 7), arms=D1_ARMS, root=None) -> pd.DataFrame:
    """gbar_arm(N) = median over the N's frozen instances (all draws and q)."""
    from ..instances.instance import load_instances_table
    inst = load_instances_table(root)
    rows = []
    for N in N_values:
        ids = sorted(inst[inst["N"] == N]["inst_id"])
        for arm in arms:
            g = [gbar_instance(i, arm, root) for i in ids]
            rows.append({"N": int(N), "arm": arm, "kind": "sector" if arm in CONFINED_ARMS else "band",
                         "gbar": float(np.nanmedian(g)), "n_inst": len(ids)})
    return pd.DataFrame(rows)


# --- the store ----------------------------------------------------------------------------------------------
_NOT_CFG = ("inst_id", "restart", "seed", "seed_ga", "inst_adhoc",
            "N", "q", "draw_id", "run_dir")       # instance / restart identity and the columns added for filtering


def config_key(rec: dict) -> str:
    from ..store.records import config_of
    cfg = {k: v for k, v in config_of(rec).items() if k not in _NOT_CFG}
    cfg = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in cfg.items()}
    return json.dumps(cfg, sort_keys=True, default=str)


def config_label(rec: dict) -> str:
    lab = f"{rec.get('effort_kind')}={int(rec.get('effort'))}"
    if rec.get("schedule") not in (None, "") and not (isinstance(rec.get("schedule"), float)):
        lab += f"|{rec['schedule']}"
    return lab


def d1_spec(lam_star: dict | None = None, ring_order: str = "lex", a4_cap: dict | None = None,
            db_step_units: str = "normalized") -> dict:
    """Registry filters of the runs that enter D1 (a filter value None = the key must be missing / NaN).
    lam_star: {N: lambda*(N)} for the penalty arms (S9); None = no lambda filter (S5 exercise only).
    a4_cap: {N: recursion cap} (S9): A4 enters with effort = cap(N) only (S8: the smoke / timing records have other
    efforts; one trajectory per instance); None = no effort filter. db_step_units: A4's step convention ("normalized",
    PLAN §1.5 as corrected in S8b; "plan" only to read the flag's runs)."""
    sector = dict(D1_CELL, ring_order=ring_order, sector_source="ga")
    return {"A0": {"init": "random", "_lam": lam_star},
            "A1": dict(sector, init="random"),
            "A2c": dict(sector, ramp_sign=-1),
            "A3": {"_lam": lam_star, "metric": "M1", "n_steps": 300},     # S7: the sweep config (no smoke / timing runs)
            "A4": dict(D1_CELL, connectivity="adaptive", sector_source="ga", ring_order=ring_order,
                       step_units=db_step_units, _effort=a4_cap)}


def _match(df: pd.DataFrame, arm: str, flt: dict) -> pd.DataFrame:
    sub = df[(df["arm"] == arm) & (df["status"] == "done")]
    if "inst_adhoc" in sub:
        sub = sub[sub["inst_adhoc"].isna() | (sub["inst_adhoc"] == False)]   # noqa: E712 (frozen instances only)
    if "evidence" in sub:                         # S9a: informational runs (arms.base.EVIDENCE_KEY) enter no rule
        sub = sub[sub["evidence"].isna()]
    for k, v in flt.items():
        if k == "_lam":
            if v:
                sub = sub[[bool(np.isclose(l, v.get(int(n), np.nan))) for l, n in zip(sub["lam"], sub["N"])]]
            continue
        if k == "_effort":
            if v:
                sub = sub[[int(e) == int(v.get(int(n), -1)) for e, n in zip(sub["effort"], sub["N"])]]
            continue
        if k not in sub:
            return sub.iloc[0:0]
        sub = sub[sub[k] == v]
    return sub


def curves_from_store(root=None, spec: dict | None = None, arms=D1_ARMS) -> pd.DataFrame:
    from ..store.index import load_registry
    from ..store.io import load_npz
    from ..store.paths import runs_dir
    reg = load_registry(root).copy()
    if reg.empty:
        return pd.DataFrame(columns=["arm", "N", "draw_id", "inst_id", "q", "cfg", "cfg_label", "restart", "g2q",
                                     "ar", "run_id"])
    from ..store.ids import parse_inst_id
    parsed = [parse_inst_id(i) for i in reg["inst_id"]]
    reg["N"] = [p["N"] for p in parsed]
    reg["q"] = [p["q"] for p in parsed]
    reg["draw_id"] = [i.split("q")[0] for i in reg["inst_id"]]
    spec = d1_spec() if spec is None else spec
    rows = []
    base = runs_dir(root)
    for arm in arms:
        if arm not in spec:
            continue
        sub = _match(reg, arm, spec[arm])
        for _, r in sub.iterrows():
            rec = {k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}
            tr = load_npz(base / r["run_dir"] / "trajectory.npz")
            rows.append({"arm": arm, "N": int(r["N"]), "draw_id": r["draw_id"], "inst_id": r["inst_id"],
                         "q": float(r["q"]), "cfg": config_key(rec), "cfg_label": config_label(rec),
                         "restart": int(r["restart"]), "g2q": np.asarray(tr["g2q_ii"], dtype=np.int64),
                         "ar": np.asarray(tr["ar_f"], dtype=np.float64), "run_id": r["run_id"]})
    return pd.DataFrame(rows)
