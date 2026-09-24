"""`gsp plan`: the OFAT envelope of PLAN §1.2 -> run specs (queue files, one JSON object per line), with the run
counts per arm and per cell printed before anything runs (S6).

A **spec** is what `gsp run` needs to call `Arm.run(inst_id, cell, effort, **kw)`, plus the `run_id` that the arm's
own `config` hashes it to. The run_id is computed here through the registered arm (`arms.base.make_arm`), so the
runner skips done runs without rebuilding anything, checks at run time that the plan is not stale, and
`gsp missing` matches the store by run_id.

Registry-driven expansion of `configs/envelope.yaml` (`arms:`, and `optional_tier:` with optional=True):
  penalty arms   (encoding penalty: A0, A2p, A3, A6, A3d) at every N of the arm's `N` list, on every accepted
                 draw x q (90 instances per N), lam = lambda*(N)
  confined arms  (A1, A2c: `cells: all_21`) on the 21 confined cells; A4 (`cells: ring_rows_18`) on the 18 ring-row
                 cells with the connectivity label `adaptive` (same sector file, PLAN §1.2)
  efforts        `depths` (A0, A1, A3; A0 also the gate-matched depths L0(N, L1) of §1.3 at the N of the baseline
                 cells), `ramp_depths` x `schedules` (A2), `recursion_steps` -> one trajectory to the cap (A4, A6)
  restarts       r = 0 .. R-1 with R = the arm's `restarts` (A0's gate-matched depths: `gate_matched_restarts`,
                 default R, cut 2 of §1.8)
  kwargs         candidates restart / lam / schedule_tag / ring_order / step_units (S8b: A4 / A6, from the envelope
                 entry), filtered by the signature of the registered arm's `config`, so the arms of S7 / S8 plug in
                 through `make_arm` alone
A spec is a **placeholder** (counted, never queued) when its arm is not registered yet, or when it needs an S9
input that is not set: lambda*(N), the gate-matched depths, the recursion cap. Those come from envelope.yaml
(`lambda_star`, `gate_matched_depths`, `recursion_cap`) or override it through `PlanParams`.

The lambda pilot of §1.4 (`pilot_specs`) is a separate set: A0, depth 7, r = 0, q = 1.5, the five lambda, the first
10 accepted draws at N <= 7 and the first 5 at N >= 8.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .._version import HARNESS_VERSION
from ..store.io import atomic_write_bytes
from ..store.paths import configs_dir, queues_dir

PLAN_SCHEMA = 1
KW_CANDIDATES = ("restart", "lam", "schedule_tag", "ring_order", "step_units")
PENALTY_ROW = "penalty|N{N}"
REASONS = ("arm_not_registered", "lambda_star", "gate_matched_depth", "recursion_cap")


class PlanError(ValueError):
    pass


# --- inputs --------------------------------------------------------------------------------------------------
def envelope_path() -> Path:
    return configs_dir() / "envelope.yaml"


def load_envelope(path=None) -> dict:
    p = Path(path) if path else envelope_path()
    env = yaml.safe_load(p.read_text())
    env["_path"] = str(p)
    env["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return env


@dataclass
class PlanParams:
    """The S9 inputs (None / missing = placeholder) and the D-9 ring order."""
    lam_star: dict = field(default_factory=dict)          # N -> lambda*(N)
    gate_matched: dict = field(default_factory=dict)      # N -> {L1: L0}
    recursion_cap: dict = field(default_factory=dict)     # arm -> {N: cap}
    ring_order: str = "lex"                               # D-9 (open, O-1): lex until Sensei decides

    @classmethod
    def from_envelope(cls, env: dict, **override) -> "PlanParams":
        ls = env.get("lambda_star") or {}
        gm = env.get("gate_matched_depths") or {}
        rc = env.get("recursion_cap") or {}
        p = cls(lam_star={int(k): float(v) for k, v in ls.items() if v is not None},
                gate_matched={int(N): {int(a): int(b) for a, b in (m or {}).items()} for N, m in gm.items()},
                recursion_cap={str(a): {int(N): int(k) for N, k in (m or {}).items()} for a, m in rc.items()})
        for k, v in override.items():
            if v is None:
                continue
            if k in ("lam_star",):
                p.lam_star.update(v)
            elif k == "gate_matched":
                for N, m in v.items():
                    p.gate_matched.setdefault(int(N), {}).update(m)
            elif k == "recursion_cap":
                for a, m in v.items():
                    p.recursion_cap.setdefault(a, {}).update(m)
            elif k == "ring_order":
                p.ring_order = v
            else:
                raise KeyError(k)
        return p

    def as_dict(self) -> dict:
        return {"lam_star": {str(k): v for k, v in sorted(self.lam_star.items())},
                "gate_matched": {str(N): {str(a): b for a, b in sorted(m.items())}
                                 for N, m in sorted(self.gate_matched.items())},
                "recursion_cap": {a: {str(N): k for N, k in sorted(m.items())}
                                  for a, m in sorted(self.recursion_cap.items())},
                "ring_order": self.ring_order}


@dataclass
class PlanFilter:
    """Subset selectors (None = everything). `draws` = the first k accepted draws of each N (of the k24 subset on
    the K = 24 cells); `restarts` = at most R restarts (r < R); `efforts` / `schedules` / `axes` / `tags` whitelist."""
    arms: tuple | None = None
    N: tuple | None = None
    draws: int | None = None
    q: tuple | None = None
    axes: tuple | None = None
    efforts: tuple | None = None
    restarts: int | None = None
    schedules: tuple | None = None
    tags: tuple | None = None

    def active(self) -> bool:
        return any(getattr(self, k) is not None for k in self.__dataclass_fields__)

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items() if v is not None}


# --- helpers ---------------------------------------------------------------------------------------------------
def cell_label(conn: str, rule: str, K: int, N: int) -> str:
    return f"{conn}|{rule}|K{int(K)}|N{int(N)}"


def effort_label(spec: dict) -> str:
    kind, e = spec.get("effort_kind"), spec.get("effort")
    if spec.get("tag") == "gate_matched" and e is None:
        return f"L0({spec.get('matched_L1')})?"
    if e is None:
        return "k=cap?" if kind == "steps" else "?"
    if kind == "ramp_depth":
        return f"p{e}/{spec.get('schedule')}"
    if kind == "steps":
        return f"k{e}"
    return f"L{e}" + (f"(~L1={spec['matched_L1']})" if spec.get("tag") == "gate_matched" else "")


def _config_params(arm) -> tuple[set, bool]:
    sig = inspect.signature(type(arm).config)
    names = {p.name for p in sig.parameters.values()}
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return names, var_kw


@contextmanager
def _cached_sectors():
    """Memoize `sectors.select.load_sector` while a plan computes run_ids (A1 / A2c configs read the sector file
    for seed_ga; ~60k specs share ~1800 (instance, rule, K) files). Restored on exit; run-time loading is unchanged."""
    from ..sectors import select
    orig = select.load_sector
    cache: dict = {}

    def cached(inst_id, rule, K, root=None):
        key = (inst_id, rule, int(K), None if root is None else str(root))
        if key not in cache:
            cache[key] = orig(inst_id, rule, K, root)
        return cache[key]

    select.load_sector = cached
    try:
        yield
    finally:
        select.load_sector = orig


class _Instances:
    def __init__(self, root=None):
        self.root = root
        self._c: dict = {}

    def get(self, inst_id):
        if inst_id not in self._c:
            from ..instances.adhoc import load_any
            inst, _, adhoc = load_any(inst_id, self.root)
            self._c[inst_id] = (inst, adhoc)
        return self._c[inst_id]


# --- expansion -------------------------------------------------------------------------------------------------
def _instances_table(root=None) -> pd.DataFrame:
    from ..instances.instance import load_instances_table
    it = load_instances_table(root)
    return it.sort_values(["N", "accept_rank", "q"], kind="stable").reset_index(drop=True)


def _select_instances(it: pd.DataFrame, N: int, flt: PlanFilter, k24_only: bool = False,
                      draws: int | None = None, qs=None) -> list:
    g = it[it["N"] == N]
    if k24_only:
        g = g[g["k24_eligible"].astype(bool)]
    order = list(dict.fromkeys(g["draw_id"]))
    for k in (draws, flt.draws):
        if k is not None:
            order = order[:int(k)]
    g = g[g["draw_id"].isin(order)]
    for qq in (qs, flt.q):
        if qq is not None:
            g = g[g["q"].isin([float(x) for x in qq])]
    return list(g.itertuples(index=False))


def _arm_entries(env: dict, optional: bool) -> list:
    """(name, entry, tier) in envelope order; optional-tier entries without effort keys inherit from `like`."""
    out = [(k, dict(v), "main") for k, v in (env.get("arms") or {}).items()]
    if optional:
        arms = env.get("arms") or {}
        for k, v in (env.get("optional_tier") or {}).items():
            v = dict(v or {})
            if "like" in v:
                base = dict(arms[v["like"]])
                base.update({a: b for a, b in v.items() if a != "like"})
                v = base
            if "encoding" not in v:
                continue                              # A5 / A3c: notes only (S14), nothing to expand
            out.append((k, v, "optional"))
    return out


def _efforts(name: str, entry: dict, N: int, params: PlanParams, gm_N: set):
    """[(effort_kind, effort | None, schedule | None, tag, matched_L1 | None, restarts, reason | None)]."""
    R = int(entry.get("restarts", 1))
    out = []
    if "depths" in entry:
        for L in entry["depths"]:
            out.append(("depth", int(L), None, "main", None, R, None))
        if entry.get("gate_matched_depths") is not None and N in gm_N:
            Rg = int(entry.get("gate_matched_restarts", R))
            m = params.gate_matched.get(N, {})
            for L1 in entry["depths"]:
                L0 = m.get(int(L1))
                out.append(("depth", None if L0 is None else int(L0), None, "gate_matched", int(L1), Rg,
                            None if L0 is not None else "gate_matched_depth"))
    elif "ramp_depths" in entry:
        from ..train.schedules import SCHEDULES
        for tag, val in (entry.get("schedules") or {}).items():
            if tag not in SCHEDULES or tuple(map(float, val)) != tuple(SCHEDULES[tag]):
                raise PlanError(f"{name}: envelope schedule {tag}={val} != train.schedules {SCHEDULES.get(tag)}")
        for p in entry["ramp_depths"]:
            for tag in (entry.get("schedules") or {}):
                out.append(("ramp_depth", int(p), tag, "main", None, R, None))
    elif "recursion_steps" in entry:
        cap = params.recursion_cap.get(name, {}).get(N)
        out.append(("steps", None if cap is None else int(cap), None, "main", None, R,
                    None if cap is not None else "recursion_cap"))
    else:
        raise PlanError(f"{name}: no depths / ramp_depths / recursion_steps in the envelope entry")
    return out


def expand(env: dict | None = None, params: PlanParams | None = None, flt: PlanFilter | None = None,
           optional: bool = False, root=None) -> list[dict]:
    """Every spec of the envelope (runnable and placeholder), without run_ids (see `resolve`)."""
    env = env or load_envelope()
    params = params or PlanParams.from_envelope(env)
    flt = flt or PlanFilter()
    from ..arms.base import registered_arms
    reg = set(registered_arms())
    it = _instances_table(root)
    cells = list(env["confined_cells"])
    gm_N = {int(c["N"]) for c in cells if c.get("axis") == "baseline"}
    cut = env.get("scaling_cut") or {}
    cut_N = {int(x) for x in cut.get("N", [])}
    specs = []
    for name, entry, tier in _arm_entries(env, optional):
        if flt.arms is not None and name not in flt.arms:
            continue
        enc = entry["encoding"]
        units = []                                   # (row, cell_label, axis, cell dict | None, N, instances)
        if enc == "penalty":
            for N in entry["N"]:
                N = int(N)
                if flt.N is not None and N not in flt.N:
                    continue
                kw_cut = {"draws": cut.get("draws"), "qs": cut.get("q")} if N in cut_N else {}
                insts = _select_instances(it, N, flt, **kw_cut)
                lab = PENALTY_ROW.format(N=N)
                units.append((lab, lab, "penalty", None, N, insts))
        else:
            which = entry.get("cells")
            label = entry.get("connectivity_label")
            for c in cells:
                if which == "ring_rows_18" and c["connectivity"] != "ring":
                    continue
                N = int(c["N"])
                if flt.N is not None and N not in flt.N:
                    continue
                if flt.axes is not None and c.get("axis") not in flt.axes:
                    continue
                conn = label or c["connectivity"]
                row = cell_label(c["connectivity"], c["rule"], c["K"], N)
                insts = _select_instances(it, N, flt, k24_only=c.get("draws") == "k24_eligible")
                units.append((row, cell_label(conn, c["rule"], c["K"], N), c.get("axis"),
                              {"connectivity": conn, "rule": c["rule"], "K": int(c["K"])}, N, insts))
        for row, lab, axis, cell, N, insts in units:
            if enc == "penalty" and flt.axes is not None and "penalty" not in flt.axes:
                continue
            lam = params.lam_star.get(N) if enc == "penalty" else None
            for kind, eff, sched, tag, L1, R, reason in _efforts(name, entry, N, params, gm_N):
                if flt.tags is not None and tag not in flt.tags:
                    continue
                if flt.efforts is not None and eff not in flt.efforts and not (eff is None and tag == "gate_matched"
                                                                              and L1 in flt.efforts):
                    continue
                if flt.schedules is not None and sched is not None and sched not in flt.schedules:
                    continue
                Rn = R if flt.restarts is None else min(R, int(flt.restarts))
                reasons = [] if name in reg else ["arm_not_registered"]
                if enc == "penalty" and lam is None:
                    reasons.append("lambda_star")
                if reason:
                    reasons.append(reason)
                for inst in insts:
                    for r in range(Rn):
                        kw = {"restart": r}
                        if enc == "penalty":
                            kw["lam"] = lam
                        if sched is not None:
                            kw["schedule_tag"] = sched
                        if enc == "confined":
                            kw["ring_order"] = params.ring_order
                        if entry.get("step_units") is not None:        # S8b: A4 / A6 (DB-QITE step convention)
                            kw["step_units"] = str(entry["step_units"])
                        specs.append({
                            "arm": name, "encoding": enc, "tier": tier, "secondary": bool(entry.get("secondary", False)),
                            "N": N, "draw_id": inst.draw_id, "e": int(inst.e), "inst_id": inst.inst_id,
                            "q": float(inst.q), "accept_rank": int(inst.accept_rank),
                            "row": row, "cell_label": lab, "axis": axis, "cell": cell,
                            "effort_kind": kind, "effort": eff, "schedule": sched, "restart": r, "lam": lam,
                            "ring_order": params.ring_order if enc == "confined" else None,
                            "tag": tag, "matched_L1": L1, "kw": kw, "run_id": None,
                            "placeholder": ",".join(reasons)})
    return specs


def pilot_specs(env: dict | None = None, flt: PlanFilter | None = None, root=None) -> list[dict]:
    """The lambda pilot of PLAN §1.4 (S9 runs it; configs equal to sweep configs are reused through run_id)."""
    env = env or load_envelope()
    flt = flt or PlanFilter()
    pl = env["lambda_pilot"]
    it = _instances_table(root)
    specs = []
    for N in env["arms"][pl["arm"]]["N"]:
        N = int(N)
        if flt.N is not None and N not in flt.N:
            continue
        k = int(pl["draws"]["N_le_7" if N <= 7 else "N_ge_8"])
        insts = _select_instances(it, N, flt, draws=k, qs=[pl["q"]])
        lab = PENALTY_ROW.format(N=N)
        for lam in pl["lam"]:
            for inst in insts:
                specs.append({
                    "arm": pl["arm"], "encoding": "penalty", "tier": "pilot", "secondary": False, "N": N,
                    "draw_id": inst.draw_id, "e": int(inst.e), "inst_id": inst.inst_id, "q": float(inst.q),
                    "accept_rank": int(inst.accept_rank), "row": lab, "cell_label": lab, "axis": "penalty",
                    "cell": None, "effort_kind": "depth", "effort": int(pl["depth"]), "schedule": None,
                    "restart": int(pl["restart"]), "lam": float(lam), "ring_order": None, "tag": "pilot",
                    "matched_L1": None, "kw": {"restart": int(pl["restart"]), "lam": float(lam)}, "run_id": None,
                    "placeholder": ""})
    return specs


def resolve(specs: list[dict], root=None, log=None) -> list[dict]:
    """Fill `run_id` (and the arm's effort_kind) of every runnable spec through the registered arm's `config`,
    filter `kw` to what that `config` accepts, and merge specs that hash to the same run_id (the same
    configuration planned twice, e.g. a gate-matched depth equal to 5 / 7 / 9). Two specs whose planned kwargs
    differ but hash to one run_id are an error (a kwarg the arm dropped)."""
    from ..arms.base import make_arm
    arms, params = {}, {}
    insts = _Instances(root)
    out, seen = [], {}
    with _cached_sectors():
        for s in specs:
            if s["placeholder"]:
                out.append(s)
                continue
            a = s["arm"]
            if a not in arms:
                arms[a] = make_arm(a)
                params[a] = _config_params(arms[a])
            arm = arms[a]
            names, var_kw = params[a]
            kw = s["kw"] if var_kw else {k: v for k, v in s["kw"].items() if k in names}
            inst, adhoc = insts.get(s["inst_id"])
            cfg = arm.config(inst, s["cell"], s["effort"], None, adhoc=adhoc, root=root, **kw)
            s = dict(s, kw=kw, run_id=cfg.run_id, effort_kind=cfg.effort_kind, planned_kw=s["kw"])
            prev = seen.get(s["run_id"])
            if prev is not None:
                if prev["planned_kw"] != s["planned_kw"] or prev["inst_id"] != s["inst_id"]:
                    raise PlanError(f"{a}: specs {prev['planned_kw']} and {s['planned_kw']} hash to one run_id "
                                    f"{s['run_id']} (a planned kwarg is ignored by {a}.config)")
                prev["tag"] = prev["tag"] if prev["tag"] == s["tag"] else f"{prev['tag']}+{s['tag']}"
                continue
            seen[s["run_id"]] = s
            out.append(s)
    for s in out:
        s.pop("planned_kw", None)
    if log:
        log(f"resolved {sum(1 for s in out if s['run_id'])} run_ids")
    return out


ARM_ORDER = ("A0", "A1", "A2p", "A2c", "A3", "A4", "A6", "A3d")


def _arm_rank(a):
    return ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER)


def order_specs(specs: list[dict], env: dict, order: str = "draw") -> list[dict]:
    """`draw`: N, draw (accept order), q, then arm / cell / effort / restart / schedule -- every arm of a draw
    together, so a partial sweep has complete draws. `arm`: arm first."""
    cells = [cell_label(c["connectivity"], c["rule"], c["K"], c["N"]) for c in env["confined_cells"]]
    row_rank = {r: i for i, r in enumerate(cells)}

    def key(s):
        rr = row_rank.get(s["row"], -1)
        eff = s["effort"] if s["effort"] is not None else 10 ** 9
        common = (eff, s.get("matched_L1") or 0, s["restart"], s.get("schedule") or "", s.get("lam") or 0.0)
        if order == "arm":
            return (_arm_rank(s["arm"]), s["N"], rr, s["accept_rank"], s["q"]) + common
        return (s["N"], s["accept_rank"], s["q"], _arm_rank(s["arm"]), rr) + common

    if order not in ("draw", "arm"):
        raise ValueError(order)
    return sorted(specs, key=key)


def build_plan(env=None, params=None, flt=None, optional=False, pilot=False, order="draw", resolve_ids=True,
               root=None, log=None) -> list[dict]:
    env = env or load_envelope()
    specs = pilot_specs(env, flt, root) if pilot else expand(env, params, flt, optional, root)
    if resolve_ids:
        specs = resolve(specs, root, log)
    return order_specs(specs, env, order)


# --- counts ----------------------------------------------------------------------------------------------------
def counts(specs: list[dict]) -> dict:
    df = pd.DataFrame([{"arm": s["arm"], "row": s["row"], "tag": s["tag"], "tier": s["tier"],
                        "runnable": not s["placeholder"], "placeholder": s["placeholder"]} for s in specs])
    if df.empty:
        return {"total": 0, "runnable": 0, "per_arm": {}, "per_cell": {}}
    per_arm = {}
    for a, g in df.groupby("arm", sort=False):
        ph = g[~g.runnable]
        per_arm[a] = {"total": int(len(g)), "runnable": int(g.runnable.sum()), "placeholder": int(len(ph)),
                      "by_tag": {t: int(n) for t, n in g.groupby("tag").size().items()},
                      "placeholder_reasons": {r: int(n) for r, n in ph.groupby("placeholder").size().items()},
                      "tier": str(g["tier"].iloc[0])}
    per_cell = {}
    for (row, a), g in df.groupby(["row", "arm"], sort=False):
        per_cell.setdefault(row, {})[a] = {"total": int(len(g)), "runnable": int(g.runnable.sum())}
    return {"total": int(len(df)), "runnable": int(df.runnable.sum()), "per_arm": per_arm, "per_cell": per_cell}


def format_counts(c: dict, env: dict, title: str = "") -> str:
    lines = [title] if title else []
    if not c["total"]:
        return "\n".join(lines + ["(no specs)"])
    arms = sorted(c["per_arm"], key=_arm_rank)
    lines.append(f"{'arm':<5} {'total':>8} {'runnable':>9} {'placeholder':>12}  tags / placeholder reasons")
    for a in arms:
        p = c["per_arm"][a]
        tags = ", ".join(f"{t} {n}" for t, n in p["by_tag"].items())
        why = "; ".join(f"{r} {n}" for r, n in p["placeholder_reasons"].items())
        tier = "" if p["tier"] == "main" else f" [{p['tier']}]"
        lines.append(f"{a:<5} {p['total']:>8} {p['runnable']:>9} {p['placeholder']:>12}  {tags}"
                     f"{' | ' + why if why else ''}{tier}")
    lines.append(f"{'all':<5} {c['total']:>8} {c['runnable']:>9} {c['total'] - c['runnable']:>12}")
    lines.append("")
    cells = [cell_label(x["connectivity"], x["rule"], x["K"], x["N"]) for x in env["confined_cells"]]
    rows = [r for r in c["per_cell"] if r.startswith("penalty")]
    rows = sorted(rows, key=lambda r: int(r.split("N")[-1])) + [r for r in cells if r in c["per_cell"]]
    rows += [r for r in c["per_cell"] if r not in rows]
    w = max(len(r) for r in rows) + 1
    lines.append("runs per cell (runnable/total where they differ):")
    lines.append(f"{'cell':<{w}}" + "".join(f"{a:>13}" for a in arms))
    for r in rows:
        vals = []
        for a in arms:
            x = c["per_cell"].get(r, {}).get(a)
            vals.append("" if x is None else (str(x["total"]) if x["runnable"] == x["total"]
                                              else f"{x['runnable']}/{x['total']}"))
        lines.append(f"{r:<{w}}" + "".join(f"{v:>13}" for v in vals))
    return "\n".join(lines)


# --- queue files -----------------------------------------------------------------------------------------------
QUEUE_KEYS = ("run_id", "arm", "inst_id", "cell", "effort", "kw", "effort_kind", "encoding", "N", "draw_id", "q",
              "row", "cell_label", "axis", "restart", "lam", "schedule", "ring_order", "tag", "matched_L1", "tier")


def spec_line(s: dict) -> str:
    return json.dumps({k: s.get(k) for k in QUEUE_KEYS}, sort_keys=False, separators=(",", ":"))


def write_queue(specs: list[dict], path, meta: dict | None = None) -> dict:
    """Runnable specs -> JSONL (one per line); `{stem}.plan.json` beside it holds the provenance and counts."""
    path = Path(path)
    run = [s for s in specs if not s["placeholder"]]
    if any(s["run_id"] is None for s in run):
        raise PlanError("unresolved spec (call resolve first)")
    atomic_write_bytes(path, ("\n".join(spec_line(s) for s in run) + ("\n" if run else "")).encode())
    side = dict(meta or {})
    side.update({"plan_schema": PLAN_SCHEMA, "queue": str(path), "n_runs": len(run),
                 "n_placeholders": len(specs) - len(run),
                 "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "harness_version": HARNESS_VERSION})
    atomic_write_bytes(sidecar_path(path), json.dumps(side, indent=1, sort_keys=True).encode())
    return side


def sidecar_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".plan.json")


def read_queue(path) -> list[dict]:
    specs = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlanError(f"{path}:{i}: not JSON ({exc})") from None
        for k in ("arm", "inst_id", "effort"):
            if k not in s:
                raise PlanError(f"{path}:{i}: spec misses {k!r}")
        s.setdefault("cell", None)
        s.setdefault("kw", {})
        s.setdefault("run_id", None)
        s.setdefault("placeholder", "")
        s["_line"] = i
        specs.append(s)
    return specs


def default_queue_path(name: str, root=None) -> Path:
    return queues_dir(root) / f"{name}.jsonl"
