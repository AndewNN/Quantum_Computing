"""The TQE rerun (reports/tqe_rerun_plan.md): the SCBX x CU paper's AR_F appendix study, Exp 1-4, on the harness.

SP = A0 (penalty QAOA, X mixer), SC = A1 (ring, violation-ranked brute-force top-K, `sector_source="bf"`), both on
the boosted circuit (fp64, lex ring order, GSP signs), q 1.5, restart r = 0 (one random init per instance). The
instances are the paper's draws e = 0-9 at every N (old recipe, no acceptance filter, N = 3 included), built ad hoc
(`instances.adhoc`), so their restart seeds are written into every spec here (the completed-work formula of
PLAN §1.1, the one the frozen seed table holds) and never derived at run time.

    Exp1  group-wise alpha (the paper's hand-chosen factors)  random init   weight decay 0.01
    Exp2  instance-wise alpha (Jh boost, the default)         random init   0.01
    Exp3  instance-wise                                        linear ramp   0.01
    Exp4  instance-wise                                        linear ramp   0

Exp1 runs only where a group-wise alpha exists: SP N 3-8 (paper Table), SC N 3-7 (paper) and N 8 = 1300
(CUDA/run_approx_1.sh). No alpha is extrapolated to N 9-10.

Priority order of the queue (the approved plan: paper grid -> SC K24 -> SP N9-10 -> SC K48), with SC K12 at N 8-10
right after the paper grid; within a part, Exp by Exp. `exps` restricts the Exps planned (the ramp of Exp3/4 waits
for Sensei's pick of (dbeta, dgamma); `ramp` sets it).

`split` assigns the specs to machines greedily in priority order (each spec to the machine that would finish it
first under the time model), so every machine works through the same priority front.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..store.paths import reports_dir

LAMS = (0.0005, 0.005, 0.05, 0.5, 5.0)
DEPTHS = (5, 7, 9)
DRAWS = tuple(range(10))
Q = 1.5
RAMP_DEFAULT = (1.5, 3.0)                    # the paper's (dbeta, dgamma), plan default; Sensei may re-pick

# Paper Table (SP, X mixer): alpha[lam][N]
ALPHA_SP = {
    0.0005: {3: 20000, 4: 10000, 5: 5000, 6: 2500, 7: 1800, 8: 1400},
    0.005: {3: 15000, 4: 7500, 5: 3750, 6: 2000, 7: 1250, 8: 800},
    0.05: {3: 1000, 4: 500, 5: 400, 6: 400, 7: 400, 8: 250},
    0.5: {3: 100, 4: 50, 5: 40, 6: 40, 7: 40, 8: 25},
    5.0: {3: 10, 4: 5, 5: 4, 6: 4, 7: 4, 8: 2.5},
}
# SC (H_obj): paper N 3-7, N 8 from CUDA/run_approx_1.sh
ALPHA_SC = {3: 30000, 4: 18000, 5: 10500, 6: 5250, 7: 2600, 8: 1300}

# Mean iterations per Exp of the published runs (tqe_rerun_plan.md "likely"); the bound is 300.
ITERS = {"SP": {1: 280, 2: 160, 3: 225, 4: 225}, "SC": {1: 250, 2: 240, 3: 185, 4: 185}}

RING = {"connectivity": "ring", "rule": "violation"}


def restart_seed_formula(N: int, e: int, r: int = 0) -> int:
    """PLAN §1.1: 4001 + 4099 e + 4999 N + 5099 r (what results/instances/seed_table.csv holds)."""
    return 4001 + 4099 * int(e) + 4999 * int(N) + 5099 * int(r)


def inst_id(N: int, e: int) -> str:
    return f"N{N:02d}e{e:03d}q{Q}"


def exp_kw(method: str, exp: int, N: int, lam=None, ramp=RAMP_DEFAULT) -> dict | None:
    """The config kwargs of one Exp (None: Exp1 has no group-wise alpha here)."""
    kw = {"restart": 0, "circuit_boosted": True}
    if method == "SC":
        kw.update(ring_order="lex", sector_source="bf")
    else:
        kw["lam"] = float(lam)
    if exp == 1:
        a = ALPHA_SP.get(float(lam), {}).get(N) if method == "SP" else ALPHA_SC.get(N)
        if a is None:
            return None
        kw["alpha"] = float(a)
    if exp in (3, 4):
        kw.update(init="ramp", ramp=[float(ramp[0]), float(ramp[1])])
    if exp == 4:
        kw["weight_decay"] = 0.0
    return kw


def _spec(method, exp, N, e, L, kw, K=None, part=""):
    arm = "A0" if method == "SP" else "A1"
    cell = None if method == "SP" else dict(RING, K=int(K))
    lam = kw.get("lam")
    lab = (f"TQE {part} Exp{exp} {method} N{N}" + (f" K{K}" if K else f" lam{lam:g}") + f" L{L} e{e}")
    return {"arm": arm, "inst_id": inst_id(N, e), "cell": cell, "effort": int(L), "kw": kw,
            "seed": restart_seed_formula(N, e, 0), "N": int(N), "draw_id": inst_id(N, e)[:7], "q": Q,
            "lam": lam, "restart": 0, "tag": f"tqe_exp{exp}", "label": lab, "tier": part,
            "method": method, "exp": int(exp), "K": None if K is None else int(K)}


def part_specs(part: str, exps, ramp=RAMP_DEFAULT) -> list[dict]:
    """One part of the priority order; within it: Exp, method (SC first: fewer, longer runs), N, lam/K, L, e."""
    grids = {
        "paper": [("SC", 12, range(3, 8)), ("SP", None, range(3, 9))],
        "sc_k12_ext": [("SC", 12, range(8, 11))],
        "sc_k24": [("SC", 24, range(3, 11))],
        "sp_ext": [("SP", None, range(9, 11))],
        "sc_k48": [("SC", 48, range(8, 11))],
    }[part]
    out = []
    for exp in exps:
        for method, K, Ns in grids:
            for N in Ns:
                for lam in (LAMS if method == "SP" else (None,)):
                    kw = exp_kw(method, exp, N, lam, ramp)
                    if kw is None:
                        continue
                    for L in DEPTHS:
                        for e in DRAWS:
                            out.append(_spec(method, exp, N, e, L, dict(kw), K, part))
    return out


PARTS = ("paper", "sc_k12_ext", "sc_k24", "sp_ext", "sc_k48")


def all_specs(exps=(1, 2, 3, 4), parts=PARTS, ramp=RAMP_DEFAULT) -> list[dict]:
    out = []
    for p in parts:
        out += part_specs(p, exps, ramp)
    return out


def check_root(root=None) -> Path:
    """The TQE results root must hold no frozen instances: every TQE instance is ad hoc there (`inst_adhoc` is
    hashed), so a planner and a runner on the same root agree on every run_id."""
    from ..store.paths import instances_dir, results_root
    if list(instances_dir(root).glob("inst_*.npz")):
        raise ValueError(f"{instances_dir(root)} holds frozen instances: plan / run the TQE queues on a separate "
                         "root (GSP_RESULTS=.../results_tqe)")
    return results_root(root)


def resolve_specs(specs: list[dict], root=None) -> list[dict]:
    """Fill run_id through the arm's config (ad hoc instances, the spec's seed) on the TQE root."""
    from ..arms.base import make_arm
    from ..instances.adhoc import load_any
    check_root(root)
    arms, insts, out, seen = {}, {}, [], set()
    for s in specs:
        arm = arms.setdefault(s["arm"], make_arm(s["arm"]))
        if s["inst_id"] not in insts:
            insts[s["inst_id"]] = load_any(s["inst_id"], root)
        inst, _, adhoc = insts[s["inst_id"]]
        cfg = arm.config(inst, s["cell"], s["effort"], s["seed"], adhoc=adhoc, root=root, **s["kw"])
        if cfg.run_id in seen:
            raise ValueError(f"duplicate run_id {cfg.run_id} ({s['label']})")
        seen.add(cfg.run_id)
        out.append(dict(s, run_id=cfg.run_id, effort_kind=cfg.effort_kind, encoding=cfg.encoding))
    return out


# --- time model (results/tables/tqe_bench.json: RTX 4080, fp64, boosted circuit) ------------------------------------
def _bench(path=None):
    path = Path(path) if path else reports_dir().parent / "results" / "tables" / "tqe_bench.json"
    rows = json.loads(Path(path).read_text())
    return {(r["arm"], r["N"], r["K"], r["L"]): (r["obs"], r["log"]) for r in rows}


def iter_seconds(method: str, N: int, L: int, K=None, bench=None) -> float:
    """s / iteration = (2L + 1) observe + logger; L 7 interpolated between 5 and 9; N 3 = 0.9 x N 4."""
    b = bench or _bench()
    key_n = max(N, 4)
    k = None if method == "SP" else K

    if L in (5, 9):
        obs, log = b[(method, key_n, k, L)]
        t = (2 * L + 1) * obs + log
    else:
        o5, g5 = b[(method, key_n, k, 5)]
        o9, g9 = b[(method, key_n, k, 9)]
        w = (L - 5) / 4
        t = (2 * L + 1) * (o5 + w * (o9 - o5)) + (g5 + w * (g9 - g5))
    return t * (0.9 if N < 4 else 1.0)


def est_seconds(spec: dict, bench=None, bound: bool = False) -> float:
    it = 300 if bound else ITERS[spec["method"]][spec["exp"]]
    return it * iter_seconds(spec["method"], spec["N"], spec["effort"], spec.get("K"), bench) + 3.0


def split(specs: list[dict], speeds: dict, bench=None) -> dict:
    """Greedy in priority order: each spec goes to the machine with the earliest finish time (speed = relative
    throughput, local 4080 = 1.0). Returns {machine: [specs]} (priority order kept) and the est. hours."""
    b = bench or _bench()
    t = {m: 0.0 for m in speeds}
    out = {m: [] for m in speeds}
    for s in specs:
        c = est_seconds(s, b)
        m = min(speeds, key=lambda k: t[k] + c / speeds[k])
        t[m] += c / speeds[m]
        out[m].append(s)
    return out, {m: t[m] / 3600 for m in t}


TQE_KEYS = ("run_id", "arm", "inst_id", "cell", "effort", "kw", "seed", "effort_kind", "encoding", "N", "draw_id",
            "q", "lam", "restart", "K", "method", "exp", "tier", "tag", "label")


def write_lines(specs: list[dict], path) -> None:
    """Atomic (temp + rename): a dynamic runner reading the file sees the old or the new version, never half."""
    from ..store.io import atomic_write_bytes
    body = "".join(json.dumps({k: s.get(k) for k in TQE_KEYS}, separators=(",", ":")) + "\n" for s in specs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(Path(path), body.encode())


def summary(specs: list[dict], bench=None) -> dict:
    b = bench or _bench()
    by = {}
    for s in specs:
        k = (s["tier"], s["method"], s["exp"])
        n, h, hb = by.get(k, (0, 0.0, 0.0))
        by[k] = (n + 1, h + est_seconds(s, b) / 3600, hb + est_seconds(s, b, True) / 3600)
    return {f"{k[0]} {k[1]} Exp{k[2]}": {"runs": v[0], "gpu_h_likely": round(v[1], 2), "gpu_h_bound": round(v[2], 2)}
            for k, v in by.items()}


__all__ = ["all_specs", "part_specs", "resolve_specs", "split", "summary", "est_seconds", "exp_kw",
           "restart_seed_formula", "ALPHA_SP", "ALPHA_SC", "PARTS", "RAMP_DEFAULT"]
