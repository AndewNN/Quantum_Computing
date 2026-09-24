"""Arms A4 (confined DB-QITE) and A6 (penalty DB-QITE), PLAN §1.5-§1.7, D-2 (S8).

The recursion (`circuits.dbqite`): U_{k+1} = e^{i r H} U_k e^{i r |0..0><0..0|} U_k^dagger e^{-i r H} U_k, r = sqrt(s),
three copies of U_k per step, never projected back into the sector (Rule C1).
  A4  start U_0 = the S3 star prep over the cell's kept strings (the GA sector file of the ring-row cell, lex order:
      the star centre u_1 = the first kept string), H = H_obj, sigma_H over the K kept strings. Cell connectivity is
      the label "adaptive" (the step couples every pair; PLAN §1.2).
  A6  start U_0 = H^n, H = H(lam) (lam = lambda*(N) from S9; 0.005 until then), sigma_H over all 2^n strings.
The loop (`run_dbqite`, observe only): at step k, the energy <H> of the compiled step U_k(s) for every s of the grid
{0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8} / sigma_H; the greedy argmin (first minimum) is kept. Every candidate energy and
every chosen s is logged. E(U_0) is measured once by observe as a diagnostic (energy drop of step 1; not charged).
The post-update logger (`DBLogger`, get_state) gives the §1.7 metrics, the variance Var(H) and p_sector.

Charging (PLAN §1.5 / §1.7): step k runs |grid| = 7 energy circuits of U_k; g2q(t) = sum_{k <= t} 7 c(U_k) with
c(U_k) of `circuits.dbqite.recursion_counts` (the 3^k growth). counts.json carries `charge_model: series` with the
per-circuit counts of U_0..U_T and the cumulative charges (`metrics.resources` reads them).

Config extras (hashed): start ("star" | "hadamard"), grid (the 7 values), grid_unit ("sigma_H"), step_units (always
written explicitly: "normalized" = the circuit carries H / sigma_H with r = sqrt(g), the default since S8b (PLAN §1.5
corrected, O-13 resolved); "plan" = the S0b wording, a flag), A4 also ring_order (the star centre; "lex") and
sector_source ("ga"). R = 1 (deterministic): restart must be 0; seed = the draw's restart seed r = 0 (the post-run
sample only). Effort = the number of recursion steps (the cap).

Leakage bookkeeping (S8b; Rule C1 state level, O-14): every step logs norm = sum |psi|^2, norm_err = norm - 1 and, for A4,
in_mass (= p_sector), out_mass (the off-sector mass, summed directly), leak_raw = 1 - p_sector (S5's estimator, which
includes the norm drift) and leak_rel = out_mass / norm (norm-free). Runs stored before S8b carry p_sector / out_mass /
norm_err only; `backfill_leakage` replays them into a sidecar `leakage.npz` (files added, nothing rewritten) and
`db_leakage(run_dir)` reads either source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..circuits import dbqite as db
from ..circuits import preserving as pr
from ..circuits.simopt import merge_x
from ..circuits.xmixer import h_start
from ..metrics.state import METRIC_KEYS, metric_context
from .base import Arm, Outcome, RunConfig, draw_of, make_extras, restart_seed
from .qaoa import sector_view

GRID_TAG = ",".join(f"{g:g}" for g in db.GRID)
DB_DEFAULTS = {"grid": GRID_TAG, "grid_unit": "sigma_H"}


def grid_values(cfg_grid: str) -> tuple:
    vals = tuple(float(v) for v in str(cfg_grid).split(","))
    if vals != db.GRID:
        raise ValueError(f"unsupported grid {cfg_grid!r} (PLAN §1.5 fixes {GRID_TAG})")
    return vals


# --- circuits of a config ------------------------------------------------------------------------------------------
def confined_db_circuit(inst, rule: str, K: int, sector_source: str = "ga", ring_order: str = "lex",
                        units: str = db.DEFAULT_STEP_UNITS, root=None) -> tuple:
    """(DBCircuit, SectorView) of A4 on one instance and cell."""
    from ..compile import transpile as tp
    from ..stats.c1 import mc_gate_count
    sv = sector_view(inst, rule, int(K), sector_source, root)
    circ = pr.build_circuit(sv, "ring", ring_order)
    prep = pr.prep_gates(circ)
    ham = db.DBHam.from_ising(inst.H_obj)
    sig = db.sigma_H(ham.diagonal(), sv.idx)
    cnt = tp.circuit_counts(circ)["prep"]
    C = db.DBCircuit(circ.n, merge_x(prep), ham, kind="confined", sigma=sig, units=units,
                     start_counts={k: int(cnt[k]) for k in db.COUNT_KEYS}, start_mc=mc_gate_count(prep),
                     start_abstract=prep,
                     meta={"K": circ.K, "ring_order": ring_order, "order": [int(x) for x in circ.order],
                           "sector_source": sv.source})
    return C, sv


def penalty_db_circuit(inst, lam: float, units: str = db.DEFAULT_STEP_UNITS) -> db.DBCircuit:
    """DBCircuit of A6: H^n start, H(lam) un-boosted, sigma_H over all 2^n strings."""
    if lam is None:
        raise ValueError("A6 needs lam (lambda*(N) from S9; 0.005 until then)")
    H = inst.hamiltonian(float(lam))
    ham = db.DBHam.from_ising(H)
    sig = db.sigma_H(ham.diagonal())
    return db.DBCircuit(H.n, h_start(H.n), ham, kind="penalty", sigma=sig, units=units,
                        start_counts={k: 0 for k in db.COUNT_KEYS}, start_mc=0, start_abstract=h_start(H.n),
                        meta={"lam": float(lam)})


# --- the loop --------------------------------------------------------------------------------------------------------
@dataclass
class DBResult:
    s: np.ndarray            # (T,) chosen step sizes (un-boosted units)
    grid_idx: np.ndarray     # (T,) index of the chosen grid value
    grid_s: np.ndarray       # (G,) the grid in s units
    E_loop: np.ndarray       # (T + 1,) observe energy of U_t (t = 0: the diagnostic E(U_0))
    E_cand: np.ndarray       # (T + 1, G) candidate energies of step t (row 0 NaN)
    wall_hist: np.ndarray    # (T + 1,) loop wall clock when psi_t was reached (logger excluded)
    logger_s: float
    n_steps: int

    def params_hist(self) -> np.ndarray:
        T = self.n_steps
        P = np.full((T + 1, max(T, 1)), np.nan)
        for t in range(1, T + 1):
            P[t, :t] = self.s[:t]
        return P


def run_dbqite(engine, grid_s, n_steps: int, logger=None, measure_e0: bool = True) -> DBResult:
    """The greedy DB-QITE loop. `engine.energy(s_list)` is the only thing the update uses (observe on the GPU);
    `logger(t, s_list)` gets a copy after every step and at t = 0, and nothing it does reaches the update."""
    grid_s = np.asarray(grid_s, dtype=np.float64)
    G, T = grid_s.size, int(n_steps)
    s_hist: list = []
    E_loop = np.full(T + 1, np.nan)
    E_cand = np.full((T + 1, G), np.nan)
    idx = np.zeros(T, dtype=np.int64)
    wall = np.zeros(T + 1)
    log_s = 0.0
    t0 = time.perf_counter()
    if measure_e0:
        E_loop[0] = engine.energy([])
    wall[0] = time.perf_counter() - t0
    if logger is not None:
        tl = time.perf_counter()
        logger(0, [])
        log_s += time.perf_counter() - tl
    for k in range(1, T + 1):
        Es = np.array([engine.energy(s_hist + [float(g)]) for g in grid_s])
        j = int(np.argmin(Es))
        s_hist.append(float(grid_s[j]))
        idx[k - 1] = j
        E_cand[k] = Es
        E_loop[k] = Es[j]
        wall[k] = time.perf_counter() - t0 - log_s
        if logger is not None:
            tl = time.perf_counter()
            logger(k, list(s_hist))
            log_s += time.perf_counter() - tl
    return DBResult(s=np.array(s_hist), grid_idx=idx, grid_s=grid_s, E_loop=E_loop, E_cand=E_cand, wall_hist=wall,
                    logger_s=log_s, n_steps=T)


class NumpyDBEngine:
    """The CPU reference engine: energies of the exact reflection formula (D-2), for tests and the example check."""

    def __init__(self, circ: db.DBCircuit, psi0: np.ndarray | None = None):
        self.circ = circ
        self.diag = circ.ham.diagonal()
        self.psi0 = db.start_state(circ.start, circ.n) if psi0 is None else np.asarray(psi0, dtype=np.complex128)
        self._cache = ((), self.psi0)

    def state(self, s_list) -> np.ndarray:
        s = tuple(float(v) for v in s_list)
        pre, psi = self._cache
        if len(pre) > len(s) or s[:len(pre)] != pre:
            pre, psi = (), self.psi0
        for j in range(len(pre), len(s)):
            psi = db.formula_step(self.diag, psi, *db.r_pair(s[j], self.circ.units, self.circ.sigma))
            if j < len(s) - 1:
                self._cache = (s[:j + 1], psi)
        return psi

    def energy(self, s_list) -> float:
        p = np.abs(self.state(s_list)) ** 2
        return float(p @ self.diag)


# --- the logger ------------------------------------------------------------------------------------------------------
class DBLogger:
    """Post-update logger: the state of U_t (get_state), the §1.7 metrics (ctx), Var(H) of the arm's H (un-boosted) and,
    for A4, p_sector / out_mass. With ctx None it logs energy and variance only (the example check)."""

    def __init__(self, circ: db.DBCircuit, ctx=None, sector_idx=None):
        self.circ = circ
        self.ctx = ctx
        self.diag = circ.ham.diagonal()
        self.sector_idx = None if sector_idx is None else np.asarray(sector_idx, dtype=np.int64)
        self.t: list = []
        self.rows: list = []
        self.last_state: np.ndarray | None = None

    def __call__(self, t: int, s_list) -> None:
        psi = self.circ.state(s_list)
        self.last_state = psi
        prob = np.abs(psi) ** 2
        row = self.ctx.evaluate(prob) if self.ctx is not None else {"energy": float(prob @ self.diag)}
        e = float(prob @ self.diag)
        row["variance"] = float(prob @ self.diag ** 2 - e ** 2)
        row.update(leakage_fields(prob, self.sector_idx))
        self.t.append(int(t))
        self.rows.append(row)

    def arrays(self) -> dict:
        keys = list(self.rows[0].keys()) if self.rows else list(METRIC_KEYS)
        out = {"t_logged": np.array(self.t, dtype=np.int64)}
        for k in keys:
            out[k] = np.array([r[k] for r in self.rows], dtype=np.float64)
        return out


def leakage_fields(prob: np.ndarray, sector_idx=None) -> dict:
    """norm, norm_err and (with a sector) in_mass, out_mass, leak_raw = 1 - in_mass, leak_rel = out_mass / norm."""
    prob = np.asarray(prob, dtype=np.float64)
    norm = float(prob.sum())
    out = {"norm": norm, "norm_err": norm - 1.0}
    if sector_idx is not None:
        mask = np.zeros(prob.size, dtype=bool)
        mask[np.asarray(sector_idx, dtype=np.int64)] = True
        ins = float(prob[mask].sum())
        om = float(prob[~mask].sum())
        out.update({"in_mass": ins, "out_mass": om, "leak_raw": 1.0 - ins, "leak_rel": om / norm if norm > 0 else np.nan})
    return out


LEAK_FIELDS = ("norm", "norm_err", "in_mass", "out_mass", "leak_raw", "leak_rel")
LEAKAGE_FILE = "leakage.npz"


def db_leakage(run_dir) -> dict | None:
    """The per-step leakage fields of a stored A4 / A6 run: from trajectory.npz (runs from S8b on) or from the replayed
    sidecar leakage.npz (`backfill_leakage`, the S8 runs). None if neither has them."""
    from pathlib import Path
    from ..store.io import load_npz
    d = Path(run_dir)
    tr = load_npz(d / "trajectory.npz")
    if "norm" in tr:
        return {"t": tr["t"], **{k: tr[k] for k in LEAK_FIELDS if k in tr}, "source": "trajectory"}
    if (d / LEAKAGE_FILE).exists():
        z = load_npz(d / LEAKAGE_FILE)
        return {"t": z["t"], **{k: z[k] for k in LEAK_FIELDS if k in z}, "source": "replay"}
    return None


def backfill_leakage(run_dir, root=None, force: bool = False) -> dict:
    """Replay every psi_t (t = 0..T) of a stored A4 / A6 run from its run.json and params, write leakage.npz with the
    LEAK_FIELDS per step (files added only), and check the replay against the stored trajectory (p_sector, out_mass,
    norm_err must agree exactly: same kernel, same arguments)."""
    from pathlib import Path
    from ..metrics.postrun import rebuild
    from ..store.io import load_npz, save_npz
    from ..store.records import read_record
    d = Path(run_dir)
    if (d / LEAKAGE_FILE).exists() and not force:
        return {"status": "exists"}
    rec = read_record(d / "run.json")
    if rec["arm"] not in ("A4", "A6") or rec.get("status") != "done":
        raise ValueError(f"{rec['run_id']}: not a done A4 / A6 run")
    tr = load_npz(d / "trajectory.npz")
    _, _, _, C, sector_idx, _ = rebuild(rec, root)
    s = [float(v) for v in np.asarray(tr["params"][-1]) if np.isfinite(v)]
    T = int(tr["t"][-1])
    rows = [leakage_fields(np.abs(C.state(s[:t])) ** 2, sector_idx) for t in range(T + 1)]
    out = {"t": np.arange(T + 1, dtype=np.int64)}
    for k in rows[0]:
        out[k] = np.array([r[k] for r in rows], dtype=np.float64)
    chk = {}
    for k, ref in (("in_mass", "p_sector"), ("out_mass", "out_mass"), ("norm_err", "norm_err")):
        if k in out and ref in tr:
            chk[f"replay_{k}_max_abs"] = float(np.max(np.abs(out[k] - np.asarray(tr[ref], dtype=np.float64))))
    save_npz(d / LEAKAGE_FILE, {**out, "run_id": np.array(rec["run_id"]), "source": np.array("replay_S8b"),
                                **{k: np.float64(v) for k, v in chk.items()}})
    return {"status": "written", "T": T, **chk}


# --- counts ----------------------------------------------------------------------------------------------------------
def db_counts(C: db.DBCircuit, res: DBResult) -> dict:
    """counts.json of an A4 / A6 run (module doc; `charge_model: series`)."""
    T, G = res.n_steps, int(res.grid_s.size)
    rc = C.counts(T)
    per_circuit = {k: int(rc[k][T]) for k in db.COUNT_KEYS}
    ch = db.charged_series(rc, G, T)
    cum = {"circuits": ch["circuits_charged"].tolist(), "cx_ii": ch["g2q_ii"].tolist(), "cx_iii": ch["g2q_iii"].tolist()}
    for k in ("t_ii", "t_iii", "tdepth_ii", "tdepth_iii"):
        per = np.asarray(rc[k][:T + 1], dtype=np.int64)
        cum[k] = np.cumsum(np.r_[0, G * per[1:]]).astype(np.int64).tolist()
    cost = C.cost_counts()
    return {"effort_unit": "step", "charge_model": "series", "grid_size": G, "circuits_per_unit": G,
            "per_circuit": per_circuit, "per_unit": {k: G * v for k, v in per_circuit.items()},
            "start": dict(C.start_counts), "layer": {k: int(cost[k]) for k in db.COUNT_KEYS},
            "phase": dict(C.phase_counts()),
            "series": {"per_circuit": {k: [int(v) for v in rc[k]] for k in db.COUNT_KEYS},
                       "cumulative": cum, "mc_gates": rc["mc_gates"], "u0_copies": rc["u0_copies"],
                       "cost_layers": rc["cost_layers"], "phases": rc["phases"]},
            "n": C.n, "T": T, "start_mc": C.start_mc, "n_zz": int(cost["n_zz"]),
            "note": ("per_circuit = U_T (the last step's circuit); per_unit = the last step's charge (|grid| x U_T); "
                     "layer = one cost exponential e^{-irH}; phase = the open-controlled phase; series = U_0..U_T and "
                     "the cumulative charges (step k runs |grid| circuits of U_k); no cross-segment cancellation")}


# --- the arms --------------------------------------------------------------------------------------------------------
def _units(cfg) -> str:
    """A stored config always names its step convention (hashed); a config without one is refused, never defaulted."""
    u = cfg.extra("step_units")
    if u not in db.STEP_UNITS:
        raise ValueError(f"{cfg.arm} config without a valid step_units ({u!r})")
    return u


class DBArm(Arm):
    effort_kind = "steps"
    start_kind = "?"

    def _extras(self, units: str, **kw) -> tuple:
        if units not in db.STEP_UNITS:
            raise ValueError(f"step_units must be one of {db.STEP_UNITS}")
        return make_extras(start=self.start_kind, step_units=units, **DB_DEFAULTS, **kw)

    def _ctx(self, cfg, inst, rulers, C, sector_idx):
        lam = cfg.lam if C.kind == "penalty" else None
        ctx = metric_context(inst, rulers, lam, sector_idx=sector_idx)
        return ctx

    def execute(self, cfg: RunConfig, inst, rulers, cell, logger: bool = True, root=None) -> Outcome:
        t0 = time.perf_counter()
        C, sector_idx = self.ansatz(cfg, inst, root)
        grid = grid_values(cfg.extra("grid"))
        if cfg.extra("grid_unit") != "sigma_H":
            raise ValueError(f"unsupported grid_unit {cfg.extra('grid_unit')!r}")
        grid_s = np.asarray(grid) / C.sigma
        ctx = self._ctx(cfg, inst, rulers, C, sector_idx)
        log = DBLogger(C, ctx, sector_idx) if logger else None
        setup_s = time.perf_counter() - t0
        res = run_dbqite(C, grid_s, int(cfg.effort), logger=log)
        t1 = time.perf_counter()
        T = res.n_steps
        s_list = res.s.tolist()
        if log is not None:
            rows = log.arrays()
            rows.pop("t_logged")
            final_psi = log.last_state
        else:
            final_psi = C.state(s_list)
            last = ctx.evaluate_state(final_psi)
            rows = {k: np.r_[np.full(T, np.nan), v] for k, v in last.items()}
        post_s = time.perf_counter() - t1
        counts = db_counts(C, res)
        ch = db.charged_series(C.counts(T), int(grid_s.size), T)
        traj = {"t": np.arange(T + 1, dtype=np.int64), "params": res.params_hist(), **ch, "wall": res.wall_hist}
        for k, v in rows.items():
            traj[k] = np.asarray(v, dtype=np.float64)
        pairs = np.array(C.pairs(s_list)).reshape(-1, 2)
        nan1 = np.array([np.nan])
        traj["s_k"] = np.r_[nan1, res.s]
        traj["g_k"] = np.r_[nan1, res.s * C.sigma]
        traj["r_H"] = np.r_[nan1, pairs[:, 0]] if T else nan1
        traj["r_rho"] = np.r_[nan1, pairs[:, 1]] if T else nan1
        traj["grid_idx"] = np.r_[-1, res.grid_idx].astype(np.int64)
        traj["E_loop"] = res.E_loop
        traj["E_cand"] = res.E_cand
        traj["energy_drop"] = np.r_[nan1, res.E_loop[:-1] - res.E_loop[1:]]
        ser = counts["series"]
        traj["cx_ii_circuit"] = np.asarray(ser["per_circuit"]["cx_ii"], dtype=np.int64)
        traj["cx_iii_circuit"] = np.asarray(ser["per_circuit"]["cx_iii"], dtype=np.int64)
        traj["mc_gates"] = np.asarray(ser["mc_gates"], dtype=np.int64)
        prefixes = [s_list[:t] for t in range(T + 1)]
        traj["sim_gates"] = np.array([C.sim_gates(p) for p in prefixes], dtype=np.int64)
        traj["tokens"] = np.array([len(C.tokens(p)) for p in prefixes], dtype=np.int64)
        traj["step_wall"] = np.r_[np.nan, np.diff(res.wall_hist)]
        loop_vs_log = (float(np.nanmax(np.abs(res.E_loop - rows["energy"]))) if log is not None else None)
        final = {k: float(v[-1]) for k, v in rows.items() if k in METRIC_KEYS or k == "p_sector"}
        metrics = dict(final)
        ar0 = float(rows["ar_f"][0])
        drops = traj["energy_drop"][1:]
        metrics.update({
            "ar_f_init": ar0 if np.isfinite(ar0) else None, "iterations": T, "steps": T,
            "circuits_charged": int(ch["circuits_charged"][-1]), "g2q_ii": int(ch["g2q_ii"][-1]),
            "g2q_iii": int(ch["g2q_iii"][-1]), "E_loop_final": float(res.E_loop[-1]),
            "E_loop_init": float(res.E_loop[0]),
            "energy_drop_total": float(res.E_loop[0] - res.E_loop[-1]),
            "energy_increase_steps": int(np.sum(drops < 0)) if T else 0,
            "s_top_chosen": int(np.sum(res.grid_idx == grid_s.size - 1)),
            "s_bottom_chosen": int(np.sum(res.grid_idx == 0)),
            "variance_final": float(rows["variance"][-1]) if "variance" in rows else None,
            "cx_ii_final_circuit": int(counts["per_circuit"]["cx_ii"]),
            "sim_gates_final": int(traj["sim_gates"][-1])})
        if "p_sector" in rows:
            leak = 1.0 - np.asarray(rows["p_sector"])
            metrics["leak_max"] = float(np.nanmax(leak))                          # signed (S8 name, kept)
            metrics["out_mass_max"] = float(np.nanmax(rows["out_mass"])) if "out_mass" in rows else None
            if "leak_rel" in rows:
                metrics["leak_raw_abs_max"] = float(np.nanmax(np.abs(rows["leak_raw"])))
                metrics["leak_rel_max"] = float(np.nanmax(rows["leak_rel"]))
        if "norm_err" in rows:
            metrics["norm_err_abs_max"] = float(np.nanmax(np.abs(rows["norm_err"])))
        per_step = np.diff(res.wall_hist)
        timings = {"setup_s": setup_s, "train_s": float(res.wall_hist[-1]), "logger_s": res.logger_s,
                   "post_s": post_s, "per_step_s": float(res.wall_hist[-1] / T) if T else None,
                   "last_step_s": float(per_step[-1]) if T else None,
                   "e0_s": float(res.wall_hist[0])}
        diag = {"n": C.n, "sigma_H": C.sigma, "step_units": C.units, "grid_size": int(grid_s.size),
                "grid_s_min": float(grid_s[0]), "grid_s_max": float(grid_s[-1]),
                "r_rho_max": float(np.max(pairs[:, 1])) if T else None,
                "r_H_max": float(np.max(pairs[:, 0])) if T else None,
                "start_mc": C.start_mc, "sim_gates_u0": C.prog.n_u0, "sim_gates_cost": C.prog.n_cost,
                "cx_ii_start": int(C.start_counts["cx_ii"]), "cx_ii_cost_layer": int(counts["layer"]["cx_ii"]),
                "cx_ii_phase": int(counts["phase"]["cx_ii"]), "cx_iii_phase": int(counts["phase"]["cx_iii"]),
                "loop_vs_logger": loop_vs_log}
        if C.kind == "confined":
            diag["K_actual"] = int(C.meta["K"])
        return Outcome(metrics=metrics, timings=timings, diagnostics=diag, trajectory=traj, counts=counts,
                       final_state=final_psi)


class A4(DBArm):
    """Confined DB-QITE: star prep over the cell's kept strings, H_obj (connectivity label "adaptive")."""
    name = "A4"
    encoding = "confined"
    start_kind = "star"

    def config(self, inst, cell, effort, seed, *, restart: int = 0, ring_order: str = "lex",
               sector_source: str = "ga", step_units: str = db.DEFAULT_STEP_UNITS, adhoc: bool = False,
               root=None) -> RunConfig:
        if int(restart) != 0:
            raise ValueError("A4 is deterministic (R = 1): restart must be 0")
        if cell is None or cell.get("connectivity") != "adaptive":
            raise ValueError("A4 runs on the ring-row cells with the connectivity label 'adaptive' (PLAN §1.2)")
        if ring_order not in pr.RING_ORDERS:
            raise ValueError(ring_order)
        seed = restart_seed(draw_of(inst.inst_id), 0, root) if seed is None else int(seed)
        sv = sector_view(inst, cell["rule"], int(cell["K"]), sector_source, root)
        return RunConfig(arm=self.name, encoding=self.encoding, inst_id=inst.inst_id, effort_kind=self.effort_kind,
                         effort=int(effort), K=int(cell["K"]), rule=cell["rule"], connectivity="adaptive",
                         restart=0, seed=seed, seed_ga=sv.seed_ga,
                         extras=self._extras(step_units, ring_order=ring_order, sector_source=sector_source,
                                             inst_adhoc=True if adhoc else None))

    def ansatz(self, cfg, inst, root=None):
        C, sv = confined_db_circuit(inst, cfg.rule, cfg.K, cfg.extra("sector_source", "ga"),
                                    cfg.extra("ring_order", "lex"), _units(cfg), root)
        return C, sv.idx


class A6(DBArm):
    """Penalty DB-QITE: H^n start, H(lam) (secondary: reported, votes in no rule)."""
    name = "A6"
    encoding = "penalty"
    start_kind = "hadamard"

    def config(self, inst, cell, effort, seed, *, lam=None, restart: int = 0, step_units: str = db.DEFAULT_STEP_UNITS,
               adhoc: bool = False, root=None) -> RunConfig:
        if int(restart) != 0:
            raise ValueError("A6 is deterministic (R = 1): restart must be 0")
        if lam is None:
            raise ValueError("A6 needs lam (lambda*(N) from S9)")
        seed = restart_seed(draw_of(inst.inst_id), 0, root) if seed is None else int(seed)
        return RunConfig(arm=self.name, encoding=self.encoding, inst_id=inst.inst_id, effort_kind=self.effort_kind,
                         effort=int(effort), restart=0, lam=float(lam), seed=seed,
                         extras=self._extras(step_units, inst_adhoc=True if adhoc else None))

    def ansatz(self, cfg, inst, root=None):
        return penalty_db_circuit(inst, cfg.lam, _units(cfg)), None


ARMS = {"A4": A4, "A6": A6}
__all__ = ["A4", "A6", "DBLogger", "DBResult", "NumpyDBEngine", "run_dbqite", "db_counts", "confined_db_circuit",
           "penalty_db_circuit", "grid_values"]
