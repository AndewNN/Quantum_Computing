"""S4 legacy checks (PLAN §5 S4 "Legacy equivalence first"): writes results/tables/s4_legacy.json.

    ~/anaconda3/envs/gsp/bin/python scripts/s4_legacy_checks.py [--skip-train-replay]

One GPU process (refuses to start if another holds the GPU). Uses the frozen copies in tests/legacy (the old
Pauli-expansion kernel `kernel_qaoa_Preserved`, `basis_T_to_pauli*`, `process_ansatz_values`) and the completed
runs in CUDA/experiments_approx_Q2_RAND_S1.0_W0.01_Jh (read-only). Checks:
  pauli_port        the numpy port of the old merged Pauli expansion (`legacy_pauli`) == the legacy copy (as maps)
  old_mixer_sign    the old kernel == the dense product prod_P exp(+i beta c_P P) in the legacy order
  equivalence       new A1 (rank ring) vs the old Pauli kernel at N = 4, K = 12: fidelity of old(gamma, beta) with
                    new(gamma, -beta) (the sign mapped) and with new(gamma, beta), random parameters
  eval_replay       the stored final parameters of the 10 N = 5, L = 5 runs through the harness circuits: AR_F vs the
                    stored AR2 (A0 on the new layered circuit; A1 on the old Pauli kernel and on the new mixer)
  init_check        the harness init (seed table, Eq. 4.11 / legacy) vs the stored first iterates
  train_replay      the harness trainer on the old kernels vs the stored trajectories (N = 5, e = 0, 1, 2)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GSP))

import gsp._threads  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP = GSP.parent / "CUDA" / "experiments_approx_Q2_RAND_S1.0_W0.01_Jh"
REP_P = EXP / "exp_L1_q1.5" / "report_Preserving12_boost_Jh_AR2.csv"
REP_X = EXP / "exp_L0.005_q1.5" / "report_X_boost_Jh_AR2.csv"
NPZ_P = EXP / "exp_L1_q1.5" / "expectation_Preserving12_boost_Jh.npz"
NPZ_X = EXP / "exp_L0.005_q1.5" / "expectation_X_boost_Jh.npz"
LAM_X = 0.005
L_REF = 5


def bkey(b) -> str:
    b = float(b)
    return str(int(b)) if b.is_integer() else repr(b)


def old_preserving(inst, K: int, L: int, circuit_boost: float = 1.0):
    """The completed work's Preserving kernel arguments for an instance (rank-order BF ring)."""
    import cudaq
    from tests.legacy import qaoaCUDAQ_instance as LI
    from tests.legacy import qaoaCUDAQ_kernels as LK
    from gsp.arms.qaoa import legacy_rank
    from gsp.circuits import legacy_pauli as lp
    n = inst.n
    order = legacy_rank(inst, K)
    bases = lp.bases_of(order, n)
    ws, cs = LK.basis_T_to_pauli_serial(bases, lp.ring_T(K), n)
    H = -LI.qubo_to_ising(inst.QU_obj, 0.0).canonicalize()
    if circuit_boost != 1.0:
        H = H * circuit_boost
    i1, c1, a2, b2, c2 = LI.process_ansatz_values(H)
    fixed = (int(n), int(L), i1, c1, a2, b2, c2, [cudaq.pauli_word(w) for w in ws], list(map(float, cs)),
             LK.reversed_str_bases_to_init_state(bases, n))
    return order, ws, cs, fixed, LK.kernel_qaoa_Preserved


def new_a1(inst, order, L: int):
    from gsp.circuits import ansatz as az, preserving as pr
    from gsp.arms.qaoa import SectorView
    sv = SectorView(n=inst.n, idx=np.sort(order), rank_idx=np.asarray(order), source="bf", seed_ga=None)
    return az.confined_ansatz(inst.H_obj, inst.boost_obj, pr.build_circuit(sv, "ring", "rank"), L)


def pauli_apply(psi, word: str, n: int):
    """P|psi> for a Pauli word (character k = qubit k = bit n-1-k of the classical index)."""
    idx = np.arange(1 << n)
    flip = 0
    phase = np.ones(1 << n, dtype=np.complex128)
    for k, c in enumerate(word):
        b = (idx >> (n - 1 - k)) & 1
        if c == "X":
            flip |= 1 << (n - 1 - k)
        elif c == "Y":
            flip |= 1 << (n - 1 - k)
            phase *= np.where(b == 0, 1j, -1j)
        elif c == "Z":
            phase *= np.where(b == 0, 1.0, -1.0)
    out = np.zeros_like(psi)
    out[idx ^ flip] = phase * psi
    return out


def check_pauli_port(backend) -> dict:
    from tests.legacy import qaoaCUDAQ_kernels as LK
    from gsp.arms.qaoa import legacy_rank
    from gsp.circuits import legacy_pauli as lp
    from gsp.instances.adhoc import load_any
    rows = []
    for iid in ("N04e004q1.5", "N05e000q1.5", "N05e002q1.5"):
        inst, _, _ = load_any(iid)
        bases = lp.bases_of(legacy_rank(inst, 12), inst.n)
        T = lp.ring_T(12)
        ws, cs = LK.basis_T_to_pauli_serial(bases, T, inst.n)
        w2, c2 = lp.merged_terms(bases, T)
        d1, d2 = dict(zip(ws, cs)), dict(zip(w2, c2))
        rows.append({"inst_id": iid, "n_words": len(ws), "same_map": d1 == d2 and c2.dtype == cs.dtype,
                     "mm_p": float(np.min(np.abs(cs))), "mm_p_port": float(lp.legacy_mm_p(legacy_rank(inst, 12), inst.n))})
    return {"rows": rows, "pass": all(r["same_map"] and r["mm_p"] == r["mm_p_port"] for r in rows)}


def check_old_mixer_sign(backend) -> dict:
    """Old kernel with gamma = 0 (mixer layers only) vs the dense product of exp(+i beta c P), legacy order."""
    from gsp.instances.instance import load_instance
    inst = load_instance("N04e004q1.5")
    n, L = inst.n, 2
    order, ws, cs, fixed, kern = old_preserving(inst, 12, L)
    rng = np.random.default_rng(11)
    out = []
    for _ in range(3):
        th = np.r_[np.zeros(L), rng.uniform(-1, 1, L)]
        s_old = backend.get_state_classical(kern, n, list(th), *fixed)
        psi = np.zeros(1 << n, dtype=np.complex128)
        psi[order] = 1 / np.sqrt(len(order))
        for ell in range(L):
            for w, c in zip(ws, cs):
                t = float(c) * th[L + ell]
                psi = np.cos(t) * psi + 1j * np.sin(t) * pauli_apply(psi, w, n)
        out.append(float(np.max(np.abs(psi - s_old))))
    return {"max_abs_diff_plus_sign": max(out), "pass": max(out) <= 1e-10,
            "note": "exp_pauli(theta, P) = exp(+i theta P) on CUDA-Q 0.15.1; the old mixer = prod over merged words"}


def check_equivalence(backend, n_samples: int = 20) -> dict:
    from gsp.instances.instance import load_instance
    from gsp.train.init import mm_i_eq411
    inst = load_instance("N04e004q1.5")
    n, L, K = inst.n, 5, 12
    order, ws, cs, fixed, kern = old_preserving(inst, K, L)
    A = new_a1(inst, order, L)
    rng = np.random.default_rng(2026)
    gam = mm_i_eq411(A.kappa_min())
    rows = []
    for scale in (0.05, 0.1, 0.3, 1.0, np.pi):
        f_map, f_nomap, leak_old, leak_new = [], [], [], []
        for _ in range(n_samples):
            g = rng.uniform(-1, 1, L) * gam
            b = rng.uniform(-1, 1, L) * scale
            s_old = backend.get_state_classical(kern, n, list(np.r_[g, b]), *fixed)
            s_new = A.state(np.r_[g, -b])
            s_new2 = A.state(np.r_[g, b])
            f_map.append(abs(np.vdot(s_old, s_new)) ** 2)
            f_nomap.append(abs(np.vdot(s_old, s_new2)) ** 2)
            leak_old.append(1 - float(np.sum(np.abs(s_old[order]) ** 2)))
            leak_new.append(1 - float(np.sum(np.abs(s_new[order]) ** 2)))
        rows.append({"beta_scale": float(scale), "fid_mapped_min": float(np.min(f_map)),
                     "fid_mapped_mean": float(np.mean(f_map)), "fid_unmapped_mean": float(np.mean(f_nomap)),
                     "leak_old_max": float(np.max(leak_old)), "leak_new_max": float(np.max(leak_new))})
    return {"inst_id": "N04e004q1.5", "K": K, "L": L, "ring": "rank (bf)", "gamma": "Eq. 4.11 (pi / kappa_min)",
            "n_samples": n_samples, "rows": rows,
            "equal": all(r["fid_mapped_min"] >= 1 - 1e-12 for r in rows)}


def check_eval_replay(backend) -> dict:
    from gsp.circuits import ansatz as az
    from gsp.instances.adhoc import load_any
    from gsp.instances.encode import jh_boost
    from gsp.metrics.state import metric_context
    rp, rx = pd.read_csv(REP_P), pd.read_csv(REP_X)
    zp, zx = np.load(NPZ_P), np.load(NPZ_X)
    rows = []
    for e in range(10):
        inst, rul, adhoc = load_any(f"N05e{e:03d}q1.5")
        # A1: old Pauli kernel, un-boosted circuit
        row = rp[(rp.Assets == 5) & (rp.Layer == L_REF) & (rp.Exp == e) & (rp.Point == 0)].iloc[0]
        th = zp[f"A5_p{L_REF}_E{e}_S0_b{bkey(inst.boost_obj)}_params"]
        order, ws, cs, fixed, kern = old_preserving(inst, 12, L_REF)
        ctx = metric_context(inst, rul, None)
        m_old = ctx.evaluate_state(backend.get_state_classical(kern, inst.n, list(th), *fixed))
        A = new_a1(inst, order, L_REF)
        m_new = ctx.evaluate_state(A.state(np.r_[th[:L_REF], -th[L_REF:]]))
        m_boost = ctx.evaluate_state(backend.get_state_classical(
            kern, inst.n, list(th), *old_preserving(inst, 12, L_REF, inst.boost_obj)[3]))
        # A0: new layered circuit
        H = inst.hamiltonian(LAM_X)
        a = jh_boost(H)
        rowx = rx[(rx.Assets == 5) & (rx.Layer == L_REF) & (rx.Exp == e) & (rx.Point == 0)].iloc[0]
        thx = zx[f"A5_p{L_REF}_E{e}_S0_b{bkey(a)}_params"]
        A0 = az.penalty_ansatz(H, L_REF)
        m_x = metric_context(inst, rul, LAM_X).evaluate_state(A0.state(thx))
        A0b = az.penalty_ansatz(H, L_REF, circuit_boosted=True)
        m_xb = metric_context(inst, rul, LAM_X).evaluate_state(A0b.state(thx))
        rows.append({"e": e, "adhoc": adhoc, "F_size": rul.F_size,
                     "A1_stored": float(row.Approximate_ratio), "A1_old_kernel": m_old["ar_f"],
                     "A1_old_kernel_boosted_circuit": m_boost["ar_f"], "A1_new_mixer_beta_neg": m_new["ar_f"],
                     "A0_stored": float(rowx.Approximate_ratio), "A0_new": m_x["ar_f"],
                     "A0_new_boosted_circuit": m_xb["ar_f"]})
    df = pd.DataFrame(rows)
    return {"rows": rows,
            "A0_max_abs_diff": float((df.A0_new - df.A0_stored).abs().max()),
            "A0_boosted_max_abs_diff": float((df.A0_new_boosted_circuit - df.A0_stored).abs().max()),
            "A1_old_max_abs_diff": float((df.A1_old_kernel - df.A1_stored).abs().max()),
            "A1_old_boosted_max_abs_diff": float((df.A1_old_kernel_boosted_circuit - df.A1_stored).abs().max()),
            "A1_new_max_abs_diff": float((df.A1_new_mixer_beta_neg - df.A1_stored).abs().max())}


def check_init() -> dict:
    """theta_1 = theta_0 (1 - lr wd) - lr g/(|g| + 1e-8), so |theta_1 - theta_0 (1 - 1e-4)| <= lr = 0.01 (= lr when
    the gradient component is resolved, 0 when fp32 lost it). A wrong init would miss by ~|theta_0| (1e2..1e6)."""
    from gsp.arms.base import restart_seed
    from gsp.circuits import ansatz as az
    from gsp.circuits.legacy_pauli import legacy_mm_p
    from gsp.arms.qaoa import legacy_rank
    from gsp.instances.adhoc import load_any
    from gsp.train.init import init_params
    from gsp.instances.encode import jh_boost
    zp, zx = np.load(NPZ_P), np.load(NPZ_X)
    rows = []
    for e in range(10):
        inst, _, _ = load_any(f"N05e{e:03d}q1.5")
        seed = restart_seed(f"N05e{e:03d}", 0)
        H = inst.hamiltonian(LAM_X)
        A0 = az.penalty_ansatz(H, L_REF)
        x0 = init_params(A0, seed)
        tr = zx[f"A5_p{L_REF}_E{e}_S0_b{bkey(jh_boost(H))}"]
        dx = float(np.abs(tr[0, 3:5] - x0[:2] * (1 - 1e-4)).max())
        A1 = new_a1(inst, legacy_rank(inst, 12), L_REF)
        x1 = init_params(A1, seed, legacy=True)
        trp = zp[f"A5_p{L_REF}_E{e}_S0_b{bkey(inst.boost_obj)}"]
        dp = float(np.abs(trp[0, 3:5] - x1[:2] * (1 - 1e-4)).max())
        rows.append({"e": e, "seed": seed, "A0_step_dev": float(dx), "A1_legacy_step_dev": float(dp),
                     "A1_gamma_range": float(np.pi / min(A1.kappa_min(), float(legacy_mm_p(A1.circ.order, A1.n)))),
                     "A1_kappa_min": A1.kappa_min()})
    return {"rows": rows, "max_dev": max(max(r["A0_step_dev"], r["A1_legacy_step_dev"]) for r in rows),
            "pass": all(r["A0_step_dev"] <= 0.01 + 1e-6 and r["A1_legacy_step_dev"] <= 0.01 + 1e-6 for r in rows)}


def check_train_replay(backend, es=(0, 1, 2)) -> dict:
    from gsp.arms.base import restart_seed
    from gsp.circuits import ansatz as az
    from gsp.instances.adhoc import load_any
    from gsp.instances.encode import jh_boost
    from gsp.metrics.state import metric_context
    from gsp.train.adamw import train_adamw
    from gsp.train.init import init_params
    rp, rx = pd.read_csv(REP_P), pd.read_csv(REP_X)
    zp, zx = np.load(NPZ_P), np.load(NPZ_X)
    rows = []
    for e in es:
        inst, rul, _ = load_any(f"N05e{e:03d}q1.5")
        seed = restart_seed(f"N05e{e:03d}", 0)
        # A0
        H = inst.hamiltonian(LAM_X)
        a = jh_boost(H)
        A0 = az.penalty_ansatz(H, L_REF)
        t = time.perf_counter()
        res = train_adamw(A0.energy, init_params(A0, seed))
        tx = time.perf_counter() - t
        tr = zx[f"A5_p{L_REF}_E{e}_S0_b{bkey(a)}"]
        T = min(len(tr), res.n_iter)
        mx = metric_context(inst, rul, LAM_X).evaluate_state(A0.state(res.params))
        rowx = rx[(rx.Assets == 5) & (rx.Layer == L_REF) & (rx.Exp == e) & (rx.Point == 0)].iloc[0]
        rows.append({"arm": "A0", "e": e, "iters": res.n_iter, "iters_stored": len(tr),
                     "df_first10": float(np.max(np.abs(res.f_hist[:10] / a - tr[:10, 0]))),
                     "df_all": float(np.max(np.abs(res.f_hist[:T] / a - tr[:T, 0]))),
                     "ar_f": mx["ar_f"], "ar_f_stored": float(rowx.Approximate_ratio), "wall_s": tx})
        print(rows[-1], flush=True)
        # A1 on the OLD Pauli kernel (legacy init), the harness trainer
        order, ws, cs, fixed, kern = old_preserving(inst, 12, L_REF)
        op = backend.ising_op(inst.H_obj, inst.boost_obj)
        A1 = new_a1(inst, order, L_REF)
        x0 = init_params(A1, seed, legacy=True)
        t = time.perf_counter()
        res = train_adamw(lambda p: backend.observe(kern, op, list(p), *fixed), x0)
        tp_ = time.perf_counter() - t
        trp = zp[f"A5_p{L_REF}_E{e}_S0_b{bkey(inst.boost_obj)}"]
        T = min(len(trp), res.n_iter)
        m = metric_context(inst, rul, None).evaluate_state(
            backend.get_state_classical(kern, inst.n, list(res.params), *fixed))
        row = rp[(rp.Assets == 5) & (rp.Layer == L_REF) & (rp.Exp == e) & (rp.Point == 0)].iloc[0]
        rows.append({"arm": "A1_old_kernel", "e": e, "iters": res.n_iter, "iters_stored": len(trp),
                     "df_first10": float(np.max(np.abs(res.f_hist[:10] / inst.boost_obj - trp[:10, 0]))),
                     "df_all": float(np.max(np.abs(res.f_hist[:T] / inst.boost_obj - trp[:T, 0]))),
                     "ar_f": m["ar_f"], "ar_f_stored": float(row.Approximate_ratio), "wall_s": tp_})
        print(rows[-1], flush=True)
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train-replay", action="store_true")
    args = ap.parse_args(argv)
    from gsp.sim import backend
    from gsp.store.paths import tables_dir
    busy = backend.gpu_compute_pids()
    if busy:
        raise SystemExit(f"GPU busy (compute PIDs {busy}); one GPU process at a time")
    backend.set_target()
    out = {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"), **backend.runtime_info()}
    for name, fn in (("pauli_port", check_pauli_port), ("old_mixer_sign", check_old_mixer_sign),
                     ("equivalence", check_equivalence), ("eval_replay", check_eval_replay)):
        t = time.perf_counter()
        out[name] = fn(backend)
        out[name]["wall_s"] = time.perf_counter() - t
        print(name, json.dumps(out[name], default=str)[:600], flush=True)
    out["init_check"] = check_init()
    print("init_check", out["init_check"]["max_dev"], out["init_check"]["pass"], flush=True)
    if not args.skip_train_replay:
        out["train_replay"] = check_train_replay(backend)
    d = tables_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "s4_legacy.json"
    if args.skip_train_replay and path.exists():
        old = json.loads(path.read_text())
        if "train_replay" in old:
            out["train_replay"] = old["train_replay"]
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
