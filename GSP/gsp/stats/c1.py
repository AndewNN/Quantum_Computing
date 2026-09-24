"""Rule C1 (PLAN §1.7): operator level for A1's compiled mixer (S3); the state-level pieces -- leakage, the
eps_num calibration on random A1 circuits, the pass / fail rule and the leakage-vs-D growth-exponent fit (S5, at
the end of this module). S8 adds A4's generator check and runs the state level along A4.

`a1_operator_level` evolves every kept basis string |u_j> through one compiled mixer layer and compares
the circuit's action with the dense ordered product in K x K (`preserving.dense_layer`, not
expm(-i beta H_M)):
  leakage      = max_{x not kept, j} |U[x, u_j]|         (<= 1e-13; the unitary's norm is 1)
  leak_norm    = max_j || U[not kept, u_j] ||_2
  block_err    = max_{i, j} |U[u_i, u_j] - U_dense[i, j]|
engine = "numpy": the abstract circuit's gate list on `gsp.compile.npsim` (native multi-controlled
gates, or the (iii) lists with decomposed=True); engine = "cudaq": the same list on the CUDA-Q
interpreter kernel (`gsp.circuits.program`, the default engine), input |u_j> by exact X gates;
engine = "builder": the builder-API kernel of `preserving.build_kernel`, input by ry(pi * bit) (whose
cos(pi/2) = 6.1e-17 is the leakage floor). Both CUDA-Q engines go through `gsp.sim.backend.get_state`
(tests and post-run checks only).
"""

from __future__ import annotations

import numpy as np

from ..circuits import preserving as pr


def _metrics(U: np.ndarray, order: np.ndarray, ref: np.ndarray) -> dict:
    n_rows = U.shape[0]
    mask = np.ones(n_rows, dtype=bool)
    mask[order] = False
    out_block = U[mask]
    return {
        "leakage": float(np.abs(out_block).max()) if out_block.size else 0.0,
        "leak_norm": float(np.sqrt((np.abs(out_block) ** 2).sum(axis=0)).max()) if out_block.size else 0.0,
        "block_err": float(np.abs(U[order, :] - ref).max()),
        "unitarity_err": float(np.abs(U.conj().T @ U - np.eye(U.shape[1])).max()),
    }


def a1_operator_level(circ: pr.PreservingCircuit, beta, engine: str = "numpy",
                      decomposed: bool = False):
    """C1 operator check of one mixer layer at angle beta (see module doc). `beta` may be a sequence:
    then one kernel serves every value and a list of results is returned."""
    betas = [float(b) for b in np.atleast_1d(beta)]
    n, K, order = circ.n, circ.K, circ.order
    if engine == "numpy":
        from ..compile import npsim
        gates = pr.layer_gates(circ, 0, decomposed=decomposed)

        def action(b):
            psi = npsim.columns(n, order)
            npsim.apply(gates, psi, [b])
            return npsim.flat(psi)
    elif engine == "cudaq":
        from ..circuits import program
        from ..sim import backend
        gates = pr.layer_gates(circ, 0, decomposed=decomposed)
        progs = [program.encode(pr.a1_gates(circ, 0, prep=False, input_idx=int(u)) + gates, n) for u in order]
        kern = program.kernel()

        def action(b):
            U = np.empty((1 << n, K), dtype=np.complex128)
            for j, pg in enumerate(progs):
                U[:, j] = backend.get_state_classical(kern, n, *pg.args([b]))
            return U
    elif engine == "builder":
        from ..sim import backend
        bk = pr.build_kernel(circ, L=1, prep=False, input_bits=True, simulate_decomposed=decomposed)

        def action(b):
            U = np.empty((1 << n, K), dtype=np.complex128)
            for j, u in enumerate(order):
                U[:, j] = backend.get_state_classical(bk.kernel, n, bk.params(betas=[b], input_idx=int(u)))
            return U
    else:
        raise ValueError(engine)
    out = []
    for b in betas:
        r = _metrics(action(b), order, pr.dense_layer(K, circ.connectivity, b, circ.symmetrized))
        r.update({"n": n, "K": K, "connectivity": circ.connectivity, "ring_order": circ.ring_order,
                  "symmetrized": circ.symmetrized, "engine": engine, "decomposed": decomposed, "beta": b})
        out.append(r)
    return out if np.ndim(beta) else out[0]


def star_prep_fidelity(circ: pr.PreservingCircuit, engine: str = "numpy", decomposed: bool = False) -> dict:
    """|<uniform over the kept strings | prep circuit |0^n>|^2 and the smallest kept amplitude."""
    n, K, order = circ.n, circ.K, circ.order
    if engine == "numpy":
        from ..compile import npsim
        psi = npsim.columns(n, [0])
        npsim.apply(pr.prep_gates(circ, decomposed=decomposed), psi)
        v = npsim.flat(psi)[:, 0]
    elif engine == "cudaq":
        from ..circuits import program
        from ..sim import backend
        pg = program.encode(pr.prep_gates(circ, decomposed=decomposed), n)
        v = backend.get_state_classical(program.kernel(), n, *pg.args())
    elif engine == "builder":
        from ..sim import backend
        bk = pr.build_kernel(circ, L=0, prep=True, simulate_decomposed=decomposed)
        v = backend.get_state_classical(bk.kernel, n, bk.params())
    else:
        raise ValueError(engine)
    target = pr.uniform_state(K)
    fid = float(abs(np.vdot(target, v[order])) ** 2)
    mask = np.ones(1 << n, dtype=bool)
    mask[order] = False
    return {"fidelity": fid, "infidelity": 1.0 - fid, "min_amp": float(v[order].real.min()),
            "max_imag": float(np.abs(v[order].imag).max()), "leakage": float(np.abs(v[mask]).max())}


# --- S5: the state level and the growth-exponent fit ----------------------------------------------------------
# PLAN §1.7 Rule C1, state level: max_k [1 - p_feas(psi_k)] along A4 must stay <= 10 x eps_num(n, D), where
# eps_num(n, D) is the maximum leakage over 10 random A1 ring circuits on the same sector with enough layers to
# match A4's count of multi-controlled gates at step k; the leakage-vs-D exponent separates accumulated rounding
# (sqrt(D), diffusive) from a systematic violation (linear in D). S8 runs it on A4; S5 provides the pieces:
#   leakage(psi, sector_idx)             1 - p_sector (the estimator used on both sides: it includes the norm drift)
#                                        and the out-of-sector mass
#   mc_gate_count(gates)                 multi-controlled gates of a gate list (names "mc*")
#   eps_num_curve(...)                   eps_num at a list of layer counts L (random Eq. 4.11 parameters)
#   state_level(leak_k, eps_num_k)       the pass / fail of the state-level rule
#   growth_exponent(D, leak)             log-log OLS slope with its 95 % t interval and the classification

SQRT_D, LINEAR_D = 0.5, 1.0
STATE_FACTOR = 10.0
N_CALIB_CIRCUITS = 10


def leakage(psi, sector_idx) -> dict:
    """leak = 1 - p_sector (signed; rounding makes it either sign), out_mass = the mass outside the sector summed
    directly, norm_err = sum |psi|^2 - 1."""
    p = np.abs(np.asarray(psi)) ** 2
    mask = np.zeros(p.size, dtype=bool)
    mask[np.asarray(sector_idx, dtype=np.int64)] = True
    ins = float(p[mask].sum())
    return {"leak": 1.0 - ins, "out_mass": float(p[~mask].sum()), "norm_err": float(p.sum()) - 1.0}


def mc_gate_count(gates) -> int:
    return int(sum(1 for g in gates if g.name.startswith("mc")))


def eps_num_curve(inst, circ: pr.PreservingCircuit, L_values, seed: int, n_circuits: int = N_CALIB_CIRCUITS,
                  engine: str = "numpy") -> list[dict]:
    """eps_num at each L: the max over n_circuits random A1 circuits (star prep + L x (cost + mixer), Eq. 4.11
    parameters; gamma ~ U[-pi/kappa, pi/kappa], beta ~ U[-pi, pi], drawn from default_rng(seed) in order) of
    |1 - p_sector| of the final state (the magnitude: rounding gives either sign). engine "numpy" (npsim) or "cudaq" (the layered kernel's get_state; a
    post-run diagnostic, never an update loop). `seed` must come from the seed table (the draw's restart seed)."""
    from ..circuits.ansatz import confined_ansatz
    rng = np.random.default_rng(int(seed))
    out = []
    for L in L_values:
        A = confined_ansatz(inst.H_obj, inst.boost_obj, circ, int(L))
        mm = np.pi / A.kappa_min()
        mc = mc_gate_count(A.abstract_gates())
        leaks = []
        for _ in range(n_circuits):
            th = np.r_[rng.uniform(-mm, mm, int(L)), rng.uniform(-np.pi, np.pi, int(L))]
            if engine == "numpy":
                from ..compile import npsim
                psi = npsim.flat(npsim.apply(A.unrolled(), npsim.columns(A.n, [0]), th))[:, 0]
            elif engine == "cudaq":
                psi = A.state(th)
            else:
                raise ValueError(engine)
            leaks.append(leakage(psi, circ.order))
        out.append({"L": int(L), "mc_gates": mc, "eps_num": max(abs(r["leak"]) for r in leaks),
                    "max_signed_leak": max(r["leak"] for r in leaks),
                    "max_out_mass": max(r["out_mass"] for r in leaks), "n_circuits": n_circuits, "engine": engine})
    return out


def state_level(leak_k, eps_num_k, factor: float = STATE_FACTOR) -> dict:
    """PLAN §1.7: falsified iff max_k leak_k > factor x eps_num(n, D_k) at some k (eps_num_k aligned with k)."""
    leak_k = np.asarray(leak_k, dtype=np.float64)
    eps_k = np.asarray(eps_num_k, dtype=np.float64)
    thr = factor * np.abs(eps_k)
    over = leak_k > thr
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(thr > 0, leak_k / thr, np.where(leak_k > 0, np.inf, 0.0))
    return {"max_leak": float(leak_k.max()) if leak_k.size else float("nan"),
            "max_ratio": float(np.max(ratio)) if ratio.size else float("nan"),
            "passes": bool(not over.any()), "first_fail_k": int(np.argmax(over)) if over.any() else None}


def growth_exponent(D, leak, level: float = 0.95) -> dict:
    """Fit log|leak| = a + b log D (OLS; zero leaks dropped). Classification: the reference exponent nearer to b
    (0.5 = accumulated rounding, 1 = a systematic rate), provided the 95 % t interval of b excludes the other one;
    else 'ambiguous'. (Requiring the interval to CONTAIN the nearer reference would reject a clean sqrt(D) signal:
    the max over circuits of one set of random walks has autocorrelated residuals, so its OLS interval is narrow
    and biased low, e.g. 0.39 [0.35, 0.43] on synthetic sqrt(D) data.)"""
    from scipy import stats as st
    D = np.asarray(D, dtype=np.float64)
    y = np.abs(np.asarray(leak, dtype=np.float64))
    ok = (D > 0) & (y > 0) & np.isfinite(y)
    x, yy = np.log(D[ok]), np.log(y[ok])
    if x.size < 3 or np.unique(x).size < 2:
        return {"exponent": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(x.size),
                "dropped": int((~ok).sum()), "class": "ambiguous"}
    r = st.linregress(x, yy)
    tq = st.t.ppf(0.5 + level / 2, x.size - 2)
    lo, hi = r.slope - tq * r.stderr, r.slope + tq * r.stderr
    b = r.slope
    if abs(b - SQRT_D) < abs(b - LINEAR_D) and hi < LINEAR_D:
        cls = "sqrt(D)"
    elif abs(b - LINEAR_D) < abs(b - SQRT_D) and lo > SQRT_D:
        cls = "linear"
    else:
        cls = "ambiguous"
    return {"exponent": float(r.slope), "lo": float(lo), "hi": float(hi), "se": float(r.stderr),
            "intercept": float(r.intercept), "n": int(x.size), "dropped": int((~ok).sum()), "class": cls}


# --- S8: A4 (confined DB-QITE) at both levels ---------------------------------------------------------------------
# Operator level (PLAN §1.7, n <= 12): the dense generator W_k = [rho_k, H] of step k, rho_k = |psi_k><psi_k| from the
# compiled circuit's state, restricted to the sector: every element with a row or a column outside the kept strings,
# relative to ||W_k||_2. For a normalized pure state ||[rho, H]||_2 = sqrt(Var_psi(H)) (W = sigma (|psi><psi_perp| -
# h.c.)); the function also returns the numerically computed norm at n <= 8 as a check. The threshold is 1e-13.
# `a4_step_operator` (extra, small n): the compiled step V = e^{i r_H H} U_k P U_k^dagger e^{-i r_H H} as a gate list on
# every kept basis string (npsim) against the exact reflection formula e^{i r_H H} (I + (e^{i r_rho} - 1) rho_k)
# e^{-i r_H H}: off-sector amplitudes and the K x K block error.
# State level: `a4_state_level` = leak_k = 1 - p_sector(psi_k) along the A4 trajectory against 10 x eps_num(n, D_k) from
# `eps_num_curve` at the A1 depth whose multi-controlled gate count matches U_k's (`matched_layers`), plus the growth
# exponent of |leak| vs D on both sides.
OPERATOR_TOL = 1e-13


def a4_generator_check(psi, diag, sector_idx, dense: bool = True) -> dict:
    """Off-sector part of W = [rho, H] for one state (classical order). dense=True builds the 2^n x 2^n matrix."""
    psi = np.asarray(psi, dtype=np.complex128)
    d = np.asarray(diag, dtype=np.float64)
    nrm = float(np.vdot(psi, psi).real)
    E = float((np.abs(psi) ** 2) @ d) / nrm
    var = float((np.abs(psi) ** 2) @ (d - E) ** 2) / nrm
    w_norm = float(np.sqrt(max(var, 0.0))) * nrm
    mask = np.zeros(psi.size, dtype=bool)
    mask[np.asarray(sector_idx, dtype=np.int64)] = True
    if dense:
        W = np.outer(psi, psi.conj()) * (d[None, :] - d[:, None])      # (rho H - H rho)_{xy} = psi_x psi_y^* (d_y - d_x)
        off = max(float(np.abs(W[~mask, :]).max(initial=0.0)), float(np.abs(W[:, ~mask]).max(initial=0.0)))
        w_max = float(np.abs(W).max())
        w_norm_num = float(np.linalg.norm(W, 2)) if psi.size <= 256 else None
        del W
    else:          # the same maximum without the matrix (|W_xy| = |W_yx|: off-sector rows suffice), row blocks
        a = np.abs(psi)
        sub = np.flatnonzero(~mask)
        off = 0.0
        for i in range(0, sub.size, 256):
            r = sub[i:i + 256]
            off = max(off, float((a[r][:, None] * a[None, :] * np.abs(d[None, :] - d[r][:, None])).max()))
        w_max, w_norm_num = None, None
    return {"off_max": off, "w_norm": w_norm, "rel": off / w_norm if w_norm > 0 else (0.0 if off == 0 else np.inf),
            "w_max": w_max, "w_norm_numeric": w_norm_num, "variance": var, "norm_err": nrm - 1.0,
            "out_amp_max": float(np.abs(psi[~mask]).max(initial=0.0)), "dense": dense}


def a4_step_operator(C, s_prefix, s_next: float, sector_idx) -> dict:
    """The compiled step from psi_k (k = len(s_prefix)) with step s_next, on every kept basis string (npsim)."""
    from ..circuits import dbqite as db
    from ..compile import decompose as dc
    from ..compile import npsim
    from ..compile.decompose import Angle, Gate
    rH, rr = db.r_pair(s_next, C.units, C.sigma)
    Uk = C.gates(list(s_prefix))

    def cost(a):
        return [g if g.angle is None or g.angle.pidx < 0 else Gate(g.name, g.qubits, Angle(g.angle.coef * a))
                for g in C.cost]

    step = cost(rH) + dc.inverse(Uk) + dc.mcphase_native(C.n, Angle(rr)) + Uk + cost(-rH)
    idx = np.asarray(sector_idx, dtype=np.int64)
    V = npsim.flat(npsim.apply(step, npsim.columns(C.n, idx)))
    psi = C.formula_state(list(s_prefix))
    d = C.ham.diagonal()
    cols = np.zeros((1 << C.n, idx.size), dtype=np.complex128)
    cols[idx, np.arange(idx.size)] = 1.0
    ph = np.exp(-1j * rH * d)[:, None]
    phi = ph * cols
    phi = phi + (np.exp(1j * rr) - 1.0) * np.outer(psi, psi.conj() @ phi)
    Vex = np.conj(ph) * phi
    mask = np.ones(1 << C.n, dtype=bool)
    mask[idx] = False
    return {"k": len(s_prefix), "leakage": float(np.abs(V[mask]).max(initial=0.0)),
            "block_err": float(np.abs(V[idx] - Vex[idx]).max()),
            "exact_leakage": float(np.abs(Vex[mask]).max(initial=0.0)), "gates": len(step)}


def matched_layers(mc_target: int, mc_prep: int, mc_layer: int) -> int:
    """The A1 depth L whose multi-controlled gate count mc_prep + L mc_layer first reaches mc_target (>= 1)."""
    return max(1, int(np.ceil((int(mc_target) - int(mc_prep)) / max(1, int(mc_layer)))))


def a4_state_level(inst, circ, leak_k, mc_k, seed: int, engine: str = "cudaq", n_circuits: int = N_CALIB_CIRCUITS,
                   factor: float = STATE_FACTOR) -> dict:
    """Rule C1 state level along one A4 trajectory. leak_k / mc_k: k = 1..T (1 - p_sector of psi_k and the
    multi-controlled gates of U_k). eps_num at the matched A1 ring depths (same sector, `eps_num_curve`)."""
    mc_prep = mc_gate_count(pr.prep_gates(circ))
    mc_layer = mc_gate_count(pr.layer_gates(circ, 0))
    Ls = [matched_layers(m, mc_prep, mc_layer) for m in mc_k]
    curve = eps_num_curve(inst, circ, Ls, seed, n_circuits=n_circuits, engine=engine)
    eps = [c["eps_num"] for c in curve]
    rule = state_level(leak_k, eps, factor)
    D_a1 = [c["mc_gates"] for c in curve]
    return {"mc_k": [int(m) for m in mc_k], "L_matched": Ls, "mc_a1": D_a1, "leak_k": [float(v) for v in leak_k],
            "eps_num_k": eps, "eps_out_mass_k": [c["max_out_mass"] for c in curve], **rule,
            "growth_a4": growth_exponent(mc_k, leak_k), "growth_a1": growth_exponent(D_a1, eps),
            "mc_prep": mc_prep, "mc_layer": mc_layer, "n_circuits": n_circuits, "engine": engine}
