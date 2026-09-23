"""Rule C1, operator level for A1's compiled mixer (PLAN §1.7, §5 S3). S8 adds A4's generator check
and the state-level rule.

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
