"""The overlap kernel of arm A3 (S7): ONE `@cudaq.kernel`, compiled once per process, that runs every M1 overlap
circuit of the harness and reads P(0...0) through a flag qubit (PLAN §1.5, "A3's P(0...0)").

Circuit (n system qubits + 1 flag qubit, flag = qubit n):
    start segment (H on every qubit)
    m forward layers   U_l(pb)      l = 0 .. m-1         (gamma_l, beta_l) = (pb[l], pb[L + l])
    m inverse layers   U_l(pa)^dag  l = m-1 .. 0         (gamma_l, beta_l) = (pa[l], pa[L + l])
    end segment (H on every qubit)
    flag = 1: X on every system qubit, X on the flag controlled by all n system qubits, X on every system qubit
              (a multi-controlled X with open controls), so <Z_flag> = 1 - 2 P(0...0) and
              P(0...0) = |<psi_m(pa)|psi_m(pb)>|^2 = (1 - <Z_flag>) / 2.
    flag = 0: no flag gate (the flag stays |0>): the state whose |0...0> amplitude is the statevector reference.
Only layers up to m appear: the layers after the last one in which pa and pb differ cancel exactly
(`VarQITE_Estimation_Choices.md` §1, §3.1).

Gate data as in the layered kernel (`program.encode_layered` packing; `overlap.encode_overlap` builds it):
  code[g] = op + 16 * (target + 32 * (n_controls + 32 * (slot + 4 * ctl_start)))
      op: 0 x, 1 h, 6 cx, 7 rx, 8 ry, 9 rz, 12 crz (single-control gates only; A3 runs on A0's circuit);
      slot: 0 constant angle coef[g], 1 coef[g] * gamma_l, 2 coef[g] * beta_l.
  seg = [start_b, start_e, layer_b, layer_e, inverse_b, inverse_e, end_b, end_e] (indices into code / coef).
The update loop calls it through `backend.observe` only (never get_state; PLAN §3.3).
"""

import cudaq


@cudaq.kernel
def kernel_overlap(n: int, L: int, m: int, code: list[int], coef: list[float], ctl: list[int], seg: list[int],
                   pa: list[float], pb: list[float], flag: int):
    q = cudaq.qvector(n)
    f = cudaq.qubit()
    for rep in range(2 * m + 2):
        g0 = seg[0]
        g1 = seg[1]
        ell = 0
        side = 1
        if rep > 2 * m:
            g0 = seg[6]
            g1 = seg[7]
        elif rep > m:
            g0 = seg[4]
            g1 = seg[5]
            ell = 2 * m - rep
            side = 0
        elif rep > 0:
            g0 = seg[2]
            g1 = seg[3]
            ell = rep - 1
        for g in range(g0, g1):
            cc = code[g]
            o = cc % 16
            cc = cc // 16
            tq = cc % 32
            cc = cc // 32
            cc = cc // 32
            sl = cc % 4
            c = cc // 4
            ang = coef[g]
            if sl == 1:
                if side == 1:
                    ang = coef[g] * pb[ell]
                else:
                    ang = coef[g] * pa[ell]
            elif sl == 2:
                if side == 1:
                    ang = coef[g] * pb[L + ell]
                else:
                    ang = coef[g] * pa[L + ell]
            if o == 12:
                rz.ctrl(ang, q[ctl[c]], q[tq])
            elif o == 9:
                rz(ang, q[tq])
            elif o == 7:
                rx(ang, q[tq])
            elif o == 1:
                h(q[tq])
            elif o == 0:
                x(q[tq])
            elif o == 6:
                x.ctrl(q[ctl[c]], q[tq])
            elif o == 8:
                ry(ang, q[tq])
    if flag == 1:
        x(q)
        x.ctrl(q, f)
        x(q)
