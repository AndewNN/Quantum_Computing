"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for the S1 reproduction tests.

Provenance
----------
source : Utils/qaoaCUDAQ.py (repo Quantum_Computing, committed at 8634ca8,
         sha256 d5c1b1244cfc54bf645ecbffac87291a01bdb15b34209e5d5c53b868052dc1b5)
lines  : 147-210  po_normalize, ret_cov_to_QUBO, qubo_to_ising, process_ansatz_values
         653-668  all_state_to_return
         679-726  find_budget
         746-750  to_sig
copied : 2026-09-24 (GSP session S1), byte-for-byte via sed; only this header and the
         import block below were added. The completed work ran these under CUDA-Q 0.13.0
         (env cudaq13); here they run under the gsp env (CUDA-Q 0.15.1).
"""
# --- import block (added; the originals come from the top of Utils/qaoaCUDAQ.py) ---
from typing import List, Tuple
import math
import numpy as np
import cudaq
from cudaq import spin

# --- Utils/qaoaCUDAQ.py:147-210 (verbatim) ---
def po_normalize(B, P, ret, cov):
    # print("cp 0")
    P_b = P / B
    ret_b = ret * P_b
    cov_b = np.diag(P_b) @ cov @ np.diag(P_b)
    
    n_max = np.int32(np.floor(np.log2(B/P))) + 1
    # print("n_max:", n_max)
    n_qs = np.cumsum(n_max)
    n_qs = np.insert(n_qs, 0, 0)
    n_qubit = n_qs[-1]
    C = np.zeros((len(P), n_qubit))
    # print("cp 1")
    for i in range(len(P)):
         for j in range(n_max[i]):
              C[i, n_qs[i] + j] = 2**j
    # print("cp 2")

    P_bb = C.T @ P_b
    ret_bb = C.T @ ret_b
    # print("ret_bb:", ret_bb)
    cov_bb = C.T @ cov_b @ C
    return P_bb, ret_bb, cov_bb, int(n_qubit), n_max, C

def ret_cov_to_QUBO(ret: np.ndarray, cov: np.ndarray, P: np.ndarray, lamb: float, q:float) -> np.ndarray: # Max return, Min variance
    di = np.diag(ret + 2*lamb*P)
    mat = lamb * np.outer(P, P) + q * cov
    return di - mat

def qubo_to_ising(qubo: np.ndarray, lamb: float) -> cudaq.SpinOperator:
    spin_op = -lamb * spin.i(0)
    for i in range(qubo.shape[0]):
        for j in range(qubo.shape[1]):
                if i != j and qubo[i, j] != 0:
                    spin_op += qubo[i, j] * ((spin.i(i) - spin.z(i)) / 2 * (spin.i(j) - spin.z(j)) / 2)
                elif i == j and qubo[i, j] != 0:
                    spin_op += qubo[i, j] * (spin.i(i) - spin.z(i)) / 2
    return spin_op

def process_ansatz_values(H: cudaq.SpinOperator) -> Tuple[List[int], List[float], List[int], List[int], List[float]]:
    HH = H.get_raw_data()
    idxs = [[j - len(HH[0][i])//2 for j in range(len(HH[0][i])) if HH[0][i][j]] for i in range(len(HH[0]))]

    HH = [(idxs[i], HH[1][i], sum(HH[0][i])) for i in range(len(HH[0]))]
    HH = sorted(HH, key=lambda x: (x[2], x[0]), reverse=False)

    idx_1 = []
    coeff_1 = []
    idx_2_a, idx_2_b = [], []
    coeff_2 = []
    # print("HH:", HH)
    for i in range(len(HH)):
        if HH[i][1].real == 0:
            continue
        if HH[i][2] == 1:
            idx_1.append(HH[i][0][0])
            coeff_1.append(HH[i][1].real)
        elif HH[i][2] == 2:
            idx_2_a.append(HH[i][0][0])
            idx_2_b.append(HH[i][0][1])
            coeff_2.append(HH[i][1].real)
    # print(HH)

    return idx_1, coeff_1, idx_2_a, idx_2_b, coeff_2

# --- Utils/qaoaCUDAQ.py:653-668 (verbatim) ---
def all_state_to_return(qb, lam, QUBO): # QUBO of Max problem
    '''
    IMPORTANT: 
        QUBO: must be QUBO of MAX problem!!!
    '''
    # print("all 0")
    ll = np.zeros((qb, 1<<qb), dtype=np.float32)
    a_0 = np.zeros(1<<qb, dtype=np.float32)
    a_1 = np.ones(1<<qb, dtype=np.float32)
    idxx = np.arange(1<<qb, dtype=np.int32)
    for i in range(qb):
        ll[i] = np.where(idxx%(1<<(qb-i))<(1<<(qb-i-1)), a_0,  a_1)
    l = ll.T.copy()
    ss = l @ QUBO
    ss = (ss.reshape(-1, 1, qb) @ l.reshape(-1, qb, 1))
    return ss.reshape(-1) - lam

# --- Utils/qaoaCUDAQ.py:679-726 (verbatim) ---
def find_budget(target_qubit, P, min_P, max_P, min_mix_mode = False):
    def rdd(a, coeff, order = 7, s = 1e-9):
        return round(a + coeff * s, order)
    
    n_assets = len(P)
    MI, MA = min_P, max_P * ((1 << math.ceil(target_qubit/n_assets))-1)
    mi, ma = MI, MA
    cou = 0
    mid = (mi + ma)/2
    while (N := np.sum(np.int32(np.floor(np.log2(mid/P))) + 1)) != target_qubit:
        if N < target_qubit:
            mi = mid
        else:
            ma = mid
        # print()
        mid = rdd((mi + ma)/2, 0, 7)
        cou += 1
        if cou > 100:
            assert False, "Cannot find budget for target qubit uwaaaaa (Should not happen, Please tell trusted adult lol)"
    MID = mid
    if not min_mix_mode:
        return MID
    
    mi, ma = MI, MID
    cou = 0
    mid = (mi + ma)/2
    while mid != ma:
        if np.sum(np.int32(np.floor(np.log2(mid/P))) + 1) < target_qubit:
            mi = mid
        else:
            ma = mid
        mid = rdd((mi + ma)/2, 1, 7)
        cou += 1
    MIN = mid

    mi, ma = MID, MA
    cou = 0
    mid = (mi + ma)/2
    while mid != ma:
        if np.sum(np.int32(np.floor(np.log2(mid/P))) + 1) > target_qubit:
            ma = mid
        else:
            mi = mid
        mid = rdd((mi + ma)/2, 1, 7)
        cou += 1
    MAX = mid

    return MIN, MAX

# --- Utils/qaoaCUDAQ.py:746-750 (verbatim) ---
def to_sig(x, sig=3):
    x = float(x)
    res = float(f"{x:.{sig}}")
    res = int(res) if res.is_integer() else res
    return res
