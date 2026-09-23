"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for the S4 legacy-equivalence tests.

Provenance
----------
source : Utils/qaoaCUDAQ.py (repo Quantum_Computing, last committed at 15965ad,
         sha256 d5c1b1244cfc54bf645ecbffac87291a01bdb15b34209e5d5c53b868052dc1b5)
lines  : 243-460  init_pauli, transform_pauli, get_pauli, _get_pauli_objects, get_pauli_serializable,
                  basis_T_to_pauli_parallel, basis_T_to_pauli, reversed_str_bases_to_init_state
         462-519  kernel_qaoa_X, kernel_qaoa_Preserved, kernel_flipped
         670-677  get_init_states
copied : 2026-09-24 (GSP session S4), byte-for-byte by line slicing; only this header, the import block
         below and the marked helper at the end were added. The completed work ran these under CUDA-Q 0.13.0
         (env cudaq13); here they run under the gsp env (CUDA-Q 0.15.1), where `exp_pauli(theta, q, P)`
         applies exp(+i theta P) (measured in S3).
"""
# --- import block (added; the originals come from the top of Utils/qaoaCUDAQ.py) ---
from typing import List, Tuple
from math import sqrt, pi
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import numpy as np
import psutil
import cudaq
from cudaq import spin

# --- Utils/qaoaCUDAQ.py:243-519 (verbatim) ---
def init_pauli(x, y):
    if x == "0" and y == "0":
        A = spin.i(0) + spin.z(0)
        # B = spin.i(0) + spin.z(0)
        B = 0
    elif x == "0" and y == "1":
        A = spin.x(0)
        B = -spin.y(0)
    elif x == "1" and y == "0":
        A = spin.x(0)
        B = spin.y(0)
    elif x == "1" and y == "1":
        A = spin.i(0) - spin.z(0)
        # B = spin.i(0) - spin.z(0)
        B = 0
    return A, B

def transform_pauli(x, y, idx, A, B):
    if x == "0" and y == "0":
        A_, B_ = 0.5 * A * (spin.i(idx) + spin.z(idx)), 0.5 * B * (spin.i(idx) + spin.z(idx))
    elif x == "0" and y == "1":
        A_, B_ = 0.5 * (A * spin.x(idx) + B * spin.y(idx)), 0.5 * (B * spin.x(idx) - A * spin.y(idx))
    elif x == "1" and y == "0":
        A_, B_ = 0.5 * (A * spin.x(idx) - B * spin.y(idx)), 0.5 * (B * spin.x(idx) + A * spin.y(idx))
    elif x == "1" and y == "1":
        A_, B_ = 0.5 * A * (spin.i(idx) - spin.z(idx)), 0.5 * B * (spin.i(idx) - spin.z(idx))
    return A_, B_


#import numba
#@numba.jit(nogil=True, nopython=True)
def get_pauli(X, Y):
    A, B = init_pauli(X[0], Y[0])
    for i in range(1, len(X)):
        A, B = transform_pauli(X[i], Y[i], i, A, B)
    # print(f"Done by thread    : '{threading.current_thread().name}'")
    return A

def _get_pauli_objects(X, Y):
    A, B = init_pauli(X[0], Y[0])
    for i in range(1, len(X)):
        A, B = transform_pauli(X[i], Y[i], i, A, B)
    # print(f"Done by thread    : '{threading.current_thread().name}'")
    return A

def get_pauli_serializable(X, Y, n_qubits):
    """
    Computes the Pauli spin operators but returns their data representation
    (a list of tuples) which is safe for multiprocessing.
    """
    A_obj = _get_pauli_objects(X, Y)

    serializable_A = []
    for term in A_obj:
        coeff = term.evaluate_coefficient()
        pauli_word = term.get_pauli_word(n_qubits)
        if coeff.real != 0 and len(pauli_word) > 0:
            serializable_A.append((coeff.real, pauli_word))
    # print(len(serializable_A))
    return serializable_A
        
def basis_T_to_pauli_parallel(bases: List[str], T: np.ndarray, n_qubits: int) -> Tuple[List[cudaq.pauli_word], np.ndarray]:
    # print("Compute pauli")
    st_pauli_compute = time.time()
    A_all, B_all = 0, 0
    cou = 0
    # left_t = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1<<30)
    # for i in range(T.shape[0]):
    #     for j in range(i + 1, T.shape[1]):
    #         if T[i, j] == 0:
    #             continue
    #         A_now, B_now = get_pauli(bases[i], bases[j])
    #         A_all += T[i, j] * A_now
    #         # B_all += T[i, j] * B_now
    #         cou += 1
    #         print("Cou:", cou)
    #         # print("Ram left:", psutil.virtual_memory().total / (1<<30), psutil.virtual_memory().available / (1<<30), (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1<<30))
    #         left_now = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1<<30)
    #         print("Ram used:", left_now - left_t)
    #         left_t = left_now
    #         print(A_all.term_count)

    indices = [(i, j) for i in range(T.shape[0]) for j in range(i + 1, T.shape[1]) if T[i, j] != 0]
    indices_bases = [(bases[i], bases[j]) for i in range(T.shape[0]) for j in range(i + 1, T.shape[1]) if T[i, j] != 0]
    max_threads = max_processes = psutil.cpu_count(logical=True)
    worker_func = partial(get_pauli_serializable, n_qubits=n_qubits)
    
    # with Pool() as pool:
    #     results = pool.starmap(get_pauli, indices, chunksize=6)
    # print("Threads:", max_threads)
    # print("cpu_affinity:", psutil.Process().cpu_affinity())
#    os.system("taskset -p 0xff %d" % os.getpid())

#    with ThreadPool(processes=max_threads) as pool:
#        # starmap preserves order, which is important for cancellation
#        results = pool.starmap(
#            get_pauli,
#            indices_bases,
# #            chunksize=max(1, len(indices) // (max_threads))
# #            chunksize=100
#            chunksize=1
#        )

    
    with Pool(processes=max_processes) as pool:
        chunksize = max(1, len(indices) // max_processes)
        # print("Working with chunksize:", chunksize)
        results = pool.starmap(worker_func, indices_bases, chunksize=chunksize)

    # print("Pauli compute:", time.time() - st_pauli_compute)

    # print("Start merging...")
    st_merge = time.time()
    """
    for idx, (i, j) in enumerate(indices):
        A_now = results[idx]
        A_all += T[i, j] * A_now
        # B_all += T[i, j] * B_now
    print("Merge:", time.time() - st_merge)
    

    st_list = time.time()
    ret_s, ret_c = [], []
    for i in A_all:
        s = i.get_pauli_word(n_qubits)
        c = i.evaluate_coefficient()
        if len(s) > 0 and c.real != 0:
            # ret_s.append(pauli_to_int(s))
            ret_s.append(s)
            # print(s)
            ret_c.append(c.real)
    print("Listing:", time.time() - st_list)
    """

    summed_terms = defaultdict(float)

    for idx, (i, j) in enumerate(indices):
        serializable_A = results[idx]
        T_val = T[i, j]

        for coeff, pauli_word in serializable_A:
            summed_terms[pauli_word] += T_val * coeff

    # print("Merge done in:", time.time() - st_merge)

    st_list = time.time()
    ret_s, ret_c = [], []
    for pauli_word, final_coeff in summed_terms.items():
        if final_coeff != 0:
            ret_s.append(pauli_word)
            ret_c.append(final_coeff)
    # print("Listing done in:", time.time() - st_list)e

    return ret_s, np.array(ret_c)

def basis_T_to_pauli(bases: List[str], T: np.ndarray, n_qubits: int) -> Tuple[List[cudaq.pauli_word], np.ndarray]:
    def init_pauli(x, y):
        if x == "0" and y == "0":
            A = spin.i(0) + spin.z(0)
            # B = spin.i(0) + spin.z(0)
            B = 0
        elif x == "0" and y == "1":
            A = spin.x(0)
            B = -spin.y(0)
        elif x == "1" and y == "0":
            A = spin.x(0)
            B = spin.y(0)
        elif x == "1" and y == "1":
            A = spin.i(0) - spin.z(0)
            # B = spin.i(0) - spin.z(0)
            B = 0
        return A, B

    def transform_pauli(x, y, idx, A, B):
        if x == "0" and y == "0":
            A_, B_ = 0.5 * A * (spin.i(idx) + spin.z(idx)), 0.5 * B * (spin.i(idx) + spin.z(idx))
        elif x == "0" and y == "1":
            A_, B_ = 0.5 * (A * spin.x(idx) + B * spin.y(idx)), 0.5 * (B * spin.x(idx) - A * spin.y(idx))
        elif x == "1" and y == "0":
            A_, B_ = 0.5 * (A * spin.x(idx) - B * spin.y(idx)), 0.5 * (B * spin.x(idx) + A * spin.y(idx))
        elif x == "1" and y == "1":
            A_, B_ = 0.5 * A * (spin.i(idx) - spin.z(idx)), 0.5 * B * (spin.i(idx) - spin.z(idx))
        return A_, B_
    
    def get_pauli(X, Y):
        A, B = init_pauli(X[0], Y[0])
        for i in range(1, len(X)):
            A, B = transform_pauli(X[i], Y[i], i, A, B)
        return A, B
        
    A_all, B_all = 0, 0
    for i in range(T.shape[0]):
        for j in range(i + 1, T.shape[1]):
            A_now, B_now = get_pauli(bases[i], bases[j])
            A_all += T[i, j] * A_now
            B_all += T[i, j] * B_now
    
    ret_s, ret_c = [], []

    for i in A_all:
        s = i.get_pauli_word(n_qubits)
        c = i.evaluate_coefficient()
        if len(s) > 0 and c.real != 0:
            # ret_s.append(pauli_to_int(s))
            ret_s.append(s)
            # print(s)
            ret_c.append(c.real)
    
    return ret_s, np.array(ret_c)

def reversed_str_bases_to_init_state(bases: List[str], n_qb: int) -> np.ndarray:
    assert len(bases[0]) == n_qb, f"Length of bases: {len(bases[0])} must match number of qubits: {n_qb}"

    init_state = np.zeros(2**n_qb, dtype=cudaq.complex())
    for base in bases:
        base_i = int(base[::-1], 2)
        init_state[base_i] = 1.0 / sqrt(len(bases))
    return init_state

@cudaq.kernel
def kernel_qaoa_X(thetas: List[float], qubit_count: int, layer_count: int, idx_1: List[int], coeff_1: List[float], idx_2_a: List[int], idx_2_b: List[int], coeff_2: List[float]):
    qreg = cudaq.qvector(qubit_count)
    # qreg = cudaq.qvector(3)
    h(qreg)

    for i in range(layer_count):
        # for idxs, coeff, l in sorted_raw_ham:
        #     if l == 1:
        #         rz(2 * coeff * thetas[i], qreg[idxs[0]])
        #     elif l == 2:
        #         x.ctrl(qreg[idxs[0]], qreg[idxs[1]])
        #         rz(2 * coeff * thetas[i], qreg[idxs[1]])
        #         x.ctrl(qreg[idxs[0]], qreg[idxs[1]])
        # for i in range(qubit_count):
        #     rx(2.0 * thetas[layer_count + i], qreg[i])

        for j in range(len(idx_1)):
            rz(2 * coeff_1[j] * thetas[i], qreg[idx_1[j]])
        
        for j in range(len(idx_2_a)):
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
            rz(2 * coeff_2[j] * thetas[i], qreg[idx_2_b[j]])
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])

        for j in range(qubit_count):
            rx(2.0 * thetas[layer_count + i], qreg[j])
            
@cudaq.kernel
def kernel_qaoa_Preserved(thetas: List[float], qubit_count: int, layer_count: int, idx_1: List[int], coeff_1: List[float], idx_2_a: List[int], idx_2_b: List[int], coeff_2: List[float], mixer_str: List[cudaq.pauli_word], mixer_coeff: List[float], init_sup: List[complex]):
    # qreg = cudaq.qvector(qubit_count)
    # h(qreg)

    # qreg = cudaq.qvector([0.+0j, 0.577350269, 0.577350269, 0., 0.577350269, 0., 0., 0.])
    # qreg = cudaq.qvector([0.+0j, 0.577350269, 0.577350269, 0., 0., 0., 0., 0., 0.577350269, 0., 0., 0., 0., 0., 0., 0.])
    qreg = cudaq.qvector(init_sup)
    
    for i in range(layer_count):

        for j in range(len(idx_1)):
            rz(2 * coeff_1[j] * thetas[i], qreg[idx_1[j]])
        
        for j in range(len(idx_2_a)):
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
            rz(2 * coeff_2[j] * thetas[i], qreg[idx_2_b[j]])
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])

        for j in range(len(mixer_str)):
            exp_pauli(mixer_coeff[j] * thetas[layer_count + i], qreg, mixer_str[j])

        # for j in range(qubit_count):
        #     rx(2.0 * thetas[layer_count + i], qreg[j])

@cudaq.kernel
def kernel_flipped(state: cudaq.State, n_qb: int):
    q = cudaq.qvector(state)
    for i in range(n_qb//2):
        swap(q[i], q[n_qb - 1 - i])

# --- Utils/qaoaCUDAQ.py:670-677 (verbatim) ---
def get_init_states(state_return, N, n_qubits, feasible=None):
    sorted_idx = np.argsort(state_return)
    # print(state_return[sorted_idx[:N]])
    init_states = []
    for i in sorted_idx[:N]:
        init_states.append(bin(i)[2:].zfill(n_qubits))
    # print("state_return_last:", state_return[sorted_idx[N-1]])
    return init_states


# --- added helper (NOT verbatim): basis_T_to_pauli_parallel without the process pool ---
def basis_T_to_pauli_serial(bases, T, n_qubits):
    """Exactly the merge of `basis_T_to_pauli_parallel` (same `get_pauli_serializable` per transition, same
    (i, j) order -- `pool.starmap` preserves order -- same `summed_terms` dict), with a list comprehension
    in place of `Pool.starmap`, so a test never forks a process that holds a CUDA context."""
    indices = [(i, j) for i in range(T.shape[0]) for j in range(i + 1, T.shape[1]) if T[i, j] != 0]
    indices_bases = [(bases[i], bases[j]) for i in range(T.shape[0]) for j in range(i + 1, T.shape[1]) if T[i, j] != 0]
    worker_func = partial(get_pauli_serializable, n_qubits=n_qubits)
    results = [worker_func(*ib) for ib in indices_bases]
    summed_terms = defaultdict(float)
    for idx, (i, j) in enumerate(indices):
        serializable_A = results[idx]
        T_val = T[i, j]
        for coeff, pauli_word in serializable_A:
            summed_terms[pauli_word] += T_val * coeff
    ret_s, ret_c = [], []
    for pauli_word, final_coeff in summed_terms.items():
        if final_coeff != 0:
            ret_s.append(pauli_word)
            ret_c.append(final_coeff)
    return ret_s, np.array(ret_c)
