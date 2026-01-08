"""
weighted_qaoa.py

A module for constructing a weighted QAOA circuit and corresponding Hamiltonian
for the Max-Cut problem using CUDAQ and NetworkX.

This module includes:
- QAOA kernels for applying weighted problem and mixer unitaries.
- A function to generate the Hamiltonian for the weighted Max-Cut problem.
- An adapter function to extract graph parameters from a NetworkX graph and run the QAOA circuit.
"""

import networkx as nx
from typing import List, Tuple
import numpy as np
from math import sqrt, pi
import math
import pandas as pd
import os
# Import cudaq and its associated spin operators.
import cudaq
from cudaq import spin
import psutil
#from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import threading
import time
from collections import defaultdict
from functools import partial

# QAOA subkernel for a weighted edge rotation.
@cudaq.kernel
def qaoaProblem(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit, alpha: float):
    """
    Build the QAOA gate sequence between two qubits (representing an edge)
    with a weighted rotation.
    
    Parameters
    ----------
    qubit_0 : cudaq.qubit
        Qubit corresponding to the first vertex of an edge.
    qubit_1 : cudaq.qubit
        Qubit corresponding to the second vertex of an edge.
    weighted_alpha : float
        Angle parameter multiplied by the edge's weight.
    """
    # Apply a weighted controlled-Z-like rotation:
    x.ctrl(qubit_0, qubit_1)
    rz(2.0 * alpha, qubit_1)
    x.ctrl(qubit_0, qubit_1)

# Main QAOA kernel for the weighted Max-Cut problem.
@cudaq.kernel
def kernel_qaoa(qubit_count: int, layer_count: int, edges_src: List[int],
                edges_tgt: List[int], thetas: List[float]):
    """
    Build the QAOA circuit for weighted max cut of a graph.
    
    Parameters
    ----------
    qubit_count : int
        Number of qubits (same as number of nodes).
    layer_count : int
        Number of QAOA layers.
    edges_src : List[int]
        List of source nodes for each edge.
    edges_tgt : List[int]
        List of target nodes for each edge.
    thetas : List[float]
        Free parameters to be optimized (length should be 2 * layer_count).
    """
    # Allocate qubits in a quantum register.
    qreg = cudaq.qvector(qubit_count)
    # Place qubits in an equal superposition state.
    h(qreg)

    # QAOA circuit with alternating problem and mixer layers.
    for i in range(layer_count):
        # Apply the weighted problem unitary for each edge.
        for edge in range(len(edges_src)):
            qubit_u = edges_src[edge]
            qubit_v = edges_tgt[edge]
            qaoaProblem(qreg[qubit_u], qreg[qubit_v], thetas[i])
        # Apply the mixer unitary on all qubits.
        for j in range(qubit_count):
            rx(2.0 * thetas[i + layer_count], qreg[j])

def create_qaoa_networkx(G: nx.Graph, is_weighted: bool = True) -> Tuple[List[Tuple[int, int]], List[int], List[int], List[float]]:
    """
    Prepare and run the QAOA circuit from a weighted NetworkX graph.
    
    Parameters
    ----------
    G : nx.Graph
        A weighted graph where edge weights are stored under the key 'weight'.
    is_weighted : bool
        Flag to determine whether the graph is weighted.
    """
    # Extract nodes and determine the qubit count.
    nodes = list(G.nodes())
    qubit_count = len(nodes)

    # Extract edge information.
    edges = list(G.edges())
    edges_src: List[int] = []
    edges_tgt: List[int] = []
    edge_weights: List[float] = []
    for u, v in edges:
        edges_src.append(u)
        edges_tgt.append(v)
        # Retrieve the weight for the edge; default to 1.0 if not provided.
        weight = G[u][v].get('weight', 1.0)
        edge_weights.append(weight if is_weighted else 1.0)

    return (edges, edges_src, edges_tgt, edge_weights)

def hamiltonian_max_cut(edges_src: List[int], edges_tgt: List[int],
                        edge_weights: List[float]) -> cudaq.SpinOperator:
    """
    Generate the Hamiltonian for finding the max cut of a weighted graph.
    
    Parameters
    ----------
    edges_src : List[int]
        List of the first (source) node for each edge.
    edges_tgt : List[int]
        List of the second (target) node for each edge.
    edge_weights : List[float]
        List of weights for each edge.
    
    Returns
    -------
    cudaq.SpinOperator
        Hamiltonian for finding the max cut of the weighted graph.
    """
    hamiltonian = 0
    for edge in range(len(edges_src)):
        qubitu = edges_src[edge]
        qubitv = edges_tgt[edge]
        weight = edge_weights[edge]
        # Multiply the term by the weight of the edge.
        hamiltonian += 0.5 * weight * (spin.z(qubitu) * spin.z(qubitv) -
                                       spin.i(qubitu) * spin.i(qubitv))
    return hamiltonian


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

def state_to_return(s, QU, lamb):
    l = np.array(list(map(int, s)))
    ss = l @ QU @ l.T
    return ss + lamb

def pauli_to_int(pauli_str: str) -> int:
    value = 0
    for i, char in enumerate(pauli_str):
        if char == 'I':
            value |= (1 << (2 * i)) * 0
        elif char == 'X':
            value |= (1 << (2 * i)) * 1
        elif char == 'Y':
            value |= (1 << (2 * i)) * 2
        elif char == 'Z':
            value |= (1 << (2 * i)) * 3
    return value

def int_to_pauli(value: int, n_qubits: int) -> str:
    pauli_str = ""
    for i in range(n_qubits):
        if (value >> 2*i) % 4 == 0:
            pauli_str += 'I'
        elif (value >> 2*i) % 4 == 1:
            pauli_str += 'X'
        elif (value >> 2*i) % 4 == 2:
            pauli_str += 'Y'
        elif (value >> 2*i) % 4 == 3:
            pauli_str += 'Z'
    return pauli_str

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
    print(f"Done by thread    : '{threading.current_thread().name}'")
    return A

def _get_pauli_objects(X, Y):
    A, B = init_pauli(X[0], Y[0])
    for i in range(1, len(X)):
        A, B = transform_pauli(X[i], Y[i], i, A, B)
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
        return serializable_A
        
def basis_T_to_pauli_parallel(bases: List[str], T: np.ndarray, n_qubits: int) -> Tuple[List[cudaq.pauli_word], np.ndarray]:
    print("Compute pauli")
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
        print("Working with chunksize:", chunksize)
        results = pool.starmap(worker_func, indices_bases, chunksize=chunksize)

    print("Pauli compute:", time.time() - st_pauli_compute)

    print("Start merging...")
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

    print("Merge done in:", time.time() - st_merge)

    st_list = time.time()
    ret_s, ret_c = [], []
    for pauli_word, final_coeff in summed_terms.items():
        if final_coeff != 0:
            ret_s.append(pauli_word)
            ret_c.append(final_coeff)
    print("Listing done in:", time.time() - st_list)

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

@cudaq.kernel
def kernel_cmpz_Preserved(thetas: List[float], qubit_count: int, layer_count: int, params: List[float], init_sup: List[complex]):
    qreg = cudaq.qvector(init_sup)

    for i in range(layer_count):
        for j in range(len(params) // 6):
            typee = params[6 * j]
            idx = int(params[6 * j + 1])
            zeta = params[6*j+2] + thetas[i] * params[6*j+3] + thetas[i+layer_count] * params[6*j+4]

            if typee  == 0: # RX
                rx(zeta, qreg[idx])
            elif typee == 1: # RY
                ry(zeta, qreg[idx])
            elif typee == 2: # RZ
                rz(zeta, qreg[idx])
            elif typee == 3: # H
                h(qreg[idx])
            elif typee == 4: # CX-control
                cx(qreg[idx], qreg[int(params[6*j+5])])


def prepare_preserving_ansatz(qubit_count: int, idx_1: List[int], coeff_1: List[float], idx_2_a: List[int], idx_2_b: List[int], coeff_2: List[float], mixer_str: List[cudaq.pauli_word], mixer_coeff: List[float]):
    def generate_list():
        return [[] for _ in range(qubit_count)]
    type_l, zeta_l, entang_l, runnum_l = [generate_list() for _ in range(4)]
    cou = 0
    all_gate, mk = [], []
    def remove(idx):
        tp = type_l[idx].pop()
        zt = zeta_l[idx].pop()
        et = entang_l[idx].pop()
        rn = runnum_l[idx].pop()
        mk[rn] = False
        # print("-", idx, rn, tp, zt, et)
    def add(typee, idx, zeta, idx_en):
        nonlocal cou
        # print("+", idx, cou, typee, zeta, idx_en)
        type_l[idx].append(typee)
        zeta_l[idx].append(zeta)
        entang_l[idx].append(idx_en)
        runnum_l[idx].append(cou)
        mk.append(typee != 5)
        all_gate.append([typee, idx, *zeta, idx_en])
        cou += 1
    def nplize(a):
        for i in range(len(a)):
            a[i] = np.array(a[i], dtype=np.float32)
        return a
    def is_zero(val, bound=1e-8):
        return abs(val) < bound
    def push(typee, idx, zeta=[0, 0, 0], idx_en=-1): # typee[0: RX, 1: RY, 2: RZ, 3: H, 4: CX-control, 5: CX-target], zeta: (const_coeff, problem_coeff, mixer_coeff)
        nonlocal cou
        # print("*", idx, cou, typee, zeta, idx_en)
        if len(type_l[idx]) > 0 and type_l[idx][-1] == typee:
            if typee <= 3:
                if not (is_zero(zeta_l[idx][-1][0] + zeta[0]) and is_zero(zeta_l[idx][-1][1] + zeta[1]) and is_zero(zeta_l[idx][-1][2] + zeta[2])):
                    for i in range(3):
                        zeta_l[idx][-1][i] += zeta[i]
                        all_gate[runnum_l[idx][-1]][i + 2] = zeta_l[idx][-1][i]
                    print("^", idx, runnum_l[idx][-1], typee, zeta_l[idx][-1])
                else:
                    remove(idx)
            if typee == 4:
                if len(entang_l[entang_l[idx][-1]]) > 0 and idx_en == entang_l[idx][-1] and idx == entang_l[entang_l[idx][-1]][-1]:
                    remove(idx)
                    remove(entang_l[idx][-1])
                else:
                    add(typee, idx, zeta, idx_en)
                    add(5, idx_en, [0, 0, 0], idx)
        else:
            add(typee, idx, zeta, idx_en)
            if typee == 4:
                add(5, idx_en, [0, 0, 0], idx)

            # print(idx)
    def push_pauli_string(strr, coeff):
        for i, p in enumerate(strr):
            if p == "X":
                push(3, i)
            elif p == "Y":
                push(0, i, [pi/2, 0, 0])
                # push(3, i)
        ll = -1
        for i, p in enumerate(strr):
            if p in ["X", "Y", "Z"]:
                if ll != -1:
                    push(4, ll, idx_en=i)
                    # push(5, i, idx_en=ll)
                ll = i
        push(2, ll, [0, 0, -2 * coeff])
        ll = -1
        for i, p in reversed(list(enumerate(strr))):
            if p in ["X", "Y", "Z"]:
                if ll != -1:
                    push(4, i, idx_en=ll)
                    # push(5, ll, idx_en=i)
                ll = i
        for i, p in enumerate(strr):
            if p == "X":
                push(3, i)
            elif p == "Y":
                # push(3, i)
                push(0, i, [-pi/2, 0, 0])
    
    for j in range(len(idx_1)):
        push(2, idx_1[j], [0, 2 * coeff_1[j], 0])
    for j in range(len(idx_2_a)):
        push(4, idx_2_a[j], idx_en=idx_2_b[j])
        # push(5, idx_2_b[j], idx_en=idx_2_a[j])
        push(2, idx_2_b[j], [0, 2 * coeff_2[j], 0])
        push(4, idx_2_a[j], idx_en=idx_2_b[j])
        # push(5, idx_2_b[j], idx_en=idx_2_a[j])
    for j in range(len(mixer_str)):
        push_pauli_string(mixer_str[j], mixer_coeff[j])

    # print(all_gate[6])
    return [nplize(i) for i in[type_l, zeta_l, entang_l, runnum_l]] + nplize([all_gate, mk])

optimizer_names = ["Nelder-Mead", "COBYLA", "SPSA", "Adam", "GradientDescent"]
def get_optimizer(idx):
    optimizer1 = cudaq.optimizers.NelderMead()
    optimizer2 = cudaq.optimizers.COBYLA()
    optimizer3 = cudaq.optimizers.SPSA()
    optimizer4 = cudaq.optimizers.Adam()
    optimizer5 = cudaq.optimizers.GradientDescent()

    optimizer = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5][idx]
    optimizer_name = optimizer_names[idx]
    FIND_GRAD = True if optimizer.requires_gradients() else False
    return optimizer, optimizer_name, FIND_GRAD

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

def get_init_states(state_return, N, n_qubits):
    sorted_idx = np.argsort(state_return)
    # print(state_return[sorted_idx[:N]])
    init_states = []
    for i in sorted_idx[:N]:
        init_states.append(bin(i)[2:].zfill(n_qubits))
    # print("state_return_last:", state_return[sorted_idx[N-1]])
    return init_states

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

def write_df(df_dir, report_col, *data, idx=None):
    if idx is None or idx is not None and ((idx == 0 and not os.path.exists(df_dir)) or (os.path.exists(df_dir) and pd.read_csv(df_dir).shape[0] == idx)):
        df_new = pd.DataFrame(np.array(data).reshape(1, -1), columns=report_col)
        m_df = os.path.exists(df_dir)
        # print(df_dir)
        df_new.to_csv(df_dir, mode='a' if m_df else 'w', header=(not m_df), index=False)
    else:
        assert os.path.exists(df_dir), f"CSV file does not exist. (write idx: {idx})"
        df = pd.read_csv(df_dir)
        assert df.shape[0] > idx, f"df shape: {df.shape}, writing idx: {idx}"
        df.iloc[idx] = np.array(data).reshape(1, -1)
        df.to_csv(df_dir, index=False)

def clip_df(df, restore_iter):
    assert restore_iter <= len(df), "restore_iter exceeds the number of iterations in the CSV file."
    df = df.iloc[:restore_iter]
    return df


# Optional: a test routine when the module is executed as a script.
if __name__ == "__main__":
    # Create a weighted graph using NetworkX.
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(2, 3, weight=3.0)
    G.add_edge(3, 0, weight=4.0)
    G.add_edge(2, 4, weight=5.0)
    G.add_edge(3, 4, weight=6.0)

    # Define the number of layers and free parameters (2 per layer).
    layer_count = 5
    thetas = [0.1] * (2 * layer_count)  # Example parameter values

    # Run the QAOA circuit using the NetworkX graph.
    create_qaoa_networkx(G, layer_count, thetas)

    # For demonstration, extract edge parameters and build the Hamiltonian.
    edges = list(G.edges())
    edges_src = [u for u, _ in edges]
    edges_tgt = [v for _, v in edges]
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in edges]
    ham = hamiltonian_max_cut(edges_src, edges_tgt, edge_weights)
    print("Hamiltonian:", ham)
