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

# Import cudaq and its associated spin operators.
import cudaq
from cudaq import spin

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
    P_b = P / B
    ret_b = ret * P_b
    cov_b = np.diag(P_b) @ cov @ np.diag(P_b)
    
    n_max = np.int32(np.floor(np.log2(B/P))) + 1
    print("n_max:", n_max)
    n_qs = np.cumsum(n_max)
    n_qs = np.insert(n_qs, 0, 0)
    n_qubit = n_qs[-1]
    C = np.zeros((len(P), n_qubit))
    for i in range(len(P)):
         for j in range(n_max[i]):
              C[i, n_qs[i] + j] = 2**j

    P_bb = C.T @ P_b
    ret_bb = C.T @ ret_b
    print("ret_bb:", ret_bb)
    cov_bb = C.T @ cov_b @ C
    return P_bb, ret_bb, cov_bb, int(n_qubit)

def ret_cov_to_QUBO(ret: np.ndarray, cov: np.ndarray, P: np.ndarray, lamb: float, q:float) -> np.ndarray:
    di = np.diag(ret + lamb * (P*P + 2*P))
    mat = 2 * lamb * np.outer(P, P) + q * cov
    return di - mat

def qubo_to_ising(qubo: np.ndarray, lamb: float) -> cudaq.SpinOperator:
    spin_op = lamb
    for i in range(qubo.shape[0]):
        for j in range(qubo.shape[1]):
                if i != j:
                    spin_op += qubo[i, j] * ((spin.i(i) - spin.z(i)) / 2 * (spin.i(j) - spin.z(j)) / 2)
                else:
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
    for i in range(len(HH)):
        if HH[i][2] == 1:
            idx_1.append(HH[i][0][0])
            coeff_1.append(HH[i][1].real)
        elif HH[i][2] == 2:
            idx_2_a.append(HH[i][0][0])
            idx_2_b.append(HH[i][0][1])
            coeff_2.append(HH[i][1].real)
    # print(HH)

    return idx_1, coeff_1, idx_2_a, idx_2_b, coeff_2






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