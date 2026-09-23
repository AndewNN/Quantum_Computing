"""`kernel_qaoa_X` of the completed work, copied verbatim from Utils/qaoaCUDAQ.py:462-488 (committed at 15965ad;
debug comments dropped, nothing else changed). The harness runs A0 through the layered kernel; this copy is the
reference of the equivalence test and the "before" engine of the S4 timing.
"""

from typing import List

import cudaq


@cudaq.kernel
def kernel_qaoa_X(thetas: List[float], qubit_count: int, layer_count: int, idx_1: List[int], coeff_1: List[float], idx_2_a: List[int], idx_2_b: List[int], coeff_2: List[float]):
    qreg = cudaq.qvector(qubit_count)
    h(qreg)

    for i in range(layer_count):
        for j in range(len(idx_1)):
            rz(2 * coeff_1[j] * thetas[i], qreg[idx_1[j]])

        for j in range(len(idx_2_a)):
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])
            rz(2 * coeff_2[j] * thetas[i], qreg[idx_2_b[j]])
            x.ctrl(qreg[idx_2_a[j]], qreg[idx_2_b[j]])

        for j in range(qubit_count):
            rx(2.0 * thetas[layer_count + i], qreg[j])


def fixed_args(ct, L: int) -> tuple:
    """The old `ansatz_fixed_param` of mode X: (n, L, idx_1, coeff_1, idx_2_a, idx_2_b, coeff_2)."""
    return (int(ct.n), int(L), list(ct.idx_1), list(ct.coeff_1), list(ct.idx_2a), list(ct.idx_2b), list(ct.coeff_2))
