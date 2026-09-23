"""Small test kernels for the backend and bit-order checks (S1). Not an arm circuit.

Kernels live only in gsp/circuits/ (PLAN §3.5), and `@cudaq.kernel` needs its source in a file.
"""

import cudaq


@cudaq.kernel
def kernel_basis(n: int, bits: list[int]):
    """|x> with x_i = bits[i] on qubit i."""
    q = cudaq.qvector(n)
    for i in range(n):
        if bits[i] == 1:
            x(q[i])


@cudaq.kernel
def kernel_probe(n: int, thetas: list[float]):
    """A generic entangled state: Ry layer, CNOT chain, Rz + Rx layer (3n angles)."""
    q = cudaq.qvector(n)
    for i in range(n):
        ry(thetas[i], q[i])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    for i in range(n):
        rz(thetas[n + i], q[i])
        rx(thetas[2 * n + i], q[i])
