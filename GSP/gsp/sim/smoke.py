"""GPU smoke test of the `gsp` env: `~/anaconda3/envs/gsp/bin/python -m gsp.sim.smoke`.

Runs observe / get_state / sample on the nvidia fp64 target at n = 8 and n = 20 through the
backend, checks the numbers against closed forms, and prints versions and timings. Refuses to
start if another process holds a CUDA compute context (one GPU process at a time).
"""

from .. import _threads  # noqa: F401

import sys
import time

import numpy as np


def main() -> int:
    from . import backend

    busy = backend.gpu_compute_pids()
    if busy:
        print(f"GPU busy (compute PIDs {busy}); not starting", file=sys.stderr)
        return 2
    from ..circuits.probe import kernel_basis, kernel_probe

    backend.set_target("nvidia", "fp64")
    info = backend.runtime_info()
    print(info)
    ok = True
    for n in (8, 20):
        th = [0.3] * n + [0.0] * (2 * n)          # Ry(0.3) on every qubit, CNOT chain, no Rz/Rx
        t0 = time.perf_counter()
        z0 = backend.observe(kernel_probe, backend.z_op(0), n, th)
        t1 = time.perf_counter()
        psi = backend.get_state(kernel_probe, n, th)
        t2 = time.perf_counter()
        counts = backend.sample(kernel_probe, n, th, shots_count=1000)
        t3 = time.perf_counter()
        norm = float(np.vdot(psi, psi).real)
        good = abs(z0 - np.cos(0.3)) < 1e-12 and abs(norm - 1) < 1e-12 and sum(counts.values()) == 1000
        ok &= good
        print(f"n={n:2d}  <Z0>={z0:.15f} (cos 0.3 = {np.cos(0.3):.15f})  |psi|^2-1={norm - 1:.1e}  "
              f"observe {t1 - t0:.3f}s  get_state {t2 - t1:.3f}s  sample {t3 - t2:.3f}s  "
              f"{'OK' if good else 'FAIL'}")
    bits = [1, 0, 0, 1, 1, 0, 1, 0]
    counts = backend.sample(kernel_basis, 8, bits, shots_count=10)
    good = list(counts) == ["10011010"]
    ok &= good
    print(f"bit order: sample key {list(counts)} for x = 10011010 -> {'OK' if good else 'FAIL'}")
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
