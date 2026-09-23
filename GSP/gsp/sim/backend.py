"""The only module that calls cudaq.set_target / observe / sample / get_state (PLAN §3.5).

Every simulator call of the harness goes through here, so the CUDA-Q 0.16 migration (whose
`sample`/`observe` API is announced to change) is a change to this file plus the self-check.
Spin operators are also built here (`spin_op`), for the same reason.

The update loops of the arms may use `observe` and `sample` only. `get_state` is for the
post-update logger, post-run metrics and tests (PLAN §1.6, §3.3).

cudaq is imported lazily: instance freezing and the other CPU-only tools never load it.

Gate fusion (S4, measured on 0.15.1 / RTX 4080, STATUS S4): the `nvidia` fp64 simulator (cusvsim) fuses gates
up to `CUDAQ_FUSION_MAX_QUBITS` qubits (default 4). For the harness circuits a limit of **1** is the fastest at
every n = 8 ... 20 (3.7x at n = 20 for A0: 70 ms -> 19 ms per circuit; 1.1-1.4x at n <= 18), and the states agree
with the unfused ones to <= 1e-15. `set_target` therefore sets it to 1 unless the variable is already set, and
`runtime_info` records the value in run.json.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache

import numpy as np

from ..instances.bits import cudaq_to_classical

DEFAULT_TARGET = "nvidia"
DEFAULT_OPTION = "fp64"
FUSION_ENV = "CUDAQ_FUSION_MAX_QUBITS"
DEFAULT_FUSION_MAX_QUBITS = "1"

_state = {"target": None, "option": None}


def _cudaq():
    import cudaq  # noqa: WPS433 (lazy on purpose)
    return cudaq


def set_target(name: str = DEFAULT_TARGET, option: str | None = DEFAULT_OPTION) -> None:
    """Select the simulator. Default: `nvidia` with option `fp64` (PLAN §3.3), gate fusion limited to one
    qubit (module docstring) unless CUDAQ_FUSION_MAX_QUBITS is already set."""
    os.environ.setdefault(FUSION_ENV, DEFAULT_FUSION_MAX_QUBITS)
    cq = _cudaq()
    if option:
        cq.set_target(name, option=option)
    else:
        cq.set_target(name)
    _state["target"], _state["option"] = name, option


def ensure_target() -> None:
    if _state["target"] is None:
        set_target()


def current_target() -> tuple[str | None, str | None]:
    return _state["target"], _state["option"]


def observe(kernel, op, *args) -> float:
    """<psi(args)| op |psi(args)> (exact on a statevector target, no shots)."""
    ensure_target()
    return float(_cudaq().observe(kernel, op, *args).expectation())


def sample(kernel, *args, shots_count: int = 1000) -> dict[str, int]:
    """Measurement counts. Keys are bitstrings with qubit 0 first, i.e. x_0 first: read as a
    binary number they are the classical index (`gsp.instances.bits`)."""
    ensure_target()
    result = _cudaq().sample(kernel, *args, shots_count=shots_count)
    return {k: int(v) for k, v in result.items()}


def get_state(kernel, *args) -> np.ndarray:
    """Statevector in CUDA-Q order (q_0 = LSB). Not for update loops."""
    ensure_target()
    return np.array(_cudaq().get_state(kernel, *args), dtype=np.complex128, copy=True)


def get_state_classical(kernel, n: int, *args) -> np.ndarray:
    """Statevector reordered to classical order (x_0 = MSB), matching the rulers' indices."""
    return cudaq_to_classical(get_state(kernel, *args), n)


def spin_op(const: float, h, J, n: int | None = None):
    """Build `const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j` as a cudaq SpinOperator.

    Terms are added in the order of `Ising.terms()`; zero coefficients are skipped. The
    identity is attached to qubit 0 as in the old `qubo_to_ising`.
    """
    cq = _cudaq()
    spin = cq.spin
    h = np.asarray(h, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    n = len(h) if n is None else n
    op = float(const) * spin.i(0)
    for i in range(n):
        if h[i] != 0:
            op += float(h[i]) * spin.z(i)
    for a in range(n):
        for b in range(a + 1, n):
            if J[a, b] != 0:
                op += float(J[a, b]) * spin.z(a) * spin.z(b)
    return op


def ising_op(H, alpha: float = 1.0):
    """SpinOperator of a `gsp.instances.encode.Ising`, times alpha (the boost)."""
    Hs = H.scaled(alpha) if alpha != 1.0 else H
    return spin_op(Hs.const, np.where(Hs.has_h, Hs.h, 0.0), np.where(Hs.has_J, Hs.J, 0.0), Hs.n)


def z_op(i: int):
    return _cudaq().spin.z(i)


@lru_cache(maxsize=1)
def _nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=20).stdout.strip().splitlines()
        name, driver = [s.strip() for s in out[0].split(",")]
        return {"gpu_name": name, "driver_version": driver}
    except Exception:
        return {"gpu_name": None, "driver_version": None}


def runtime_info() -> dict:
    """What run.json records about the simulator (PLAN §3.5)."""
    cq = _cudaq()
    info = {"cudaq_version": cq.__version__, "target": _state["target"],
            "target_option": _state["option"], "fusion_max_qubits": os.environ.get(FUSION_ENV)}
    info.update(_nvidia_smi())
    return info


def gpu_compute_pids() -> list[int]:
    """PIDs of processes holding a CUDA compute context (the `nvidia-smi` guard), minus ours."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=20).stdout
    except Exception:
        return []
    me = os.getpid()
    return [int(t) for t in out.split() if t.strip().isdigit() and int(t) != me]
