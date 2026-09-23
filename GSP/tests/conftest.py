"""Shared test setup.

- BLAS/OpenMP threads are pinned to 1 before numpy loads (PLAN §3.3), so energy sums are computed
  exactly as in the harness entry points.
- `slow` tests run only with --runslow.
- `gpu` tests run only if CUDA-Q imports and no other process holds a CUDA compute context
  (the one-GPU-process rule, PLAN §3.3); otherwise they are skipped with the reason.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked slow")


def _gpu_block_reason():
    try:
        import cudaq  # noqa: F401
    except Exception as exc:  # pragma: no cover
        return f"cudaq not importable: {exc!r}"
    from gsp.sim.backend import gpu_compute_pids
    pids = gpu_compute_pids()
    if pids:
        return f"GPU busy (compute PIDs {pids}); one GPU process at a time"
    return None


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    gpu_reason = None
    gpu_checked = False
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(pytest.mark.skip(reason="slow: use --runslow"))
        if "gpu" in item.keywords:
            if not gpu_checked:
                gpu_reason = _gpu_block_reason()
                gpu_checked = True
            if gpu_reason:
                item.add_marker(pytest.mark.skip(reason=gpu_reason))


@pytest.fixture(scope="session")
def market():
    from gsp.instances.data import load_market
    return load_market()
