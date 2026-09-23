"""Pin BLAS/OpenMP thread pools to one thread (PLAN §3.3).

Import this module before numpy in every entry point. The 24-thread BLAS spin once cost 13x.
"""

import os

_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def pin_threads() -> None:
    for var in _VARS:
        os.environ[var] = "1"


pin_threads()
