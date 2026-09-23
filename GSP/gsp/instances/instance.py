"""Loading frozen instances (the read side of `gsp instances freeze`).

    inst = load_instance("N05e003q1.5")
    H = inst.hamiltonian(lam)      # H(lam) = H_obj + lam Pen, old convention, un-boosted Ising
    alpha = jh_boost(H)            # the boost of the Hamiltonian being run
    rul = load_rulers("N05e003q1.5")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..store.io import load_npz
from ..store.paths import inst_path, instances_parquet_path, rulers_path, seed_table_path
from .encode import Ising, hamiltonian, qubo_lambda
from .rulers import Rulers


def _ising(z: dict, prefix: str) -> Ising:
    h = z[f"{prefix}_h"]
    return Ising(n=int(h.size), const=float(z[f"{prefix}_const"]), h=h, J=z[f"{prefix}_J"],
                 has_h=z[f"{prefix}_has_h"], has_J=z[f"{prefix}_has_J"])


@dataclass(frozen=True)
class Instance:
    inst_id: str
    draw_id: str
    N: int
    e: int
    q: float
    n: int
    eps: float
    B: float
    P: np.ndarray
    ret: np.ndarray
    cov: np.ndarray
    tickers: np.ndarray
    P_bb: np.ndarray
    ret_bb: np.ndarray
    cov_bb: np.ndarray
    QU_obj: np.ndarray
    QU_pen: np.ndarray
    H_obj: Ising
    Pen: Ising
    boost_obj: float
    arrays: dict

    def qubo(self, lam: float) -> np.ndarray:
        """QU(lam, q) of the MAX problem, the old `ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lam, q)`."""
        return qubo_lambda(self.ret_bb, self.cov_bb, self.P_bb, lam, self.q)

    def hamiltonian(self, lam: float) -> Ising:
        """H(lam) (MIN problem, un-boosted), built exactly as the old `-qubo_to_ising(QU, lam)`."""
        if lam == 0:
            return self.H_obj
        return hamiltonian(self.ret_bb, self.cov_bb, self.P_bb, lam, self.q)


def instance_from_arrays(z: dict) -> Instance:
    return Instance(
        inst_id=str(z["inst_id"]), draw_id=str(z["draw_id"]), N=int(z["N"]), e=int(z["e"]),
        q=float(z["q"]), n=int(z["n"]), eps=float(z["eps"]), B=float(z["B"]),
        P=z["P"], ret=z["ret"], cov=z["cov"], tickers=z["tickers"],
        P_bb=z["P_bb"], ret_bb=z["ret_bb"], cov_bb=z["cov_bb"],
        QU_obj=z["QU_obj"], QU_pen=z["QU_pen"],
        H_obj=_ising(z, "obj"), Pen=_ising(z, "pen"), boost_obj=float(z["boost_obj"]), arrays=z,
    )


def load_instance(inst_id: str, root=None) -> Instance:
    return instance_from_arrays(load_npz(inst_path(inst_id, root)))


def rulers_from_arrays(z: dict) -> Rulers:
    return Rulers(
        n=int(z["n"]), eps=float(z["eps"]), band_idx=z["band_idx"].astype(np.int64),
        band_pen=z["band_pen"], f_band=z["f_band"], E_min=float(z["E_min"]), E_max=float(z["E_max"]),
        xstar_idx=z["xstar_idx"].astype(np.int64), top10_idx=z["top10_idx"].astype(np.int64),
        top10_size=int(z["top10_size"]), gap_band=float(z["gap_band"]), n_distinct=int(z["n_distinct"]),
        F_size=int(z["F_size"]), f_all_min=float(z["f_all_min"]), f_all_max=float(z["f_all_max"]),
        n_direct_disagree=int(z["n_direct_disagree"]), n_near_boundary=int(z["n_near_boundary"]),
    )


def load_rulers(inst_id: str, root=None) -> Rulers:
    return rulers_from_arrays(load_npz(rulers_path(inst_id, root)))


def load_instances_table(root=None) -> pd.DataFrame:
    return pd.read_parquet(instances_parquet_path(root))


def load_seed_table(root=None) -> pd.DataFrame:
    """The seed table with exact floats (written as repr, read back round-trip)."""
    return pd.read_csv(seed_table_path(root), float_precision="round_trip",
                       dtype={"asset_idx": str, "asset_idx_raw": str, "tickers": str})
