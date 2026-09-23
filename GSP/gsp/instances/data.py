"""The market data behind every instance (PLAN §1.1, port of PO_new_ApproxRatio.py:386-394).

Covariance: dataset/top_50_us_stocks_data_20250526_011226_covariance.csv
Return/price: dataset/top_50_us_stocks_returns_price.csv
Price filter strictly 108 < Price < 216; the covariance is restricted to the filtered tickers,
in their order. The CSVs are read with pandas' default parser, as the completed work did.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COV_FILE = "top_50_us_stocks_data_20250526_011226_covariance.csv"
RET_FILE = "top_50_us_stocks_returns_price.csv"
PRICE_MIN, PRICE_MAX = 108, 216


def default_dataset_dir() -> Path:
    env = os.environ.get("GSP_DATASET_DIR")
    if env:
        return Path(env)
    # GSP/gsp/instances/data.py -> repo root is three levels above the package dir
    return Path(__file__).resolve().parents[3] / "dataset"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Market:
    """The filtered market, in the row order the draw code indexes."""

    tickers: np.ndarray        # (M,) str
    names: np.ndarray          # (M,) str, Company_Name
    raw_index: np.ndarray      # (M,) int, row labels in the returns CSV (the old `asset_idx_raw`)
    ret: np.ndarray            # (M,) float64, Average_Return
    price: np.ndarray          # (M,) float64, Price
    cov: np.ndarray            # (M, M) float64
    sources: dict              # {file name: sha256}

    @property
    def size(self) -> int:
        return len(self.tickers)


def load_market(dataset_dir=None) -> Market:
    d = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    cov_path, ret_path = d / COV_FILE, d / RET_FILE
    # --- PO_new_ApproxRatio.py:386-394, same pandas calls ---
    data_cov_pd = pd.read_csv(cov_path)
    data_ret_p_pd = pd.read_csv(ret_path)
    data_ret_p_pd = data_ret_p_pd[(data_ret_p_pd["Price"] > PRICE_MIN) & (data_ret_p_pd["Price"] < PRICE_MAX)]
    data_cov_pd = data_cov_pd.loc[data_cov_pd["Ticker"].isin(data_ret_p_pd["Ticker"])].reset_index(drop=True)
    data_cov_pd = data_cov_pd[["Ticker"] + data_cov_pd["Ticker"].tolist()]
    # ---
    if list(data_cov_pd["Ticker"]) != list(data_ret_p_pd["Ticker"]):
        raise ValueError("covariance and return CSVs list the filtered tickers in different orders")
    cov = data_cov_pd.drop("Ticker", axis=1).to_numpy()
    rp = data_ret_p_pd.drop("Ticker", axis=1)
    raw_index = rp.index.to_numpy()
    names = rp["Company_Name"].to_numpy().astype(str)
    num = rp.drop("Company_Name", axis=1).to_numpy()
    return Market(
        tickers=data_ret_p_pd["Ticker"].to_numpy().astype(str),
        names=names,
        raw_index=np.asarray(raw_index, dtype=np.int64),
        ret=np.ascontiguousarray(num[:, 0], dtype=np.float64),
        price=np.ascontiguousarray(num[:, 1], dtype=np.float64),
        cov=np.ascontiguousarray(cov, dtype=np.float64),
        sources={COV_FILE: sha256_file(cov_path), RET_FILE: sha256_file(ret_path)},
    )
