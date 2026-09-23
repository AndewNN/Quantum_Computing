"""The draw recipe, seeds, acceptance and ids of PLAN §1.1.

Ports (copied, not imported): `find_budget` from Utils/qaoaCUDAQ.py:679-726 and the draw from
CUDA/PO_new_ApproxRatio.py:446-485. The RNG calls are replayed in the same order on a private
`np.random.RandomState` seeded like the old global `np.random.seed` (same Mersenne-Twister
stream): asset choice, then `set_state` back to the seeded state, then the budget weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import PRICE_MAX, PRICE_MIN, Market

# --- the fixed design (PLAN §1.1) ---
QUBITS_PER_ASSET = 2
EPS = 0.1
N_VALUES = tuple(range(4, 11))
DRAWS_PER_N = 30
Q_VALUES = (1.0, 1.5, 3.0)
K_REQ = 12                                     # |F_eps| >= 12 at every N (D-14, revised in S1b)
K24 = 24                                       # the K = 24 cells (N = 5, 6) use accepted draws
K24_CELL_N = (5, 6)                            #   with |F_eps| >= 24 only (PLAN §1.1, §1.2)
N_RESTARTS = 5
GA_RULES = ("violation", "objective")          # rule_idx 0, 1
DUPLICATE_ASSET = False                        # the old default
MAX_SEED_INDEX = 10_000                        # safety cap on e while looking for 30 accepted draws


def draw_seed(N: int, e: int) -> int:
    return 911 + 991 * e + 997 * N


def restart_seed(N: int, e: int, r: int) -> int:
    return 4001 + 4099 * e + 4999 * N + 5099 * r


def ga_seed(N: int, e: int, rule_idx: int) -> int:
    return 6007 + 6101 * e + 6199 * N + 6203 * rule_idx


def k_req(N: int) -> int:
    """Acceptance threshold on |F_eps|: 12 at every N (S1b; S1 had 24 at N = 5, 6)."""
    return K_REQ


def k24_eligible(F_eps: int, accepted: bool = True) -> bool:
    """An accepted draw whose band can host a K = 24 sector (used by the N = 5, 6 K = 24 cells)."""
    return bool(accepted) and int(F_eps) >= K24


def draw_id(N: int, e: int) -> str:
    return f"N{N:02d}e{e:03d}"


def q_tag(q: float) -> str:
    """1.0 -> '1.0', 1.5 -> '1.5', 3.0 -> '3.0' (PLAN example `N05e003q1.5`)."""
    return repr(float(q))


def inst_id(N: int, e: int, q: float) -> str:
    return f"{draw_id(N, e)}q{q_tag(q)}"


# --- Utils/qaoaCUDAQ.py:679-726, copied verbatim (only indentation-neutral) ---
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
# --- end of copy ---


@dataclass(frozen=True)
class Draw:
    N: int
    e: int
    seed: int
    asset_idx: np.ndarray       # positions in the filtered market (the old `asset_idx`)
    asset_idx_raw: np.ndarray   # row labels in the returns CSV (the old `asset_idx_raw`)
    tickers: np.ndarray
    names: np.ndarray
    P: np.ndarray               # prices
    ret: np.ndarray
    cov: np.ndarray
    w: float                    # the budget weight
    B_min: float
    B_max: float
    B: float

    @property
    def draw_id(self) -> str:
        return draw_id(self.N, self.e)


def make_draw(market: Market, N: int, e: int) -> Draw:
    """Replay PO_new_ApproxRatio.py:446-485 for (N, e) with Q = 2 qubits per asset."""
    seed = draw_seed(N, e)
    rng = np.random.RandomState(seed)
    state = rng.get_state()
    asset_idx = rng.choice(market.size, N, replace=DUPLICATE_ASSET)
    cov = market.cov[asset_idx, :][:, asset_idx]
    P = market.price[asset_idx]
    ret = market.ret[asset_idx]
    rng.set_state(state)
    w = rng.uniform(0, 1)
    B_mi, B_ma = find_budget(QUBITS_PER_ASSET * N, P, PRICE_MIN, PRICE_MAX, min_mix_mode=True)
    B = B_mi * w + B_ma * (1 - w)
    return Draw(
        N=N, e=e, seed=seed,
        asset_idx=np.asarray(asset_idx, dtype=np.int64),
        asset_idx_raw=market.raw_index[asset_idx],
        tickers=market.tickers[asset_idx],
        names=market.names[asset_idx],
        P=P, ret=ret, cov=cov,
        w=float(w), B_min=float(B_mi), B_max=float(B_ma), B=float(B),
    )
