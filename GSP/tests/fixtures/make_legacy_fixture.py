"""Build tests/fixtures/legacy_completed_draws.json from the completed work (read-only).

Provenance
----------
Run once in S1 (2026-09-24) with the gsp env. Sources (read-only, never edited):
  CUDA/experiments_approx_Q2_RAND_S1.0_W0.01_Jh/exp_L{lam}_q1.5/expectation_*_boost_Jh.npz
      keys A{N}_E{e}_{P,ret,cov,idx}: the asset data each completed run used
  CUDA/experiments_approx_Q2_RAND_S1.0_W0.01_Jh/exp_L{lam}_q1.5/report_*_boost_Jh.csv
      columns Assets, Exp, Budget, Boost (read with float_precision="round_trip")
(CUDA/debug/QU_L1/QUBO_L1.npz was examined and NOT used: its penalty QUBOs match no draw of
today's recipe (relative differences 0.1-1.4), so it predates the current price window/dataset.)
Floats are stored as float.hex() so the JSON round-trips exactly.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "CUDA", "experiments_approx_Q2_RAND_S1.0_W0.01_Jh")


def hx(a):
    return [float(v).hex() for v in np.asarray(a, dtype=np.float64).ravel()]


def main():
    draws = {}
    boosts = {}
    for d in sorted(glob.glob(os.path.join(SRC, "exp_L*_q1.5"))):
        lam_tag = os.path.basename(d)[len("exp_L"):-len("_q1.5")]
        (npz,) = glob.glob(os.path.join(d, "expectation_*_boost_Jh.npz"))
        (rep,) = [f for f in glob.glob(os.path.join(d, "report_*_boost_Jh.csv")) if "_AR2" not in f]
        mode = "Preserving" if "Preserving" in os.path.basename(rep) else "X"
        z = np.load(npz)
        df = pd.read_csv(rep, float_precision="round_trip")
        for key in z.files:
            if not key.endswith("_idx"):
                continue
            pre = key[:-4]
            N, e = int(pre.split("_")[0][1:]), int(pre.split("_")[1][1:])
            rows = df[(df["Assets"] == N) & (df["Exp"] == e)]
            # Budget: one completed row (N=8, e=9, L=9, a split-run merge) differs by 1 ulp, so the
            # fixture keeps every distinct double seen for the draw.
            assert rows["Boost"].nunique() == 1
            rec = draws.setdefault(f"N{N}e{e}", {
                "N": N, "e": e,
                "asset_idx_raw": [int(v) for v in z[pre + "_idx"]],
                "P": hx(z[pre + "_P"]), "ret": hx(z[pre + "_ret"]),
                "B_values": [],
                "cov_by_run": {},
            })
            rec["B_values"] = sorted(set(rec["B_values"]) | {float(v).hex() for v in rows["Budget"]})
            rec["cov_by_run"][lam_tag] = hx(z[pre + "_cov"])
            boosts.setdefault(f"N{N}e{e}", {})[f"{mode}_L{lam_tag}"] = float(rows["Boost"].iloc[0])
    out = {
        "_provenance": __doc__,
        "q": 1.5,
        "qubits_per_asset": 2,
        "draws": draws,
        "boost": boosts,
    }
    with open(os.path.join(HERE, "legacy_completed_draws.json"), "w") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    print(len(draws), "draws")


if __name__ == "__main__":
    main()
