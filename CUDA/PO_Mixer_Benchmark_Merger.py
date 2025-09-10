import argparse
import os
import numpy as np
import pandas as pd
import shutil

LOOP = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment parameter sweep")

    # Qubits (list of ints)
    parser.add_argument(
        "-Q", "--qubit",
        nargs="+", type=int, default=[5],
        help="List of target qubits, e.g. -Q 4 5 6"
    )

    # Assets (list of ints)
    parser.add_argument(
        "-A", "--asset",
        nargs="+", type=int, default=[3, 4, 5],
        help="List of asset counts, e.g. -A 3 4 5"
    )

    # Lambda
    parser.add_argument(
        "-L", "--lamb",
        type=int, default=4,
        help="Budget Penalty (float)"
    )

    # Bases (list of ints)
    parser.add_argument(
        "-B", "--bases",
        nargs="+", type=int, default=[3, 6, 12],
        help="List of num initial bases, e.g. -B 3 6 12 25"
    )

    # Volatility
    parser.add_argument(
        "-q",
        type=int, default=0,
        help="Volatility Weight (float)"
    )

    # end iter (run from start to end-1)
    parser.add_argument(
        "-ed", "--end_iter",
        type=int, default=LOOP,
        help="Number of iterations (int)"
    )

    return parser.parse_args()

args = parse_args()
TARGET_QUBIT_IN = args.qubit
N_ASSETS_IN = args.asset
LAMB = args.lamb # Budget Penalty
Q = args.q # Volatility Weight
num_init_bases_in = args.bases
LOOP = args.end_iter
modes = ["X", "Preserving"]

def find_dir_start_with(str):
    base_path = "./experiments"
    ret = []
    for dir_name in os.listdir(base_path):
        if dir_name.startswith(str):
            st, ed = dir_name.split("_it")[-1].split("-")
            ret.append((dir_name, int(st), int(ed)))
    return sorted(ret, key=lambda x: (x[1], x[2]))

for TARGET_QUBIT in TARGET_QUBIT_IN:
    for N_ASSETS in N_ASSETS_IN:
        for num_init_bases in num_init_bases_in:
            dir_name = f"exp_Q{TARGET_QUBIT}_A{N_ASSETS}_L{LAMB}_q{Q}_B{num_init_bases}"
            dir_path = f"./experiments/{dir_name}"

            os.makedirs(f"{dir_path}", exist_ok=True)
            os.makedirs(f"{dir_path}/expectations_X", exist_ok=True)
            os.makedirs(f"{dir_path}/expectations_Preserving", exist_ok=True)

            ret = find_dir_start_with(dir_name + "_it")
            mk = np.zeros(LOOP, dtype=bool)
            for dirr, st, ed in ret:
                mk[st:ed+1] = True
            assert np.all(mk), f"Not all iterations are covered for {dir_name}, missing {np.where(mk==False)[0]}"

            now, pdd = 0, [None, None]
            for dirr, st, ed in ret:
                dir_path_cp = f"./experiments/{dirr}"
                print(dir_path_cp)
                for m_i, mode in enumerate(modes):
                    for i in range(now, min(ed+1, LOOP)):
                        shutil.copyfile(f"{dir_path_cp}/expectations_{mode}/expectations_{i}.npy", f"{dir_path}/expectations_{mode}/expectations_{i}.npy")
                    if pdd[m_i] is None:
                        pdd[m_i] = pd.read_csv(f"{dir_path_cp}/{mode}.csv")
                    else:
                        pd_r = pd.read_csv(f"{dir_path_cp}/{mode}.csv")
                        pd_r = pd_r.iloc[now-st:min(ed+1, LOOP)-st]
                        pdd[m_i] = pd.concat([pdd[m_i], pd_r], ignore_index=True)
                now = ed + 1
            for m_i, mode in enumerate(modes):
                pdd[m_i].to_csv(f"{dir_path}/{mode}.csv", index=False)