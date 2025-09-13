import argparse
import os
import numpy as np
import pandas as pd
import shutil

LOOP = 100
modes = ["X", "Preserving"]
report_col = ["Approximate_ratio", "MaxProb_ratio", "init_1_time", "init_2_time", "optim_time", "observe_time"]

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

    # modes (list of str)
    parser.add_argument(
        "-m", "--mode",
        nargs="+", type=str, default=modes,
        help="List of modes, e.g. -m X Preserving"
    )

    return parser.parse_args()

args = parse_args()
TARGET_QUBIT_IN = args.qubit
N_ASSETS_IN = args.asset
LAMB = args.lamb # Budget Penalty
Q = args.q # Volatility Weight
num_init_bases_in = args.bases
LOOP = args.end_iter
modes = args.mode

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

            if "X" in modes and "Preserving" in modes:
                df_X = pd.read_csv(f"{dir_path}/X.csv")
                df_P = pd.read_csv(f"{dir_path}/Preserving.csv")

                approx_X = df_X["Approximate_ratio"].mean()
                approx_P = df_P["Approximate_ratio"].mean()
                print(f"Approximate ratio for X: {approx_X:.4f}, Preserving: {approx_P:.4f}")

                maxprob_X = df_X["MaxProb_ratio"].mean()
                maxprob_P = df_P["MaxProb_ratio"].mean()
                print(f"MaxProb ratio for X: {maxprob_X:.4f}, Preserving: {maxprob_P:.4f}")

                init_1_time_X = df_X["init_1_time"].mean()
                init_2_time_X = df_X["init_2_time"].mean()
                optim_time_X = df_X["optim_time"].mean()
                observe_time_X = df_X["observe_time"].mean()

                init_1_time_P = df_P["init_1_time"].mean()
                init_2_time_P = df_P["init_2_time"].mean()
                optim_time_P = df_P["optim_time"].mean()
                observe_time_P = df_P["observe_time"].mean()

                print(f"Init 1 time for X: {init_1_time_X*1000:.2f} ms, Preserving: {init_1_time_P*1000:.2f} ms")
                print(f"Init 2 time for X: {init_2_time_X*1000:.2f} ms, Preserving: {init_2_time_P*1000:.2f} ms")
                print(f"Optim time for X: {optim_time_X*1000:.2f} ms, Preserving: {optim_time_P*1000:.2f} ms")
                print(f"Observe time for X: {observe_time_X*1000:.2f} ms, Preserving: {observe_time_P*1000:.2f} ms")

                col_result = ["Mode"] + report_col
                df_result = pd.DataFrame(columns=col_result)
                df_result.loc[0] = ["X", approx_X, maxprob_X, init_1_time_X, init_2_time_X, optim_time_X, observe_time_X]
                df_result.loc[1] = ["Preserving", approx_P, maxprob_P, init_1_time_P, init_2_time_P, optim_time_P, observe_time_P]
                df_result.to_csv(f"{dir_path}/result.csv", index=False)