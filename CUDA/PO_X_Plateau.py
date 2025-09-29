import numpy as np
import pandas as pd
import cudaq
import sys
import os
import torch
from tqdm import tqdm
import shutil
import argparse
sys.path.append(os.path.abspath(".."))
from Utils.qaoaCUDAQ import po_normalize, ret_cov_to_QUBO, qubo_to_ising, process_ansatz_values,\
    basis_T_to_pauli, reversed_str_bases_to_init_state, kernel_qaoa_Preserved, kernel_qaoa_X, kernel_flipped, get_optimizer, find_budget,\
    all_state_to_return, get_init_states, write_df, clip_df

import joblib

cudaq.set_target("nvidia")
pd.set_option('display.width', 1000)
np.random.seed(109)
rand_state = np.random.get_state()
np.random.seed(50)
state = np.random.get_state()

report_col = ["N", "Sum_1", "Sum_2"]

LOOP = 100
N = 2000
oversample_factor = 5
over_budget_bound = 1.0 # valid budget in [0, B * over_budget_bound]
min_P, max_P = 125, 250

def file_copy(src, dst):
    try:
        shutil.copyfile(src, dst)
    except shutil.SameFileError:
        pass

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

    # Volatility
    parser.add_argument(
        "-q",
        type=int, default=0,
        help="Volatility Weight (float)"
    )

    # QAOA Layers
    parser.add_argument(
        "-l", "--layer",
        type=int, default=5,
        help="Number of QAOA layers (int)"
    )

    # start iter
    # parser.add_argument(
    #     "-st", "--start_iter",
    #     type=int, default=0,
    #     help="Number of iterations (int)"
    # )

    # end iter (run from start to end-1)
    parser.add_argument(
        "-ed", "--end_iter",
        type=int, default=LOOP,
        help="Number of iterations (int)"
    )

    # Number of samples
    parser.add_argument(
        "-N",
        type=int, default=N,
        help="Number of Samples (int)"
    )

    return parser.parse_args()

args = parse_args()

# HYPER PARAMETERS
TARGET_QUBIT_IN = args.qubit
N_ASSETS_IN = args.asset
LAMB = args.lamb # Budget Penalty
Q = args.q # Volatility Weight
LAYER = args.layer
iter_start = 0
iter_end = args.end_iter
N = args.N

# with open("rng_state.pkl", "rb") as f:
#     state = pickle.load(f)
np.random.set_state(state)
GM_loaded = joblib.load('./models/gaussian_copula.pkl')
# samples = GM_loaded.sample(50)
samples = GM_loaded.sample(int(max(N_ASSETS_IN) * LOOP * oversample_factor))
samples = samples[(samples["Price"] > min_P) & (samples["Price"] < max_P)]
# print(samples["Average_Return"].min(), samples["Average_Return"].max())
# print(samples["Price"].min(), samples["Price"].max())
print(samples.shape)
samples = samples.to_numpy()
assert samples.shape[0] > max(N_ASSETS_IN) * LOOP, "Please increase the oversample factor to get more samples ;-;"
# sns.jointplot(data=samples, x='Price', y='Average_Return', kind='reg')
# samples = samples.to_numpy()
# plt.show()


##########################################################################################

# DO NOT INTERFERE loop_state TO LET EACH SETUP SYNCHRONIZED

##########################################################################################

np.random.set_state(state)
state_init_loop = np.random.get_state()

for TARGET_QUBIT in TARGET_QUBIT_IN:
    for N_ASSETS in N_ASSETS_IN:
        if TARGET_QUBIT < N_ASSETS:
            continue

        print(f"Target Qubit: {TARGET_QUBIT}, N Assets: {N_ASSETS}, ST: {iter_start}, ED: {iter_end}")

        dir_name = f"exp_Q{TARGET_QUBIT}_A{N_ASSETS}_L{LAMB}_q{Q}"
        dir_path = f"./experiments_plateau_X/{dir_name}"
        
        os.makedirs(dir_path, exist_ok=True)

        np.random.set_state(state_init_loop)
        loop_state = np.random.get_state()
        restore_iter = iter_start
        
        for i in range(restore_iter):
            np.random.rand(N_ASSETS, N_ASSETS)
            np.random.uniform(-np.pi / 8, np.pi / 8, LAYER * 4)
        
        pbar = tqdm(range(restore_iter, iter_end))
        for i in pbar:
            pbar.set_description(f"X:init_1")
            P = samples[i * N_ASSETS:(i + 1) * N_ASSETS, 0]
            ret = samples[i * N_ASSETS:(i + 1) * N_ASSETS, 1]
            np.random.set_state(loop_state)
            cov = np.random.rand(N_ASSETS, N_ASSETS)
            cov += cov.T
            # print(cov)
            loop_state = np.random.get_state()

            q = Q # Volatility Weight
            B = find_budget(TARGET_QUBIT, P, min_P, max_P)

            P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, P, ret, cov)
            # state_return, in_budget = all_state_to_return(B, C, ret, P, over_budget_bound)

            pbar.set_description(f"X:init_2")
            lamb = LAMB

            QU = -ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
            H = qubo_to_ising(QU, lamb).canonicalize()
            idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use = process_ansatz_values(H)

            coeff_1_use, coeff_2_use = np.array(coeff_1_use), np.array(coeff_2_use)
            mm_1 = np.min(np.abs(coeff_1_use)) if len(coeff_1_use) > 0 else 1e9
            mm_2 = np.min(np.abs(coeff_2_use)) if len(coeff_2_use) > 0 else 1e9
            mm = min(mm_1, mm_2)
            mm_i = np.pi / mm


            kernel_qaoa_use = kernel_qaoa_X

            idx = 3
            layer_count = LAYER
            parameter_count = layer_count * 2
            ansatz_fixed_param = (int(n_qubit), layer_count, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use)
            it_st, sum_1, sum_2 = 0, 0.0, 0.0
            df_now = pd.read_csv(f"{dir_path}/report.csv") if os.path.exists(f"{dir_path}/report.csv") else None
            if df_now is not None and df_now.shape[0] > i:
                if df_now.iloc[i]['N'] >= N:
                    np.random.uniform(-np.pi / 8, np.pi / 8, 4 * LAYER)
                    loop_state = np.random.get_state()
                    continue
                it_st, sum_1, sum_2 = df_now.iloc[i]
                it_st = int(it_st)
            
            np.random.set_state(rand_state)
            # points = np.random.uniform(-np.pi, np.pi, (N, parameter_count))
            points = np.zeros((N, parameter_count))
            points[:, ::2] = np.random.uniform(-mm_i, mm_i, (N, layer_count))
            points[:, 1::2] = np.random.uniform(-np.pi, np.pi, (N, layer_count))

            expectations = []

            pbar.set_description(f"X:observing")
            for ii in tqdm(range(it_st, N), leave=False):
                expectations.append(float(cudaq.observe(kernel_qaoa_use, H, points[ii], *ansatz_fixed_param).expectation()))
            expectations = np.array(expectations)
            sum_1 += expectations.sum()
            sum_2 += (expectations**2).sum()

            # dff = pd.DataFrame(np.array([iter_end, sum_1, sum_2]).reshape(1, -1), columns=report_col)
            # dff.to_csv(f"{dir_path}/report.csv", index=False)
            write_df(f"{dir_path}/report.csv", report_col, N, sum_1, sum_2, idx=i)


            np.random.set_state(loop_state)
            np.random.uniform(-np.pi / 8, np.pi / 8, 4 * LAYER) # Reserved for params of X and Preserving hamiltonian
            loop_state = np.random.get_state()