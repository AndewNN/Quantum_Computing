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
# np.random.seed(109)
# rand_state = np.random.get_state()

report_col = ["Exp", "Assets", "Qubits", "N", "Sum_1", "Sum_2", "Coeff", "Budget"]
# report_col = ["exp", "N", "Sum_1", "Sum_2"]

N = 2000
TARGET_QUBIT_IN = 3
TARGET_ASSET = [3, 4, 5, 6, 7]
min_P, max_P = 125, 250
Z = None

def file_copy(src, dst):
    try:
        shutil.copyfile(src, dst)
    except shutil.SameFileError:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment parameter sweep")

    # # Experiment tag number (int)
    # parser.add_argument(
    #     "-E", "--exp",
    #     type=int, default=0,
    #     help="Experiment tag number (int)"
    # )

    # number of Experiments (int)
    parser.add_argument(
        "-E", "--exp",
        type=int, default=100,
        help="Number of Experiments (int)"
    )

    # Target Qubits per Asset
    parser.add_argument(
        "-Q", "--qubit",
        type=int, default=TARGET_QUBIT_IN,
        help="Number of qubits per asset"
    )

    # Assets (list of ints)
    parser.add_argument(
        "-A", "--asset",
        nargs="+", type=int, default=TARGET_ASSET,
        help="List of asset counts, e.g. -A 3 4 5 6 7"
    )

    # Lambda
    parser.add_argument(
        "-L", "--lamb",
        type=float, default=0.001,
        help="Budget Penalty (float)"
    )

    # Volatility
    parser.add_argument(
        "-q",
        type=float, default=1.0,
        help="Volatility Weight (float)"
    )

    # QAOA Layers
    parser.add_argument(
        "-p", "--layer",
        type=int, default=5,
        help="Number of QAOA layers (int)"
    )

    # Number of samples
    parser.add_argument(
        "-N",
        type=int, default=N,
        help="Number of Samples (int)"
    )

    # Expectation Basis
    parser.add_argument(
        "-Z", "--basis",
        type=int, nargs="+", default=Z,
        help="Expectation Basis (list of ints) e.g. -Z 0, -Z 0 7"
    )

    return parser.parse_args()

args = parse_args()

# HYPER PARAMETERS
TARGET_QUBIT_IN = args.qubit
TARGET_ASSET = args.asset
LAMB = args.lamb # Budget Penalty
Q = args.q # Volatility Weight
LAYER = args.layer
N = args.N
Z = args.basis
E = args.exp


# Dataset
data_cov_pd = pd.read_csv("../dataset/top_50_us_stocks_data_20250526_011226_covariance.csv")
data_ret_p_pd = pd.read_csv("../dataset/top_50_us_stocks_returns_price.csv")
# print(data_cov.shape, data_ret_p.shape)

f_Q = Q if not Q.is_integer() else int(Q)
f_LAMB = LAMB if not LAMB.is_integer() else int(LAMB)
dir_name = f"exp_p{LAYER}_L{f_LAMB}_q{f_Q}"
dir_path = f"./experiments_plateau_X/{dir_name}"
file_name  = f"report_{'Hall' if Z is None else ('Z' + ''.join([str(z) for z in Z]))}.csv"

os.makedirs(dir_path, exist_ok=True)

print(f"Experiments: {E}, Max_Qubits: {TARGET_QUBIT_IN}, Assets: {TARGET_ASSET}, Lambda: {LAMB}, q: {Q}, Layers: {LAYER}, N: {N}, Basis: {Z}")
pbar_all = tqdm(range(E))
for e in pbar_all:
    H = None
    pbar = tqdm(enumerate(TARGET_ASSET), leave=False)
    for i, N_ASSETS in pbar:
        np.random.seed(911 + 991 * e + 997 * N_ASSETS)
        state = np.random.get_state()
        # asset_idx = np.random.choice(data_cov_pd.shape[0], max(TARGET_ASSET), replace=False)
        asset_idx = np.random.choice(data_cov_pd.shape[0], N_ASSETS, replace=False)
        data_cov = data_cov_pd.drop("Ticker", axis=1).to_numpy()[asset_idx, :][:, asset_idx]
        stock_names = data_ret_p_pd["Ticker"].to_numpy()[asset_idx]
        # print("Selected Stocks: ", stock_names)
        data_ret_p = data_ret_p_pd.drop("Ticker", axis=1).to_numpy()[asset_idx, :]

        data_ret = data_ret_p[:, 0]
        data_p = data_ret_p[:, 1]

        # print(data_cov.shape)

        np.random.set_state(state)
        selected_price = np.random.uniform(125, 250, N_ASSETS)
        price_factor = selected_price / data_p
        data_p = selected_price
        # data_ret = data_ret * price_factor
        # data_cov = (price_factor[None, :] * data_cov) * price_factor[:, None]

        # print(data_ret.round(5))
        # print(data_p.round(2))
        # print(stock_names)

        np.random.set_state(state)
        weighted = np.random.uniform(0, 1)
        B_mi, B_ma = find_budget(TARGET_QUBIT_IN * N_ASSETS, data_p, min_P, max_P, min_mix_mode=True)
        B = B_mi * weighted + B_ma * (1 - weighted)
        # print("Budget: ", B)


        P = data_p[:N_ASSETS]
        ret = data_ret[:N_ASSETS]
        cov = data_cov[:N_ASSETS, :N_ASSETS]

        q = Q
        P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, P, ret, cov)
        pbar.set_description(f"Assets: {N_ASSETS}, Qubits: {n_qubit}")
        # print(f"Assets: {N_ASSETS}, Qubits: {n_qubit}")
        TARGET_QUBIT = n_qubit
        lamb = LAMB

        QU = -ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
        # if N_ASSETS in [3, 4]:
        #     print(QU.shape)
        #     print(QU)
            
        # if H is None:
        H_ansatz = qubo_to_ising(QU, lamb).canonicalize()
        if Z is None:
            H = H_ansatz
        else:
            H = 1
            for i in range(len(Z)):
                H = H * cudaq.spin.z(Z[i])
        # H_1 = cudaq.spin.z(7) * cudaq.spin.z(8)
        # # H_2 = cudaq.spin.z(n_qubit//2) * cudaq.spin.z(n_qubit//2 + 1)
        # H_2 = cudaq.spin.z(7) * cudaq.spin.z(8)
        # H_2 = cudaq.spin.z(0)
        idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use = process_ansatz_values(H_ansatz)
        coeff_1_use, coeff_2_use = np.array(coeff_1_use), np.array(coeff_2_use)
        # print(idx_2_a_use, idx_2_b_use)

        # print(coeff_1_use.__abs__().min(), coeff_1_use.__abs__().max(), coeff_1_use.__abs__().mean())
        # print(coeff_2_use.__abs__().min(), coeff_2_use.__abs__().max(), coeff_2_use.__abs__().mean())
        # break

        mm_1 = np.min(np.abs(coeff_1_use)) if len(coeff_1_use) > 0 else 1e9
        mm_2 = np.min(np.abs(coeff_2_use)) if len(coeff_2_use) > 0 else 1e9
        mm_i = np.pi / min(mm_1, mm_2)

        kernel_qaoa_use = kernel_qaoa_X
        idx = 3
        layer_count = LAYER
        parameter_count = layer_count * 2
        ansatz_fixed_param = (int(n_qubit), layer_count, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use)
        it_st, sum_1, sum_2 = 0, 0.0, 0.0
        df_now = pd.read_csv(f"./experiments_plateau_X/{dir_name}/{file_name}") if os.path.exists(f"./experiments_plateau_X/{dir_name}/{file_name}") else None
        # print("\nhere\n")
        if df_now is not None:
            if df_now[(df_now["Assets"] == N_ASSETS) & (df_now["Exp"] == e)].shape[0] > 0:
                if df_now.loc[(df_now["Assets"] == N_ASSETS) & (df_now["Exp"] == e), "N"].values[0] >= N:
                    continue
                else:
                    it_st, sum_1, sum_2 = df_now.loc[(df_now["Assets"] == N_ASSETS) & (df_now["Exp"] == e), ["N", "Sum_1", "Sum_2"]].values[0]
                    it_st = int(it_st)
            else:
                df_now.loc[-1] = [e, N_ASSETS, n_qubit, 0, 0.0, 0.0, 0.0, B]
        else:
            df_now = pd.DataFrame(np.array([e, N_ASSETS, TARGET_QUBIT, 0, 0.0, 0.0, 0.0, B])[None, :], columns=report_col)
        # np.random.set_state(rand_state)
        np.random.seed(4001 + 4099 * e + 4999 * N_ASSETS)
        points = np.random.uniform(-1, 1, (N, parameter_count))
        points[:, ::2] *= mm_i
        points[:, 1::2] *= np.pi

        expectations = []

        for ii in tqdm(range(it_st, N), leave=False):
            expectations.append(float(cudaq.observe(kernel_qaoa_use, H, points[ii], *ansatz_fixed_param).expectation()))
        expectations = np.array(expectations)
        sum_1 += expectations.sum()
        sum_2 += (expectations ** 2).sum()

        df_now.sort_values(by=["Exp", "Assets"], inplace=True)
        df_now.loc[(df_now["Assets"] == N_ASSETS) & (df_now["Exp"] == e), ["N", "Sum_1", "Sum_2", "Coeff", "Budget"]] = [N, sum_1, sum_2, coeff_2_use[0], B]
        df_now.to_csv(f"./experiments_plateau_X/{dir_name}/{file_name}", index=False)
        