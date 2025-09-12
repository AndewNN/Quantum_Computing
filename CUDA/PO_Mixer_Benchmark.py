import numpy as np
import pandas as pd
import cudaq
from cudaq import spin
from typing import List, Tuple
import matplotlib.pyplot as plt
from math import sqrt
import math
import sys
import os
import torch
from tqdm import tqdm
import shutil
import argparse
sys.path.append(os.path.abspath(".."))
from Utils.qaoaCUDAQ import po_normalize, ret_cov_to_QUBO, qubo_to_ising, process_ansatz_values, state_to_return, pauli_to_int, int_to_pauli,\
    basis_T_to_pauli, reversed_str_bases_to_init_state, kernel_qaoa_Preserved, kernel_qaoa_X, kernel_flipped, get_optimizer, find_budget,\
    all_state_to_return, get_init_states, write_df, clip_df, kernel_cmpz_Preserved, prepare_preserving_ansatz

import seaborn as sns   
from copulas.multivariate import GaussianMultivariate
import joblib
import pickle

import time
# cudaq.mpi.initialize()
# cudaq.set_target("nvidia")
cudaq.set_target("nvidia")
pd.set_option('display.width', 1000)
np.random.seed(50)
state = np.random.get_state()
# with open("rng_state.pkl", "wb") as f:
#     pickle.dump(state, f)
modes = ["X", "Preserving"]
report_col = ["Approximate_ratio", "MaxProb_ratio", "init_1_time", "init_2_time", "optim_time", "observe_time"]

# PIPELINE PARAMETERS
LOOP = 100
oversample_factor = 5
over_budget_bound = 1.0 # valid budget in [0, B * over_budget_bound]
min_P, max_P = 125, 250
hamiltonian_boost = 2000

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

    # QAOA Layers
    parser.add_argument(
        "-l", "--layer",
        type=int, default=5,
        help="Number of QAOA layers (int)"
    )

    # start iter
    parser.add_argument(
        "-st", "--start_iter",
        type=int, default=0,
        help="Number of iterations (int)"
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

# HYPER PARAMETERS
init_state_ratio = 0.1
TARGET_QUBIT_IN = args.qubit
N_ASSETS_IN = args.asset
LAMB = args.lamb # Budget Penalty
Q = args.q # Volatility Weight
LAYER = args.layer

# num_init_bases = int(2**TARGET_QUBIT * init_state_ratio)
num_init_bases_in = args.bases
iter_start = args.start_iter
iter_end = args.end_iter
modes = args.mode

# print(TARGET_QUBIT_IN)
# print(N_ASSETS_IN, type(N_ASSETS_IN))
# print(LAMB)
# print(Q)

MPI_ENABLE = False

if MPI_ENABLE:
    print(cudaq.mpi.rank())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

target = cudaq.get_target()
qpu_count = target.num_qpus()
print("Number of QPUs:", qpu_count)



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

np.random.set_state(state)
state_init_loop = np.random.get_state()
ch_tr = True
for TARGET_QUBIT in TARGET_QUBIT_IN:
    for N_ASSETS in N_ASSETS_IN:
        for num_init_bases in num_init_bases_in:

            if TARGET_QUBIT < N_ASSETS:
                continue

            print(f"Target Qubit: {TARGET_QUBIT}, N Assets: {N_ASSETS}, Num Init Bases: {num_init_bases}, ST: {iter_start}, ED: {iter_end}, Modes: {modes}")
            # continue
            dir_name = f"exp_Q{TARGET_QUBIT}_A{N_ASSETS}_L{LAMB}_q{Q}_B{num_init_bases}"
            if iter_start != 0 or iter_end != LOOP:
                dir_name += f"_it{iter_start}-{iter_end-1}"
            dir_name_Xbase = f"exp_Q{TARGET_QUBIT}_A{N_ASSETS}_L{LAMB}_q{Q}_B3"
            dir_path = f"./experiments/{dir_name}"
            dir_path_Xbase = f"./experiments/{dir_name_Xbase}"
            if os.path.exists(f"{dir_path}/result.csv"):
                print("Completed")
                continue

            os.makedirs(f"{dir_path}", exist_ok=True)
            # os.makedirs(f"{dir_path}/Preserving", exist_ok=True)
            os.makedirs(f"{dir_path}/expectations_X", exist_ok=True)
            os.makedirs(f"{dir_path}/expectations_Preserving", exist_ok=True)
            # os.makedirs(f"{dir_path}/iter_states", exist_ok=True)

            # continue

            np.random.set_state(state_init_loop)
            restore_iter, tmpp = iter_start, int(1e9)
            if os.path.exists(f"{dir_path}/X.csv") or os.path.exists(f"{dir_path}/Preserving.csv"):
                for mode in modes:
                    df = pd.read_csv(f"./{dir_path}/{mode}.csv")
                    # if restore_iter > iter_start:
                    #     restore_iter = min(restore_iter, df.shape[0] + iter_start)
                    # else:
                    #     restore_iter = df.shape[0] + iter_start
                    tmpp = min(tmpp, df.shape[0] + iter_start)
                restore_iter = max(restore_iter, tmpp) # I don't know what on earth will trigger this max but this is safer YK what I mean

                for mode in modes:
                    df = pd.read_csv(f"./{dir_path}/{mode}.csv")
                    df = clip_df(df, restore_iter - iter_start)
                    df.to_csv(f"{dir_path}/{mode}.csv", index=False)

            else:
                for curr_dir, dirs, files in os.walk(dir_path):
                    for file in files:
                        os.remove(os.path.join(curr_dir, file))

            for i in range(restore_iter):
                np.random.rand(N_ASSETS, N_ASSETS)
                np.random.uniform(-np.pi / 8, np.pi / 8, LAYER * 4)

            X_exist = False
            if os.path.exists(f"{dir_path_Xbase}/X.csv") and pd.read_csv(f"{dir_path_Xbase}/X.csv").shape[0] >= iter_end:
                file_copy(f"{dir_path_Xbase}/X.csv", f"{dir_path}/X.csv")
                x_csv = pd.read_csv(f"{dir_path}/X.csv")
                x_csv = x_csv.iloc[iter_start:iter_end]
                x_csv.to_csv(f"{dir_path}/X.csv", index=False)
                # shutil.copytree(f"{dir_path_Xbase}/expectations_X", f"{dir_path}/expectations_X", dirs_exist_ok=True)
                for f_i in range(iter_start, iter_end):
                    file_copy(f"{dir_path_Xbase}/expectations_X/expectations_{f_i}.npy", f"{dir_path}/expectations_X/expectations_{f_i}.npy")
                X_exist = True

            pbar = tqdm(range(restore_iter, iter_end))
            for i in pbar:
                # if i == 1 and ch_tr:
                #     # tr = tracker.SummaryTracker()
                #     print("Start Tracking GB")
                #     ch_tr = False

                pbar.set_description("global:init_1")
                st = time.time()

                P = samples[i * N_ASSETS:(i + 1) * N_ASSETS, 0]
                ret = samples[i * N_ASSETS:(i + 1) * N_ASSETS, 1]
                # print(P)
                # print(ret)
                
                # P = np.array([195.27, 183.26, 131.3])
                # ret = np.array([0.00107, 0.00083, 0.00071])
                cov = np.random.rand(N_ASSETS, N_ASSETS)
                cov += cov.T
                q = 0 # Volatility Weight
                B = find_budget(TARGET_QUBIT, P, min_P, max_P)
                # break
                # B = 270
                P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, P, ret, cov)
                state_return, in_budget = all_state_to_return(B, C, ret, P, over_budget_bound)
                init_state = get_init_states(state_return, in_budget, num_init_bases, n_qubit)

                # print(P)
                # print(ret)
                # print(B)
                # print(init_state)
                # exit()

                feasible_state_return = state_return * in_budget
                max_return = state_return[int(init_state[0], 2)]

                init_1_time = time.time() - st
                # print(f"initial: {init_1_time*1000:.2f} ms.")

                for mode in modes:
                    if mode == "X" and X_exist:
                        np.random.uniform(-np.pi / 8, np.pi / 8, 2 * LAYER)
                        continue
                    pbar.set_description(f"{mode}:init_2")
                    st = time.time()
                    lamb = LAMB if mode == "X" else 0 # Budget Penalty

                    QU = -ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
                    H = qubo_to_ising(QU, lamb).canonicalize() * (1 if mode == "X" else hamiltonian_boost)
                    QU_0 = -ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, 0, q)
                    H_0 = qubo_to_ising(QU_0, 0).canonicalize()
                    idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use = process_ansatz_values(H)

                    kernel_qaoa_use = kernel_qaoa_X if mode == "X" else kernel_qaoa_Preserved
                    # kernel_qaoa_use = kernel_qaoa_X if mode == "X" else kernel_cmpz_Preserved

                    idx = 3
                    layer_count = LAYER
                    parameter_count = layer_count * 2
                    optimizer, optimizer_name, FIND_GRAD = get_optimizer(idx)
                    optimizer.max_iterations = 1000
                    optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8, 2 * LAYER)

                    if mode == "X":
                        ansatz_fixed_param = (int(n_qubit), layer_count, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use)
                        init_2_time = time.time() - st
                        # print(f"init for {mode}: {init_2_time*1000:.2f} ms.")
                    else:
                        n_bases = len(init_state)
                        # print("n_bases:", n_bases)
                        T = np.zeros((n_bases, n_bases), dtype=np.float32)
                        T[:-1, 1:] += np.eye(n_bases - 1, dtype=np.float32)
                        T[1:, :-1] += np.eye(n_bases - 1, dtype=np.float32)
                        T[0, -1] = T[-1, 0] = 1.0
                        # print(T)
                        mixer_s, mixer_c = basis_T_to_pauli(init_state, T, n_qubit)
                        init_bases = reversed_str_bases_to_init_state(init_state, n_qubit)

                        ansatz_fixed_param = (int(n_qubit), layer_count, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c, init_bases)

                        # preserving_gates = prepare_preserving_ansatz(n_qubit, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c.tolist())
                        # ansatz_fixed_param = (int(n_qubit), layer_count, preserving_gates, init_bases)

                        # preserving_gates, mk = prepare_preserving_ansatz(n_qubit, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c.tolist())[-2:]
                        # preserving_gates = preserving_gates[mk == 1].reshape(-1)
                        # ansatz_fixed_param = (int(n_qubit), layer_count, preserving_gates, init_bases)


                        init_2_time = time.time() - st
                        # print(f"Init for {mode}: {init_2_time*1000:.2f} ms.")
                    # print(cudaq.draw(kernel_qaoa_use, [0.5]*4, *ansatz_fixed_param[:1], 1, *ansatz_fixed_param[2:]))

                    pbar.set_description(f"{mode}:optim")
                    st = time.time()
                    expectations = []
                    # expectations2 = []
                    # expectations3 = []
                    def cost_func(parameters, cal_expectation=False):
                        # return cudaq.observe(kernel_qaoa, H, n_qubit, layer_count, parameters, 0).expectation()
                        exp_return = float(cudaq.observe(kernel_qaoa_use, H, parameters, *ansatz_fixed_param).expectation())
                        if cal_expectation:
                            exp_return_in = float(cudaq.observe(kernel_qaoa_use, H_0, parameters, *ansatz_fixed_param).expectation())
                            expectations.append([exp_return_in, exp_return])
                        return exp_return
                        #     exp_return = cudaq.observe(kernel_qaoa_use, H_0, parameters, *ansatz_fixed_param, execution=cudaq.parallel.thread).expectation()
                        #     expectations.append(exp_return)
                        # return cudaq.observe(kernel_qaoa_use, H, parameters, *ansatz_fixed_param, execution=cudaq.parallel.thread).expectation()

                    def objective(parameters):
                        expectation = cost_func(parameters, cal_expectation=True)
                        # expectations3.append(expectation)
                        return expectation

                    fd = cudaq.gradients.ForwardDifference()
                    def objective_grad_cuda(parameters):
                        expectation = cost_func(parameters, cal_expectation=True)
                        # expectations3.append(expectation)

                        gradient = fd.compute(parameters, cost_func, expectation)

                        return expectation, gradient

                    objective_func = objective_grad_cuda if FIND_GRAD else objective

                    optimal_expectation, optimal_parameters = optimizer.optimize(
                        dimensions=parameter_count, function=objective_func)
                    np.save(f"{dir_path}/expectations_{mode}/expectations_{i}.npy", np.array(expectations))
                    optim_time = time.time() - st
                    # print(f"Optimization for {mode}: {optim_time*1000:.2f} ms.")
                    
                    pbar.set_description(f"{mode}:observe")
                    st = time.time()
                    result = cudaq.get_state(kernel_qaoa_use, optimal_parameters, *ansatz_fixed_param)
                    idx_r_best = np.argmax(np.abs(result))
                    idx_best = bin(idx_r_best)[2:].zfill(n_qubit)[::-1]

                    result_r = cudaq.get_state(kernel_flipped, result, TARGET_QUBIT)
                    prob = np.abs(result_r)**2

                    approx_ratio = (prob * (feasible_state_return)).sum() / max_return
                    maxprob_ratio = state_return[int(idx_best, 2)] / max_return if in_budget[int(idx_best, 2)] else 0.0

                    observe_time = time.time() - st

                    write_df(f"{dir_path}/{mode}.csv", report_col,
                                approx_ratio, maxprob_ratio, init_1_time, init_2_time, optim_time, observe_time)
                    # gc.collect()

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