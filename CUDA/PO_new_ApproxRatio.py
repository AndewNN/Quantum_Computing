import numpy as np
import pandas as pd
import cudaq
import sys
import os
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, ExponentialLR, CyclicLR, SequentialLR
import time
from tqdm import tqdm
import shutil
import argparse
import faulthandler
import ga_solver
from math import sqrt
faulthandler.enable()
sys.path.append(os.path.abspath(".."))
from Utils.qaoaCUDAQ import po_normalize, ret_cov_to_QUBO, qubo_to_ising, process_ansatz_values, kernel_qaoa_X, find_budget, kernel_flipped,\
    kernel_qaoa_Preserved, all_state_to_return, get_init_states, basis_T_to_pauli_parallel, basis_T_to_pauli, reversed_str_bases_to_init_state, get_optimizer

'''
X A6

L0.5 b_X: 50
L0.05 b_X: 250
L0.005 b_X: 6875
L0.0005 b_X: 17500
L0.00005 b_X: 23750
L0.000005 b_X: 36875    
'''

if __name__ == "__main__":
    cudaq.set_target("nvidia")
    pd.set_option('display.width', 1000)
    # np.random.seed(109)
    # rand_state = np.random.get_state()

    # Assume that already set CUDA_VISIBLE_DEVICES
    device = torch.device("cuda:0")

    report_col = ["Assets", "Exp", "Qubits", "Approximate_ratio", "Return", "Risk", "Budget_Violations", "Budget", "MaxProb_ratio", "init_1_time", "init_2_time", "optim_time", "epochs", "observe_time"]

    TARGET_QUBIT_IN = 3
    TARGET_ASSET = [3, 4, 5, 6, 7]
    # min_P, max_P = 95, 190
    min_P, max_P = 108, 216
    # min_P, max_P = 200, 400
    hamiltonian_X_boost = 7500.0
    hamiltonian_P_boost = 7500.0
    modes = ["X", "Preserving"]
    eps = [0.1]
    SHIFT = 1e-4
    F_TOL = 1e-4
    is_GA = False

    # os.system("taskset -p 0xfffff %d" % os.getpid())

    def file_copy(src, dst):
        try:
            shutil.copyfile(src, dst)
        except shutil.SameFileError:
            pass

    def parse_argss():
        parser = argparse.ArgumentParser(description="Experiment parameter sweep")

        # number of Experiments (int)
        parser.add_argument(
            "-E", "--exp",
            type=int, default=50,
            help="Number of Experiments (int)"
        )

        # idx of Starting Experiment (int)
        parser.add_argument(
            "-E_st", "--exp_start",
            type=int, default=0,
            help="Starting Experiment index (int)"
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

        # select mode in ["X", "Preserving"]
        parser.add_argument(
            "-m", "--mode",
            type=str, default="X",
            help="Mode selection in ['X', 'Preserving']"
        )

        # number of preserving bases
        parser.add_argument(
            "-B", "--bases",
            type=int, default=12,
            help="Number of preserving bases (int)"
        )

        # parameter shift for gradient
        parser.add_argument(
            "-s", "--shift",
            type=float, default=SHIFT,
            help="Parameter shift for gradient (float)"
        )

        # Hamiltonian boost for X mixer
        parser.add_argument(
            "-b_X", "--ham_boost_X",
            type=float, default=hamiltonian_X_boost,
            help="Hamiltonian boost for X mixer (float)"
        )

        # Hamiltonian boost for Preserving mixer
        parser.add_argument(
            "-b_P", "--ham_boost_P",
            type=float, default=hamiltonian_P_boost,
            help="Hamiltonian boost for Preserving mixer (float)"
        )

        # epsilon for budget feasible set for each Asset
        parser.add_argument(
            "-eps", "--epsilon",
            nargs="+", type=float, default=eps,
            help="List of epsilon for budget feasible set for each Asset, e.g. -eps 0.1 0.2"
        )

        # disable progress bar
        parser.add_argument(
            "--no_pbar",
            action="store_true", default=False,
            help="Use tqdm progress bar (bool) e.g. --pbar True or --pbar False"
        )

        # disable create directory
        parser.add_argument(
            "--no_dir",
            action="store_true", default=False,
            help="Disable create directory (bool) e.g. --no_dir True or --no_dir False"
        )

        # optim by pytorch
        parser.add_argument(
            "--torch_optim",
            action="store_true", default=False,
            help="Use pytorch optimizer (bool) e.g. --torch_optim True or --torch_optim False"
        )

        # Overwrite old results rather than skipping
        parser.add_argument(
            "--OVERWRITE",
            action="store_true", default=False,
            help="Overwrite old results rather than skipping"
        )

        # absolute tolerance for convergence
        parser.add_argument(
            "--f_tol",
            type=float, default=F_TOL,
            help="Absolute tolerance for convergence (float)"
        )

        # create feasible set via Genetic Algorithm Approximation
        parser.add_argument(
            "--GA",
            action="store_true", default=False,
            help="Use Genetic Algorithm for feasible set approximation"
        )

        # Random init
        parser.add_argument(
            "--random_init",
            action="store_true", default=False,
            help="Use random initialization (bool) e.g. --random_init True or --random_init False"
        )

        # GA debug
        parser.add_argument(
            "--DEBUG_GA",
            action="store_true", default=False,
            help="Run one GA instance for debugging"
        )

        # duplicate Asset
        parser.add_argument(
            "--DUPLICATE_ASSET",
            action="store_true", default=False,
            help="Duplicate Asset data for testing larger Assets"
        )

        # BF debug
        parser.add_argument(
            "--DEBUG_BF",
            action="store_true", default=False,
            help="Run one BF instance for debugging"
        )

        return parser.parse_args()

    args = parse_argss()

    # HYPER PARAMETERS
    TARGET_QUBIT_IN = args.qubit
    TARGET_ASSET = args.asset
    LAMB = args.lamb # Budget Penalty
    Q = args.q # Volatility Weight
    LAYER = args.layer
    # N = args.N
    # Z = args.basis
    E = args.exp
    E_st = args.exp_start
    mode = args.mode
    num_init_bases = args.bases
    fd = cudaq.gradients.ForwardDifference()
    SHIFT = args.shift
    hamiltonian_X_boost = args.ham_boost_X
    hamiltonian_P_boost = args.ham_boost_P
    eps = args.epsilon
    is_pbar = not args.no_pbar
    is_dir = not args.no_dir
    is_torch_optim = args.torch_optim
    OVERWRITE = args.OVERWRITE
    F_TOL = args.f_tol
    random_init = args.random_init
    DEBUG_GA = args.DEBUG_GA
    DUPLICATE_ASSET = args.DUPLICATE_ASSET
    DEBUG_BF = args.DEBUG_BF

    is_GA = args.GA
    population_size = 2000
    generations = 35
    crossover_rate = 0.85
    elitism_count = 2
    tournament_size = 5

    hamiltonian_P_boost = hamiltonian_P_boost if not hamiltonian_P_boost.is_integer() else int(hamiltonian_P_boost)
    hamiltonian_X_boost = hamiltonian_X_boost if not hamiltonian_X_boost.is_integer() else int(hamiltonian_X_boost)

    LAMB = LAMB if mode == "X" else 1.0
    assert mode in modes, f"Mode {mode} not in {modes}"
    assert len(eps) == 1 or len(eps) == len(TARGET_ASSET), "Length of eps must be 1 or equal to length of TARGET_ASSET"
    if len(eps) == 1:
        eps = eps * len(TARGET_ASSET)
    eps = np.array(eps)

    # a = cudaq.spin.x(0) * cudaq.spin.x(1) * cudaq.spin.z(2) * cudaq.spin.z(3) * cudaq.spin.y(4)
    # b = cudaq.spin.y(0) * cudaq.spin.x(1) * cudaq.spin.z(2) * cudaq.spin.z(3) * cudaq.spin.y(4)
    # s_a, s_b = a.get_pauli_word(), b.get_pauli_word()
    # c_a, c_b = a.evaluate_coefficient().real, b.evaluate_coefficient().real
    # print(sys.getsizeof(s_a), sys.getsizeof(s_b))
    # print(sys.getsizeof(c_a), sys.getsizeof(c_b))
    # exit(0)

    # Dataset
    data_cov_pd = pd.read_csv("../dataset/top_50_us_stocks_data_20250526_011226_covariance.csv")
    data_ret_p_pd = pd.read_csv("../dataset/top_50_us_stocks_returns_price.csv")
    # print(np.sort(data_ret_p_pd["Price"]))
    # exit(0)

    data_ret_p_pd = data_ret_p_pd[(data_ret_p_pd["Price"] > min_P) & (data_ret_p_pd["Price"] < max_P)]
    data_cov_pd = data_cov_pd.loc[data_cov_pd["Ticker"].isin(data_ret_p_pd["Ticker"])].reset_index(drop=True)
    # print(data_cov_pd.shape, data_ret_p_pd.shape) 
    # exit(0)

    # experiments_approx_Q3/exp_p5_L0.001_q1\
    #                                        |- report_X_boost_1.csv
    #                                        |- report_Preserving_12_boost_2000.csv
    #                                        |- report_Preserving_24_boost_2000.csv
    #                                        |- expectation_X_boost_1.npz [A * E, #optim_loops(varies), 2 * LAYER]
    #                                        |- expectation_Preserving_12_boost_2000.npz [A * E, #optim_loops(varies)]
    #                                        |- expectation_Preserving_24_boost_2000.npz [A * E, #optim_loops(varies)]

    f_Q = Q if not Q.is_integer() else int(Q)
    f_LAMB = LAMB if not LAMB.is_integer() else int(LAMB)
    dir_name = f"exp_p{LAYER}_L{f_LAMB}_q{f_Q}{'_torch' if is_torch_optim else ''}"
    dir_path = f"./experiments_approx_Q{TARGET_QUBIT_IN}{'_RAND' if random_init else ''}/{dir_name}"
    file_postfix = f"{mode}{'' if mode == 'X' else str(num_init_bases)}_boost_{hamiltonian_P_boost if mode == 'Preserving' else hamiltonian_X_boost}"
    file_postfix += ("_GA" if mode == "Preserving" and is_GA else "")
    report_name = f"report_{file_postfix}.csv"
    expect_name = f"expectation_{file_postfix}.npz"

    if is_dir:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Experiments: {E}, Qubits/Asset: {TARGET_QUBIT_IN}, Assets: {TARGET_ASSET}, epsilon: {eps.tolist()}, Lambda: {LAMB}, q: {Q}, Layers: {LAYER}, mode: {mode}{f', num_init_bases: {num_init_bases}' if mode == 'Preserving' else ''}, GA: {is_GA}, boost: {hamiltonian_X_boost if mode == 'X' else hamiltonian_P_boost}")
    # if __name__ == "__main__":
        # from multiprocessing import freeze_support
        # freeze_support()
    # if is_pbar:
    #     pbar_A = tqdm(TARGET_ASSET)
    # # for i, N_ASSETS in enumerate(pbar_A):
    # for i, N_ASSETS in (enumerate(TARGET_ASSET) if not is_pbar else enumerate(pbar_A)):
    pbar_A = tqdm(TARGET_ASSET, disable=not is_pbar)
    for idx_asset, N_ASSETS in enumerate(pbar_A):
        if is_pbar:
            pbar_A.set_description(f"Assets {N_ASSETS}")
            # pbar_exp = tqdm(range(E_st, E), leave=False)
        # for e in (range(E_st, E) if not is_pbar else pbar_exp):
        pbar_exp = tqdm(range(E_st, E), leave=False, disable=not is_pbar)
        for e in pbar_exp:
        # for e in range(E):
            df_now = pd.read_csv(f"{dir_path}/{report_name}") if os.path.exists(f"{dir_path}/{report_name}") else None
            if df_now is not None:
                if not OVERWRITE and df_now[(df_now["Assets"] == N_ASSETS) & (df_now["Exp"] == e)].shape[0] > 0:
                    continue
            else :
                df_now = pd.DataFrame(columns=report_col)

            if is_pbar:
                pbar_exp.set_description("init_1 ")
            st = time.time()
            np.random.seed(911 + 991 * e + 997 * N_ASSETS)
            state = np.random.get_state()
            # asset_idx = np.random.choice(data_cov_pd.shape[0], max(TARGET_ASSET), replace=False)
            asset_idx = np.random.choice(data_cov_pd.shape[0], N_ASSETS, replace=DUPLICATE_ASSET)
            # print(asset_idx)
            # asset_idx = np.array([0, 18, 27, 32, 41])
            # data_cov = data_cov_pd.drop("Ticker", axis=1)
            data_cov = data_cov_pd.drop("Ticker", axis=1).to_numpy()[asset_idx, :][:, asset_idx]
            stock_names = data_ret_p_pd["Company_Name"].to_numpy()[asset_idx]
            # print("Selected Stocks: ", stock_names)
            data_ret_p = data_ret_p_pd.drop("Ticker", axis=1)
            # print(data_ret_p.index[asset_idx].to_numpy())
            asset_idx_raw = data_ret_p.index[asset_idx].to_numpy()
            data_ret_p = data_ret_p.drop("Company_Name", axis=1).to_numpy()[asset_idx, :]

            

            data_ret = data_ret_p[:, 0]
            data_p = data_ret_p[:, 1]

            # print(data_p.tolist())
            # print(data_ret.tolist())
            # print(data_cov.tolist())
            # print(stock_names)
            # print(asset_idx)
            # break

            if os.path.exists(f"{dir_path}/{expect_name}"):
                curr_expect = np.load(f"{dir_path}/{expect_name}")
            else:
                curr_expect = {}
            curr_expect = dict(curr_expect)
            curr_expect[f'A{N_ASSETS}_E{e}_P'] = data_p
            curr_expect[f'A{N_ASSETS}_E{e}_ret'] = data_ret
            curr_expect[f'A{N_ASSETS}_E{e}_cov'] = data_cov
            curr_expect[f'A{N_ASSETS}_E{e}_idx'] = asset_idx_raw
            np.savez_compressed(f"{dir_path}/{expect_name}", **curr_expect)


            # print(data_cov.shape)

            # np.random.set_state(state)
            # selected_price = np.random.uniform(125, 250, N_ASSETS)
            # price_factor = selected_price / data_p
            # data_p = selected_price

            np.random.set_state(state)
            weighted = np.random.uniform(0, 1)
            B_mi, B_ma = find_budget(TARGET_QUBIT_IN * N_ASSETS, data_p, min_P, max_P, min_mix_mode=True)
            B = B_mi * weighted + B_ma * (1 - weighted)


            # print(data_ret)
            # print(B)
            # print(data_p)
            # break
            # print("\n", data_p)
            # break
            if is_GA:
                mutation_rate = 1.5 / (N_ASSETS * TARGET_QUBIT_IN)
                ga = ga_solver.GeneticAlgorithm(
                    prices=data_p,
                    asset_bit_lengths=[TARGET_QUBIT_IN] * N_ASSETS,
                    budget=B,
                    population_size=population_size,
                    mutation_rate=mutation_rate,
                    crossover_rate=crossover_rate,
                    elitism_count=elitism_count,
                    tournament_size=tournament_size
                )
                st_GA = time.perf_counter()
                ga.run(generations, verbose=False)
                et_GA = time.perf_counter()
                time_GA = et_GA - st_GA
                top_inv = ga.get_top_n_individuals(num_init_bases, False)
                feasible_chromosomes_appr = [ind.chromosome for ind in top_inv]
                feasible_reversed_basis_appr = []
                for i in range(len(feasible_chromosomes_appr)):
                    chrom = feasible_chromosomes_appr[i]
                    str_b = ""
                    for aa in range(N_ASSETS):
                        str_a = ""
                        for c in range(TARGET_QUBIT_IN):
                            str_a = str(int(chrom[aa * TARGET_QUBIT_IN + c])) + str_a
                        str_b += str_a
                    feasible_reversed_basis_appr.append(str_b)
                    # print(np.abs(B-top_inv[i].total_cost)/B)

                    ## -------------
                    # st_BF = time.perf_counter()
                    # top_inv_gt = ga.get_top_n_brute_force_individuals(num_init_bases, False)
                    # et_BF = time.perf_counter()
                    # time_BF = et_BF - st_BF

                    col_GA = ["Assets", "GA_time_ms", "BF_time_ms", "mean12_eps_GA", "mean24_eps_GA", "mean12_eps_BF", "mean24_eps_BF"]

                    all_diff_ga, all_diff_bf = 0, 0
                    list_diff_ga = []
                    list_chrom_ga = []
                    for i in range(12):
                        budd = top_inv[i].total_cost
                        # budd_gt = top_inv_gt[i].total_cost
                        all_diff_ga += np.abs(budd - B) / B
                        list_diff_ga.append(np.abs(budd - B) / B)
                        list_chrom_ga.append(top_inv[i].chromosome)
                        # all_diff_bf += np.abs(budd_gt - B) / B
                    mean12_eps_GA = all_diff_ga / 12
                    mean12_eps_BF = all_diff_bf / 12

                    if num_init_bases >= 24:
                        for i in range(12, 24):
                            budd = top_inv[i].total_cost
                            # budd_gt = top_inv_gt[i].total_cost
                            all_diff_ga += np.abs(budd - B) / B
                            list_diff_ga.append(np.abs(budd - B) / B)
                            list_chrom_ga.append(top_inv[i].chromosome)
                            # all_diff_bf += np.abs(budd_gt - B) / B
                    mean24_eps_GA = all_diff_ga / 24
                    mean24_eps_BF = all_diff_bf / 24






            P = data_p[:N_ASSETS]
            ret = data_ret[:N_ASSETS]
            cov = data_cov[:N_ASSETS, :N_ASSETS]

            q = Q
            lamb = LAMB
            hamiltonian_boost = (hamiltonian_X_boost if mode == "X" else hamiltonian_P_boost)
            if DEBUG_GA ^ (not DEBUG_BF):
                P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, P, ret, cov)
                QU_lamb = ret_cov_to_QUBO(np.zeros_like(ret_bb), np.zeros_like(cov_bb), P_bb, lamb, 0.0)
            if not DEBUG_GA:
                TARGET_QUBIT = n_qubit
                # print(f"Assets: {N_ASSETS}, Qubits: {n_qubit}")

                QU = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
                QU_eval = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, 0.0, q)
                QU_return = ret_cov_to_QUBO(ret_bb, np.zeros_like(cov_bb), P_bb, 0.0, 0.0)
                QU_risk = ret_cov_to_QUBO(np.zeros_like(ret_bb), cov_bb, np.zeros_like(P_bb), 0.0, q)

                H_ansatz = -qubo_to_ising(*((QU, lamb) if mode == "X" else (QU_eval, 0.0))).canonicalize() * hamiltonian_boost
                H_lamb = -qubo_to_ising(QU_lamb, lamb).canonicalize() * hamiltonian_boost
                H_eval = -qubo_to_ising(QU_eval, 0.0).canonicalize() * hamiltonian_boost
                H_return = qubo_to_ising(QU_return, 0.0).canonicalize() * hamiltonian_boost
                H_risk = -qubo_to_ising(QU_risk, 0.0).canonicalize() * hamiltonian_boost


            # state_return = all_state_to_return(n_qubit, lamb, QU)
            if DEBUG_GA ^ (not DEBUG_BF):
                st = time.perf_counter()
                state_penalty = -all_state_to_return(n_qubit, lamb, QU_lamb) # lamb * |P^t x -1|^2
                time_BF = time.perf_counter() - st

            # print("Time GA penalty (s):", time_GA)
            # print("Time BF penalty (s):", time_BF)
            if DEBUG_BF:
                state_penalty_s = np.sort(state_penalty)
                mean12_eps_BF = np.sqrt(state_penalty_s[:12] / lamb).mean()
                mean24_eps_BF = np.sqrt(state_penalty_s[:24] / lamb).mean()
            # print(state_penalty_s[:24])
            # for i in range(24):
            #     gaa = list_diff_ga[i]
            #     bff = np.sqrt(state_penalty_s[i] / lamb)
            #     print(gaa, bff, ("****************" if bff - gaa > 1e-8 else ""))
            #     print(list_chrom_ga[i])
            #     print()
            
            df_speed = (pd.read_csv("./speed.csv") if os.path.exists("./speed.csv") else pd.DataFrame(columns=["Assets", "GA_time_ms", "BF_time_ms", "mean12_eps_GA", "mean24_eps_GA", "mean12_eps_BF", "mean24_eps_BF"]))
            new_row_speed = {
                "Assets": N_ASSETS,
                "GA_time_ms": (time_GA * 1000) if is_GA else np.nan,
                "BF_time_ms": time_BF * 1000 if (is_GA and DEBUG_BF) else np.nan,
                "mean12_eps_GA": mean12_eps_GA if (is_GA and not DUPLICATE_ASSET) else np.nan,
                "mean24_eps_GA": mean24_eps_GA if (is_GA and not DUPLICATE_ASSET) else np.nan,
                "mean12_eps_BF": mean12_eps_BF if (is_GA and DEBUG_BF) else np.nan,
                "mean24_eps_BF": mean24_eps_BF if (is_GA and DEBUG_BF) else np.nan
            }
            if is_GA and DEBUG_GA:
                if os.path.exists("./speed.csv"):
                    df_speed = pd.concat([df_speed, pd.DataFrame([new_row_speed])], ignore_index=True)
                else:
                    df_speed = pd.DataFrame([new_row_speed])
                df_speed.to_csv("./speed.csv", index=False)
                continue

            state_eval = all_state_to_return(n_qubit, 0.0, QU_eval)

            # state_optim = -all_state_to_return(n_qubit, *((lamb, QU) if mode == "X" else (0.0, QU_eval)))
            # idx_bestt = np.argmin(state_eval)
            # best_vall = state_optim[idx_bestt]
            # print(state_optim.min(), state_optim.max())
            # # print(best_vall)
            # continue

            # np.save(f"./debug/QU_L1/qubo_A{N_ASSETS}_E{e}.npy", QU_lamb)
            # continue
            # state_budgetD = np.sqrt(state_penalty/lamb)
            # print(np.sort(state_budgetD)[:12])
            # break


            # np.save(f"./debug/state_return_{file_postfix}_p{LAYER}_L{f_LAMB}_q{f_Q}_A{N_ASSETS}_Q{TARGET_QUBIT}.npy", np.array(state_return))
            # print("saved")
            # break

            # |P^t x -1| <= eps
            # lamb (P^t x -1)^2 <= lamb * eps^2
            eps_t = (eps[idx_asset]) ** 2
            idx_feasible = np.where(np.abs(state_penalty) <= eps_t)

            # direct compute
            # state_penalty_eps = np.sqrt(state_penalty / lamb)
            # idx_feasible_debug = np.where(state_penalty_eps <= eps[idx_asset])
            # print("equality:", (sorted(idx_feasible[0]) == sorted(idx_feasible_debug[0]))) 
            # break
            
            # mi_r, ma_r = state_eval[idx_feasible].min(), state_eval[idx_feasible].max()
            # print(len(idx_feasible))
            # idx_dbg = np.argsort(state_penalty)
            # print(np.sqrt(state_penalty[idx_dbg[[1]]] / lamb))
            # print("mi", mi_r, ma_r)
            # break


            init_1_time = time.time() - st

            if is_pbar:
                pbar_exp.set_description("init_2 ")
            st = time.time()
            idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use = process_ansatz_values(H_ansatz)
            coeff_1_use, coeff_2_use = np.array(coeff_1_use), np.array(coeff_2_use)
            kernel_qaoa_use = kernel_qaoa_X if mode == "X" else kernel_qaoa_Preserved
            layer_count = LAYER
            parameter_count = layer_count * 2

            if mode == "X":
                ansatz_fixed_param = (int(n_qubit), layer_count, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use)
            else:
                # init_state = get_init_states(state_return, num_init_bases, n_qubit)
                # idx_eiei = np.argsort(state_penalty)[num_init_bases-1]
                if is_GA:
                    init_state = feasible_reversed_basis_appr.copy()
                else:
                    init_state = get_init_states(state_penalty, num_init_bases, n_qubit)
                # print(sorted(feasible_reversed_basis_appr))
                # print(sorted(get_init_states(state_penalty, num_init_bases, n_qubit)))
                # break

                # # print(init_state[num_init_bases-1], P_bb)
                # uuu = np.array([int(e) for e in init_state[num_init_bases-1]])
                # print(-1 * (uuu @ P_bb - 1))
                # idxx = np.argsort(state_penalty)
                # print(np.sqrt(state_penalty[idxx[[num_init_bases-1]]] / lamb))
                # continue
                n_bases = len(init_state)
                # print("n_bases:", n_bases)
                T = np.zeros((n_bases, n_bases), dtype=np.float32)
                T[:-1, 1:] += np.eye(n_bases - 1, dtype=np.float32)
                T[1:, :-1] += np.eye(n_bases - 1, dtype=np.float32)
                T[0, -1] = T[-1, 0] = 1.0
                # print(T)
                st_pauli = time.time()
                # mixer_s, mixer_c = basis_T_to_pauli(init_state, T, n_qubit)
                mixer_s, mixer_c = basis_T_to_pauli_parallel(init_state, T, n_qubit)
                # print("num pauli string (1 layer):", len(mixer_s))
                # print("time:", time.time() - st_pauli)
                # assert False
                # mixer_s = mixer_s[:250000]
                # mixer_c = mixer_c[:250000]
                # break
                # print(init_state[0])
                # print(feasible_basis_appr[0])
                # break
                init_bases = reversed_str_bases_to_init_state(init_state, n_qubit)
                # print(init_bases)

                ansatz_fixed_param = (int(n_qubit), layer_count, idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use, mixer_s, mixer_c, init_bases)

            mm_1 = np.min(np.abs(coeff_1_use)) if len(coeff_1_use) > 0 else 1e9
            mm_2 = np.min(np.abs(coeff_2_use)) if len(coeff_2_use) > 0 else 1e9
            mm_p = 1e9
            if mode == "Preserving":
                mm_p = np.min(np.abs(mixer_c)) if len(mixer_c) > 0 else 1e9
            mm_i = np.pi / min(mm_1, mm_2, mm_p)

            idx = 3
            if not is_torch_optim:
                optimizer, optimizer_name, FIND_GRAD = get_optimizer(idx)
                optimizer.max_iterations = 300
                # print(f"\noptim iter: {optimizer.max_iterations}\noptim eps: {optimizer.eps}")
            np.random.seed(4001 + 4099 * e + 4999 * N_ASSETS)
            points = np.random.uniform(-1, 1, (parameter_count))
            points[::2] *= mm_i
            points[1::2] *= np.pi
            # print(f"Initial Parameters: {points.tolist()}")

            # result = cudaq.get_state(kernel_qaoa_use, points, *ansatz_fixed_param)
            # prob = np.abs(result)**2
            # print(np.sort(prob))

            if is_torch_optim:
                max_iter = 300
                if random_init:
                    points_cu = torch.tensor(points, dtype=torch.float64, device=device)
                else:
                    points_cu = torch.tensor(np.zeros_like(points), dtype=torch.float64, device=device)
                # print("init at:", np.round(points_cu.cpu().numpy(), 4).tolist())

                # optimizer_cu = Adam([points_cu], lr=hamiltonian_boost)
                optimizer_cu = Adam([points_cu], lr=0.01, betas=(0.95, 0.98), weight_decay=0.01, decoupled_weight_decay=True)
                # optimizer_cu = Adam([points_cu], lr=0.01, betas=(0.9, 0.999), weight_decay=0)
                # optimizer_cu = AdamW([points_cu], lr=0.01)

                # scheduler_co = CosineAnnealingWarmRestarts(optimizer_cu, T_0=300, T_mult=2)
                scheduler_co = CosineAnnealingLR(optimizer_cu, T_max=max_iter, eta_min=0.0003)
                scheduler_cu = ExponentialLR(optimizer_cu, gamma=0.987)
                scheduler_warmup = CyclicLR(optimizer_cu, base_lr=0.01, max_lr=0.012, step_size_up=10, step_size_down=10, mode='triangular2')
                # scheduler_all = SequentialLR(optimizer_cu, schedulers=[scheduler_warmup, scheduler_cu], milestones=[40])
                # scheduler_all = SequentialLR(optimizer_cu, schedulers=[scheduler_warmup, scheduler_co], milestones=[40])
                scheduler_all = scheduler_co
                # scheduler_cu = ReduceLROnPlateau(optimizer_cu, mode='min', factor=0.5, patience=20, min_lr= 1e-5)
                FIND_GRAD = True

            init_2_time = time.time() - st

            if is_pbar:
                pbar_exp.set_description("optim  ")
            # print("start optimization")
            st = time.time()
            num_iter = 0
            last_f = None
            cou_con = 0
            expectations = []
            if not is_torch_optim:
                def cost_func(parameters, cal_expectation=False):
                    # print("in 1")
                    exp_return = float(cudaq.observe(kernel_qaoa_use, H_ansatz, parameters, *ansatz_fixed_param).expectation())
                    # print("in 2")
                    if cal_expectation:
                        # if last_f is not None and abs(exp_return - last_f) < F_TOL:
                        #     raise cudaq.optimization.StopOptimization("Converged")
                        # print("in cal 1")
                        num_iter += 1
                        exp_return_eval = float(cudaq.observe(kernel_qaoa_use, H_eval, parameters, *ansatz_fixed_param).expectation())
                        # print("in cal 2")
                        exp_return_lamb = float(cudaq.observe(kernel_qaoa_use, H_lamb, parameters, *ansatz_fixed_param).expectation()) / hamiltonian_boost
                        exp_return_violate = sqrt(exp_return_lamb / lamb)
                        expectations.append([exp_return / hamiltonian_boost, exp_return_eval / hamiltonian_boost, exp_return_lamb, parameters[0], parameters[1]])
                    return exp_return

                def objective(parameters):
                    expectation = cost_func(parameters, cal_expectation=True)
                    return expectation
                
                def objective_grad_cuda(parameters):
                    expectation = cost_func(parameters, cal_expectation=True)
                    gradient = fd.compute(parameters, cost_func, expectation)
                    return expectation, gradient
                
                objective_func = objective_grad_cuda if FIND_GRAD else objective

                # optimizer.initial_parameters = points
                optimizer.initial_parameters = np.zeros_like(points)
                optimal_expectation, optimal_parameters = optimizer.optimize(
                    dimensions=parameter_count, function=objective_func)
            
            if is_torch_optim:
                optimal_expectation, optimal_parameters = None, None
                # if is_pbar:
                #     pbar_optim = tqdm(range(max_iter), leave=False)
                # for it in (range(max_iter) if not is_pbar else pbar_optim):
                pbar_optim = tqdm(range(max_iter), leave=False, disable=not is_pbar)
                for it in pbar_optim:
                    optimizer_cu.zero_grad()
                    params = points_cu.detach().clone()
                    expectation = float(cudaq.observe(kernel_qaoa_use, H_ansatz, params.cpu().numpy(), *ansatz_fixed_param).expectation())
                    # if last_f is not None:
                    #     print(abs(expectation - last_f))
                    num_iter += 1
                    if mode == "X":
                        expectation_eval = float(cudaq.observe(kernel_qaoa_use, H_eval, params.cpu().numpy(), *ansatz_fixed_param).expectation())
                    else:
                        expectation_eval = expectation
                    expectation_lamb = float(cudaq.observe(kernel_qaoa_use, H_lamb, params.cpu().numpy(), *ansatz_fixed_param).expectation()) / hamiltonian_boost
                    expectation_violate = sqrt(expectation_lamb / lamb)
                    grad = torch.zeros_like(params)
                    # print(grad.dtype)
                    for j in range(parameter_count):
                        shift = np.zeros(parameter_count)
                        shift[j] = SHIFT
                        forward = float(cudaq.observe(kernel_qaoa_use, H_ansatz, (params.cpu().numpy() + shift), *ansatz_fixed_param).expectation())
                        # backward = float(cudaq.observe(kernel_qaoa_use, H_ansatz, (params.cpu().numpy() - shift), *ansatz_fixed_param).expectation())
                        # grad[j] = (forward - backward) / (2.0 * SHIFT)
                        grad[j] = (forward - expectation) / SHIFT
                    # print(grad)
                    # print(grad.abs().mean().item())
                    points_cu.grad = grad
                    optimizer_cu.step()
                    # scheduler_cu.step()
                    # scheduler_cu.step(expectation)
                    scheduler_all.step()
                    # print(points_cu[0].item(), points_cu[1].item())
                    expectations.append([expectation/hamiltonian_boost, expectation_eval/hamiltonian_boost, expectation_lamb, points_cu[0].item(), points_cu[1].item()])
                    # if it > 3 and last_f is not None and abs(expectation - last_f) < F_TOL:
                    #     break
                    
                    cou_con = cou_con + 1 if last_f is not None and abs(expectation - last_f) < F_TOL else 0
                    if cou_con >= 3:
                        break
                    last_f = expectation
                    
                    # if optimal_expectation is None or optimal_expectation > expectation_eval:
                    #     optimal_expectation = expectation_eval
                    #     optimal_parameters = points_cu.cpu().numpy()
                    if is_pbar:
                        pbar_optim.set_description(f"Iter {it}, Exp_obj {expectation/hamiltonian_boost:.6f}, Exp_eval {expectation_eval/hamiltonian_boost:.6f}, Exp_lamb {expectation_lamb:.6f}, LR {optimizer_cu.param_groups[0]['lr']:.4f}")
                optimal_parameters = points_cu.cpu().numpy()
            if os.path.exists(f"{dir_path}/{expect_name}"):
                curr_expect = np.load(f"{dir_path}/{expect_name}")
            else:
                curr_expect = {}
            curr_expect = dict(curr_expect)
            curr_expect[f'A{N_ASSETS}_E{e}'] = np.array(expectations)
            # print("optimal parameters:", optimal_parameters.tolist())
            curr_expect[f'A{N_ASSETS}_E{e}_params'] = np.array(optimal_parameters)
            np.savez_compressed(f"{dir_path}/{expect_name}", **curr_expect)
            optim_time = time.time() - st

            if is_pbar:
                pbar_exp.set_description("observe")
            st = time.time()
            result = cudaq.get_state(kernel_qaoa_use, optimal_parameters, *ansatz_fixed_param)
            idx_r_best = np.argmax(np.abs(result))
            idx_best = bin(idx_r_best)[2:].zfill(n_qubit)[::-1]

            result_r = cudaq.get_state(kernel_flipped, result, TARGET_QUBIT)
            prob = np.abs(result_r)**2
            # print(-np.sort(-prob)[:5])

            # mi_r, ma_r = state_eval.min(), state_eval.max()
            # print(f"\n\nhere\n{idx_feasible[0].shape} {eps}\n\n")
            mi_r, ma_r = state_eval[idx_feasible].min(), state_eval[idx_feasible].max()
            # print(optimal_expectation)
            optimal_expectation = (prob * (state_eval)).sum()
            # print(optimal_expectation)
            # print(np.sort(prob))

            # print(idx_feasible[0].shape)
            if len(idx_feasible[0]) >= 2:
                approx_ratio = (optimal_expectation - mi_r) / (ma_r - mi_r)
                maxprob_ratio = (state_eval[int(idx_best, 2)] - mi_r) / (ma_r - mi_r)
            else:
                approx_ratio, maxprob_ratio = np.nan, np.nan
            budget_violation = float(cudaq.observe(kernel_qaoa_use, H_lamb, optimal_parameters, *ansatz_fixed_param).expectation()) / hamiltonian_boost
            return_final = float(cudaq.observe(kernel_qaoa_use, H_return, optimal_parameters, *ansatz_fixed_param).expectation()) / hamiltonian_boost
            risk_final = float(cudaq.observe(kernel_qaoa_use, H_risk, optimal_parameters, *ansatz_fixed_param).expectation()) / hamiltonian_boost
            #
            observe_time = time.time() - st

            # update df_now for simultaneously run experiments
            df_now = pd.read_csv(f"{dir_path}/{report_name}") if os.path.exists(f"{dir_path}/{report_name}") else pd.DataFrame(columns=report_col)

            # remove row such that Assets and Exp match
            df_now = df_now[~((df_now["Assets"] == N_ASSETS) & (df_now["Exp"] == e))]
            

            df_now.loc[-1] = [N_ASSETS, e, n_qubit, approx_ratio, return_final, risk_final, budget_violation, B, maxprob_ratio, init_1_time, init_2_time, optim_time, num_iter, observe_time]
            df_now.sort_values(by=["Assets", "Exp"], inplace=True)
            df_now.reset_index(drop=True, inplace=True)
            df_now.to_csv(f"{dir_path}/{report_name}", index=False)
