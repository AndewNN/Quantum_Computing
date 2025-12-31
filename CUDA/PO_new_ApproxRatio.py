import numpy as np
import pandas as pd
import cudaq
import sys
import os
import torch
from torch.optim import Adam, adamw
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import time
from tqdm import tqdm
import shutil
import argparse
import faulthandler
faulthandler.enable()
sys.path.append(os.path.abspath(".."))
from Utils.qaoaCUDAQ import po_normalize, ret_cov_to_QUBO, qubo_to_ising, process_ansatz_values, kernel_qaoa_X, find_budget, kernel_flipped,\
    kernel_qaoa_Preserved, all_state_to_return, get_init_states, basis_T_to_pauli, reversed_str_bases_to_init_state, get_optimizer


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

    report_col = ["Assets", "Exp", "Qubits", "Approximate_ratio", "Budget_Violations", "MaxProb_ratio", "init_1_time", "init_2_time", "optim_time", "observe_time"]

    TARGET_QUBIT_IN = 3
    TARGET_ASSET = [3, 4, 5, 6, 7]
    min_P, max_P = 125, 250
    hamiltonian_X_boost = 7500
    hamiltonian_P_boost = 7500
    modes = ["X", "Preserving"]
    eps = [0.1]
    SHIFT = 1e-6

    os.system("taskset -p 0xfffff %d" % os.getpid())

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
            type=int, default=hamiltonian_X_boost,
            help="Hamiltonian boost for X mixer (int)"
        )

        # Hamiltonian boost for Preserving mixer
        parser.add_argument(
            "-b_P", "--ham_boost_P",
            type=int, default=hamiltonian_P_boost,
            help="Hamiltonian boost for Preserving mixer (int)"
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
    # print(data_cov.shape, data_ret_p.shape)

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
    dir_path = f"./experiments_approx_Q{TARGET_QUBIT_IN}/{dir_name}"
    file_postfix = f"{mode}{'' if mode == 'X' else str(num_init_bases)}_boost_{hamiltonian_P_boost if mode == 'Preserving' else hamiltonian_X_boost}"
    report_name = f"report_{file_postfix}.csv"
    expect_name = f"expectation_{file_postfix}.npz"

    if is_dir:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Experiments: {E}, Qubits/Asset: {TARGET_QUBIT_IN}, Assets: {TARGET_ASSET}, epsilon: {eps.tolist()}, Lambda: {LAMB}, q: {Q}, Layers: {LAYER}, mode: {mode}{f', num_init_bases: {num_init_bases}' if mode == 'Preserving' else ''}, boost: {hamiltonian_X_boost if mode == 'X' else hamiltonian_P_boost}")
    # if __name__ == "__main__":
        # from multiprocessing import freeze_support
        # freeze_support()
    if is_pbar:
        pbar_A = tqdm(TARGET_ASSET)
    # for i, N_ASSETS in enumerate(pbar_A):
    for i, N_ASSETS in (enumerate(TARGET_ASSET) if not is_pbar else enumerate(pbar_A)):
        if is_pbar:
            pbar_A.set_description(f"Assets {N_ASSETS}")
            pbar_exp = tqdm(range(E), leave=False)
        for e in (range(E) if not is_pbar else pbar_exp):
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

            np.random.set_state(state)
            weighted = np.random.uniform(0, 1)
            B_mi, B_ma = find_budget(TARGET_QUBIT_IN * N_ASSETS, data_p, min_P, max_P, min_mix_mode=True)
            B = B_mi * weighted + B_ma * (1 - weighted)

            P = data_p[:N_ASSETS]
            ret = data_ret[:N_ASSETS]
            cov = data_cov[:N_ASSETS, :N_ASSETS]

            q = Q
            P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, P, ret, cov)
            # print(f"Assets: {N_ASSETS}, Qubits: {n_qubit}")
            TARGET_QUBIT = n_qubit
            lamb = LAMB

            QU = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
            QU_lamb = ret_cov_to_QUBO(np.zeros_like(ret_bb), np.zeros_like(cov_bb), P_bb, lamb, 0.0)
            QU_eval = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, 0.0, q)
            hamiltonian_boost = (hamiltonian_X_boost if mode == "X" else hamiltonian_P_boost)

            if is_torch_optim:
                H_ansatz = -qubo_to_ising(*((QU, lamb) if mode == "X" else (QU_eval, 0.0))).canonicalize()
                H_lamb = -qubo_to_ising(QU_lamb, lamb).canonicalize()
                H_eval = -qubo_to_ising(QU_eval, 0.0).canonicalize()
            else:
                H_ansatz = -qubo_to_ising(*((QU, lamb) if mode == "X" else (QU_eval, 0.0))).canonicalize() * hamiltonian_boost
                H_lamb = -qubo_to_ising(QU_lamb, lamb).canonicalize()
                H_eval = -qubo_to_ising(QU_eval, 0.0).canonicalize() * hamiltonian_boost

            # state_return = all_state_to_return(n_qubit, lamb, QU)
            state_penalty = -all_state_to_return(n_qubit, lamb, QU_lamb)
            state_eval = all_state_to_return(n_qubit, 0.0, QU_eval)


            # np.save(f"./debug/state_return_{file_postfix}_p{LAYER}_L{f_LAMB}_q{f_Q}_A{N_ASSETS}_Q{TARGET_QUBIT}.npy", np.array(state_return))
            # print("saved")
            # break

            # |P^t x -1| <= eps
            # lamb (P^t x -1)^2 <= lamb * eps^2
            eps_t = lamb * (eps[i]) ** 2
            idx_feasible = np.where(np.abs(state_penalty) <= eps_t)

            # direct compute
            # state_penalty_eps = np.sqrt(state_penalty / lamb)
            # idx_feasible_debug = np.where(state_penalty_eps <= eps[i])
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
                init_state = get_init_states(state_penalty, num_init_bases, n_qubit)
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
                mixer_s, mixer_c = basis_T_to_pauli(init_state, T, n_qubit)
                # print("num pauli string (1 layer):", len(mixer_s))
                # print("time:", time.time() - st_pauli)
                # break
                # mixer_s = mixer_s[:250000]
                # mixer_c = mixer_c[:250000]
                # break
                init_bases = reversed_str_bases_to_init_state(init_state, n_qubit)

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
                optimizer.max_iterations = 210
            np.random.seed(4001 + 4099 * e + 4999 * N_ASSETS)
            points = np.random.uniform(-1, 1, (parameter_count))
            points[::2] *= mm_i
            points[1::2] *= np.pi

            if is_torch_optim:
                points_cu = torch.tensor(points, dtype=torch.float32, device=device)
                optimizer_cu = Adam([points_cu], lr=hamiltonian_boost)
                # scheduler_cu = CosineAnnealingWarmRestarts(optimizer_cu, T_0=30, T_mult=2, eta_min=hamiltonian_boost * 0.01)
                scheduler_cu = ReduceLROnPlateau(optimizer_cu, mode='min', factor=0.5, patience=20, min_lr=hamiltonian_boost * 1e-4)
                FIND_GRAD = True
                max_iter = 210 # 30 + 60 + 120

            init_2_time = time.time() - st

            if is_pbar:
                pbar_exp.set_description("optim  ")
            # print("start optimization")
            st = time.time()
            expectations = []
            if not is_torch_optim:
                def cost_func(parameters, cal_expectation=False):
                    # print("in 1")
                    exp_return = float(cudaq.observe(kernel_qaoa_use, H_ansatz, parameters, *ansatz_fixed_param).expectation())
                    # print("in 2")
                    if cal_expectation:
                        # print("in cal 1")
                        exp_return_eval = float(cudaq.observe(kernel_qaoa_use, H_eval, parameters, *ansatz_fixed_param).expectation())
                        # print("in cal 2")
                        exp_return_lamb = float(cudaq.observe(kernel_qaoa_use, H_lamb, parameters, *ansatz_fixed_param).expectation())
                        expectations.append([exp_return / hamiltonian_boost, exp_return_eval / hamiltonian_boost, exp_return_lamb])
                    return exp_return

                def objective(parameters):
                    expectation = cost_func(parameters, cal_expectation=True)
                    return expectation
                
                def objective_grad_cuda(parameters):
                    expectation = cost_func(parameters, cal_expectation=True)
                    gradient = fd.compute(parameters, cost_func, expectation)
                    return expectation, gradient
                
                objective_func = objective_grad_cuda if FIND_GRAD else objective

                optimal_expectation, optimal_parameters = optimizer.optimize(
                    dimensions=parameter_count, function=objective_func)
            
            if is_torch_optim:
                optimal_expectation, optimal_parameters = None, None
                if is_pbar:
                    pbar_optim = tqdm(range(max_iter), leave=False)
                for it in (range(max_iter) if not is_pbar else pbar_optim):
                    optimizer_cu.zero_grad()
                    params = points_cu.detach().clone()
                    expectation = float(cudaq.observe(kernel_qaoa_use, H_ansatz, params.cpu().numpy(), *ansatz_fixed_param).expectation())
                    if mode == "X":
                        expectation_eval = float(cudaq.observe(kernel_qaoa_use, H_eval, params.cpu().numpy(), *ansatz_fixed_param).expectation())
                    else:
                        expectation_eval = expectation
                    expectation_lamb = float(cudaq.observe(kernel_qaoa_use, H_lamb, params.cpu().numpy(), *ansatz_fixed_param).expectation())
                    grad = torch.zeros_like(params)
                    for j in range(parameter_count):
                        shift = np.zeros(parameter_count)
                        shift[j] = SHIFT
                        forward = float(cudaq.observe(kernel_qaoa_use, H_ansatz, (params.cpu().numpy() + shift), *ansatz_fixed_param).expectation())
                        # backward = float(cudaq.observe(kernel_qaoa_use, H_ansatz, (params.cpu().numpy() - shift), *ansatz_fixed_param).expectation())
                        # grad[j] = (forward - backward) / (2.0 * SHIFT)
                        grad[j] = (forward - expectation) / SHIFT
                    points_cu.grad = grad
                    optimizer_cu.step()
                    scheduler_cu.step(expectation)
                    expectations.append([expectation, expectation_eval, expectation_lamb])
                    if optimal_expectation is None or optimal_expectation > expectation_eval:
                        optimal_expectation = expectation_eval
                        optimal_parameters = params.cpu().numpy()
                    if is_pbar:
                        pbar_optim.set_description(f"Iter {it}, Exp_obj {expectation:.6f}, Exp_eval {expectation_eval:.6f}, Exp_lamb {expectation_lamb:.6f}, LR {optimizer_cu.param_groups[0]['lr']:.4f}")
            
            if os.path.exists(f"{dir_path}/{expect_name}"):
                curr_expect = np.load(f"{dir_path}/{expect_name}")
            else:
                curr_expect = {}
            curr_expect = dict(curr_expect)
            curr_expect[f'A{N_ASSETS}_E{e}'] = np.array(expectations)
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
            mi_r, ma_r = state_eval[idx_feasible].min(), state_eval[idx_feasible].max()
            # print(optimal_expectation)
            optimal_expectation = (prob * (state_eval)).sum()
            # print(optimal_expectation)

            approx_ratio = (optimal_expectation - mi_r) / (ma_r - mi_r)
            maxprob_ratio = (state_eval[int(idx_best, 2)] - mi_r) / (ma_r - mi_r)
            budget_violation = float(cudaq.observe(kernel_qaoa_use, H_lamb, optimal_parameters, *ansatz_fixed_param).expectation())
            observe_time = time.time() - st

            # remove row such that Assets and Exp match
            df_now = df_now[~((df_now["Assets"] == N_ASSETS) & (df_now["Exp"] == e))]
            

            df_now.loc[-1] = [N_ASSETS, e, n_qubit, approx_ratio, budget_violation, maxprob_ratio, init_1_time, init_2_time, optim_time, observe_time]
            df_now.sort_values(by=["Assets", "Exp"], inplace=True)
            df_now.reset_index(drop=True, inplace=True)
            df_now.to_csv(f"{dir_path}/{report_name}", index=False)
