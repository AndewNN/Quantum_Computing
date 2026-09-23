"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for the S1 reproduction tests.

Provenance
----------
source : CUDA/PO_new_ApproxRatio.py (repo Quantum_Computing, committed at 8634ca8,
         sha256 2979ad15585cdbe8a67eff72bee26214f7df8d5d64c40d242c14f396f68fe68b)
lines  : 370      LAMB override for mode "Preserving"
         386-394  dataset load + price filter
         446-485  the draw: asset choice, budget weight, find_budget, B
         579-623  QUBOs, Hamiltonians, Jh boost
         660      state_penalty (brute-force band energies)
         707-711  state_eval / state_optim (diagonal energies)
         733-734  eps_t / idx_feasible (the band)
         Returned `ansatz_terms` are the un-boosted ones of line 606 (the copy stops at 623).
copied : 2026-09-24 (GSP session S1), generated with sed-like line slicing.
changes: (1) the lines are wrapped in `legacy_instance(...)`: loop-level lines are dedented by
         8 spaces, the `if DEBUG_GA ^ (not DEBUG_BF):` / `if not DEBUG_GA:` guards are kept;
         (2) `"../dataset/` is replaced by `DATASET_DIR + "/` (the script ran from CUDA/);
         (3) the constants the lines read (argparse defaults of the completed Q2 runs) are set
         in the preamble below; nothing else is changed.
"""
import os
import numpy as np
import pandas as pd

from .qaoaCUDAQ_instance import (po_normalize, ret_cov_to_QUBO, qubo_to_ising,
                                 process_ansatz_values, find_budget, all_state_to_return, to_sig)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")


def legacy_instance(N_ASSETS, e, Q, LAMB, mode="X"):
    """Replay the completed work's instance build for one (N, e, q, lambda, mode)."""
    # --- preamble: the constants of the completed Q2 runs (-Q 2 -norm Jh, defaults otherwise) ---
    TARGET_QUBIT_IN = 2
    min_P, max_P = 108, 216
    DUPLICATE_ASSET = False
    DEBUG_GA = False
    DEBUG_BF = False
    auto_boost_mode = "Jh"
    learning_rate_scale = 1.0
    hamiltonian_X_boost = hamiltonian_R_boost = hamiltonian_P_boost = 0.0
    eps = np.array([0.1])
    idx_asset = 0
    LAMB = LAMB if mode in ["X", "Ramp"] else 1.0  # PO_new_ApproxRatio.py:370 (verbatim)
    # --- PO_new_ApproxRatio.py:386-394 ---
    data_cov_pd = pd.read_csv(DATASET_DIR + "/top_50_us_stocks_data_20250526_011226_covariance.csv")
    data_ret_p_pd = pd.read_csv(DATASET_DIR + "/top_50_us_stocks_returns_price.csv")
    # print(np.sort(data_ret_p_pd["Price"]))
    # exit(0)

    data_ret_p_pd = data_ret_p_pd[(data_ret_p_pd["Price"] > min_P) & (data_ret_p_pd["Price"] < max_P)]

    data_cov_pd = data_cov_pd.loc[data_cov_pd["Ticker"].isin(data_ret_p_pd["Ticker"])].reset_index(drop=True)
    data_cov_pd = data_cov_pd[["Ticker"] + data_cov_pd["Ticker"].tolist()]

    # --- PO_new_ApproxRatio.py:446-485 ---
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

    # print(data_cov)
    # print(data_p.tolist())
    # print(data_ret.tolist())
    # print(data_cov.tolist())
    # print(stock_names)
    # print(asset_idx)
    # break


    # print(data_cov.shape)

    # np.random.set_state(state)
    # selected_price = np.random.uniform(125, 250, N_ASSETS)
    # price_factor = selected_price / data_p
    # data_p = selected_price

    np.random.set_state(state)
    weighted = np.random.uniform(0, 1)
    B_mi, B_ma = find_budget(TARGET_QUBIT_IN * N_ASSETS, data_p, min_P, max_P, min_mix_mode=True)
    B = B_mi * weighted + B_ma * (1 - weighted)

    # --- PO_new_ApproxRatio.py:579-623 ---
    P = data_p[:N_ASSETS]
    ret = data_ret[:N_ASSETS]
    cov = data_cov[:N_ASSETS, :N_ASSETS]

    q = Q
    lamb = LAMB
    hamiltonian_boost = (hamiltonian_X_boost if mode == "X" else hamiltonian_R_boost if mode == "Ramp" else hamiltonian_P_boost)
    if DEBUG_GA ^ (not DEBUG_BF):
        P_bb, ret_bb, cov_bb, n_qubit, n_max, C = po_normalize(B, P, ret, cov)
        QU_lamb = ret_cov_to_QUBO(np.zeros_like(ret_bb), np.zeros_like(cov_bb), P_bb, lamb, 0.0)
    if not DEBUG_GA:
        TARGET_QUBIT = n_qubit
        # print(f"Assets: {N_ASSETS}, Qubits: {n_qubit}")

        # QUBOs of MAX PROBLEM
        QU = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, lamb, q)
        QU_eval = ret_cov_to_QUBO(ret_bb, cov_bb, P_bb, 0.0, q)
        QU_return = ret_cov_to_QUBO(ret_bb, np.zeros_like(cov_bb), np.zeros_like(P_bb), 0.0, 0.0)
        QU_risk = ret_cov_to_QUBO(np.zeros_like(ret_bb), cov_bb, np.zeros_like(P_bb), 0.0, q)

        # Hamiltonians of MIN PROBLEM
        H_ansatz = -qubo_to_ising(*((QU, lamb) if mode in ["X", "Ramp"] else (QU_eval, 0.0))).canonicalize()
        H_lamb = -qubo_to_ising(QU_lamb, lamb).canonicalize()
        H_eval = -qubo_to_ising(QU_eval, 0.0).canonicalize()
        H_return = -qubo_to_ising(QU_return, 0.0).canonicalize()
        H_risk = -qubo_to_ising(QU_risk, 0.0).canonicalize()

        idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use = process_ansatz_values(H_ansatz)
        coeff_1_use, coeff_2_use = np.array(coeff_1_use), np.array(coeff_2_use)
        max_J = np.max(np.abs(coeff_2_use))
        max_h = np.max(np.abs(coeff_1_use))
        max_J_h = max(max_J, max_h)
        # print(f"max|J|: {to_sig(max_J)} -> {1/max_J}")
        # print(f"max|J,h|: {to_sig(max_J_h)} -> {1/max_J_h}")
        # print(f"max|h|: {to_sig(max_h)} -> {1/max_h}")
        use_norm = (max_J if auto_boost_mode == "J" else max_J_h if auto_boost_mode == "Jh" else max_h if auto_boost_mode == "h" else 1.0)
        hamiltonian_boost = 1 / use_norm if auto_boost_mode != "fixed" else hamiltonian_boost
        hamiltonian_boost = hamiltonian_boost * learning_rate_scale
        hamiltonian_boost = to_sig(hamiltonian_boost, 4)

        H_ansatz = H_ansatz * hamiltonian_boost
        H_lamb = H_lamb * hamiltonian_boost
        H_eval = H_eval * hamiltonian_boost
        H_return = H_return * hamiltonian_boost
        H_risk = H_risk * hamiltonian_boost 

    # --- PO_new_ApproxRatio.py:660 (inside `if DEBUG_GA ^ (not DEBUG_BF):`) ---
    state_penalty = -all_state_to_return(n_qubit, lamb, QU_lamb) # lamb * |P^t x -1|^2

    # --- PO_new_ApproxRatio.py:707-711 ---
    state_eval = all_state_to_return(n_qubit, 0.0, QU_eval)
    idx_optimal = np.argsort(state_eval)[-1]
    # print(state_eval[np.argsort(state_eval)[-10:]])

    state_optim = -all_state_to_return(n_qubit, *((lamb, QU) if mode in ["X", "Ramp"] else (0.0, QU_eval)))

    # --- PO_new_ApproxRatio.py:733-734 ---
    eps_t = lamb * (eps[idx_asset]) ** 2
    idx_feasible = np.where(np.abs(state_penalty) <= eps_t)

    return dict(asset_idx=asset_idx, asset_idx_raw=asset_idx_raw, stock_names=stock_names,
                P=P, ret=ret, cov=cov, weighted=weighted, B_mi=B_mi, B_ma=B_ma, B=B,
                P_bb=P_bb, ret_bb=ret_bb, cov_bb=cov_bb, n_qubit=n_qubit, n_max=n_max, C=C,
                QU=QU, QU_eval=QU_eval, QU_lamb=QU_lamb, QU_return=QU_return, QU_risk=QU_risk,
                H_ansatz=H_ansatz, H_lamb=H_lamb, H_eval=H_eval, hamiltonian_boost=hamiltonian_boost,
                ansatz_terms=(idx_1_use, coeff_1_use, idx_2_a_use, idx_2_b_use, coeff_2_use),
                state_penalty=state_penalty, state_eval=state_eval, state_optim=state_optim,
                idx_feasible=idx_feasible[0], lamb=lamb, q=q)
