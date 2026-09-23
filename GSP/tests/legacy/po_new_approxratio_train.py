"""FROZEN LEGACY COPY. Do not edit, do not "fix". Reference for the S4 trainer-port test.

Provenance
----------
source : CUDA/PO_new_ApproxRatio.py (repo Quantum_Computing, committed at 8634ca8,
         sha256 2979ad15585cdbe8a67eff72bee26214f7df8d5d64c40d242c14f396f68fe68b)
lines  : 812-818  mm_1, mm_2, mm_p, mm_i (the random-init gamma range)
         833-926  the restart: seed, init, Adam(decoupled_weight_decay) + CosineAnnealingLR, the forward-FD
                  loop, the f_tol stop rule
copied : 2026-09-24 (GSP session S4), by line slicing.
changes: (1) the lines are wrapped in `legacy_train(...)`: lines 812-818 are dedented by 8 spaces and lines
         833-926 by 12, so both sit at function level; (2) the names the lines read from the script's scope
         (the kernel, the Hamiltonians, the fixed arguments, the coefficients, the flags and the constants
         SHIFT = F_TOL = 1e-4 of lines 51-52) are the function's parameters; (3) the function returns
         (points_cu after the loop, expectations, num_iter). Nothing else is changed.
"""
from math import sqrt

import cudaq
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def legacy_train(kernel_qaoa_use, H_ansatz, H_eval, H_lamb, ansatz_fixed_param, coeff_1_use, coeff_2_use, mixer_c,
                 mode, e, N_ASSETS, i_s, layer_count, hamiltonian_boost, lamb, device, random_init=True,
                 is_LR_init=False, delta_beta=0.3, delta_gamma=0.6, WEIGHT_DECAY=0.01, SHIFT=1e-4, F_TOL=1e-4,
                 is_pbar=False):
    parameter_count = layer_count * 2
    # --- CUDA/PO_new_ApproxRatio.py:812-818 ---
    # print("init done")
    mm_1 = np.min(np.abs(coeff_1_use)) if len(coeff_1_use) > 0 else 1e9
    mm_2 = np.min(np.abs(coeff_2_use)) if len(coeff_2_use) > 0 else 1e9
    mm_p = 1e9
    if mode == "Preserving":
        mm_p = np.min(np.abs(mixer_c)) if len(mixer_c) > 0 else 1e9
    mm_i = np.pi / min(mm_1, mm_2, mm_p)
    # --- CUDA/PO_new_ApproxRatio.py:833-926 ---
    num_iter = 0
    last_f = None
    cou_con = 0
    expectations = []

    if mode != "Ramp":
        np.random.seed(4001 + 4099 * e + 4999 * N_ASSETS + 5099 * i_s)
        points = np.random.uniform(-1, 1, (parameter_count))
        # points[::2] *= mm_i
        # points[1::2] *= np.pi
        points[:layer_count] *= mm_i
        points[layer_count:] *= np.pi
        # print(f"Initial Parameters: {points.tolist()}")

        # result = cudaq.get_state(kernel_qaoa_use, points, *ansatz_fixed_param)
        # prob = np.abs(result)**2
        # print(np.sort(prob))

        max_iter = 300
        if random_init:
            points_cu = torch.tensor(points, dtype=torch.float64, device=device)
        elif is_LR_init:
            # same convention as mode "Ramp": beta is negated so the ramp follows the ground state
            # (Eq. 3 of arXiv:2405.09169 is e^{+i beta H_B}, while rx(2*beta) here is e^{-i beta X})
            points_cu = torch.tensor(np.zeros_like(points), dtype=torch.float64, device=device)
            for itt in range(layer_count):
                points_cu[itt] = delta_gamma * (itt+1) / layer_count
                points_cu[layer_count + itt] = -delta_beta * (1 - itt/layer_count)
        else:
            points_cu = torch.tensor(np.zeros_like(points), dtype=torch.float64, device=device)
        # print("init at:", np.round(points_cu.cpu().numpy(), 4).tolist())

        # optimizer_cu = Adam([points_cu], lr=hamiltonian_boost)
        optimizer_cu = Adam([points_cu], lr=0.01, betas=(0.95, 0.98), weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)
        # optimizer_cu = Adam([points_cu], lr=0.01, betas=(0.95, 0.98), weight_decay=0.01, decoupled_weight_decay=True)
        # optimizer_cu = Adam([points_cu], lr=0.01, betas=(0.9, 0.999), weight_decay=0)
        # optimizer_cu = AdamW([points_cu], lr=0.01)

        # scheduler_co = CosineAnnealingWarmRestarts(optimizer_cu, T_0=300, T_mult=2)
        scheduler_co = CosineAnnealingLR(optimizer_cu, T_max=max_iter, eta_min=0.0003)
        # scheduler_cu = ExponentialLR(optimizer_cu, gamma=0.987)
        # scheduler_warmup = CyclicLR(optimizer_cu, base_lr=0.01, max_lr=0.012, step_size_up=10, step_size_down=10, mode='triangular2')
        # scheduler_all = SequentialLR(optimizer_cu, schedulers=[scheduler_warmup, scheduler_cu], milestones=[40])
        # scheduler_all = SequentialLR(optimizer_cu, schedulers=[scheduler_warmup, scheduler_co], milestones=[40])
        scheduler_all = scheduler_co
        # scheduler_cu = ReduceLROnPlateau(optimizer_cu, mode='min', factor=0.5, patience=20, min_lr= 1e-5)
        FIND_GRAD = True


        optimal_expectation, optimal_parameters = None, None
        # if is_pbar:
        #     pbar_optim = tqdm(range(max_iter), leave=False)
        # for it in (range(max_iter) if not is_pbar else pbar_optim):
        pbar_optim = tqdm(range(max_iter), leave=False, disable=not is_pbar)
        for it in pbar_optim:
            optimizer_cu.zero_grad()
            params = points_cu.detach().clone()
            # print(params.cpu().numpy())
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

    # --- end of copy ---
    return points_cu.detach().cpu().numpy(), expectations, num_iter
