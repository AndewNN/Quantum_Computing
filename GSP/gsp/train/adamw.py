"""The AdamW trainer of A0 / A1 (PLAN §1.6), ported from CUDA/PO_new_ApproxRatio.py:851-926.

Verbatim settings: `torch.optim.Adam(lr=0.01, betas=(0.95, 0.98), weight_decay=0.01,
decoupled_weight_decay=True)` (= AdamW) and `CosineAnnealingLR(T_max=300, eta_min=3e-4)`, at most 300
iterations, float64 parameters; each iteration: f = energy(theta), the forward-FD gradient (`gradients.py`),
`optimizer.step()`, `scheduler.step()`, then the stop rule: stop once |f_t - f_{t-1}| < f_tol = 1e-4 held for 3
consecutive iterations (f in the boosted units of the old code). The completed runs kept torch on the GPU; here
torch is CPU-only (torch 2.10.0+cpu), the arithmetic is float64 in both.

Update loop = observe only: `energy` is `Ansatz.energy` (backend.observe). The post-update `logger(t, params)`
is called with a COPY of the parameters at t = 0 (the init) and after every update; nothing it returns or does
reaches the update (a test asserts identical trajectories with the logger on and off). The trainer's wall clock
excludes the logger's time.

Trajectory convention: theta_0 = init, theta_t = after t updates. `f[t]` = f(theta_t) as the loop measured it
(t < T; the loop never evaluates theta_T). circuits charged up to theta_t = t (2L + 1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .gradients import ForwardFD


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 0.01
    betas: tuple = (0.95, 0.98)
    weight_decay: float = 0.01
    t_max: int = 300
    eta_min: float = 0.0003
    max_iter: int = 300
    f_tol: float = 1e-4
    patience: int = 3


@dataclass
class TrainResult:
    params: np.ndarray          # theta_T (after the last update)
    params_hist: np.ndarray     # (T + 1, P): theta_0 .. theta_T
    f_hist: np.ndarray          # (T,): f(theta_t) as measured by the loop (boosted units)
    lr_hist: np.ndarray         # (T,): learning rate of update t
    grad_norm: np.ndarray       # (T,): |grad| of update t
    wall_hist: np.ndarray       # (T + 1,): trainer wall clock (s) when theta_t was reached, logger excluded
    n_iter: int                 # T
    converged: bool             # stopped by the f_tol rule (else max_iter)
    circuits_per_iter: int      # 2L + 1
    logger_s: float             # time spent in the logger

    @property
    def circuits_charged(self) -> np.ndarray:
        return np.arange(self.n_iter + 1, dtype=np.int64) * self.circuits_per_iter


def train_adamw(energy, x0, cfg: AdamWConfig = AdamWConfig(), gradient=None, logger=None) -> TrainResult:
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    gradient = ForwardFD() if gradient is None else gradient
    torch.set_num_threads(1)
    x0 = np.array(x0, dtype=np.float64)
    parameter_count = x0.size
    t_start = time.perf_counter()
    logger_s = 0.0

    def log(t, p):
        nonlocal logger_s
        if logger is None:
            return
        tl = time.perf_counter()
        logger(t, np.array(p, dtype=np.float64, copy=True))
        logger_s += time.perf_counter() - tl

    points_cu = torch.tensor(x0, dtype=torch.float64)
    optimizer_cu = Adam([points_cu], lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay,
                        decoupled_weight_decay=True)
    scheduler_all = CosineAnnealingLR(optimizer_cu, T_max=cfg.t_max, eta_min=cfg.eta_min)

    params_hist = [x0.copy()]
    wall = [0.0]
    f_hist, lr_hist, gnorm = [], [], []
    log(0, x0)
    last_f = None
    cou_con = 0
    converged = False
    for _it in range(cfg.max_iter):
        optimizer_cu.zero_grad()
        params = points_cu.detach().clone()
        p = params.cpu().numpy()
        expectation = float(energy(p))
        g = gradient(energy, p, expectation)
        grad = torch.zeros_like(params)
        for j in range(parameter_count):
            grad[j] = g[j]
        lr_hist.append(float(optimizer_cu.param_groups[0]["lr"]))
        points_cu.grad = grad
        optimizer_cu.step()
        scheduler_all.step()
        f_hist.append(expectation)
        gnorm.append(float(np.linalg.norm(np.asarray(g, dtype=np.float64))))
        now = points_cu.detach().cpu().numpy().copy()
        params_hist.append(now)
        wall.append(time.perf_counter() - t_start - logger_s)
        log(len(f_hist), now)
        cou_con = cou_con + 1 if last_f is not None and abs(expectation - last_f) < cfg.f_tol else 0
        if cou_con >= cfg.patience:
            converged = True
            break
        last_f = expectation
    return TrainResult(params=params_hist[-1], params_hist=np.array(params_hist), f_hist=np.array(f_hist),
                       lr_hist=np.array(lr_hist), grad_norm=np.array(gnorm), wall_hist=np.array(wall),
                       n_iter=len(f_hist), converged=converged,
                       circuits_per_iter=gradient.circuits(parameter_count), logger_s=logger_s)
