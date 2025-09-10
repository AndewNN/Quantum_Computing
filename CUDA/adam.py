# adam.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.optim import Adam


ArrayLike = np.ndarray

@dataclass
class AdamDefaults:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0             # L2 (coupled) if decouple_wd=False; WD if decouple_wd=True
    decouple_wd: bool = True              # AdamW by default
    amsgrad: bool = False
    bias_correction: bool = True
    max_grad_norm: Optional[float] = None # global clipping (None = off)


class _ParamAdapter:
    """
    Small adapter that lets us accept either:
      - dicts: {"param": ndarray, "grad": ndarray}
      - objects with attributes: .data (ndarray) and .grad (ndarray or None)
    """
    def __init__(self, target: Any):
        self._t = target

    @property
    def data(self) -> ArrayLike:
        if isinstance(self._t, dict):
            return self._t["param"]
        return self._t.data

    @property
    def grad(self) -> Optional[ArrayLike]:
        if isinstance(self._t, dict):
            return self._t.get("grad", None)
        return getattr(self._t, "grad", None)

    @grad.setter
    def grad(self, value: Optional[ArrayLike]) -> None:
        if isinstance(self._t, dict):
            self._t["grad"] = value
        else:
            setattr(self._t, "grad", value)


class Adam:
    """
    A lightweight, NumPy-based Adam/AdamW optimizer.

    Parameters
    ----------
    params : Iterable
        Iterable of parameter carriers. Each item can be:
          - dict {"param": np.ndarray, "grad": np.ndarray}
          - an object with .data (np.ndarray) and .grad (np.ndarray or None)
        You may also pass *param groups*: dictionaries with a "params" list and
        optional per-group hyperparam overrides, e.g.:
          { "params": [p1, p2], "lr": 5e-4, "weight_decay": 0.01 }
    defaults : AdamDefaults
        Global defaults. Per-group overrides take precedence.
    """

    # ------------------------- public API -------------------------

    def __init__(
        self,
        params: Iterable[Any],
        defaults: AdamDefaults = AdamDefaults(),
    ):
        self.defaults = defaults
        self.param_groups: List[Dict[str, Any]] = self._build_param_groups(params)
        self.state: Dict[int, Dict[str, Any]] = {}  # per-parameter state
        self._step: int = 0

        # Initialize states lazily on first step to handle shapes properly

    def zero_grad(self) -> None:
        """Sets all gradients to zero (if present)."""
        for group in self.param_groups:
            for p_raw in group["params"]:
                p = _ParamAdapter(p_raw)
                if p.grad is not None:
                    p.grad[...] = 0.0

    def step(self) -> None:
        """Performs a single optimization step."""
        self._step += 1

        # Optional global grad clipping by norm (across all tensors in all groups)
        if any(g["max_grad_norm"] is not None for g in self.param_groups):
            self._global_clip()

        for group in self.param_groups:
            lr            = group["lr"]
            beta1, beta2  = group["betas"]
            eps           = group["eps"]
            wd            = group["weight_decay"]
            decouple_wd   = group["decouple_wd"]
            amsgrad       = group["amsgrad"]
            bias_corr     = group["bias_correction"]

            for p_raw in group["params"]:
                p = _ParamAdapter(p_raw)
                grad = p.grad
                if grad is None:
                    continue

                if grad.dtype != p.data.dtype:
                    # cast grad to param dtype (simple AMP safety)
                    grad = grad.astype(p.data.dtype, copy=False)

                state = self._get_state_for(p)
                m, v = state["m"], state["v"]

                if amsgrad:
                    vhat = state["vhat"]

                # Apply coupled L2 regularization (Adam) by adding to the gradient
                if wd != 0.0 and not decouple_wd:
                    grad = grad + wd * p.data

                # Update biased first/second moment estimates
                m[...] = beta1 * m + (1.0 - beta1) * grad
                v[...] = beta2 * v + (1.0 - beta2) * (grad * grad)
                mhat = m / (1.0 - beta1 ** self._step)

                if amsgrad:
                    np.maximum(vhat, v, out=vhat)
                    denom = vhat / (1 - beta2 ** self._step)
                    denom = np.sqrt(denom) + eps
                else:
                    denom = v / (1 - beta2 ** self._step)
                    denom = np.sqrt(denom) + eps

                # Bias correction
                if bias_corr:
                    t = self._step
                    bias_c1 = 1.0 - beta1 ** t
                    bias_c2 = 1.0 - beta2 ** t
                    step_size = lr * (math.sqrt(bias_c2) / bias_c1)
                else:
                    step_size = lr

                # Parameter update
                update = step_size * (mhat / denom)

                if decouple_wd and wd != 0.0:
                    # AdamW: decoupled weight decay
                    p.data[...] = p.data - lr * wd * p.data - update
                else:
                    p.data[...] = p.data - update

    def state_dict(self) -> Dict[str, Any]:
        """Returns a Python dict with optimizer state (for checkpointing)."""
        # Save param group settings and per-parameter states
        packed_groups = []
        for g in self.param_groups:
            g_copy = {k: v for k, v in g.items() if k != "params"}
            g_copy["param_ids"] = [id(_ParamAdapter(p).data) for p in g["params"]]
            packed_groups.append(g_copy)

        # Map states by param id
        packed_states = {pid: self._pack_state(s) for pid, s in self._enumerate_states().items()}

        return {
            "step": self._step,
            "defaults": self.defaults.__dict__.copy(),
            "param_groups": packed_groups,
            "state": packed_states,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads optimizer state produced by `state_dict`."""
        self._step = int(state_dict.get("step", 0))
        # restore defaults (non-strict—missing keys fall back to current)
        d = state_dict.get("defaults", {})
        self.defaults = AdamDefaults(**{**self.defaults.__dict__, **d})

        # Rebuild a mapping from param id to actual param
        id_map = {id(_ParamAdapter(p).data): _ParamAdapter(p) for g in self.param_groups for p in g["params"]}

        # Restore per-parameter states
        packed_states: Dict[int, Dict[str, Any]] = state_dict.get("state", {})
        self.state.clear()
        for pid, s in packed_states.items():
            pid = int(pid)
            if pid in id_map:
                p = id_map[pid]
                self._set_state_for(p, self._unpack_state(s, p.data.shape, p.data.dtype))

        # Restore group hyperparams, matching by param ids (best-effort)
        incoming_groups: List[Dict[str, Any]] = state_dict.get("param_groups", [])
        # Build a flat list of our param ids per group for alignment
        my_groups_ids = [[id(_ParamAdapter(p).data) for p in g["params"]] for g in self.param_groups]
        for g_in in incoming_groups:
            in_ids = g_in.get("param_ids", [])
            # Find the best matching group by overlap
            best = None
            best_overlap = -1
            for i, mine in enumerate(my_groups_ids):
                overlap = len(set(in_ids).intersection(mine))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = i
            if best is not None and best_overlap > 0:
                # Merge hyperparams (keep our "params" list)
                for k, v in g_in.items():
                    if k not in ("params", "param_ids"):
                        self.param_groups[best][k] = v

    # ------------------------- helpers -------------------------

    def _build_param_groups(self, params: Iterable[Any]) -> List[Dict[str, Any]]:
        # Support either a flat list of params, or list of group dicts
        groups: List[Dict[str, Any]] = []
        if not params:
            raise ValueError("Adam received an empty params iterable.")

        def _make_group(param_list: List[Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
            # fill with defaults, then apply overrides
            g = {
                "params": list(param_list),
                "lr": self.defaults.lr,
                "betas": self.defaults.betas,
                "eps": self.defaults.eps,
                "weight_decay": self.defaults.weight_decay,
                "decouple_wd": self.defaults.decouple_wd,
                "amsgrad": self.defaults.amsgrad,
                "bias_correction": self.defaults.bias_correction,
                "max_grad_norm": self.defaults.max_grad_norm,
            }
            g.update(overrides)
            # Basic validation
            b1, b2 = g["betas"]
            if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
                raise ValueError(f"Invalid betas: {g['betas']}")
            if g["lr"] < 0.0:
                raise ValueError("Invalid lr (must be >= 0).")
            if g["eps"] <= 0.0:
                raise ValueError("Invalid eps (must be > 0).")
            if g["max_grad_norm"] is not None and g["max_grad_norm"] <= 0.0:
                raise ValueError("Invalid max_grad_norm (must be > 0 or None).")
            return g

        # Normalize input
        first = next(iter(params))
        # Need to iterate again, so coerce to list
        params_list = list(params) if not isinstance(params, list) else params

        if isinstance(first, dict) and "params" in first:
            # List of param groups
            for g in params_list:
                if "params" not in g:
                    raise ValueError("Param group dicts must have a 'params' key.")
                groups.append(_make_group(g["params"], {k: v for k, v in g.items() if k != "params"}))
        else:
            # Flat list → single group
            groups.append(_make_group(params_list, {}))

        return groups

    def _get_state_for(self, p: _ParamAdapter) -> Dict[str, Any]:
        pid = id(p.data)
        if pid not in self.state:
            self.state[pid] = self._new_state_like(p.data)
        return self.state[pid]

    def _set_state_for(self, p: _ParamAdapter, s: Dict[str, Any]) -> None:
        self.state[id(p.data)] = s

    def _new_state_like(self, arr: ArrayLike) -> Dict[str, Any]:
        st = {
            "m": np.zeros_like(arr, dtype=arr.dtype),
            "v": np.zeros_like(arr, dtype=arr.dtype),
        }
        # Optional AMSGrad
        st["vhat"] = np.zeros_like(arr, dtype=arr.dtype)
        return st

    def _enumerate_states(self) -> Dict[int, Dict[str, Any]]:
        return self.state

    def _pack_state(self, s: Dict[str, Any]) -> Dict[str, Any]:
        # Convert arrays to bytes plus shape/dtype for portability
        def pack(a: ArrayLike) -> Dict[str, Any]:
            return {"shape": a.shape, "dtype": str(a.dtype), "bytes": a.tobytes()}
        out = {"m": pack(s["m"]), "v": pack(s["v"]), "vhat": pack(s["vhat"])}
        return out

    def _unpack_state(self, s: Dict[str, Any], shape: Tuple[int, ...], dtype: np.dtype) -> Dict[str, Any]:
        # Prefer stored dtype/shape; fall back to current param metadata
        def unpack(packed: Dict[str, Any]) -> ArrayLike:
            shp = tuple(packed.get("shape", shape))
            dt = np.dtype(packed.get("dtype", str(dtype)))
            arr = np.frombuffer(packed["bytes"], dtype=dt).copy()
            return arr.reshape(shp)
        return {"m": unpack(s["m"]), "v": unpack(s["v"]), "vhat": unpack(s["vhat"])}

    def _global_clip(self) -> None:
        """Global norm clip across *all* grads if any group requests it.
        Uses the smallest (most strict) max_grad_norm among groups that set it.
        """
        # Determine smallest max_grad_norm among groups that enable clipping
        max_norms = [g["max_grad_norm"] for g in self.param_groups if g["max_grad_norm"] is not None]
        if not max_norms:
            return
        max_norm = float(min(max_norms))

        # Compute global norm
        total_sq = 0.0
        grads: List[Tuple[_ParamAdapter, ArrayLike]] = []
        for group in self.param_groups:
            for p_raw in group["params"]:
                p = _ParamAdapter(p_raw)
                if p.grad is not None:
                    g = p.grad
                    grads.append((p, g))
                    total_sq += float(np.sum(g.astype(np.float64) ** 2))
        global_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
        if global_norm == 0.0:
            return

        # Scale if needed
        if global_norm > max_norm:
            scale = max_norm / (global_norm + 1e-12)
            for p, g in grads:
                p.grad[...] = g * scale


# ------------------------- minimal example -------------------------
if __name__ == "__main__":
    # Tiny demo: optimize y = x^2 to zero with different per-parameter settings
    rng = np.random.default_rng(0)

    # Build "parameters"
    x1 = {"param": rng.normal(size=(3,)), "grad": None}
    x2 = {"param": rng.normal(size=(3,)), "grad": None}
    print(x1)
    print(x2)

    # Two param groups: x1 gets stronger weight decay
    opt = Adam(
        params=[
            {"params": [x1], "lr": 1e-2, "weight_decay": 0.1, "decouple_wd": True},
            {"params": [x2], "lr": 1e-2, "weight_decay": 0.0},
        ],
        # defaults=AdamDefaults(amsgrad=True, max_grad_norm=1.0),
        defaults=AdamDefaults(amsgrad=True),
    )

    for step in range(200):
        # Simple objective: f = ||x1||^2 + ||x2||^2
        x1["grad"] = 2.0 * x1["param"]
        x2["grad"] = 2.0 * x2["param"]

        opt.step()
        opt.zero_grad()

    print("x1:", x1["param"])
    print("x2:", x2["param"])