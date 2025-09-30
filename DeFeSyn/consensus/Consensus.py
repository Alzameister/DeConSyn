from copy import deepcopy
import numpy as np
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _is_tensor(x):
    return _HAS_TORCH and isinstance(x, torch.Tensor)


class Consensus:
    """
    Asynchronous consensus with dynamic epsilon (learning factor).
    Works elementwise over dicts of numpy arrays / torch tensors.
    """

    def __init__(self, model_type="CTGAN", alpha: float = 0.5, snapshot_on_device: bool = True):
        self.model_type = model_type
        self.alpha = float(alpha)
        self.degree = 0
        self.eps = 1.0
        self.prev_eps = 1.0
        self.x0 = None                       # snapshot of x_i at window start
        self.snapshot_on_device = snapshot_on_device  # set False to keep x0 on CPU

    def set_degree(self, degree: int):
        if self.degree != 0:
            return
        self.degree = max(1, int(degree))
        self.eps = self.alpha / self.degree
        self.prev_eps = self.eps

    def _clone_like(self, x):
        # Snapshot helper: clone/detach torch tensors; copy numpy arrays
        if _is_tensor(x):
            y = x.detach().clone()
            if not self.snapshot_on_device:
                y = y.to("cpu")
            return y
        elif isinstance(x, np.ndarray):
            return x.copy()
        else:
            raise TypeError(f"Unsupported leaf type: {type(x)}")

    def _snapshot(self, x_dict):
        # Shallow structure copy + per-leaf clone/copy
        if not isinstance(x_dict, dict):
            raise TypeError(f"Expected dict, got {type(x_dict)}")
        if x_dict and all(isinstance(v, dict) for v in x_dict.values()): # CTGAN
            return {
                k: {p: self._clone_like(vv) for p, vv in v.items()}
                for k, v in x_dict.items()
            }
        else: # Tabddpm
            return {k: self._clone_like(v) for k, v in x_dict.items()}

    def start_consensus_window(self, x_i: dict):
        # Take a value snapshot for correction term; avoid deepcopy on GPU tensors
        self.x0 = self._snapshot(x_i)
        self.prev_eps = self.eps

    def _blend_inplace(self, A, B, A0, eps: float, corr: float):
        """
        In-place update for both nested (CTGAN) and flat (TabDDPM) dicts.
        A[k] = A[k]*(1 - eps - corr) + eps*B[k] + corr*A0[k]
        """
        if not isinstance(A, dict) or not isinstance(B, dict) or not isinstance(A0, dict):
            raise TypeError("All arguments must be dicts")

        # Nested dict (CTGAN): recurse per block
        if A and all(isinstance(v, dict) for v in A.values()):
            for k in A:
                self._blend_inplace(A[k], B[k], A0[k], eps, corr)
            return

        # Flat dict (TabDDPM): blend tensors/arrays
        if _HAS_TORCH:
            torch_no_grad = torch.no_grad
        else:
            from contextlib import contextmanager
            @contextmanager
            def torch_no_grad():
                yield
        with torch_no_grad():
            for k, a in A.items():
                b = B[k]
                a0 = A0[k]
                if _is_tensor(a):
                    if _is_tensor(b):
                        if b.device != a.device: b = b.to(a.device, non_blocking=True)
                        if b.dtype != a.dtype:  b = b.to(a.dtype)
                    if _is_tensor(a0):
                        if a0.device != a.device: a0 = a0.to(a.device, non_blocking=True)
                        if a0.dtype != a.dtype:  a0 = a0.to(a.dtype)
                    a.mul_(1.0 - eps - corr).add_(eps, b).add_(corr, a0)
                else:
                    a *= (1.0 - eps - corr)
                    a += eps * b
                    a += corr * a0

    def step_with_neighbor(self, x_i: dict, x_j: dict, eps_j: float) -> dict:
        if self.x0 is None:
            self.x0 = self._snapshot(x_i)
            self.prev_eps = self.eps

        new_eps = min(self.eps, float(eps_j))
        corr = 0.0 if self.prev_eps <= 0 else (1.0 - (new_eps / self.prev_eps))

        # CTGAN
        if x_i and all(isinstance(v, dict) for v in x_i.values()):
            self._blend_inplace(x_i['generator'], x_j['generator'], self.x0['generator'], new_eps, corr)
            self._blend_inplace(x_i['discriminator'], x_j['discriminator'], self.x0['discriminator'], new_eps, corr)
        # TabDDPM
        else:
            self._blend_inplace(x_i, x_j, self.x0, new_eps, corr)

        self.prev_eps = self.eps
        self.eps = new_eps
        return x_i

    def end_consensus_window(self):
        self.x0 = None

    def get_eps(self) -> float:
        return float(self.eps)
