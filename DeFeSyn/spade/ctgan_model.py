import threading
from typing import Dict, Tuple, Optional, Any

import torch

from DeFeSyn.models.CTGAN.synthesizers.ctgan import CTGAN
from DeFeSyn.spade.serialization import decode_state_dict_pair, encode_state_dict_pair, encode_state_dict_pair_blob, \
    decode_state_dict_pair_blob
from DeFeSyn.spade.snapshots import snapshot_state_dict_pair

# {"generator": {...}, "discriminator": {...}}
Weights = Dict[str, Dict[str, torch.Tensor]]


def resolve_device(device: str) -> Tuple[torch.device, bool]:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device), True
    return torch.device("cpu"), False

def coerce_dtypes(src: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in src.items():
        if torch.is_tensor(v) and k in ref and v.dtype != ref[k].dtype:
            out[k] = v.to(ref[k].dtype)
        else:
            out[k] = v
    return out

class CTGANModel:
    """
    Thin wrapper around CTGAN that:
      - manages device / cuda flag
      - maintains an atomic CPU snapshot for cheap, thread-safe reads
      - (de)serializes weights via `serialization.py`
      - snapshots via `snapshots.py`
    """

    def __init__(
            self,
            full_data,
            data,
            discrete_columns,
            epochs: int,
            verbose: bool = True,
            device: str = "cpu",
    ):
        self.full_data = full_data
        self.data = data
        self.discrete_columns = list(discrete_columns or [])
        self.epochs = int(epochs)
        self.verbose = bool(verbose)

        self.device, use_cuda_flag = resolve_device(device)
        self.model = CTGAN(epochs=self.epochs, verbose=self.verbose, cuda=use_cuda_flag)

        self.weights: Weights = {}
        self._weights_lock = threading.RLock()
        self._cpu_weights: Optional[Weights] = None

    # ----------------------------
    # Utilities
    # ----------------------------
    def _move_modules(self) -> None:
        G = getattr(self.model, "_generator", None)
        D = getattr(self.model, "_discriminator", None)
        if G is not None:
            G.to(self.device)
        if D is not None:
            D.to(self.device)

    def _refresh_cpu_snapshot(self) -> None:
        with self._weights_lock:
            self._cpu_weights = snapshot_state_dict_pair(self.model._generator, self.model._discriminator)

    # ----------------------------
    # Public
    # ----------------------------
    def fit(self) -> None:
        """Train CTGAN; refresh GPU placement and CPU snapshot afterwards."""
        self.model.fit(
            full_data=self.full_data,
            train_data=self.data,
            discrete_columns=self.discrete_columns,
            gen_state_dict=self.weights.get("generator"),
            dis_state_dict=self.weights.get("discriminator"),
        )
        self._move_modules()
        self._refresh_cpu_snapshot()

    def sample(self, num_samples: int, seed: int = 42):
        return self.model.sample(num_samples, random_state=seed)

    def is_trained(self) -> bool:
        return self._cpu_weights is not None

    def get_weights(self) -> Optional[Weights]:
        """Deep-copied CPU weights (safe for caller mutation) or None if not ready."""
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            out: Weights = {"generator": {}, "discriminator": {}}
            for side in ("generator", "discriminator"):
                for k, v in self._cpu_weights[side].items():
                    out[side][k] = v.clone() if torch.is_tensor(v) else v
            return out

    def set_weights(self, weights: Weights) -> None:
        """Load weights into CTGAN and refresh snapshot (atomic)."""
        if not weights or "generator" not in weights or "discriminator" not in weights:
            return
        with self._weights_lock:
            self._move_modules()
            G = self.model._generator
            D = self.model._discriminator

            g_ref = G.state_dict()
            d_ref = D.state_dict()
            g_norm = coerce_dtypes(weights["generator"], g_ref)
            d_norm = coerce_dtypes(weights["discriminator"], d_ref)

            G.load_state_dict(g_norm, strict=True)
            D.load_state_dict(d_norm, strict=True)

            self._cpu_weights = snapshot_state_dict_pair(G, D)

    def encode(self) -> Optional[Dict[str, Any]]:
        """JSON-serializable dict with b64(gzip(torch.save(...))) + checksum, or None if not trained."""
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            return encode_state_dict_pair_blob(self._cpu_weights["generator"], self._cpu_weights["discriminator"], as_ascii=True)
            # return encode_state_dict_pair(self._cpu_weights["generator"], self._cpu_weights["discriminator"])

    def decode(self, package: str) -> Weights:
        """Inverse of `encode`."""
        g, d = decode_state_dict_pair_blob(package)
        # g, d = decode_state_dict_pair(package)
        return {"generator": g, "discriminator": d}

    def decode_and_load(self, package: str) -> None:
        self.set_weights(self.decode(package))