import threading
from typing import Dict, Optional, Any

import torch, io, gzip, base64, hashlib

from DeFeSyn.training_framework.models.CTGAN.synthesizers.ctgan import CTGAN


class CTGANModel:
    """
    Thin wrapper around ctgan.CTGAN that:
      - manages device / cuda flag
      - maintains an atomic CPU snapshot for cheap, thread-safe reads
      - (de)serializes weights with gzip+base64 and a checksum
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
        """
        Args:
            full_data: Global/full dataset (optional, used by CTGAN.fit if provided).
            data: Partition used to train this node.
            discrete_columns: List[str] of discrete column names.
            epochs: Number of training epochs.
            verbose: Verbose training logs from CTGAN.
            device: "cpu" or "cuda[:index]".
        """
        self.full_data = full_data
        self.data = data
        self.discrete_columns = list(discrete_columns or [])
        self.epochs = int(epochs)
        self.verbose = bool(verbose)

        if str(device).startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
            use_cuda_flag = True
        else:
            self.device = torch.device("cpu")
            use_cuda_flag = False

        self.model = CTGAN(
            epochs=self.epochs,
            verbose=self.verbose,
            cuda=use_cuda_flag,
        )

        # Optional warm-start state provided by caller
        self.weights: Dict[str, Dict[str, torch.Tensor]] = {}

        # Concurrency: atomic CPU snapshots
        self._weights_lock = threading.RLock()
        self._cpu_weights: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    # ----------------------------
    # Utilities
    # ----------------------------
    def _move_modules(self) -> None:
        """Ensure CTGAN's internal modules live on the configured device."""
        G = getattr(self.model, "_generator", None)
        D = getattr(self.model, "_discriminator", None)
        if G is not None:
            G.to(self.device)
        if D is not None:
            D.to(self.device)

    def _snapshot_cpu(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        CPU snapshot of current parameters.
        Tensors are detached+cloned to remain immutable for readers.
        """
        G = self.model._generator
        D = self.model._discriminator
        gen_sd: Dict[str, Any] = {}
        dis_sd: Dict[str, Any] = {}
        for k, v in G.state_dict().items():
            gen_sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        for k, v in D.state_dict().items():
            dis_sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        return {"generator": gen_sd, "discriminator": dis_sd}

    def _refresh_cpu_snapshot(self) -> None:
        with self._weights_lock:
            self._cpu_weights = self._snapshot_cpu()

    # ----------------------------
    # Public
    # ----------------------------
    def train(self) -> None:
        """Fit CTGAN; refresh GPU placement and CPU snapshot afterwards."""
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
        """Generate synthetic samples from the trained CTGAN model."""
        return self.model.sample(num_samples, random_state=seed)

    def get_weights(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """
        Return a deep-copied CPU state dict (safe for caller mutation) or None if not ready.
        {
          "generator": {name: tensor, ...},
          "discriminator": {name: tensor, ...}
        }
        """
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            out = {"generator": {}, "discriminator": {}}
            for side in ("generator", "discriminator"):
                for k, v in self._cpu_weights[side].items():
                    out[side][k] = v.clone() if torch.is_tensor(v) else v
            return out

    def set_weights(self, weights: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Load a state dict into CTGAN and refresh the CPU snapshot (atomic)."""
        if not weights or "generator" not in weights or "discriminator" not in weights:
            return

        with self._weights_lock:
            self._move_modules()
            G = self.model._generator
            D = self.model._discriminator

            refG = G.state_dict()
            for k, v in list(weights["generator"].items()):
                if torch.is_tensor(v) and k in refG and v.dtype != refG[k].dtype:
                    weights["generator"][k] = v.to(refG[k].dtype)

            refD = D.state_dict()
            for k, v in list(weights["discriminator"].items()):
                if torch.is_tensor(v) and k in refD and v.dtype != refD[k].dtype:
                    weights["discriminator"][k] = v.to(refD[k].dtype)

            G.load_state_dict(weights["generator"], strict=True)
            D.load_state_dict(weights["discriminator"], strict=True)

            self._cpu_weights = self._snapshot_cpu()

    def encode(self) -> Optional[Dict[str, Any]]:
        """
        Return a JSON-serializable package:
        {
          "generator": b64(gzip(torch.save(state_dict))),
          "discriminator": b64(gzip(torch.save(state_dict))),
          "checksum": sha256(gen_blob + dis_blob),
          "gen_bytes": int,
          "dis_bytes": int
        }
        """
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            snap = self._cpu_weights

        def _pack(obj: dict) -> bytes:
            with io.BytesIO() as buf:
                torch.save(obj, buf)
                raw = buf.getvalue()
            return gzip.compress(raw)

        gen_comp = _pack(snap["generator"])
        dis_comp = _pack(snap["discriminator"])
        checksum = hashlib.sha256(gen_comp + dis_comp).hexdigest()
        return {
            "generator": base64.b64encode(gen_comp).decode("utf-8"),
            "discriminator": base64.b64encode(dis_comp).decode("utf-8"),
            "checksum": checksum,
            "gen_bytes": len(gen_comp),
            "dis_bytes": len(dis_comp),
        }

    def decode(self, encoded: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Decode the state_dict package produced by `encode()`."""
        gen_comp = base64.b64decode(encoded["generator"])
        dis_comp = base64.b64decode(encoded["discriminator"])
        checksum = encoded["checksum"]
        if hashlib.sha256(gen_comp + dis_comp).hexdigest() != checksum:
            raise ValueError("Checksum mismatch.")

        def _unpack(blob: bytes) -> Dict[str, torch.Tensor]:
            try:
                buf = io.BytesIO(gzip.decompress(blob))
                obj = torch.load(buf, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"decode failed: {e}")
            for k, v in list(obj.items()):
                if torch.is_tensor(v) and v.dtype == torch.float64:
                    obj[k] = v.float()
            return obj

        gen_sd = _unpack(gen_comp)
        dis_sd = _unpack(dis_comp)
        return {"generator": gen_sd, "discriminator": dis_sd}

    def decode_and_load(self, package: Dict[str, Any]) -> None:
        """Convenience: decode a package and immediately load it."""
        decoded = self.decode(package)
        self.set_weights(decoded)