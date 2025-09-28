import base64
import gzip
import hashlib
import io
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from DeFeSyn.models.CTGAN.synthesizers.ctgan import CTGAN
from DeFeSyn.models.tab_ddpm import GaussianMultinomialDiffusion, MLPDiffusion, ResNetDiffusion
from DeFeSyn.models.tab_ddpm.trainer import Trainer
from DeFeSyn.spade.serialization import encode_state_dict_pair_blob, decode_state_dict_pair_blob
from DeFeSyn.spade.snapshots import snapshot_state_dict_pair


# from DeFeSyn.models.scripts.train import train


class Model(ABC):
    def __init__(
        self,
        full_data: pd.DataFrame,
        data: pd.DataFrame,
        device: str = "cpu"
    ):
        self.full_data = full_data
        self.data = data
        self.device, self.use_cuda_flag = self._resolve_device(device)

    def _resolve_device(self, device: str) -> tuple[torch.device, bool]:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            return torch.device(device), True
        return torch.device("cpu"), False

    def _coerce_dtypes(self, src: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in src.items():
            if torch.is_tensor(v) and k in ref and v.dtype != ref[k].dtype:
                out[k] = v.to(ref[k].dtype)
            else:
                out[k] = v
        return out

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def sample(self, num_samples: int, seed: int = 42):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    @abstractmethod
    def encode(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def decode(self, encoded: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def decode_and_load(self, package: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_loss_values(self):
        pass

    @abstractmethod
    def clear_loss_values(self):
        pass

class CTGANModel(Model):
    def __init__(
            self,
            full_data,
            data,
            discrete_columns,
            epochs: int,
            verbose: bool = True,
            device: str = "cpu",
    ):
        super().__init__(full_data, data, device)
        self.discrete_columns = list(discrete_columns or [])
        self.epochs = int(epochs)
        self.verbose = bool(verbose)

        self.model = CTGAN(epochs=self.epochs, verbose=self.verbose, cuda=self.use_cuda_flag)

        self.weights: dict[str, dict[str, torch.Tensor]] = {}
        self._weights_lock = threading.RLock()
        self._cpu_weights: dict[str, dict[str, torch.Tensor]] | None  = None

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

    def _snapshot_cpu(self) -> dict[str, dict[str, torch.Tensor]]:
        """
        CPU snapshot of current parameters.
        Tensors are detached+cloned to remain immutable for readers.
        """
        G = self.model._generator
        D = self.model._discriminator
        gen_sd: dict[str, Any] = {}
        dis_sd: dict[str, Any] = {}
        for k, v in G.state_dict().items():
            gen_sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        for k, v in D.state_dict().items():
            dis_sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        return {"generator": gen_sd, "discriminator": dis_sd}

    def _refresh_cpu_snapshot(self) -> None:
        with self._weights_lock:
            self._cpu_weights = snapshot_state_dict_pair(self.model._generator, self.model._discriminator)

    # ----------------------------
    # Public
    # ----------------------------
    def fit(self) -> None:
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

    def is_trained(self) -> bool:
        return self._cpu_weights is not None

    def get_weights(self) -> dict[str, dict[str, torch.Tensor]] | None:
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
            out:  dict[str, dict[str, torch.Tensor]] = {"generator": {}, "discriminator": {}}
            for side in ("generator", "discriminator"):
                for k, v in self._cpu_weights[side].items():
                    out[side][k] = v.clone() if torch.is_tensor(v) else v
            return out

    def set_weights(self, weights: dict[str, dict[str, torch.Tensor]]) -> None:
        """Load a state dict into CTGAN and refresh the CPU snapshot (atomic)."""
        if not weights or "generator" not in weights or "discriminator" not in weights:
            return
        with self._weights_lock:
            self._move_modules()
            G = self.model._generator
            D = self.model._discriminator

            g_ref = G.state_dict()
            d_ref = D.state_dict()
            g_norm = self._coerce_dtypes(weights["generator"], g_ref)
            d_norm = self._coerce_dtypes(weights["discriminator"], d_ref)

            G.load_state_dict(g_norm, strict=True)
            D.load_state_dict(d_norm, strict=True)

            self._cpu_weights = snapshot_state_dict_pair(G, D)

    def encode(self) -> dict[str, Any] | None:
        """JSON-serializable dict with b64(gzip(torch.save(...))) + checksum, or None if not trained."""
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            return encode_state_dict_pair_blob(self._cpu_weights["generator"], self._cpu_weights["discriminator"],
                                               as_ascii=True)

    def decode(self, package: str) -> dict[str, dict[str, torch.Tensor]]:
        """Inverse of `encode`."""
        g, d = decode_state_dict_pair_blob(package)
        return {"generator": g, "discriminator": d}

    def decode_and_load(self, package: str) -> None:
        self.set_weights(self.decode(package))

    def get_loss_values(self):
        return getattr(self.model, "loss_values", None)

    def clear_loss_values(self):
        if hasattr(self.model, "loss_values"):
            self.model.loss_values = None

class TabDDPMModel(Model):
    def __init__(
            self,
            full_data,
            data,
            discrete_columns,
            epochs: int,
            verbose: bool = True,
            device: str = "cpu",
    ):
        super().__init__(full_data, data, device)

        self.epochs = int(epochs)
        self.verbose = bool(verbose)
        self.discrete_columns = discrete_columns
        self.num_classes = self._get_num_classes(full_data, discrete_columns)
        self.num_numerical_features = self.full_data.shape[1] - len(self.discrete_columns)
        # TODO: For now adult only config copied, make it configurable by input
        self.lr = 0.001809824563637657
        self.weight_decay = 0.0
        self.steps = 30000
        self.batch_size = 4096
        rtdl_params = {
            "d_layers": [1024, 512],
            "dropout":0.0
        }
        d_in = np.sum(self.num_classes) + self.num_numerical_features

        self.model = get_model(
            model_name="mlp",
            model_params={
                "num_classes": 2,
                "is_y_cond": True,
                "d_in": d_in,
                "rtdl_params": rtdl_params
            },
            n_num_features=self.num_numerical_features,
            category_sizes=self.num_classes
        )
        self.model.to(self.device)

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type="mse",
            num_timesteps=1000,
            scheduler="cosine",
            device=self.device
        )
        self.diffusion.to(self.device)

        self._weights_lock = threading.RLock()
        self._cpu_weights: dict[str, torch.Tensor] | None = None

    def fit(self):
        self.diffusion.train()
        train_iter = self._get_train_iter()
        trainer = Trainer(
            diffusion=self.diffusion,
            train_iter=train_iter,
            lr=self.lr,
            weight_decay=self.weight_decay,
            steps=self.steps,
            device=self.device
        )
        trainer.run_loop()
        self.loss_history = trainer.loss_history
        self._refresh_cpu_snapshot()

    def sample(self, num_samples: int, seed: int = 42):
        pass

    def get_weights(self):
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            return {k: v.clone() if torch.is_tensor(v) else v for k, v in self._cpu_weights.items()}

    def set_weights(self, weights):
        if not weights:
            return
        with self._weights_lock:
            ref = self.model.state_dict()
            for k, v in list(weights.items()):
                if torch.is_tensor(v) and k in ref and v.dtype != ref[k].dtype:
                    weights[k] = v.to(ref[k].dtype)
            self.model.load_state_dict(weights, strict=True)
            self._refresh_cpu_snapshot()

    def encode(self) -> dict[str, Any]:
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            with io.BytesIO() as buf:
                torch.save(self._cpu_weights, buf)
                raw = buf.getvalue()
            comp = gzip.compress(raw)
            checksum = hashlib.sha256(comp).hexdigest()
            return {
                "model": base64.b64encode(comp).decode("utf-8"),
                "checksum": checksum,
                "bytes": len(comp),
            }

    def decode(self, encoded: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
        comp = base64.b64decode(encoded["model"])
        checksum = encoded["checksum"]
        if hashlib.sha256(comp).hexdigest() != checksum:
            raise ValueError("Checksum mismatch.")
        buf = io.BytesIO(gzip.decompress(comp))
        obj = torch.load(buf, map_location="cpu")
        for k, v in list(obj.items()):
            if torch.is_tensor(v) and v.dtype == torch.float64:
                obj[k] = v.float()
        return obj

    def decode_and_load(self, package: dict[str, Any]) -> None:
        decoded = self.decode(package)
        self.set_weights(decoded)

    def get_loss_values(self):
        return getattr(self, "loss_values", None)

    def clear_loss_values(self):
        if hasattr(self, "loss_values"):
            self.loss_values = None

    def _get_num_classes(self, full_df: pd.DataFrame, discrete_columns: list[str]) -> np.ndarray:
        if not discrete_columns:
            return np.array([0], dtype=np.int64)
        return np.array([full_df[c].astype('category').cat.categories.size
                         for c in discrete_columns], dtype=np.int64)

    def _get_train_iter(self, shuffle=True):
        dataset = TabularDataset(self.data, self.discrete_columns)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
        return iter(loader)

    def _snapshot_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _refresh_cpu_snapshot(self) -> None:
        with self._weights_lock:
            self._cpu_weights = self._snapshot_cpu()

class TabularDataset(Dataset):
    def __init__(self, data: pd.DataFrame, discrete_columns: list[str]):
        self.data = data.reset_index(drop=True)
        self.discrete_columns = discrete_columns
        self.numerical_columns = [c for c in data.columns if c not in discrete_columns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x_num = torch.tensor(row[self.numerical_columns].values, dtype=torch.float32)
        x_cat = torch.tensor(row[self.discrete_columns].values, dtype=torch.long)
        return x_num, x_cat

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
):
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model
