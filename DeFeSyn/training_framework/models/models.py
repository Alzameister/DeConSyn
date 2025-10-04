import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Dataset

from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.training_framework.models.CTGAN.synthesizers.ctgan import CTGAN
from DeFeSyn.training_framework.models.tab_ddpm import GaussianMultinomialDiffusion, MLPDiffusion, ResNetDiffusion
from DeFeSyn.training_framework.models.tab_ddpm.trainer import Trainer
from DeFeSyn.io.serialization import encode_state_dict_pair_blob, decode_state_dict_pair_blob
from DeFeSyn.io.snapshots import snapshot_state_dict_pair
from DeFeSyn.io.io import get_repo_root


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
            target: str = "income"
    ):
        super().__init__(full_data, data, device)

        self.epochs = int(epochs)
        self.verbose = bool(verbose)

        self.discrete_columns = [c for c in discrete_columns if c != target]
        self.numerical_columns = [c for c in full_data.columns if c not in discrete_columns and c != target]
        self.target = target
        self._align_data()

        self.column_names = self.full_data.columns.tolist()
        self.num_classes = self._get_num_classes()
        self.y_classes = self._get_y_classes()
        self.y_dist = self._get_y_dist()
        self.num_numerical_features = self.full_data.shape[1] - len(self.discrete_columns) - 1
        self.category_mappings = {
            col: self.full_data[col].cat.categories
            for col in self.discrete_columns + ([self.target] if self.target else [])
            if pd.api.types.is_categorical_dtype(self.full_data[col])
        }

        # TODO: For now adult only config copied, make it configurable by input
        self.lr = 0.001809824563637657
        self.weight_decay = 0.0
        # TODO: Adjust stepszie, now used 100 --> 300 Rounds * 100 = 30'000 (as baseline) + 200 rounds extra for safety
        self.steps = 100
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
            y_dist=self.y_dist,
            column_names=self.column_names,
            category_mappings=self.category_mappings,
            denoise_fn=self.model,
            gaussian_loss_type="mse",
            num_timesteps=1000,
            scheduler="cosine",
            device=self.device
        )
        self.diffusion.to(self.device)
        self.loss_values = None

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
        self.loss_values = trainer.loss_history
        self._refresh_cpu_snapshot()

    def sample(self, num_samples: int, seed: int = 42):
        if not self.y_dist:
            self.y_dist = self._get_y_dist()
        torch.manual_seed(seed)
        self.diffusion.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(num_samples, self.y_dist)
            return samples

    def is_trained(self) -> bool:
        return self._cpu_weights is not None

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
            norm = self._coerce_dtypes(weights, ref)
            self.model.load_state_dict(norm, strict=True)
            self._refresh_cpu_snapshot()

    def encode(self) -> dict[str, Any]:
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            # Only one state dict (no generator/discriminator split)
            return encode_state_dict_pair_blob(self._cpu_weights, {}, as_ascii=True)

    def decode(self, encoded: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
        g, _ = decode_state_dict_pair_blob(encoded)
        return g

    def decode_and_load(self, package: dict[str, Any]) -> None:
        self.set_weights(self.decode(package))

    def get_loss_values(self):
        return getattr(self, "loss_values", None)

    def clear_loss_values(self):
        if hasattr(self, "loss_values"):
            self.loss_values = None

    def _get_num_classes(self) -> np.ndarray:
        if not self.discrete_columns:
            return np.array([0], dtype=np.int64)

        num_classes = np.array([self.full_data[c].nunique() for c in self.discrete_columns], dtype=np.int64)
        return num_classes

    def _get_y_classes(self) -> int:
        if not self.target:
            return 0
        num_classes = self.full_data[self.target].nunique()
        return num_classes

    def _get_y_dist(self) -> np.ndarray | None:
        if not self.target:
            return None
        y_counts = self.full_data[self.target].value_counts().sort_index()
        y_dist = y_counts / y_counts.sum()
        return torch.tensor(y_dist.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device)

    def _get_train_iter(self, shuffle=True):
        dataset = TabularDataset(self.data, self.discrete_columns, self.target)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
        return iter(loader)

    def _snapshot_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _refresh_cpu_snapshot(self) -> None:
        with self._weights_lock:
            self._cpu_weights = self._snapshot_cpu()

    def _align_data(self):
        ordered_columns = self.numerical_columns + self.discrete_columns + [self.target]
        self.full_data = self.full_data[ordered_columns]
        self.data = self.data[ordered_columns]

    def fit_baseline(self):
        self.steps = 30_000

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            y_dist=self.y_dist,
            column_names=self.column_names,
            category_mappings=self.category_mappings,
            denoise_fn=self.model,
            gaussian_loss_type="mse",
            num_timesteps=1000,
            scheduler="cosine",
            device=self.device
        )
        self.diffusion.to(self.device)
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
        self.loss_values = trainer.loss_history
        self._refresh_cpu_snapshot()

        # Save model under runs/tabddpm_baseline

        root = get_repo_root()
        path = root / "runs" / "tabddpm" / "tabddpm_baseline"
        os.makedirs(path, exist_ok=True)
        model_path = path / "tabddpm_adult_baseline.pkl"
        torch.save(self.diffusion, model_path)


class TabularDataset(Dataset):
    def __init__(self, data: pd.DataFrame, discrete_columns: list[str], target: str):
        self.data = data.reset_index(drop=True)
        self.target = target
        self.discrete_columns = [c for c in discrete_columns if c != target]
        self.numerical_columns = [c for c in data.columns if c not in discrete_columns and c != target]

        # Convert to numeric codes
        for col in self.discrete_columns:
            self.data[col] = self.data[col].cat.codes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x_num = row[self.numerical_columns].to_numpy(dtype=np.float32)
        x_cat = row[self.discrete_columns].to_numpy(dtype=np.int64)

        x = np.concatenate([x_num, x_cat])
        y = row[self.target]
        if pd.api.types.is_categorical_dtype(self.data[self.target]):
            y = self.data[self.target].cat.codes.iloc[idx]
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

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

def discrete_cols_of(df):
    return [c for c in df.columns if getattr(df[c].dtype, "name", "") == "category"]

if __name__ == '__main__':
    adult = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
    manifest = "manifest.yaml"
    loader = DatasetLoader(f"{adult}/{manifest}")
    full_train = loader.get_train()
    full_test = loader.get_test()
    discrete_columns = discrete_cols_of(full_train)

    model = TabDDPMModel(
        full_data=full_train,
        data=full_train,
        discrete_columns=discrete_columns,
        epochs=300,
        verbose=True,
        device="cpu",
        target="income"
    )
    model.fit_baseline()