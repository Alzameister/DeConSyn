import pickle
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
import os

from category_encoders import OrdinalEncoder

from DeFeSyn.data.data_loader import DatasetLoader, ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET
from DeFeSyn.models.CTGAN.synthesizers.ctgan import CTGAN
from DeFeSyn.models.tab_ddpm import GaussianMultinomialDiffusion, MLPDiffusion, ResNetDiffusion
from DeFeSyn.models.tab_ddpm.lib import load_config
from DeFeSyn.models.tab_ddpm.lib.data import Transformations, prepare_fast_dataloader
from DeFeSyn.models.tab_ddpm.scripts.sample import sample
from DeFeSyn.models.tab_ddpm.scripts.utils_train import make_dataset
from DeFeSyn.models.tab_ddpm.trainer import Trainer, train
from DeFeSyn.io.serialization import encode_state_dict_pair_blob, decode_state_dict_pair_blob
from DeFeSyn.io.snapshots import snapshot_state_dict_pair
from DeFeSyn.io.io import get_repo_root, get_config_dir


class Model(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        device: str = "cpu"
    ):
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
            data,
            discrete_columns,
            epochs: int,
            data_transformer = None,
            verbose: bool = True,
            device: str = "cpu",
    ):
        super().__init__(data, device)
        self.discrete_columns = list(discrete_columns or [])
        self.epochs = int(epochs)
        self.verbose = bool(verbose)
        self.data_transformer = data_transformer

        self.model = CTGAN(epochs=self.epochs, verbose=self.verbose, cuda=self.use_cuda_flag)

        self.weights: dict[str, dict[str, torch.Tensor]] = {}
        self._weights_lock = threading.RLock()
        self._cpu_weights: dict[str, dict[str, torch.Tensor]] | None  = None
        self._loss_values = pd.DataFrame(columns=["Epoch", "Generator Loss", "Discriminator Loss"])

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
            data_transformer=self.data_transformer,
            train_data=self.data,
            discrete_columns=self.discrete_columns,
            gen_state_dict=self.weights.get("generator"),
            dis_state_dict=self.weights.get("discriminator"),
        )
        loss_values = getattr(self.model, "loss_values", None)
        if loss_values is not None:
            lv_pd = pd.DataFrame(loss_values)
            # Add to existing loss values
            self._loss_values = pd.concat([self._loss_values, lv_pd], ignore_index=True)
            self._loss_values["Epoch"] = self._loss_values.index + 1 * self.epochs
            print(self._loss_values)
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

    def fit_baseline(self):
        self.epochs = 300
        self.model = CTGAN(epochs=self.epochs, verbose=True, cuda=self.use_cuda_flag)
        self.model.fit(
            train_data=self.data,
            discrete_columns=self.discrete_columns,
            gen_state_dict=self.weights.get("generator"),
            dis_state_dict=self.weights.get("discriminator"),
            data_transformer=self.data_transformer
        )
        self._move_modules()
        self._refresh_cpu_snapshot()

        # Save model under runs/ctgan_baseline

        root = get_repo_root()
        path = root / "runs" / "ctgan_baseline"
        os.makedirs(path, exist_ok=True)
        model_path = path / "ctgan_adult_baseline.pkl"
        self.model.save(model_path)

class TabDDPMModel(Model):
    def __init__(
            self,
            data,
            discrete_columns,
            real_data_path: str,
            cat_encoder: OrdinalEncoder,
            num_encoder: Any,
            y_encoder: Any,
            config: dict,
            verbose: bool = True,
            device: str = "cpu",
            target: str = "income"

    ):
        super().__init__(data, device)
        self.discrete_columns = discrete_columns
        self.target = target
        self.verbose = verbose
        self.cat_encoder = cat_encoder
        self.num_encoder = num_encoder
        self.y_encoder = y_encoder
        self.config = config


        model_type = config['model_type']
        model_params = config['model_params']
        diffusion_params = config['diffusion_params']
        train_main = config['train']['main']
        T_dict = config['train']['T']
        num_numerical_features = config['num_numerical_features']


        real_data_path = os.path.normpath(real_data_path)

        T = Transformations(**T_dict)

        self.dataset = make_dataset(
            real_data_path,
            T,
            num_classes=model_params['num_classes'],
            is_y_cond=model_params['is_y_cond'],
            change_val=False,
            cat_encoder=self.cat_encoder,
            num_encoder=self.num_encoder,
            y_encoder=self.y_encoder
        )

        # Save the encoders
        num_encoder = self.dataset.num_transform
        cat_encoder = self.dataset.cat_transform
        y_encoder = self.dataset.y_transform

        root = get_repo_root()
        path = root / "runs" / "tabddpm"
        os.makedirs(path, exist_ok=True)
        num_path = path / "num_encoder.pkl"
        cat_path = path / "cat_encoder.pkl"
        y_path = path / "y_encoder.pkl"
        with open(num_path, "wb") as f:
            pickle.dump(num_encoder, f)
        with open(cat_path, "wb") as f:
            pickle.dump(cat_encoder, f)
        with open(y_path, "wb") as f:
            pickle.dump(y_encoder, f)

        K = np.array(self.dataset.get_category_sizes('train'))
        if len(K) == 0 or T_dict.get('cat_encoding') == 'one-hot':
            K = np.array([0])
        print(K)

        num_numerical_features = self.dataset.X_num['train'].shape[1] if self.dataset.X_num is not None else 0
        d_in = np.sum(K) + num_numerical_features
        model_params['d_in'] = d_in
        print(d_in)

        print(model_params)
        self.model = get_model(
            model_type,
            model_params,
            num_numerical_features,
            category_sizes=self.dataset.get_category_sizes('train')
        )
        self.model.to(device)

        train_loader = prepare_fast_dataloader(self.dataset, split='train', batch_size=train_main['batch_size'])
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=diffusion_params['gaussian_loss_type'],
            num_timesteps=diffusion_params['num_timesteps'],
            scheduler="cosine",
            device=device
        )
        self.diffusion.to(device)
        self.diffusion.train()

        self.trainer = Trainer(
            self.diffusion,
            train_loader,
            lr=train_main['lr'],
            weight_decay=train_main['weight_decay'],
            steps=train_main['steps'],
            device=device
        )

        self._weights_lock = threading.RLock()
        self._cpu_weights: dict[str, torch.Tensor] | None = None

        self._max_step = 0
        self._step_size = train_main['steps']

    def fit(self):
        self.trainer.run_loop()
        self.loss_values = self.trainer.loss_history

        if hasattr(self, "loss_values") and self.loss_values is not None and not self.loss_values.empty:
            self.loss_values["step"] = [(i + 1) * self._step_size for i in range(len(self.loss_values))]

        self._refresh_cpu_snapshot()

    def postprocess_sample(self, X_gen, y_gen):
        # Split into numerical and categorical features
        num_features = self.num_numerical_features
        X_num = X_gen[:, :num_features]
        X_cat = X_gen[:, num_features:]

        # Inverse transform numerical features
        X_num_inv = self.dataset.num_transform.inverse_transform(X_num)

        # Inverse transform categorical features
        X_cat_inv = self.dataset.cat_transform.inverse_transform(X_cat)

        y = y_gen['y']
        y_inv = self.dataset.y_transform.inverse_transform(y.reshape(-1, 1))
        y_inv = y_inv.squeeze(1)

        # Combine back if needed
        X_final = np.hstack([X_num_inv, X_cat_inv])
        return X_final, y_inv

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

    def _snapshot_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _refresh_cpu_snapshot(self) -> None:
        with self._weights_lock:
            self._cpu_weights = self._snapshot_cpu()

    def fit_baseline(self, parent_dir, real_data_path, config):
        self.dataset = train(
            parent_dir=parent_dir,
            real_data_path=real_data_path,
            **config['train']['main'],
            **config['diffusion_params'],
            model_type=config['model_type'],
            model_params=config['model_params'],
            T_dict=config['train']['T'],
            num_numerical_features=config['num_numerical_features'],
            device=self.device,
            change_val=False
        )


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
    loader = DatasetLoader("../../data/adult", ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET)
    full_train = loader.get_train()
    full_test = loader.get_test()
    data_dir = "../../runs/tabddpm/tabddpm_baseline/base2"
    real_data_path = "../../data/adult/npy"
    transformer = loader.get_data_transformer()
    encoder = loader.get_cat_oe()
    config_dir = get_config_dir()
    raw_config = config_dir / "config.toml"
    config = load_config(raw_config)
    root = get_repo_root()
    path = root / "runs" / "tabddpm" / "tabddpm_baseline"
    os.makedirs(path, exist_ok=True)
    model_path = path / "model.pt"

    model = TabDDPMModel(
        data=full_train,
        discrete_columns=discrete_cols_of(full_train),
        real_data_path=real_data_path,
        encoder=encoder,
        device="cpu",
        target=ADULT_TARGET,
        config=config
    )
    model.fit_baseline(data_dir, real_data_path, config)

    sample(
        parent_dir=data_dir,
        real_data_path=real_data_path,
        num_samples=2000,
        batch_size=2000,
        disbalance=config['sample'].get('disbalance', None),
        **config['diffusion_params'],
        model_path=model_path,
        model_type=config['model_type'],
        model_params=config['model_params'],
        T_dict=config['train']['T'],
        num_numerical_features=config['num_numerical_features'],
        device="cpu",
        seed=0,
        change_val=False
    )

    x_cat_train_p = path / 'X_cat_train.npy'
    x_num_train_p = path / 'X_num_train.npy'
    y_train_p = path / 'y_train.npy'

    x_cat_train = np.load(x_cat_train_p, allow_pickle=True)
    x_num_train = np.load(x_num_train_p, allow_pickle=True)
    y_train = np.load(y_train_p, allow_pickle=True)

    x_cat_train_pd = pd.DataFrame(x_cat_train)
    x_num_train_pd = pd.DataFrame(x_num_train)
    y_train_pd = pd.DataFrame(y_train)

    print(x_cat_train_pd)