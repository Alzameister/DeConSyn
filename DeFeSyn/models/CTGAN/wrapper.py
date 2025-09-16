import math
import threading
import torch, io, gzip, base64, hashlib

from DeFeSyn.models.CTGAN.synthesizers.ctgan import CTGAN


class CTGANModel:
    """
    CTGANModel is a wrapper for the CTGAN synthesizer from the ctgan library.
    """
    def __init__(self, full_data, data, discrete_columns, epochs, verbose=True,
                 device: str = "cpu"):
        """
        Initialize the CTGAN model.

        Args:
            data (pd.DataFrame): The training data.
            discrete_columns (list): List of discrete columns in the data.
            epochs (int): Number of training epochs.
        """
        self.full_data = full_data
        self.data = data
        self.discrete_columns = discrete_columns
        self.epochs = epochs
        self.verbose = verbose
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
            use_cuda_flag = True          # ctgan expects bool
        else:
            self.device = torch.device("cpu")
            use_cuda_flag = False
        self.model = CTGAN(
            epochs=self.epochs,
            verbose=self.verbose,
            cuda=use_cuda_flag
        )
        self.weights: dict = {}
        self._weights_lock = threading.RLock()
        # Last CPU snapshot (immutable tensors) for cheap reads
        self._cpu_weights: dict | None = None
        self._legacy_serialize = True  # keep existing format flag


    def _move_modules(self):
        # ctgan builds networks before/inside fit; guard attributes
        G = getattr(self.model, "_generator", None)
        D = getattr(self.model, "_discriminator", None)
        if G is not None:
            G.to(self.device)
        if D is not None:
            D.to(self.device)

    def _snapshot_cpu(self) -> dict:
        """
        Atomic (under lock) CPU snapshot of current model parameters.
        Tensors are detached and cloned to avoid later mutation.
        """
        G = self.model._generator
        D = self.model._discriminator
        gen_sd = {}
        dis_sd = {}
        for k, v in G.state_dict().items():
            gen_sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        for k, v in D.state_dict().items():
            dis_sd[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        return {"generator": gen_sd, "discriminator": dis_sd}

    def _refresh_cpu_snapshot(self):
        with self._weights_lock:
            self._cpu_weights = self._snapshot_cpu()

    def train(self):
        """
        Train the CTGAN model on the provided data.
        """
        self.model.fit(
            full_data=self.full_data,
            train_data=self.data,
            discrete_columns=self.discrete_columns,
            gen_state_dict=self.weights.get('generator', None),
            dis_state_dict=self.weights.get('discriminator', None)
        )
        self._move_modules()
        self._refresh_cpu_snapshot()

    def sample(self, num_samples, seed=42):
        """
        Generate synthetic samples from the trained CTGAN model.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            pd.DataFrame: Generated synthetic samples.
        """
        return self.model.sample(num_samples, random_state=seed)

    def get_weights(self):
        """
        Get the weights of the trained CTGAN model.

        Returns:
            dict: Weights of the CTGAN model.
        """
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            # Deep copy (clone tensors) so caller can mutate freely
            out = {"generator": {}, "discriminator": {}}
            for side in ("generator", "discriminator"):
                for k, v in self._cpu_weights[side].items():
                    if torch.is_tensor(v):
                        out[side][k] = v.clone()
                    else:
                        out[side][k] = v
            return out

    def load_weights(self, weights):
        """
        Load weights into the CTGAN model.

        Args:
            weights (dict): Weights to load into the model.
        """
        if not weights or "generator" not in weights or "discriminator" not in weights:
            return
        with self._weights_lock:
            self._move_modules()
            G = self.model._generator
            D = self.model._discriminator

            # Coerce dtypes/devices
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
            # Update snapshot (still inside lock to keep atomic view)
            self._cpu_weights = self._snapshot_cpu()

    def encode(self):
        """Return a JSON-serializable package containing the weights."""
        with self._weights_lock:
            if self._cpu_weights is None:
                return None
            snap = self._cpu_weights  # immutable tensors (cloned earlier)

        def _pack(obj: dict) -> bytes:
            with io.BytesIO() as buf:
                torch.save(obj, buf, _use_new_zipfile_serialization=not self._legacy_serialize)
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

    def decode(self, encoded):
        """Decode the state_dict from a JSON-serializable package."""
        gen_comp = base64.b64decode(encoded["generator"])
        dis_comp = base64.b64decode(encoded["discriminator"])
        checksum = encoded["checksum"]
        if hashlib.sha256(gen_comp + dis_comp).hexdigest() != checksum:
            raise ValueError("Checksum mismatch.")
        def _unpack(blob: bytes) -> dict:
            try:
                buf = io.BytesIO(gzip.decompress(blob))
                obj = torch.load(buf, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"decode failed: {e}")
            # Normalize dtypes (float64 -> float32)
            for k, v in list(obj.items()):
                if torch.is_tensor(v) and v.dtype == torch.float64:
                    obj[k] = v.float()
            return obj

        gen_sd = _unpack(gen_comp)
        dis_sd = _unpack(dis_comp)
        return {"generator": gen_sd, "discriminator": dis_sd}

    def decode_and_load(self, package: dict):
        decoded = self.decode(package)
        self.load_weights(decoded)

def _is_float_tensor(t: torch.Tensor) -> bool:
    return t.is_floating_point()

@torch.no_grad()
def state_dict_snapshot(module: torch.nn.Module, device: str = "cpu") -> dict[str, torch.Tensor]:
    """
    Frozen copy of float tensors (params + float buffers), on CPU.
    """
    snap = {}
    for k, v in module.state_dict().items():
        if isinstance(v, torch.Tensor) and _is_float_tensor(v):
            snap[k] = v.detach().to(device).clone()
    return snap

@torch.no_grad()
def l2_delta_between_snapshots(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """
    Efficient L2 norm of (b - a) without concatenating huge vectors.
    Assumes same keys & shapes.
    """
    s = 0.0
    for k, va in a.items():
        vb = b[k]
        diff = vb - va
        s += float((diff * diff).sum().item())
    return math.sqrt(s)

@torch.no_grad()
def l2_norm_snapshot(a: dict[str, torch.Tensor]) -> float:
    s = 0.0
    for v in a.values():
        s += float((v * v).sum().item())
    return math.sqrt(s)


@torch.no_grad()
def gan_snapshot(generator: torch.nn.Module, discriminator: torch.nn.Module, device="cpu"):
    g = state_dict_snapshot(generator, device)
    d = state_dict_snapshot(discriminator, device)
    return {("G", k): v for k, v in g.items()} | {("D", k): v for k, v in d.items()}

@torch.no_grad()
def get_gan_snapshot(ctgan_model: CTGANModel, device: str= "cpu"):
    """Return snapshot dict or None if G/D aren't ready yet."""
    try:
        G = getattr(ctgan_model, "_generator", None)
        D = getattr(ctgan_model, "_discriminator", None)
        if G is None or D is None:
            return None
        g = state_dict_snapshot(G, device)
        d = state_dict_snapshot(D, device)
        return {("G", k): v for k, v in g.items()} | {("D", k): v for k, v in d.items()}
    except Exception:
        return None