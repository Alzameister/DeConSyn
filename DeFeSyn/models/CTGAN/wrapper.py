import math
from pathlib import Path

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


    def _move_modules(self):
        # ctgan builds networks before/inside fit; guard attributes
        G = getattr(self.model, "_generator", None)
        D = getattr(self.model, "_discriminator", None)
        if G is not None:
            G.to(self.device)
        if D is not None:
            D.to(self.device)

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
        self.weights = self.get_weights()

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
        # Always return to CPU for consensus and communication
        G = self.model._generator
        D = self.model._discriminator
        return {
            'generator': {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                          for k, v in G.state_dict().items()},
            'discriminator': {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in D.state_dict().items()}
        }

    def load_weights(self, weights):
        """
        Load weights into the CTGAN model.

        Args:
            weights (dict): Weights to load into the model.
        """
        if "generator" not in weights or "discriminator" not in weights:
            return
        self._move_modules()  # ensure modules on target device (GPU if configured)
        self.model._generator.load_state_dict(weights["generator"])
        self.model._discriminator.load_state_dict(weights["discriminator"])
        self.weights = self.get_weights()

    def encode(self):
        """Return a JSON-serializable package containing the weights."""
        try:
            generator = self.weights['generator']
            discriminator = self.weights['discriminator']
        except KeyError:
            return None
        cooked = {
            'generator': {},
            'discriminator': {}
        }
        for k, v in generator.items():
            if torch.is_tensor(v):
                t = v.detach().cpu()
                if torch.is_floating_point(t):
                    t = t.to(torch.float64)
                cooked['generator'][k] = t
            else:
                cooked['generator'][k] = v
        for k, v in discriminator.items():
            if torch.is_tensor(v):
                t = v.detach().cpu()
                if torch.is_floating_point(t):
                    t = t.to(torch.float64)
                cooked['discriminator'][k] = t
            else:
                cooked['discriminator'][k] = v

        gen_buffer = io.BytesIO()
        torch.save(cooked['generator'], gen_buffer)
        gen_raw = gen_buffer.getvalue()
        gen_compressed = gzip.compress(gen_raw)
        gen_encoded = base64.b64encode(gen_compressed).decode('utf-8')

        dis_buffer = io.BytesIO()
        torch.save(cooked['discriminator'], dis_buffer)
        dis_raw = dis_buffer.getvalue()
        dis_compressed = gzip.compress(dis_raw)
        dis_encoded = base64.b64encode(dis_compressed).decode('utf-8')

        checksum = hashlib.sha256(gen_compressed + dis_compressed).hexdigest()
        return {
            'generator': gen_encoded,
            'discriminator': dis_encoded,
            'checksum': checksum
        }

    def decode(self, encoded_state_dict):
        """Decode the state_dict from a JSON-serializable package."""
        gen_encoded = encoded_state_dict['generator']
        dis_encoded = encoded_state_dict['discriminator']
        checksum = encoded_state_dict['checksum']

        gen_compressed = base64.b64decode(gen_encoded)
        dis_compressed = base64.b64decode(dis_encoded)

        if hashlib.sha256(gen_compressed + dis_compressed).hexdigest() != checksum:
            raise ValueError("Checksum mismatch. The weights may be corrupted.")

        gen_buffer = io.BytesIO(gzip.decompress(gen_compressed))
        dis_buffer = io.BytesIO(gzip.decompress(dis_compressed))
        generator = torch.load(gen_buffer, map_location='cpu')
        discriminator = torch.load(dis_buffer, map_location='cpu')
        return {
            'generator': generator,
            'discriminator': discriminator
        }


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