import copy
import os
from functools import lru_cache
from pathlib import Path
import re

import torch

from DeFeSyn.models import CTGAN

REPO_MARKERS = {".git", "pyproject.toml", "poetry.lock", "setup.cfg", "setup.py"}
torch.serialization.add_safe_globals([CTGAN])

@lru_cache(maxsize=1)
def get_repo_root(start: str | Path | None = None) -> Path:
    """
    Return absolute path to repo root.
    Order: ENV override -> walk up from 'start' (or CWD) -> CWD fallback.
    """
    env = os.getenv("DEFESYN_REPO_ROOT")
    if env:
        return Path(env).resolve()

    p = Path(start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if any((parent / m).exists() for m in REPO_MARKERS):
            return parent
    return Path.cwd().resolve()

def get_config_dir(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else get_repo_root()
    return (root / "DeFeSyn" / "models" / "tab_ddpm" / "configs").resolve()

def runs_dir(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else get_repo_root()
    return (root / "runs").resolve()

# ---------- Path utils ----------
# <repo>/runs/<run_id>/agent_<NN>/iter-<IIIII>[-<phase>].(npz|pt)
_PATH_RE = re.compile(
    r".*/runs/(?P<run>[^/\\]+)/agent_(?P<node>\d+)/iter-(?P<iter>\d+)(?:-(?P<phase>[^./\\]+))?\.(?P<ext>npz|pt)$"
)

def make_path(run_id: str, node_id: int, iteration: int, phase: str | None, ext: str,
              repo_root: str | Path | None = None) -> Path:
    phase_part = f"-{phase}" if phase else ""
    fname = f"iter-{iteration:05d}{phase_part}.{ext}"
    return runs_dir(repo_root) / run_id / f"agent_{node_id:02d}" / fname

def get_run_dir(run_id: str, node_id: int, repo_root: str | Path | None = None) -> Path:
    p = runs_dir(repo_root) / run_id / f"agent_{node_id:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_meta_from_path(path: str | Path) -> dict:
    s = str(path).replace("\\", "/")
    m = _PATH_RE.match(s)
    if not m:
        return {"run_id": None, "node_id": None, "iteration": None, "phase": None, "ext": None}
    g = m.groupdict()
    return {
        "run_id": g["run"],
        "node_id": int(g["node"]),
        "iteration": int(g["iter"]),
        "phase": g.get("phase"),
        "ext": g["ext"],
    }

def save_weights_pt(state_dict: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_sd = {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in state_dict.items()}
    torch.save(cpu_sd, path)

def save_model_pickle(model, path: str | Path, *, keep_discriminator=True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = copy.deepcopy(model)  # don't mutate the live object

    # Put internal modules on CPU for portability
    try:
        m.set_device(torch.device("cpu"))
    except Exception:
        pass

    if getattr(m, "_generator", None) is not None:
        m._generator.to("cpu").eval()

    if getattr(m, "_discriminator", None) is not None:
        if keep_discriminator:
            m._discriminator.to("cpu").eval()
        else:
            # Optional: drop it to shrink the file (not needed for sampling)
            m._discriminator = None

    torch.save(m, path)

def load_weights_pt(path: str | Path, device="cpu") -> dict:
    return torch.load(path, map_location=device)

def load_model_pickle(path: str | Path, device="cpu"):
    model = torch.load(path, map_location="cpu", weights_only=False)
    try:
        model.set_device(torch.device(device))
    except Exception:
        pass
    return model