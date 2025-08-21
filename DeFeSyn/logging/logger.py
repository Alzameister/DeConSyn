import os
import re
import sys
from functools import lru_cache

import torch
import warnings
from pathlib import Path

import numpy as np
from loguru import logger
from datetime import datetime


def init_logging(run_id: str | None = None, level: str = "INFO") -> str:
    if run_id is None:
        current_time = datetime.now()
        run_id = f"run-{current_time.strftime('%Y%m%d-%H%M%S')}"
    project_root = Path(__file__).resolve().parent.parent.parent  # Adjust the number of .parent calls based on your directory structure
    log_dir = project_root / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "| <level>{level: <7}</level> "
            "| n={extra[node_id]:0>2} {extra[jid]} "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
    )

    logger.add(
        log_dir / "events.jsonl",
        level="INFO",
        serialize=True,
        enqueue=True,
        rotation="50 MB",
        retention="14 days",
        filter=lambda rec: rec["extra"].get("stream") == "event",
    )

    warnings.filterwarnings("ignore", category=UserWarning)

    # Set safe defaults so the console format doesn't KeyError before you bind per-agent.
    logger.configure(extra={"node_id": "--", "jid": ""})

    return run_id

def bind_defaults(log):
    return log.bind(node_id="--", jid="")

def flatten_params_numpy(weights: dict) -> np.ndarray:
    """
    Convert model weights dict → single 1D float32 vector.
    """
    parts = []
    for k in sorted(weights.keys()):
        w = weights[k]
        if hasattr(w, "detach"):  # torch tensor
            w = w.detach().cpu().numpy()
        arr = np.asarray(w, dtype=np.float32).ravel()
        parts.append(arr)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

# -------- Repo root detection --------
REPO_MARKERS = {".git", "pyproject.toml", "poetry.lock", "setup.cfg", "setup.py"}

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

def load_weights_pt(path: str | Path, device="cpu") -> dict:
    return torch.load(path, map_location=device)