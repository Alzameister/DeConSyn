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


def init_logging(run_id: str | None = None,
                 level: str = "INFO",
                 agents: int | None = None,
                 epochs: int | None = None,
                 iterations: int | None = None,
                 topology: str | None = None) -> str:
    if run_id is None:
        current_time = datetime.now()
        parts = []
        if agents is not None:
            parts.append(f"{agents}Agents")
        if epochs is not None:
            parts.append(f"{epochs}Epochs")
        if iterations is not None:
            parts.append(f"{iterations}Iterations")
        if topology is not None:
            safe_topology = re.sub(r"[^a-zA-Z0-9]", "", topology)
            parts.append(safe_topology)
        meta = "-" + "-".join(parts) if parts else ""
        run_id = f"run-{current_time.strftime('%Y%m%d-%H%M%S')}{meta}"
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
        "| <level>{level: <7}</level> "
        "| n={extra[node_id]:0>2} {extra[jid]} "
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )

    # 1) Console sink (stderr)
    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        format=console_fmt,
    )

    logger.add(
        log_dir / "console.log",
        level=level,
        enqueue=True,
        format=console_fmt,
        rotation="50 MB",
        retention="14 days",
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