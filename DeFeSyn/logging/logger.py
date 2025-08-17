import sys
import uuid
import warnings
from pathlib import Path
from loguru import logger


def init_logging(run_id: str | None = None, level: str = "INFO") -> str:
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    log_dir = Path("logs") / run_id
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
        log_dir / "agents.jsonl",
        level="INFO",
        serialize=True,
        enqueue=True,
        rotation="50 MB",
        retention="14 days",
    )

    warnings.filterwarnings("ignore", category=UserWarning)

    # Set safe defaults so the console format doesn't KeyError before you bind per-agent.
    logger.configure(extra={"node_id": "--", "jid": ""})

    return run_id

def bind_defaults(log):
    return log.bind(node_id="--", jid="")