import asyncio
import os
import argparse
import contextlib
import signal
from pathlib import Path

import spade

from loguru import logger
from joblib import parallel_config

from DeFeSyn.data.data_loader import DatasetLoader, ADULT_CATEGORICAL_COLUMNS, ADULT_TARGET
from DeFeSyn.data.data_transformer import DataTransformer
from DeFeSyn.logging.logger import init_logging
from DeFeSyn.training_framework.agent.node_agent import NodeAgent, NodeConfig, NodeData
from DeFeSyn.utils.graph import Graph
from DeFeSyn.utils.seed import set_global_seed

ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
ADULT_MANIFEST = "manifest.yaml"
SEED = 42
set_global_seed(SEED)

# =========================
# UTILS
# =========================

async def _shutdown_agents(agents: list[NodeAgent]) -> None:
    for a in agents:
        try:
            a.presence.set_unavailable()
        except Exception:
            pass

    await asyncio.gather(*[a.stop() for a in agents], return_exceptions=True)

    await asyncio.sleep(0.3)

    try:
        await spade.quit_spade()
    except Exception:
        pass

    try:
        logger.remove()
    except Exception:
        pass

def _csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _csv_bools(s: str) -> list[bool]:
    m = {"true": True, "false": False, "1": True, "0": False}
    return [m[x.strip().lower()] for x in s.split(",") if x.strip().lower() in m]

# =========================
# CLI
# =========================
async def run(
    nr_agents: int,
    epochs: int,
    max_iterations: int,
    alpha: float,
    data_root: str,
    # manifest: str,
    topology: str,
    xmpp_domain: str,
    password: str,
    seed: int,
    n_jobs: int,
    log_level: str = "INFO",
    k: int = 4,
    p: float = 0.1,
    model_type: str = "tabddpm"
):
    # logging + seed
    run_id = init_logging(level=log_level.upper(),
                          agents=nr_agents,
                          epochs=epochs,
                          iterations=max_iterations,
                          topology=topology)
    set_global_seed(seed)

    # prepare data
    csv_path = data_root + "/csv"
    transformer = DataTransformer(data_root, ADULT_TARGET, ADULT_CATEGORICAL_COLUMNS)
    transformer.save_csv(Path(csv_path))
    npy_path = data_root + "/npy"

    logger.info(f"CSV saved to {csv_path}")

    loader = DatasetLoader(csv_path, ADULT_CATEGORICAL_COLUMNS)
    full_train = loader.get_train()
    full_test = loader.get_test()
    splits = loader.split(nr_agents, seed=seed)

    def partition_for(i: int) -> NodeData:
        part_train = splits[i]
        return NodeData(part_train=part_train, full_train=full_train, full_test=full_test)
    logger.info(f"Data loaded from {data_root} and partitioned for {nr_agents} agents.")
    # Head of partitions
    for i in range(nr_agents):
        part = splits[i]
        transformer.save_split_npy(
            part,
            Path(npy_path) / "splits" / str(nr_agents),
            i,
            ADULT_CATEGORICAL_COLUMNS,
            ADULT_TARGET
        )
        logger.info(f"Agent {i} partition: {part.shape}, head:\n{part.head(3)}")

    # neighbors
    if topology.lower() == "ring":
        neighbors_map = Graph.ring(nr_agents)
    elif topology.lower() == "full":
        neighbors_map = Graph.full(nr_agents)
    elif topology.lower() == "small-world":
        neighbors_map = Graph.small_world(nr_agents, k=k, p=p, seed=seed)
    else:
        raise ValueError("Unsupported topology. Use 'ring', 'full' or 'small-world'.")

    # agents
    agents: list[NodeAgent] = []
    try:
        for i in range(nr_agents):
            PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "FeDeSyn")
            parent_dir = os.path.join(PROJECT_ROOT, "runs", run_id)
            cfg = NodeConfig(
                jid=f"agent{i}@{xmpp_domain}",
                id=i,
                password=password,
                epochs=epochs,
                max_iterations=max_iterations,
                alpha=alpha,
                run_id=run_id,
                model_type=model_type,
                real_data_path=npy_path + f"/splits/{nr_agents}/split_{i}" ,
                real_full_data_path=npy_path,
                parent_dir=parent_dir + f"/agent_0{i}", # TODO: Correct formatting
                target=ADULT_TARGET
            )
            data = partition_for(i)
            a = NodeAgent(cfg=cfg, data=data, neighbors=neighbors_map[cfg.jid])
            agents.append(a)

        with parallel_config(n_jobs=n_jobs, prefer=None):
            await asyncio.gather(*[a.start(auto_register=True) for a in agents])
            logger.info(f"{nr_agents} agents started (epochs={epochs}, iters={max_iterations}, alpha={alpha}).")

            while True:
                await asyncio.sleep(2)
                if all(a.is_final for a in agents):
                    for a in agents:
                        a.fsm_done.set()
                    break

            await asyncio.gather(*[a.fsm_done.wait() for a in agents])
            logger.info("All FSMs finished — stopping agents...")

            await asyncio.gather(*[a.stop() for a in agents], return_exceptions=True)

            await asyncio.gather(*[spade.wait_until_finished(a) for a in agents], return_exceptions=True)

            logger.info("Agents stopped.")
    finally:
        await _shutdown_agents(agents)
        logger.info("Agents stopped cleanly.")

async def cli_async(args: argparse.Namespace) -> int:
    if os.name == "nt":
        with contextlib.suppress(Exception):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, loop.stop)

    if args.command == "run":
        await run(
            nr_agents=args.agents,
            epochs=args.epochs,
            max_iterations=args.iterations,
            alpha=args.alpha,
            data_root=args.data_root,
            topology=args.topology,
            k=args.k,
            p=args.p,
            xmpp_domain=args.xmpp_domain,
            password=args.password,
            seed=args.seed,
            n_jobs=args.n_jobs,
            log_level=args.log_level,
            model_type=args.model_type
        )
        return 0

    raise ValueError("Unknown command")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="defesyn-agents", description="Run DeFeSyn SPADE node agents.")
    sub = p.add_subparsers(dest="command", required=True)

    # Shared args
    def add_shared(sp):
        sp.add_argument("--agents", type=int, default=4, help="Number of agents (default: 4)")
        sp.add_argument("--alpha", type=float, default=1.0, help="ACo-L alpha (default: 1.0)")
        sp.add_argument("--data-root", default=ADULT_PATH, help="Dataset root directory")
        sp.add_argument("--topology", choices=["ring", "full", "small-world"], default="ring", help="Neighbor topology")
        sp.add_argument("--k", type=int, default=4, help="Number of nearest neighbors for small-world (default: 4)")
        sp.add_argument("--p", type=float, default=0.1, help="Rewiring probability for small-world (default: 0.1)")
        sp.add_argument("--xmpp-domain", default="localhost", help="XMPP domain for agent JIDs (default: localhost)")
        sp.add_argument("--password", default="password", help="XMPP password for all agents (default: password)")
        sp.add_argument("--seed", type=int, default=SEED, help=f"Global seed (default: {SEED})")
        sp.add_argument("--n-jobs", type=int, default=1, help="joblib parallel workers (default: 1)")
        sp.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging level")
        sp.add_argument("--model-type", default="tabddpm", choices=["tabddpm", "ctgan"], help="Model type to use (default: tabddpm)")

    # run
    sp_run = sub.add_parser("run", help="Run a single experiment")
    add_shared(sp_run)
    sp_run.add_argument("--epochs", type=int, required=True, help="Epochs")
    sp_run.add_argument("--iterations", type=int, required=True, help="Max iterations")

    # sweep
    sp_sweep = sub.add_parser("sweep", help="Run a sweep over epochs × iterations")
    add_shared(sp_sweep)
    sp_sweep.add_argument("--epochs-list", required=True, help='Comma-separated epochs list, e.g. "5,10,15"')
    sp_sweep.add_argument("--iterations-list", required=True, help='Comma-separated iterations list, e.g. "20,60"')

    return p

def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(cli_async(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

if __name__ == "__main__":
    raise SystemExit(cli())

