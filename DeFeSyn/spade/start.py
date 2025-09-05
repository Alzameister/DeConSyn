import asyncio
import os

import requests
import spade
from loguru import logger
from joblib import parallel_config
from requests.auth import HTTPBasicAuth

from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.logging.logger import init_logging
from DeFeSyn.spade.NodeAgent import NodeAgent, NodeConfig, NodeData
from DeFeSyn.utils.seed import set_global_seed

ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
ADULT_MANIFEST = "manifest.yaml"
SEED = 42
set_global_seed(SEED)

def agent_jid(i: int) -> str:
    return f"agent{i}@localhost"

def build_neighbors(n: int) -> dict[int, list[str]]:
    return {
        i: [agent_jid((i + 1) % n), agent_jid((i + n - 1) % n)]
        for i in range(n)
    }

async def shutdown_agents(agents: list[NodeAgent]) -> None:
    # 1) announce we're going offline
    for a in agents:
        try:
            a.presence.set_unavailable()
        except Exception:
            pass

    # 2) stop all agents
    await asyncio.gather(*[a.stop() for a in agents], return_exceptions=True)

    # 3) let slixmpp flush sockets/file handles
    await asyncio.sleep(0.3)

    # 4) optional: ask SPADE to cleanup globals if available
    try:
        # older/newer SPADE may or may not have this — safe to try
        await spade.quit_spade()      # type: ignore[attr-defined]
    except Exception:
        pass

    # 5) release log file handles (prevents WinError 5 on rerun)
    try:
        logger.remove()  # remove all sinks; or remove specific sink ids if you track them
    except Exception:
        pass

async def epoch_experiments():
    NR_AGENTS = 4
    EPOCHS = [5]
    MAX_ITERATIONS = [60]
    ALPHA = 1.0

    # --- Data prep (outside agents)
    loader = DatasetLoader(manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}")
    full_train = loader.get_train()
    full_test = loader.get_test()
    # If you still need to split & persist:
    data_dir, manifest_name = loader.split(NR_AGENTS, save_path=f"{ADULT_PATH}/{NR_AGENTS}")
    split_loader = DatasetLoader(manifest_path=f"{data_dir}/{manifest_name}")

    def partition_for(i: int) -> NodeData:
        # pick this agent's partition: keys look like "part-<i>-train" etc.
        train_name = next(n for n in split_loader.resource_names() if "-train" in n and n.endswith(f"part-{i}"))
        part_train = split_loader.get(train_name)
        return NodeData(part_train=part_train, full_train=full_train, full_test=full_test)

    # --- Neighbors (ring)
    neighbors_map = build_neighbors(NR_AGENTS)

    for e, m in zip(EPOCHS, MAX_ITERATIONS):
        run_id = init_logging(level="INFO")
        logger.info(f"Starting experiment with EPOCHS={e}, MAX_ITERATIONS={m}")

        # --- Create agents
        agents: list[NodeAgent] = []
        try:
            for i in range(NR_AGENTS):
                cfg = NodeConfig(
                    jid=agent_jid(i),
                    id=i,
                    password="password",
                    epochs=e,
                    max_iterations=m,
                    alpha=ALPHA,
                    run_id=run_id,
                )
                data = partition_for(i)
                a = NodeAgent(
                    cfg=cfg,
                    data=data,
                    neighbors=neighbors_map[i],
                )
                agents.append(a)

            # --- Start & run
            await asyncio.gather(*[a.start(auto_register=True) for a in agents])
            logger.info(f"{NR_AGENTS} agents started.")

            # FSM is set up in agent.setup(), nothing else to do
            await asyncio.gather(*[spade.wait_until_finished(a) for a in agents])
            logger.info("Agents finished.")

            await asyncio.gather(*[a.stop() for a in agents])
            logger.info("Agents stopped.")
        finally:
            await shutdown_agents(agents)
            logger.info("Agents stopped cleanly.")

async def main():
    run_id = init_logging(level="INFO")
    NR_AGENTS = 4
    EPOCHS = 15
    MAX_ITERATIONS = 20
    ALPHA = 1.0

    # --- Data prep (outside agents)
    loader = DatasetLoader(manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}")
    full_train = loader.get_train()
    full_test  = loader.get_test()
    # If you still need to split & persist:
    data_dir, manifest_name = loader.split(NR_AGENTS, save_path=f"{ADULT_PATH}/{NR_AGENTS}")
    split_loader = DatasetLoader(manifest_path=f"{data_dir}/{manifest_name}")

    def partition_for(i: int) -> NodeData:
        # pick this agent's partition: keys look like "part-<i>-train" etc.
        train_name = next(n for n in split_loader.resource_names() if "-train" in n and n.endswith(f"part-{i}"))
        part_train = split_loader.get(train_name)
        return NodeData(part_train=part_train, full_train=full_train, full_test=full_test)

    # --- Neighbors (ring)
    neighbors_map = build_neighbors(NR_AGENTS)

    # --- Create agents
    agents: list[NodeAgent] = []
    try:
        for i in range(NR_AGENTS):
            cfg = NodeConfig(
                jid=agent_jid(i),
                id=i,
                password="password",
                epochs=EPOCHS,
                max_iterations=MAX_ITERATIONS,
                alpha=ALPHA,
                run_id=run_id,
            )
            data = partition_for(i)
            a = NodeAgent(
                cfg=cfg,
                data=data,
                neighbors=neighbors_map[i],
            )
            agents.append(a)

        # --- Start & run
        await asyncio.gather(*[a.start(auto_register=True) for a in agents])
        logger.info(f"{NR_AGENTS} agents started.")

        # FSM is set up in agent.setup(), nothing else to do
        await asyncio.gather(*[spade.wait_until_finished(a) for a in agents])
        logger.info("Agents finished.")

        await asyncio.gather(*[a.stop() for a in agents])
        logger.info("Agents stopped.")
    finally:
        await shutdown_agents(agents)
        logger.info("Agents stopped cleanly.")


# =========================
# CLI SUPPORT (drop-in)
# =========================
import argparse
import contextlib
import signal

def _csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _csv_bools(s: str) -> list[bool]:
    m = {"true": True, "false": False, "1": True, "0": False}
    return [m[x.strip().lower()] for x in s.split(",") if x.strip().lower() in m]

def build_neighbors_full(n: int) -> dict[int, list[str]]:
    return {i: [agent_jid(j) for j in range(n) if j != i] for i in range(n)}

async def run_once(
    nr_agents: int,
    epochs: int,
    max_iterations: int,
    alpha: float,
    data_root: str,
    manifest: str,
    topology: str,
    xmpp_domain: str,
    password: str,
    seed: int,
    n_jobs: int,
    log_level: str = "INFO",
):
    # logging + seed
    run_id = init_logging(level=log_level.upper(),
                          agents=nr_agents,
                          epochs=epochs,
                          iterations=max_iterations)
    set_global_seed(seed)

    # prepare data
    loader = DatasetLoader(manifest_path=f"{data_root}/{manifest}")
    full_train = loader.get_train()
    full_test  = loader.get_test()

    data_dir, manifest_name = loader.split(nr_agents, save_path=f"{data_root}/{nr_agents}")
    split_loader = DatasetLoader(manifest_path=f"{data_dir}/{manifest_name}")

    def partition_for(i: int) -> NodeData:
        train_name = next(n for n in split_loader.resource_names() if "-train" in n and n.endswith(f"part-{i}"))
        part_train = split_loader.get(train_name)
        return NodeData(part_train=part_train, full_train=full_train, full_test=full_test)

    # neighbors
    if topology.lower() == "ring":
        neighbors_map = build_neighbors(nr_agents)
    elif topology.lower() == "full":
        neighbors_map = build_neighbors_full(nr_agents)
    else:
        raise ValueError("Unsupported topology. Use 'ring' or 'full'.")

    # agents
    agents: list[NodeAgent] = []
    try:
        for i in range(nr_agents):
            cfg = NodeConfig(
                jid=f"agent{i}@{xmpp_domain}",
                id=i,
                password=password,
                epochs=epochs,
                max_iterations=max_iterations,
                alpha=alpha,
                run_id=run_id,
            )
            data = partition_for(i)
            a = NodeAgent(cfg=cfg, data=data, neighbors=neighbors_map[i])
            agents.append(a)

        with parallel_config(n_jobs=n_jobs, prefer=None):
            await asyncio.gather(*[a.start(auto_register=True) for a in agents])
            logger.info(f"{nr_agents} agents started (epochs={epochs}, iters={max_iterations}, alpha={alpha}).")

            # wait for completion
            await asyncio.gather(*[spade.wait_until_finished(a) for a in agents])

            logger.info("Agents finished.")
            await asyncio.gather(*[a.stop() for a in agents])
            logger.info("Agents stopped.")
    finally:
        await shutdown_agents(agents)
        logger.info("Agents stopped cleanly.")

async def cli_async(args: argparse.Namespace) -> int:
    # Windows event loop policy (as you had)
    if os.name == "nt":
        with contextlib.suppress(Exception):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]

    # graceful Ctrl+C
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, loop.stop)

    if args.command == "run":
        await run_once(
            nr_agents=args.agents,
            epochs=args.epochs,
            max_iterations=args.iterations,
            alpha=args.alpha,
            data_root=args.data_root,
            manifest=args.manifest,
            topology=args.topology,
            xmpp_domain=args.xmpp_domain,
            password=args.password,
            seed=args.seed,
            n_jobs=args.n_jobs,
            log_level=args.log_level,
        )
        return 0

    if args.command == "sweep":
        iterations = _csv_ints(args.iterations_list)
        for i, e in enumerate(_csv_ints(args.epochs_list)):
            await asyncio.sleep(5)
            it = iterations[i]
            logger.info(f"=== Sweep run: epochs={e}, iterations={it} ===")
            await run_once(
                nr_agents=args.agents,
                epochs=e,
                max_iterations=it,
                alpha=args.alpha,
                data_root=args.data_root,
                manifest=args.manifest,
                topology=args.topology,
                xmpp_domain=args.xmpp_domain,
                password=args.password,
                seed=args.seed,
                n_jobs=args.n_jobs,
                log_level=args.log_level,
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
        sp.add_argument("--manifest", default=ADULT_MANIFEST, help="Manifest filename (default: manifest.yaml)")
        sp.add_argument("--topology", choices=["ring", "full"], default="ring", help="Neighbor topology")
        sp.add_argument("--xmpp-domain", default="localhost", help="XMPP domain for agent JIDs (default: localhost)")
        sp.add_argument("--password", default="password", help="XMPP password for all agents (default: password)")
        sp.add_argument("--seed", type=int, default=SEED, help=f"Global seed (default: {SEED})")
        sp.add_argument("--n-jobs", type=int, default=1, help="joblib parallel workers (default: 1)")
        sp.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging level")

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

