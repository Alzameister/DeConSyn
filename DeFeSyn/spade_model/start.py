import asyncio
import os

import requests
import spade
from loguru import logger
from joblib import parallel_config
from requests.auth import HTTPBasicAuth

from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.logging.logger import init_logging
from DeFeSyn.spade_model.NodeAgent import NodeAgent, NodeConfig, NodeData

ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
ADULT_MANIFEST = "manifest.yaml"

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


if __name__ == "__main__":
    with parallel_config(n_jobs=1, prefer=None):
        # Windows: prefer Selector loop (more predictable with sockets & debuggers)
        if os.name == "nt":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
            except Exception:
                pass

        asyncio.run(epoch_experiments())
