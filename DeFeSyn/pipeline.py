import asyncio
from loguru import logger
import spade

from DeFeSyn.spade_model.agents.NodeAgent import NodeAgent
from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.logging.logger import init_logging

def agent_jid(i: int) -> str:
    return f"agent{i}@localhost"

async def main():
    run_id = init_logging(level="INFO")

    # ---- Setup
    ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
    ADULT_MANIFEST = "manifest.yaml"
    NR_AGENTS = 2
    EPOCHS = 1
    MAX_ITERATIONS = 2

    # ---- Load and split dataset
    data_dir = f"{ADULT_PATH}/{NR_AGENTS}"
    logger.info(f"Splitting dataset into {NR_AGENTS} parts...")
    loader = DatasetLoader(manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}")
    full_data = loader.concat()
    train = loader.get_train()
    test = loader.get_test()
    data_dir, manifest_name = loader.split(NR_AGENTS, save_path=data_dir)
    logger.info(f"Finished splitting dataset into {NR_AGENTS} parts...")

    # ---- Topology (ring). Swap for full-mesh if you prefer.
    neighbors_map = {
        i: [agent_jid((i + 1) % NR_AGENTS)]  # ring neighbor
        for i in range(NR_AGENTS)
    }
    # Full-mesh alternative:
    # neighbors_map = {i: [agent_jid(j) for j in range(NR_AGENTS) if j != i] for i in range(NR_AGENTS)}

    # Setup agents
    logger.info(f"Loading {NR_AGENTS} agents...")
    agents = []
    for i in range(NR_AGENTS):
        logger.info(f"[{i}] {i} / {NR_AGENTS}")

        a = NodeAgent(
            jid=agent_jid(i),
            id=i,
            password="password",
            full_data=full_data,
            full_train_data=train,
            full_test_data=test,
            data_source=data_dir,
            manifest_file_name=manifest_name,
            epochs=EPOCHS,
            max_iterations=MAX_ITERATIONS,
            neighbors=neighbors_map[i],
        )

        agents.append(a)

    # ---- Start agents
    await asyncio.gather(*[a.start(auto_register=True) for a in agents])
    logger.info(f"{NR_AGENTS} agents started.")

    # ---- Add FSMs (START_STATE will subscribe + wait for availability)
    await asyncio.gather(*[a.setup_fsm() for a in agents])
    logger.info("FSM behaviors added to agents.")

    # ---- Wait for completion and stop
    await asyncio.gather(*[spade.wait_until_finished(a) for a in agents])
    logger.info("Agents finished their tasks.")

    await asyncio.gather(*[a.stop() for a in agents])
    logger.info("Agents stopped.")

if __name__ == "__main__":
    try:
        spade.run(main())
    except RuntimeError as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())





