import asyncio
import logging
import warnings

import spade

from DeFeSyn.spade_model.agents.NodeAgent import NodeAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)

from DeFeSyn.data.DataLoader import DatasetLoader

async def main():
    # Setup
    ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
    ADULT_MANIFEST = "manifest.yaml"
    NR_AGENTS = 2
    EPOCHS = 1
    MAX_ITERATIONS = 2

    # Loader Pipeline
    data_dir = f"{ADULT_PATH}/{NR_AGENTS}"
    logging.info(f"Splitting dataset into {NR_AGENTS} parts...")
    loader = DatasetLoader(manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}")
    full_data = loader.concat()
    train = loader.get_train()
    test = loader.get_test()
    data_dir, manifest_name = loader.split(NR_AGENTS, save_path=data_dir)
    logging.info(f"Finished splitting dataset into {NR_AGENTS} parts...")

    # Setup agents
    logging.info(f"Loading {NR_AGENTS} agents...")
    agents = []
    for i in range(NR_AGENTS):
        logging.info(f"[{i}] {i} / {NR_AGENTS}")
        jid = f"agent_{i}@localhost"
        password = "password"
        agent = NodeAgent(
            jid=jid,
            password=password,
            id=i,
            full_data=full_data,
            full_train_data=train,
            full_test_data=test,
            data_source=data_dir,
            manifest_file_name=manifest_name,
            epochs=EPOCHS,
            max_iterations=MAX_ITERATIONS
        )
        agents.append(agent)

    await asyncio.gather(*[agent.start(auto_register=True) for agent in agents])
    logging.info(f"{NR_AGENTS} agents started.")

    # TODO: Network topology setup? For now, ring topology is taken.
    for i, agent in enumerate(agents):
        if i == len(agents) - 1:
            agent.presence.subscribe(f"agent_0@localhost")
            logging.info(f"Agent {agent.jid} subscribed to agent_0@localhost")
            continue
        agent.presence.subscribe(f"agent_{i+1}@localhost")
        logging.info(f"Agent {agent.jid} subscribed to agent_{i+1}@localhost")

    await asyncio.sleep(2)

    for i, agent in enumerate(agents):
        await asyncio.gather(agent.setup_fsm())
    # await asyncio.gather(*[agent.setup_fsm() for agent in agents])
    logging.info("FSM behaviors added to agents.")

    await asyncio.gather(*[spade.wait_until_finished(agent) for agent in agents])
    logging.info("Agents finished their tasks.")

    await asyncio.gather(*[agent.stop() for agent in agents])
    logging.info("Agents stopped.")

if __name__ == "__main__":
    try:
        spade.run(main())
    except RuntimeError as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())





