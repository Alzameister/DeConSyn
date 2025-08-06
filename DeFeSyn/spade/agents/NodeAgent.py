import logging
import asyncio

import spade
from spade.agent import Agent

from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.spade.behaviors.FSMBehavior import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# TODO: Refactor to a config file or environment variable
ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
ADULT_MANIFEST = "adult.yaml"

class NodeAgent(Agent):
    """
    NodeAgent is an agent that implements a finite state machine (FSM) to run a Decentralized Federated Learning (DeFeSyn) framework.
    """
    def __init__(self, jid: str, password: str, data_source: str, manifest_file_name: str, epochs: int=100):
        super().__init__(jid, password)
        self.queue = asyncio.Queue()
        self.data_source = data_source
        self.manifest_file_name = manifest_file_name
        self.epochs = epochs

        logging.info(f"Loading data from {self.data_source}/{self.manifest_file_name}...")
        self.loader = DatasetLoader(manifest_path=f"{self.data_source}/{self.manifest_file_name}")
        self.resource_names = self.loader.resource_names()
        logging.info(f"Available resources: {self.resource_names}")
        self.data = {}
        for name in self.resource_names:
            key = name.split("-")[-1]
            logging.info(f"{key}: {len(self.loader.get(name))} rows")
            self.data[key] = self.loader.get(name)
        self.data['full'] = self.loader.concat()
        logging.info(f"Data loaded: {self.data['full']}")
        logging.info(f"Total length of all resources: {len(self.data)} rows")
        logging.info("All resources loaded and saved to agent's data attribute.")


    async def setup(self):
        await super().setup()
        logging.info("Setting up NodeAgent...")
        fsm = NodeFSMBehaviour()
        fsm.add_state(name=TRAINING_STATE,  state=TrainingState(), initial=True)
        fsm.add_state(name=PULL_STATE,      state=PullState())
        fsm.add_state(name=PUSH_STATE,      state=PushState())
        fsm.add_state(name=RECEIVE_STATE,   state=ReceiveState())

        fsm.add_transition(source=TRAINING_STATE,   dest=PULL_STATE)
        fsm.add_transition(source=TRAINING_STATE,   dest=PUSH_STATE)
        fsm.add_transition(source=PULL_STATE,       dest=PUSH_STATE)
        fsm.add_transition(source=PUSH_STATE,       dest=TRAINING_STATE)

        self.add_behaviour(fsm)
        logging.info("NodeAgent setup complete.")


async def main():
    agent = NodeAgent(
        jid="agent@localhost",
        password="password",
        data_source=ADULT_PATH,
        manifest_file_name=ADULT_MANIFEST,
        epochs=2
    )
    await agent.start()
    logging.info("Agent started")

    await spade.wait_until_finished(agent)
    await agent.stop()
    logging.info("Agent finished")

if __name__ == "__main__":
    try:
        spade.run(main())
    except RuntimeError as e:
        # Fallback for environments with a running event loop (like PyCharm)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())