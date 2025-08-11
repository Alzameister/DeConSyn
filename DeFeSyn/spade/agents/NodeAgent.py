import asyncio
import json
import warnings
from logging import Logger

import spade
from spade.agent import Agent
from spade.message import Message

from DeFeSyn.consensus.Consensus import Consensus
from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.spade.behaviors.FSMBehavior import *
from DeFeSyn.spade.behaviors.ReceiveBehavior import ReceiveBehavior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# TODO: Refactor to a config file or environment variable
ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
ADULT_MANIFEST = "manifest.yaml"

class NodeAgent(Agent):
    """
    NodeAgent is an agent that implements a finite state machine (FSM) to run a Decentralized Federated Learning (DeFeSyn) framework.
    """
    def __init__(self, jid: str, id: int, password: str, data_source: str, manifest_file_name: str, epochs: int=100):
        super().__init__(jid, password)
        self.id = id
        self.logger: Logger = logging.getLogger(f"agent_{self.id}")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.queue: asyncio.Queue = asyncio.Queue()
        self.data_source = data_source
        self.manifest_file_name = manifest_file_name
        self.epochs = epochs
        self.model: CTGANModel = None  # Placeholder for the model instance
        self.weights: dict = {} # Placeholder for model weights
        self.consensus: Consensus = Consensus()

        self.logger.info(f"Loading data from {self.data_source}/{self.manifest_file_name}...")
        self.loader = DatasetLoader(manifest_path=f"{self.data_source}/{self.manifest_file_name}")
        self.resource_names = self.loader.resource_names()
        self.logger.info(f"Available resources: {self.resource_names}")
        self.data: dict = {}
        for name in self.resource_names:
            if f"part-{self.id}" in name:
                key = name.split("-")[1]
                logging.info(f"{key}: {len(self.loader.get(name))} rows")
                self.data[key] = self.loader.get(name)
        self.data['full'] = self.loader.concat()
        self.logger.info(f"Data loaded: {self.data['full']}")
        self.logger.info(f"Total length of all resources: {len(self.data)} rows")
        self.logger.info("All resources loaded and saved to agent's data attribute.")

    async def setup(self):
        await super().setup()
        self.logger.info("Setting up NodeAgent...")

        self.presence.set_available()
        self.presence.on_subscribe = lambda jid: asyncio.create_task(self._on_subscribe(jid))
        self.presence.on_subscribed = lambda jid: asyncio.create_task(self._on_subscribed(jid))
        self.presence.on_unsubscribed = lambda jid: asyncio.create_task(self._on_unsubscribed(jid))

        self.add_behaviour(ReceiveBehavior())
        self.logger.info("NodeAgent setup complete.")

    async def setup_fsm(self):
        """
        Set up the finite state machine (FSM) for the NodeAgent.
        This method initializes the FSM with the states and transitions required for the agent's operation.
        """
        self.logger.info("Setting up FSM...")
        fsm = NodeFSMBehaviour()
        fsm.add_state(name=TRAINING_STATE, state=TrainingState(), initial=True)
        fsm.add_state(name=PULL_STATE, state=PullState())
        fsm.add_state(name=PUSH_STATE, state=PushState())

        fsm.add_transition(source=TRAINING_STATE, dest=PULL_STATE)
        fsm.add_transition(source=TRAINING_STATE, dest=PUSH_STATE)
        fsm.add_transition(source=PULL_STATE, dest=PUSH_STATE)
        fsm.add_transition(source=PUSH_STATE, dest=TRAINING_STATE)

        self.add_behaviour(fsm)
        self.logger.info("FSM setup complete.")

    async def _on_subscribe(self, jid):
        self.logger.info(f"[{self.jid}] got subscribe from {jid}")
        await self.presence.approve_subscription(jid)
        await self.presence.subscribe(jid)  # ensure mutual subscription

    async def _on_subscribed(self, jid):
        self.logger.info(f"[{self.jid}] subscription completed with {jid}")

    async def _on_unsubscribed(self, jid):
        self.logger.info(f"[{self.jid}] unsubscribed by {jid}")

    async def send_to_peer(self, peer_jid: str, payload: dict):
        """
        Send a message to a peer agent.
        Args:
            peer_jid (str): JID of the peer agent.
            payload (dict): Payload to send.
        """
        self.logger.info(f"[{self.jid}] sending to {peer_jid}")
        msg = Message(to=str(peer_jid))
        msg.set_metadata("performative", "inform")
        msg.set_metadata("type", "gossip")
        msg.set_metadata("content-type", "application/octet-stream+b64")
        msg.body = json.dumps(payload)  # or just a plain string
        await self.send(msg)
        self.logger.info(f"[{self.jid}] message sent to {peer_jid}")

async def main():
    nr_agents = 2
    epochs = 1
    data_dir = f"{ADULT_PATH}/{nr_agents}"
    logging.info(f"Splitting dataset into {nr_agents} parts...")
    loader = DatasetLoader(manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}")
    data_dir, manifest_name = loader.split(nr_agents, save_path=data_dir)

    logging.info("Starting NodeAgents...")
    agent_1 = NodeAgent(
        jid="agent1@localhost",
        id=1,
        password="password",
        data_source=data_dir,
        manifest_file_name=manifest_name,
        epochs=epochs
    )
    agent_2 = NodeAgent(
        jid="agent2@localhost",
        id=2,
        password="password",
        data_source=data_dir,
        manifest_file_name=manifest_name,
        epochs=epochs
    )

    await asyncio.gather(
        agent_1.start(auto_register=True),
        agent_2.start(auto_register=True)
    )
    logging.info("All Agents started")

    agent_1.presence.subscribe("agent2@localhost"),
    agent_2.presence.subscribe("agent1@localhost")
    logging.info("Agents subscribed to each other")

    await asyncio.gather(
        agent_1.setup_fsm(),
        agent_2.setup_fsm()
    )
    logging.info("FSM behaviors added")

    await asyncio.gather(
        spade.wait_until_finished(agent_1),
        spade.wait_until_finished(agent_2)
    )
    logging.info("Agents finished their tasks")

    await asyncio.gather(
        agent_2.stop(),
        agent_1.stop()
    )
    logging.info("Agent finished")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        spade.run(main())
    except RuntimeError as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())