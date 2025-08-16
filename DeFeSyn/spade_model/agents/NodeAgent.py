import warnings
from logging import Logger
import pandas as pd
import spade
import spade.agent
from spade.agent import Agent

from DeFeSyn.consensus.Consensus import Consensus
from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.spade_model.behaviors.FSMBehavior import *
from DeFeSyn.spade_model.behaviors.ReceiveBehavior import ReceiveBehavior, PushReceiveBehavior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning)

# TODO: Refactor to a config file or environment variable
ADULT_PATH = "C:/Users/trist/OneDrive/Dokumente/UZH/BA/05_Data/adult"
ADULT_MANIFEST = "manifest.yaml"

class NodeAgent(Agent):
    """
    NodeAgent is an agent that implements a finite state machine (FSM) to run a Decentralized Federated Learning (DeFeSyn) framework.
    """
    def __init__(self,
                 jid: str,
                 id: int,
                 password: str,
                 full_data: pd.DataFrame,
                 full_train_data: pd.DataFrame,
                 full_test_data: pd.DataFrame,
                 data_source: str,
                 manifest_file_name: str,
                 epochs: int=100,
                 max_iterations: int=10):
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
        self.push_queue: asyncio.Queue = asyncio.Queue()
        self.data_source = data_source
        self.manifest_file_name = manifest_file_name
        self.epochs = epochs
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.weights: dict = {} # Placeholder for model weights
        self.consensus: Consensus = Consensus()

        self.logger.info(f"Loading data from {self.data_source}/{self.manifest_file_name}...")
        self.loader = DatasetLoader(manifest_path=f"{self.data_source}/{self.manifest_file_name}")
        self.resource_names = self.loader.resource_names()
        self.logger.info(f"Available resources: {self.resource_names}")

        # TODO: Set full_training_data for CTGAN transformer
        self.full_data = full_data
        self.full_train_data = full_train_data
        self.full_test_data = full_test_data
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

        self.model: CTGANModel = None  # Placeholder for the model instance
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

    async def setup(self):
        await super().setup()
        self.logger.info("Setting up NodeAgent...")

        self.presence.set_available()
        self.presence.on_subscribe = lambda jid: asyncio.create_task(self._on_subscribe(jid))
        self.presence.on_subscribed = lambda jid: asyncio.create_task(self._on_subscribed(jid))
        self.presence.on_unsubscribed = lambda jid: asyncio.create_task(self._on_unsubscribed(jid))

        receive_template = Template(metadata={"performative": "inform", "type": "gossip"})
        self.add_behaviour(ReceiveBehavior(), receive_template)

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
        self.presence.approve_subscription(jid)
        await self.presence.subscribe(jid)  # ensure mutual subscription

    async def _on_subscribed(self, jid):
        self.logger.info(f"[{self.jid}] subscription completed with {jid}")

    async def _on_unsubscribed(self, jid):
        self.logger.info(f"[{self.jid}] unsubscribed by {jid}")

async def main():
    nr_agents = 2
    epochs = 1
    max_iterations = 2
    data_dir = f"{ADULT_PATH}/{nr_agents}"
    logging.info(f"Splitting dataset into {nr_agents} parts...")
    loader = DatasetLoader(manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}")
    full_data = loader.concat()
    train = loader.get_train()
    test = loader.get_test()
    data_dir, manifest_name = loader.split(nr_agents, save_path=data_dir)

    logging.info("Starting NodeAgents...")
    agent_1 = NodeAgent(jid="agent0@localhost", id=0, password="password", full_data=full_data, full_test_data=test, full_train_data=train,
                        data_source=data_dir,
                        manifest_file_name=manifest_name, epochs=epochs, max_iterations=max_iterations)
    agent_2 = NodeAgent(jid="agent1@localhost", id=1, password="password", full_data=full_data, full_test_data=test, full_train_data=train, data_source=data_dir,
                        manifest_file_name=manifest_name, epochs=epochs, max_iterations=max_iterations)

    await asyncio.gather(
        agent_1.start(auto_register=True),
        agent_2.start(auto_register=True)
    )
    logging.info("All Agents started")

    agent_1.presence.subscribe("agent1@localhost")
    agent_2.presence.subscribe("agent0@localhost")
    logging.info("Agents subscribed to each other")

    await asyncio.sleep(2)  # Wait for subscriptions to be established

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
    try:
        spade.run(main())
    except RuntimeError as e:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())