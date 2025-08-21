import pandas as pd
import spade
import spade.agent
from loguru import logger
from spade.agent import Agent

from DeFeSyn.consensus.Consensus import Consensus
from DeFeSyn.data.DataLoader import DatasetLoader
from DeFeSyn.logging.logger import init_logging
from DeFeSyn.spade_model.behaviors.FSMBehavior import *
from DeFeSyn.spade_model.behaviors.ReceiveBehavior import ReceiveBehavior, BarrierReceiver

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
                 max_iterations: int=10,
                 neighbors: list[str] | None = None,
                 run_id: str | None = None
             ):
        super().__init__(jid, password)
        self.start_ready_from = None
        self.start_expected = None
        self.start_ready_event = None
        self.id = id
        self.run_id = run_id
        self.data_source = data_source
        self.manifest_file_name = manifest_file_name
        self.epochs = epochs
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.neighbors = neighbors or []
        self.participants = self.neighbors + [self.jid]

        self.log = logger.bind(node_id=id, jid=jid)
        self.event = self.log.bind(stream="event")

        self.log.info(f"Loading data from {self.data_source}/{self.manifest_file_name}...")
        self.loader = DatasetLoader(manifest_path=f"{self.data_source}/{self.manifest_file_name}")
        self.resource_names = self.loader.resource_names()
        self.log.info(f"Available resources: {self.resource_names}")
        self.full_data = full_data
        self.full_train_data = full_train_data
        self.full_test_data = full_test_data
        self.data: dict = {}
        for name in self.resource_names:
            if f"part-{self.id}" in name:
                key = name.split("-")[1]
                self.log.info(f"{key}: {len(self.loader.get(name))} rows")
                self.data[key] = self.loader.get(name)
        self.data['full'] = self.loader.concat()
        self.log.info(f"Data loaded: {self.data['full']}")
        self.log.info(f"Total length of all resources: {len(self.data)} rows")
        self.log.info("All resources loaded and saved to agent's data attribute.")

        self.model: CTGANModel = None  # Placeholder for the model instance
        self.consensus: Consensus = Consensus()

        self.queue: asyncio.Queue = asyncio.Queue()
        self.push_queue: asyncio.Queue = asyncio.Queue()
        self.weights: dict = {}  # Placeholder for model weights
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

    async def setup(self):
        await super().setup()
        self.log.info("Setting up NodeAgent...")

        self.presence.set_available()
        self.presence.on_subscribe = lambda jid: asyncio.create_task(self._on_subscribe(jid))
        self.presence.on_subscribed = lambda jid: asyncio.create_task(self._on_subscribed(jid))
        self.presence.on_unsubscribed = lambda jid: asyncio.create_task(self._on_unsubscribed(jid))

        receive_template = Template(metadata={"performative": "inform", "type": "gossip"})
        self.add_behaviour(ReceiveBehavior(), template=receive_template)

        self.start_ready_from = {str(self.jid)}  # include self
        self.start_expected = set(map(str, getattr(self, "participants", [self.jid]))) | {str(self.jid)}
        self.start_ready_event = asyncio.Event()

        barrier_tmpl = Template(metadata={"performative": "inform", "type": "barrier", "stage": "start"})
        self.add_behaviour(BarrierReceiver(), template=barrier_tmpl)

        self.log.info("NodeAgent setup complete.")

    async def setup_fsm(self):
        """
        Set up the finite state machine (FSM) for the NodeAgent.
        This method initializes the FSM with the states and transitions required for the agent's operation.
        """
        self.log.info("Setting up FSM...")
        fsm = NodeFSMBehaviour()
        fsm.add_state(name=START_STATE, state=StartState(), initial=True)
        fsm.add_state(name=TRAINING_STATE, state=TrainingState())
        fsm.add_state(name=PULL_STATE, state=PullState())
        fsm.add_state(name=PUSH_STATE, state=PushState())
        fsm.add_state(name=FINAL_STATE, state=FinalState())

        fsm.add_transition(source=START_STATE, dest=TRAINING_STATE)
        fsm.add_transition(source=TRAINING_STATE, dest=PULL_STATE)
        fsm.add_transition(source=TRAINING_STATE, dest=PUSH_STATE)
        fsm.add_transition(source=PULL_STATE, dest=PUSH_STATE)
        fsm.add_transition(source=PUSH_STATE, dest=TRAINING_STATE)
        fsm.add_transition(source=TRAINING_STATE, dest=FINAL_STATE)

        self.add_behaviour(fsm)
        self.log.info("FSM setup complete.")

    async def _on_subscribe(self, jid):
        self.log.info(f"[{self.jid}] got subscribe from {jid}")
        self.presence.approve_subscription(jid)
        await self.presence.subscribe(jid)  # ensure mutual subscription

    async def _on_subscribed(self, jid):
        self.log.info(f"[{self.jid}] subscription completed with {jid}")

    async def _on_unsubscribed(self, jid):
        self.log.info(f"[{self.jid}] unsubscribed by {jid}")

def agent_jid(i: int) -> str:
    return f"agent{i}@localhost"

async def main():
    run_id = init_logging(level="INFO")
    nr_agents = 2
    epochs = 1
    max_iterations = 2

    # ---- Data prep
    data_dir = f"{ADULT_PATH}/{nr_agents}"
    logger.info(f"Splitting dataset into {nr_agents} parts...")
    loader = DatasetLoader(manifest_path=f"{ADULT_PATH}/{ADULT_MANIFEST}")
    full_data = loader.concat()
    train = loader.get_train()
    test = loader.get_test()
    data_dir, manifest_name = loader.split(nr_agents, save_path=data_dir)
    logger.info(f"Finished splitting dataset into {nr_agents} parts...")

    # ---- Topology (ring). Swap for full-mesh if you prefer.
    neighbors_map = {
        i: [agent_jid((i + 1) % nr_agents)]  # ring neighbor
        for i in range(nr_agents)
    }
    # Full-mesh alternative:
    # neighbors_map = {i: [agent_jid(j) for j in range(nr_agents) if j != i] for i in range(nr_agents)}

    # ---- Create agents (pass neighbors)
    logger.info(f"Loading {nr_agents} agents...")
    agents = []
    for i in range(nr_agents):
        a = NodeAgent(
            jid=agent_jid(i),
            id=i,
            password="password",
            full_data=full_data,
            full_train_data=train,
            full_test_data=test,
            data_source=data_dir,
            manifest_file_name=manifest_name,
            epochs=epochs,
            max_iterations=max_iterations,
            neighbors=neighbors_map[i],
        )
        agents.append(a)

    # ---- Start agents
    await asyncio.gather(*[a.start(auto_register=True) for a in agents])
    logger.info(f"{nr_agents} agents started.")

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