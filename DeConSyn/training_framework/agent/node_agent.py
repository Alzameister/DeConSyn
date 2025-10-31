import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import QuantileTransformer
from spade.agent import Agent
from spade.template import Template
from loguru import logger

from DeConSyn.models.CTGAN.data_transformer import DataTransformer
from DeConSyn.training_framework.consensus.consensus import Consensus
from DeConSyn.models.models import Model
from DeConSyn.training_framework.fsm.fsm_behaviour import NodeFSMBehaviour, TRAINING_STATE, START_STATE, PULL_STATE, PUSH_STATE, FINAL_STATE
from DeConSyn.training_framework.fsm.states.final import FinalState
from DeConSyn.training_framework.fsm.states.pull import PullState
from DeConSyn.training_framework.fsm.states.push import PushState
from DeConSyn.training_framework.fsm.states.train import TrainingState
from DeConSyn.training_framework.fsm.states.start import StartState
from DeConSyn.training_framework.communication.receive_behaviour import ReceiveBehaviour
from DeConSyn.training_framework.communication.barrier_behaviour import BarrierHelloBehaviour, BarrierAckRouter
from DeConSyn.training_framework.communication.presence_behaviour import PresenceBehaviour
from DeConSyn.io.io import get_repo_root


@dataclass(frozen=True)
class NodeConfig:
    jid: str
    id: int
    password: str
    epochs: int = 100
    max_iterations: int = 10
    alpha: float = 0.5
    run_id: str | None = None
    device: str = "auto"
    model_type: str = "tabddpm",
    real_data_path: str | None = None,
    target: str | None = None,
    cat_encoder: OrdinalEncoder | None = None,
    num_encoder: QuantileTransformer | None = None,
    y_encoder: OrdinalEncoder | None = None,
    data_transformer: DataTransformer | None = None,
    config: dict | None = None

@dataclass(frozen=True)
class NodeData:
    """Data the agent needs at runtime."""
    # Per-agent partition(s)
    part_train: pd.DataFrame
    # Global context (For CTGAN Initialization/Evaluation)
    full_train: pd.DataFrame
    full_test: pd.DataFrame


class NodeAgent(Agent):
    """
    Implementation of a SPADE Agent that runs a FSM to perform decentralized federated learning.
    """
    def __init__(
            self,
            cfg: NodeConfig,
            data: NodeData,
            neighbors: list[str] | None = None,
            consensus: Consensus | None = None,
            repo_root: Path | None = None
    ):
        super().__init__(cfg.jid, cfg.password)

        # --- identity / config
        self.id: int = cfg.id
        self.run_id: str | None = cfg.run_id
        self.epochs: int = cfg.epochs
        self.max_iterations: int = cfg.max_iterations
        self.current_iteration: int = 0

        # logging
        self.log = logger.bind(node_id=self.id, jid=self.jid)
        self.event = self.log.bind(stream="event")

        self.model_type: str = cfg.model_type
        self.real_data_path = cfg.real_data_path
        self.target = cfg.target
        self.cat_encoder = cfg.cat_encoder
        self.num_encoder = cfg.num_encoder
        self.y_encoder = cfg.y_encoder
        self.data_transformer = cfg.data_transformer
        if self.model_type.lower() == "tabddpm":
            self.config = cfg.config
        self.log.info("Using model: {}", self.model_type)

        if cfg.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = cfg.device
        self.log.info("Using device: {}", self.device)

        # --- topology
        self.neighbors: list[str] = list(neighbors or [])
        self.participants: list[str] = self.neighbors + [self.jid]
        self.active_neighbors: set[str] = set()
        self.log.info("Neighbors: {}", self.neighbors)

        # --- data
        self.data: dict[str, pd.DataFrame] = {
            "train": data.part_train,
            "full_train": data.full_train,
            "full_test": data.full_test,
        }

        # --- infra
        self.repo_dir: Path = repo_root or get_repo_root()
        self.consensus: Consensus = consensus or Consensus(alpha=cfg.alpha)

        # queues for gossip + barrier tokens
        # Dictionary of neighbor JID and request SPADE message
        self.pending_gossip: dict = {str(n): {} for n in self.neighbors}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.push_queue: asyncio.Queue = asyncio.Queue()
        self.barrier_queues: dict[str, asyncio.Queue[str]] = {}
        self.pending_gossip_replies: list = []
        self.fsm_done: asyncio.Event = asyncio.Event()

        # model + state (filled by FSM on first TRAIN)
        self.model: Model = None  # CTGANModel instance after first TRAIN
        self.weights: dict = {}  # latest local weights
        self.loss_values = pd.DataFrame(columns=["Epoch", "Generator Loss", "Discriminator Loss"])

        self.is_final = False

    async def setup(self):
        self.log.info("Setting up NodeAgent...")

        # Advertise presence; PresenceBehavior will manage callbacks + subscriptions
        self.presence.set_available()

        # 1) Presence (ACo-L): auto-subscribe + track active neighbors + update degree/epsilon
        self.add_behaviour(PresenceBehaviour(poll_secs=2.0))
        self.add_behaviour(
            ReceiveBehaviour(),
            template=Template(metadata={"performative": "inform", "type": "gossip-req"})
        )
        self.add_behaviour(
            BarrierHelloBehaviour(),
            template=Template(metadata={"performative": "inform", "type": "barrier-hello"})
        )
        self.add_behaviour(
            BarrierAckRouter(),
            template=Template(metadata={"performative": "inform", "type": "barrier-ack"})
        )

        await self._setup_fsm()

        self.log.info("NodeAgent setup complete.")

    async def _setup_fsm(self):
        # Declare states once, in a single, readable structure
        states = {
            START_STATE: StartState(),
            TRAINING_STATE: TrainingState(),
            PULL_STATE: PullState(),
            PUSH_STATE: PushState(),
            FINAL_STATE: FinalState(),
        }

        # Declare all allowed transitions compactly
        transitions = [
            (START_STATE, TRAINING_STATE),
            (TRAINING_STATE, PULL_STATE),
            (TRAINING_STATE, PUSH_STATE),
            (PULL_STATE, PUSH_STATE),
            (PUSH_STATE, TRAINING_STATE),
            (TRAINING_STATE, FINAL_STATE),
        ]

        fsm = NodeFSMBehaviour(states=states, transitions=transitions, initial=START_STATE)
        self.add_behaviour(fsm)

def make_rid(self) -> str:
    # short, unique per sender
    return f"{self.jid.localpart}-{uuid.uuid4().hex[:12]}"

