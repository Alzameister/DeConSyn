import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from spade.agent import Agent
from spade.template import Template
from loguru import logger

from DeFeSyn.consensus.Consensus import Consensus
from DeFeSyn.spade.FSMBehaviour import NodeFSMBehaviour, TRAINING_STATE, StartState, TrainingState, \
    PullState, PushState, FinalState
from DeFeSyn.spade.ReceiveBehaviour import BarrierHelloResponder, BarrierAckRouter, ReceiveBehaviour
from DeFeSyn.spade.FSMBehaviour import START_STATE, PULL_STATE, PUSH_STATE, FINAL_STATE
from DeFeSyn.spade.PresenceBehaviour import PresenceBehaviour
from DeFeSyn.utils.io import get_repo_root


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
        self.queue: asyncio.Queue = asyncio.Queue()
        self.push_queue: asyncio.Queue = asyncio.Queue()
        self.barrier_queues: dict[str, asyncio.Queue[str]] = {}

        # model + state (filled by FSM on first TRAIN)
        self.model = None  # CTGANModel instance after first TRAIN
        self.weights: dict = {}  # latest local weights
        self.loss_values = pd.DataFrame(columns=["Epoch", "Generator Loss", "Discriminator Loss"])


    async def setup(self):
        self.log.info("Setting up NodeAgent...")

        # Advertise presence; PresenceBehavior will manage callbacks + subscriptions
        self.presence.set_available()

        # 1) Presence (ACo-L): auto-subscribe + track active neighbors + update degree/epsilon
        self.add_behaviour(PresenceBehaviour(poll_secs=2.0))
        self.add_behaviour(
            ReceiveBehaviour(),
            template=Template(metadata={"performative": "inform", "type": "gossip"})
        )
        self.add_behaviour(
            BarrierHelloResponder(),
            template=Template(metadata={"performative": "inform", "type": "barrier-hello"})
        )
        self.add_behaviour(
            BarrierAckRouter(),
            template=Template(metadata={"performative": "inform", "type": "barrier-ack"})
        )

        await self._setup_fsm()

        self.log.info("NodeAgent setup complete.")

    async def _setup_fsm(self):
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
