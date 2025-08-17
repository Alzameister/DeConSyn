import asyncio
import json
import time
import uuid

from loguru import logger
import random

import pandas as pd
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from DeFeSyn.models.CTGAN.wrapper import CTGANModel

TRAINING_STATE = "TRAINING_STATE"
PULL_STATE = "PULL_STATE"
PUSH_STATE = "PUSH_STATE"
RECEIVE_STATE = "RECEIVE_STATE"
FINAL_STATE = "FINAL_STATE"

class NodeFSMBehaviour(FSMBehaviour):
    """
    NodeFSMBehaviour is a finite state machine (FSM) behavior for a NodeAgent in the DeFeSyn framework.
    It manages the states of the agent during the synthetic data generation model training process.
    It includes states for training, pulling, pushing, and receiving data.
    """
    async def on_start(self):
        self.agent.log.info(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        self.agent.log.info(f"FSM finished at state {self.current_state}")
        await self.agent.stop()

class TrainingState(State):
    """
    TrainingState is a state in the FSM that handles the training of a Synthetic Data Generation model.
    """
    def __init__(self):
        super().__init__()
        self.epochs = None
        self.data: dict = None

    async def run(self):
        self.agent.current_iteration += 1
        it = self.agent.current_iteration
        if self.agent.current_iteration > self.agent.max_iterations:
            self.agent.log.info("Max iterations reached. Exiting…")
            self.set_next_state(FINAL_STATE)
        else:
            self.agent.log.info(f"Starting FSM iteration {it} → TRAIN")
            self.epochs = self.epochs or self.agent.epochs
            self.data = self.data or self.agent.data

            if 'train' not in self.data:
                self.agent.log.error("TRAIN: No training split in agent.data; Cannot proceed with training.")
                self.set_next_state(FINAL_STATE)

            data = self.data['train']
            discrete_cols = [c for c in data.columns if data[c].dtype.name == "category"]

            self.agent.log.info(f"TRAIN: CTGAN epochs={self.epochs} | discrete_cols={discrete_cols}")

            if not self.agent.model:
                self.agent.log.info("TRAIN: Init CTGAN model")
                self.agent.model = CTGANModel(
                    full_data=self.agent.full_train_data,
                    data=data,
                    discrete_columns=discrete_cols,
                    epochs=self.epochs
                )

            if self.agent.weights:
                self.agent.log.info("TRAIN: Warm start: loading weights")
                self.agent.model.load_weights(self.agent.weights)
            else:
                self.agent.log.info("TRAIN: Cold start (no weights)")

            t0 = time.perf_counter()
            self.agent.model.train()
            ms = (time.perf_counter() - t0) * 1000.0
            self.agent.loss_values = self.agent.model.model.loss_values
            self.agent.weights = self.agent.model.get_weights()
            self.agent.log.info(f"TRAIN done in {ms:.1f} ms; weights updated")

            # TODO: Structured TRAIN event (last losses if available)

            self.agent.log.info(f"TRAIN: iteration {it} completed -> transition PULL")
            self.set_next_state(PULL_STATE)

class PullState(State):
    async def run(self):
        if self.agent.queue.empty():
            self.agent.log.info("PULL: queue empty → transition PUSH")
        else:
            it = self.agent.current_iteration
            self.agent.log.info("PULL: processing queue…")
            consumed = []
            q_before = self.agent.queue.qsize()
            while not self.agent.queue.empty():
                msg = await self.agent.queue.get()
                if msg.get_metadata("performative") == "inform" and msg.get_metadata("type") == "gossip":
                    self.agent.log.info(f"PULL: got weights from {msg.sender}")
                    received_weights = self.agent.model.decode(json.loads(msg.body))

                    # Perform consensus averaging with the received weights
                    self.agent.log.info("PULL: Consensus averaging")
                    self.agent.weights = self.agent.weights = self.agent.consensus.average(
                        x_i = self.agent.weights,
                        x_j = received_weights
                    )
                    consumed.append({
                        "neighbor": str(msg.sender),
                        "msg_id": msg.get_metadata("msg_id"),
                        "version": int(msg.get_metadata("version") or it)
                    })
            q_after = self.agent.queue.qsize()
            self.agent.log.info(f"PULL: averaged {len(consumed)} updates (queue {q_before}→{q_after})")

            # TODO: Structured PULL event

        self.agent.log.info(f"PULL transition PUSH")
        self.set_next_state(PUSH_STATE)

class PushState(State):
    async def run(self):
        class WaitResponse(OneShotBehaviour):
            def __init__(self, fut):
                super().__init__()
                self.fut = fut

            async def run(self):
                future = await self.receive(timeout=30.0)
                self.fut.set_result(future)

        it = self.agent.current_iteration

        fut = asyncio.get_running_loop().create_future()
        template = Template(metadata={"performative": "inform", "type": "gossip-reply"})
        self.agent.add_behaviour(WaitResponse(fut), template)

        contacts = self.agent.presence.get_contacts()
        neighbors = [jid for jid, c in contacts.items() if c.is_available()]
        peer = random.choice(neighbors)
        await asyncio.sleep(random.uniform(0.5, 1.5))

        pkg = self.agent.model.encode()
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)

        msg = Message(to=str(peer))
        msg.set_metadata("performative", "inform")
        msg.set_metadata("type", "gossip")
        msg.set_metadata("content-type", "application/octet-stream+b64")
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("version", version)
        msg.body = json.dumps(pkg)

        self.agent.log.info(f"PUSH: send → {peer}")
        await self.send(msg)

        # TODO: Structured PUSH event

        self.agent.log.info(f"Waiting for RESPOND from {peer}")
        reply = await fut
        if reply is None:
            # TODO: Handle timeout or no response on other side?
            self.agent.log.warning(f"No RESPOND from {peer}")
            # TODO: Structured RESPONSE event
        else:
            self.agent.log.info(f"RESPOND from {reply.sender}")
            received_weights = self.agent.model.decode(json.loads(reply.body))
            self.agent.weights = self.agent.consensus.average(x_i=self.agent.weights, x_j=received_weights)
            self.agent.log.info(f"RESPOND: weights updated from {reply.sender}")

            # TODO: # Structured RESPOND received

        self.agent.log.info(f"PUSH: transition RECEIVE")
        self.set_next_state(TRAINING_STATE)

class FinalState(State):
    async def run(self):
        # TODO: Cleanup / Final Reporting?
        self.agent.log.info("FSM completed. Stopping agent.")

        # TODO: Structured FINAL event