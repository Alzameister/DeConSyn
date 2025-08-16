import asyncio
import json
import logging
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

class NodeFSMBehaviour(FSMBehaviour):
    """
    NodeFSMBehaviour is a finite state machine (FSM) behavior for a NodeAgent in the DeFeSyn framework.
    It manages the states of the agent during the synthetic data generation model training process.
    It includes states for training, pulling, pushing, and receiving data.
    """
    async def on_start(self):
        logging.info(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        logging.info(f"FSM finished at state {self.current_state}")
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
        if self.agent.current_iteration > self.agent.max_iterations:
            self.agent.logger.info("Maximum number of iterations reached. Exiting...")
            return

        self.agent.logger.info(f"Starting FSM Iteration {self.agent.current_iteration}")
        self.agent.logger.info("Starting training state…")
        if not self.epochs:
            self.epochs = self.agent.epochs
        if not self.data:
            self.data = self.agent.data
        if 'train' not in self.data:
            self.agent.logger.error("No data available for training. Please check the data source.")
            self.set_next_state(PULL_STATE)
            return

        data = self.data['train']
        discrete_cols = [col for col in data.columns
                         if data[col].dtype.name == "category"]

        self.agent.logger.info(f"Using {self.epochs} epochs for training.")
        self.agent.logger.info(f"Identified {len(discrete_cols)} discrete columns: {discrete_cols}")

        if not self.agent.model:
            self.agent.logger.info("Initializing CTGAN model for training.")
            self.agent.model = CTGANModel(
                full_data=self.agent.full_train_data,
                data=data,
                discrete_columns=discrete_cols,
                epochs=self.epochs
            )

        if self.agent.weights:
            self.agent.logger.info("Loading weights into CTGAN model for warm start.")
            self.agent.model.load_weights(self.agent.weights)
            self.agent.logger.info("Weights loaded.")
        else:
            self.agent.logger.info("No model weights found. Performing cold start.")

        self.agent.logger.info("Starting CTGAN training…")
        self.agent.model.train()
        self.agent.logger.info("CTGAN training complete.")
        # TODO: Saving of metrics
        # TODO: Save loss values based on FSM epochs --> I.e. save loss values after each epoch while training, for each time FSM is in TRAINING_STATE
        self.agent.loss_values = self.agent.model.model.loss_values
        self.agent.weights = self.agent.model.get_weights()
        self.agent.logger.info("Weights obtained from CTGAN model.")

        self.set_next_state(PULL_STATE)

class PullState(State):
    async def run(self):
        if self.agent.queue.empty():
            self.agent.logger.info("No Data received to pull. Transitioning to PushState.")
        else:
            self.agent.logger.info("Pulling data from other agents...")
            while not self.agent.queue.empty():
                msg = await self.agent.queue.get()
                if msg.get_metadata("performative") == "inform" and msg.get_metadata("type") == "gossip":
                    self.agent.logger.info(f"Processing model weights from {msg.sender}.")
                    received_weights = self.agent.model.decode(json.loads(msg.body))
                    self.agent.logger.info("Model weights decoded successfully.")

                    # Perform consensus averaging with the received weights
                    self.agent.logger.info("Performing consensus averaging with received weights.")
                    new_weights = self.agent.weights = self.agent.consensus.average(
                        x_i = self.agent.weights,
                        x_j = received_weights
                    )
                    self.agent.weights = new_weights
            self.agent.logger.info("Consensus averaging complete. New weights obtained.")

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

        fut = asyncio.get_running_loop().create_future()
        template = Template(metadata={"performative": "inform", "type": "gossip-reply"})
        self.agent.add_behaviour(WaitResponse(fut), template)
        # TODO
        contacts = self.agent.presence.get_contacts()
        neighbors = [jid for jid, c in contacts.items() if c.is_available()]
        peer = random.choice(neighbors)
        await asyncio.sleep(random.uniform(0.5, 1.5))

        self.agent.logger.info(f"Pushing model weights to {peer}...")
        pkg = self.agent.model.encode()
        msg = Message(to=str(peer))
        msg.set_metadata("performative", "inform")
        msg.set_metadata("type", "gossip")
        msg.set_metadata("content-type", "application/octet-stream+b64")
        msg.body = json.dumps(pkg)
        await self.send(msg)
        self.agent.logger.info(f"[{self.agent.jid}] message sent to {peer}")

        self.agent.logger.info(f"Waiting for response from {peer}...")
        reply = await fut
        if reply is None:
            # TODO: Handle timeout or no response on other side?
            self.agent.logger.warning(f"No response received from {peer}.")
        else:
            response_msg = reply

            self.agent.logger.info(f"Processing model weights from {response_msg.sender}.")
            received_weights = self.agent.model.decode(json.loads(response_msg.body))
            self.agent.logger.info("Model weights decoded successfully.")
            # Perform consensus averaging with the received weights
            self.agent.logger.info("Performing consensus averaging with received weights.")
            new_weights = self.agent.weights = self.agent.consensus.average(
                x_i=self.agent.weights,
                x_j=received_weights
            )
            self.agent.weights = new_weights
            self.agent.logger.info("Consensus averaging complete. New weights obtained.")

        self.set_next_state(TRAINING_STATE)