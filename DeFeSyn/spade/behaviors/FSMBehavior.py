import json
import logging
import random

from spade.behaviour import FSMBehaviour, State
from spade.message import Message

from DeFeSyn.models.CTGAN.CTGAN import CTGANModel

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

    # TODO: Only train if we have received new weights?
    async def run(self):
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

                    # TODO: Send new weights back to sender or original weights before averaging?
                    self.agent.logger.info(f"Sending weights back to {msg.sender}...")
                    pkg = self.agent.model.encode()
                    response_msg = Message(to=str(msg.sender))
                    response_msg.set_metadata("performative", "inform")
                    response_msg.set_metadata("type", "gossip-reply")
                    response_msg.set_metadata("content-type", "application/octet-stream+b64")
                    response_msg.body = json.dumps(pkg)
                    await self.send(response_msg)
                    self.agent.logger.info(f"Response sent to {msg.sender} with new weights.")

        self.set_next_state(PUSH_STATE)

class PushState(State):
    async def run(self):
        # TODO
        contacts = self.agent.presence.get_contacts()
        neighbors = [jid for jid, c in contacts.items() if c.is_available()]
        peer = random.choice(neighbors)

        self.agent.logger.info(f"Pushing model weights to {peer}...")
        pkg = self.agent.model.encode()
        msg = Message(to=str(peer))
        msg.set_metadata("performative", "inform")
        msg.set_metadata("type", "gossip")
        msg.set_metadata("content-type", "application/octet-stream+b64")
        msg.body = json.dumps(pkg)
        await self.send(msg)
        self.agent.logger.info(f"[{self.agent.jid}] message sent to {peer}")

        # TODO: Wait for response from the peer
        self.agent.logger.info(f"Waiting for response from {peer}...")
        while self.agent.push_queue.empty():
            await self.agent.sleep(0.1)

        response_msg = await self.agent.push_queue.get()
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