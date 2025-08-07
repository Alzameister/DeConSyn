import logging

from spade.behaviour import FSMBehaviour, State

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
        self.data = None

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

        ctgan = CTGANModel(
            data=data,
            discrete_columns=discrete_cols,
            epochs=self.epochs
        )
        self.agent.logger.info("Starting CTGAN training…")
        ctgan.train()
        self.agent.logger.info("CTGAN training complete.")
        # TODO: Save model + weights for Consensus
        weights = ctgan.get_weights()
        self.agent.logger.info("Weights obtained from CTGAN model.")

        self.set_next_state(PULL_STATE)

class PullState(State):
    # TODO: Docstring
    async def run(self):
        # TODO: Implement
        self.set_next_state(PUSH_STATE)

class PushState(State):
    # TODO: Docstring
    async def run(self):
        # TODO: Implement
        self.set_next_state(TRAINING_STATE)

class ReceiveState(State):
    # TODO: Docstring
    async def run(self):
        # TODO: Implement
        raise NotImplementedError("Receive state is not implemented yet.")