from spade.behaviour import FSMBehaviour, State

TRAINING_STATE = "TRAINING_STATE"
PULL_STATE = "PULL_STATE"
PUSH_STATE = "PUSH_STATE"
RECEIVE_STATE = "RECEIVE_STATE"

class NodeFSMBehaviour(FSMBehaviour):
    # TODO: Docstring
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()

class TrainingState(State):
    # TODO: Docstring
    async def run(self):
        # TODO: Implement
        raise NotImplementedError("Training state is not implemented yet.")

class PullState(State):
    # TODO: Docstring
    async def run(self):
        # TODO: Implement
        raise NotImplementedError("Pull state is not implemented yet.")

class PushState(State):
    # TODO: Docstring
    async def run(self):
        # TODO: Implement
        raise NotImplementedError("Push state is not implemented yet.")

class ReceiveState(State):
    # TODO: Docstring
    async def run(self):
        # TODO: Implement
        raise NotImplementedError("Receive state is not implemented yet.")