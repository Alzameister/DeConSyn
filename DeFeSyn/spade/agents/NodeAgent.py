import asyncio

import spade
from spade.agent import Agent
from DeFeSyn.spade.behaviors.FSMBehavior import *


class NodeAgent(Agent):
    """
    NodeAgent is an agent that implements a finite state machine (FSM) to run a Decentralized Federated Learning (DeFeSyn) framework.
    """

    async def setup(self):
        fsm = NodeFSMBehaviour()
        fsm.add_state(name=TRAINING_STATE, state=TrainingState(), initial=True)
        fsm.add_state(name=PULL_STATE, state=PullState())
        fsm.add_state(name=PUSH_STATE, state=PushState())
        fsm.add_state(name=RECEIVE_STATE, state=ReceiveState())

        fsm.add_transition(source=TRAINING_STATE, dest=PULL_STATE)
        fsm.add_transition(source=TRAINING_STATE, dest=PUSH_STATE)
        fsm.add_transition(source=PULL_STATE, dest=PUSH_STATE)
        fsm.add_transition(source=PUSH_STATE, dest=TRAINING_STATE)

        self.add_behaviour(fsm)


async def main():
    agent = NodeAgent("agent@localhost", "password")
    await agent.start()
    print("Agent started")

    await spade.wait_until_finished(agent)
    await agent.stop()
    print("Agent finished")

if __name__ == "__main__":
    try:
        spade.run(main())
    except RuntimeError as e:
        # Fallback for environments with a running event loop (like PyCharm)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())