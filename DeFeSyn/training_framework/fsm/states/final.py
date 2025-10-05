import contextlib

from DeFeSyn.training_framework.fsm.states.state import BaseState


class FinalState(BaseState):
    async def run(self):
        self.log.info("FSM completed. Stopping agent.")
        with contextlib.suppress(Exception):
            self.agent.presence.set_unavailable()
