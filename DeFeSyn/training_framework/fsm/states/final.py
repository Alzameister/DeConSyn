import asyncio
import contextlib

import torch
from spade.template import Template

from DeFeSyn.training_framework.communication.wait_response_behaviour import WaitResponseBehaviour
from DeFeSyn.training_framework.fsm.fsm_behaviour import MT_INFORM, T_GOSSIP_WEIGHTS, clear_memory
from DeFeSyn.training_framework.fsm.states.state import BaseState


class FinalState(BaseState):
    async def run(self):
        self.log.warning("FSM completed. Entering final state: serving and participating in consensus.")
        self.agent.is_final = True

        while True:
            neighbors = [
                k for k, v in self.agent.pending_gossip.items()
                if isinstance(v, dict) and v.get("want_pull") is True
            ]

            for neighbor in neighbors:
                rid, _ = await self._send_gossip_request(neighbor, self.agent.current_iteration, kind="pull")
                fut = asyncio.get_running_loop().create_future()
                waiter = WaitResponseBehaviour(fut, neighbor, timeout=60.0)
                self.agent.add_behaviour(
                    waiter,
                    Template(metadata={"performative": MT_INFORM, "type": T_GOSSIP_WEIGHTS, "rid": rid})
                )
                reply = await fut
                with contextlib.suppress(Exception):
                    waiter.kill()

                if reply and reply.body:
                    received = self._decode_weights(reply.body)
                    eps_j = self._msg_eps(reply, fallback=self.agent.consensus.get_eps())
                    with torch.no_grad():
                        self.agent.weights = self.agent.consensus.step_with_neighbor(
                            x_i=self.agent.weights,
                            x_j=received,
                            eps_j=eps_j,
                        )
                        if self.agent.model:
                            self.agent.model.set_weights(self.agent.weights)
                    self.log.warning("Received updated weights from {}", neighbor)
                self.agent.pending_gossip.pop(neighbor, None)
            clear_memory()
            await asyncio.sleep(2)