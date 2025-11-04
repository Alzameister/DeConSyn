import asyncio
import contextlib
import time

import torch
from spade.template import Template

from DeConSyn.training_framework.communication.wait_response_behaviour import WaitResponseBehaviour
from DeConSyn.training_framework.fsm.fsm_behaviour import PUSH_STATE, MT_INFORM, T_GOSSIP_WEIGHTS, clear_memory
from DeConSyn.training_framework.fsm.states.state import BaseState


class PullState(BaseState):
    def get_request_agents(self) -> list[str]:
        """
        Returns a list of neighbor keys where 'want_pull' == True in their value dict.
        """
        return [
            k for k, v in self.agent.pending_gossip.items()
            if isinstance(v, dict) and v.get("weights") is not None
        ]

    async def run(self):
        neighbors = self.get_request_agents()
        if not neighbors:
            self.log.info("PULL: no pending pulls → transition PUSH")
            self.set_next_state(PUSH_STATE)
            return

        self.log.info("requests: {}", neighbors)
        self.log.info("PULL: processing {} reciprocal pulls …", len(neighbors))
        self.report("PULL before Consume")

        consumed = 0
        dict_before = len(neighbors)
        t0 = time.perf_counter()
        pkg = self.agent.model.encode()

        for neighbor in neighbors:
            neighbor_data = self.agent.pending_gossip.get(neighbor)
            received = neighbor_data.get("weights")
            eps_j = neighbor_data.get("eps")
            msg_id = neighbor_data.get("msg_id")

            with torch.no_grad():
                self.agent.weights = self.agent.consensus.step_with_neighbor(
                    x_i=self.agent.weights,
                    x_j=received,
                    eps_j=eps_j,
                )
                if self.agent.model:
                    self.agent.model.set_weights(self.agent.weights)

            consumed += 1
            self.log.info(
                "PULL: consensus step with {} (eps_i: {:.6f} → {:.6f}, used eps_j={:.6f})",
                neighbor, self.agent.consensus.prev_eps, self.agent.consensus.get_eps(), eps_j,
            )
            self.ev(
                "PULL_RECV", "ok",
                local_step=self.agent.current_iteration,
                neighbor_id=str(neighbor),
                msg_id=str(msg_id),
                version=int(self.agent.current_iteration),
                timeout=False,
            )
            self.agent.pending_gossip.pop(neighbor, None)

        ms = (time.perf_counter() - t0) * 1000.0
        dict_after = len(self.get_request_agents())

        self.log.info("PULL: averaged {} updates (dict size {}→{})", consumed, dict_before, dict_after)
        self.ev(
            "MIX", "consensus",
            consumed_count=int(consumed),
            queue_before=int(dict_before),
            queue_after=int(dict_after),
            mix_ms=float(ms),
        )

        clear_memory()
        self.report("PULL")
        self.set_next_state(PUSH_STATE)
