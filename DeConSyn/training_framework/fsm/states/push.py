import asyncio
import contextlib
import gc

import torch
from spade.template import Template

from DeConSyn.training_framework.communication.wait_response_behaviour import WaitResponseBehaviour
from DeConSyn.training_framework.fsm.fsm_behaviour import pick_random_peer, TRAINING_STATE, MT_INFORM, T_GOSSIP_WEIGHTS, \
    clear_memory
from DeConSyn.training_framework.fsm.states.state import BaseState


class PushState(BaseState):
    async def run(self):
        it = self.agent.current_iteration
        active = self._active_neighbors()
        peer = pick_random_peer(active)
        self.log.info("PUSH: active neighbors: {}", active)
        self.log.info("PUSH: peer {}", peer)

        if not peer:
            self.log.warning("PUSH: no available neighbors; skipping this round")
            self._persist_weights(it)
            self._persist_model(it)
            self.set_next_state(TRAINING_STATE)
            return

        rid, _ver = await self._send_gossip_request(peer, it, kind="push")

        fut = asyncio.get_running_loop().create_future()
        waiter = WaitResponseBehaviour(fut, peer)
        self.agent.add_behaviour(
            waiter,
            Template(metadata={"performative": MT_INFORM, "type": T_GOSSIP_WEIGHTS, "rid": rid})
        )

        self.log.info("Waiting for weights from {} (rid={})", peer, rid)
        reply = await fut
        with contextlib.suppress(Exception):
            waiter.kill()

        if not reply:
            self.log.warning("No weights from {}", peer)
            self._persist_weights(it)
            self._persist_model(it)
            self.set_next_state(TRAINING_STATE)
            return

        self.log.info("Received weights from {}", peer)
        try:
            payload = reply.body
            if payload is None:
                self.log.error("PUSH: missing payload from %s (rid=%s)", peer, rid)
                # decide: retry, pick another neighbor, or skip consensus this round
                return
            received = self._decode_weights(reply.body)
        finally:
            reply.body = None

        eps_j = self._msg_eps(reply, fallback=self.agent.consensus.get_eps())
        with torch.no_grad():
            new_w = self.agent.consensus.step_with_neighbor(
                x_i=self.agent.weights, x_j=received, eps_j=eps_j
            )

        if new_w is not None:
            self.agent.weights = new_w
            if self.agent.model:
                self.agent.model.set_weights(self.agent.weights)
        else:
            self.log.warning("PUSH: consensus step returned None → skipping weight update")

        def _valid_weights(w) -> bool:
            if not isinstance(w, dict):
                return False
            # CTGAN: expects generator/discriminator dicts
            if "generator" in w and "discriminator" in w:
                return (
                        isinstance(w["generator"], dict)
                        and isinstance(w["discriminator"], dict)
                        and all(v is not None for v in w["generator"].values())
                        and all(v is not None for v in w["discriminator"].values())
                )
            # TabDDPM: expects flat dict of tensors
            return all(torch.is_tensor(v) and v is not None for v in w.values())

        if not _valid_weights(self.agent.weights):
            self.log.warning(
                "PUSH: consensus produced invalid weights → skipping persist and forcing cold start next round")

        self.log.info(
            "PUSH: consensus step applied (eps_i: {:.6f} → {:.6f}, used eps_j={:.6f})",
            self.agent.consensus.prev_eps, self.agent.consensus.get_eps(), eps_j,
        )

        try:
            v = int(reply.get_metadata("version"))
        except Exception:
            v = -1
        self.ev(
            "PUSH_RECV", "ok",
            local_step=self.agent.current_iteration,
            neighbor_id=str(reply.sender),
            msg_id=str(reply.get_metadata("msg_id")),
            version=v,
            timeout=False,
        )

        self._persist_weights(it)
        self._persist_model(it)

        with contextlib.suppress(Exception):
            waiter.kill()

        # del waiter, fut, reply, received
        gc.collect()
        clear_memory()
        self.report("PUSH")
        self.set_next_state(TRAINING_STATE)

    def _push_event(self, it: int, ms: float):
        self.agent.event.bind(
            event="PUSH",
            local_step=it,
            epoch_ms=float(ms),
        ).info("push")

    def _receive_event(self, sender, msg_id, msg_version, timeout: bool = False):
        try:
            v = int(msg_version)
        except Exception:
            v = -1
        self.agent.event.bind(
            event="PUSH_RECV",
            local_step=self.agent.current_iteration,
            neighbor_id=str(sender),
            msg_id=str(msg_id),
            version=v,
            timeout=timeout
        ).info("ok" if not timeout else "timeout")
