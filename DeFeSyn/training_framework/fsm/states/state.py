import uuid
from abc import ABC

import psutil
import torch
from spade.behaviour import State
from spade.message import Message

from DeFeSyn.io.io import make_path, save_weights_pt, save_model_pickle
from DeFeSyn.training_framework.fsm.fsm_behaviour import parse_float, MT_INFORM, T_BARRIER_HELLO, T_GOSSIP_REQ, \
    T_GOSSIP_WEIGHTS, _bytes_len


class BaseState(State, ABC):
    @property
    def log(self):
        return self.agent.log

    def ev(self, event: str, msg: str = "info", **fields):
        """Safe wrapper around agent.event.bind(...).info(...)."""
        try:
            self.agent.event.bind(event=event, **fields).info(msg)
        except Exception as e:
            self.agent.log.debug("ev() failed for event='{}': {}", event, e)

    def _active_neighbors(self) -> set[str]:
        active = getattr(self.agent, "active_neighbors", None)
        if isinstance(active, set):
            self.agent.log.info("Active neighbors: {}", active)
            return active
        contacts = self.agent.presence.get_contacts()
        return {str(jid) for jid, c in contacts.items() if c.is_available()}

    # ---- Encoding helpers ----
    def _encode_weights(self) -> str:
        if not getattr(self.agent, "model", None) or not self.agent.model.is_trained():
            raise RuntimeError("Model not trained; cannot encode weights.")
        blob = self.agent.model.encode()  # returns base85+zlib string now
        if not blob:
            raise RuntimeError("Encoding returned empty payload; model snapshot missing.")
        return blob

    def _decode_weights(self, body: str) -> dict:
        return self.agent.model.decode(body)

    def _msg_eps(self, msg, fallback: float) -> float:
        return parse_float(msg.get_metadata("eps"), default=fallback)

    def report(self, label: str = ""):
        rss_mb = psutil.Process().memory_info().rss / 1024 ** 2
        gpu_mb = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
        behaviours = getattr(self.agent, "behaviours", None)
        num_behaviours = len(behaviours) if behaviours is not None else 0
        self.log.info("{} STATE MEM: rss={:.1f}MB gpu={:.1f}MB behaviours={}",
                      label, rss_mb, gpu_mb, num_behaviours)

    # ---- Persistence helpers ----
    def _persist_weights(self, it: int):
        should = (self.agent.current_iteration % 10 == 0) or (self.agent.current_iteration == self.agent.max_iterations) or (self.agent.current_iteration == 1)
        if not should:
            return
        p = make_path(
            run_id=self.agent.run_id,
            node_id=self.agent.id,
            iteration=it,
            phase="weights",
            ext="pt",
            repo_root=self.agent.repo_dir,
        )
        save_weights_pt(state_dict=self.agent.weights, path=p)

    def _persist_model(self, it: int):
        should = (self.agent.current_iteration % 25 == 0) or (self.agent.current_iteration == self.agent.max_iterations)
        if not should:
            return
        p = make_path(
            run_id=self.agent.run_id,
            node_id=self.agent.id,
            iteration=it,
            phase="model",
            ext="pkl",
            repo_root=self.agent.repo_dir,
        )
        pt = make_path(
            run_id=self.agent.run_id,
            node_id=self.agent.id,
            iteration=it,
            phase="model",
            ext="pt",
            repo_root=self.agent.repo_dir
        )
        model = self.agent.model
        if hasattr(model, "diffusion"):
            save_model_pickle(model=model.diffusion, path=p)
        elif hasattr(model, "model"):
            save_model_pickle(model=model.model, path=p)

    # ---- Barrier helpers ----
    async def _send_barrier_hello(self, to_jid: str, token: str):
        msg = Message(to=str(to_jid))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_BARRIER_HELLO)
        msg.set_metadata("token", token)
        await self.send(msg)

    async def _send_gossip_request(self, peer: str, it: int, kind: str):
        rid = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)

        msg = Message(to=str(peer))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_GOSSIP_REQ)
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("rid", rid)
        msg.set_metadata("version", version)
        msg.set_metadata("kind", kind)
        await self.send(msg)
        self.ev("REQUEST", "send", neighbor=str(peer), msg_id=rid, ver=int(version), kind=str(kind))

        return rid, version

    async def _send_gossip_weights(self, peer: str, it: int, blob: str, rid: str):
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)
        msg = Message(to=str(peer))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_GOSSIP_WEIGHTS)
        msg.set_metadata("content-type", "application/x-weights")
        msg.set_metadata("content-encoding", "b85+zlib")
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("rid", rid)
        msg.set_metadata("version", version)
        msg.set_metadata("eps", f"{self.agent.consensus.get_eps():.12f}")
        msg.body = blob
        await self.send(msg)
        self.ev("WEIGHTS", "send", neighbor=str(peer), msg_id=msg_id, ver=int(version),
                bytes=int(_bytes_len(msg.body or b"")), rid=rid)
        return msg_id, version
