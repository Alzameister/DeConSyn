import asyncio
import contextlib
import time
import uuid
from typing import Optional

from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

from DeFeSyn.spade.communication.PresenceBehaviour import PresenceBehaviour


class ReceiveAckBehaviour(CyclicBehaviour):
    """Handles inbound gossip messages:"""
    async def _try_encode_weights(self) -> Optional[dict]:
        try:
            t0 = time.perf_counter()
            pkg = self.agent.model.encode()
            _ = (time.perf_counter() - t0) * 1000.0
            return pkg if pkg else None
        except Exception as e:
            self.agent.log.warning("RESPOND: encode failed: {}", e)
            return None

    def _safe_eps(self) -> Optional[float]:
        try:
            return float(self.agent.consensus.get_eps())
        except Exception:
            return None

    async def run(self):
        msg = await self.receive(timeout=0.1)
        if not msg:
            return

        msg_id = msg.get_metadata("msg_id") or f"in-{uuid.uuid4().hex[:6]}"
        mtype = (msg.get_metadata("type") or "").strip()
        perf = (msg.get_metadata("performative") or "").strip()
        kind = (msg.get_metadata("kind") or "push").strip()
        sender = PresenceBehaviour.strip_jid(msg.sender)


        if not (perf == "inform" and mtype == "gossip-req"):
            return

        rid = msg.get_metadata("rid") or msg.get_metadata("msg_id") or f"rid-{uuid.uuid4().hex[:8]}"
        version_raw = msg.get_metadata("version")
        try:
            msg_version = int(version_raw) if version_raw is not None else -1
        except Exception:
            msg_version = -1

        if kind == "push":
            self.agent.pending_gossip[sender] = {"want_pull": True, "seen_at": self.agent.current_iteration, "rid": rid}
        else:
            pass

        blob = await self._try_encode_weights()
        if not blob:
            self.agent.log.warning(
                "Delaying gossip-reply to {} (weights not ready). Will flush after training.",
                sender
            )
            self.agent.pending_gossip_replies.append(msg)
            return

        resp = Message(to=sender)
        resp.set_metadata("performative", "inform")
        resp.set_metadata("type", "gossip-weights")
        resp.set_metadata("content-type", "application/x-ctgan-weights")
        resp.set_metadata("content-encoding", "b85+zlib")
        resp.set_metadata("msg_id", f"resp-{rid}")
        resp.set_metadata("rid", rid)
        resp.set_metadata("version", str(self.agent.current_iteration))
        with contextlib.suppress(Exception):
            resp.set_metadata("eps", f"{float(self.agent.consensus.get_eps()):.12f}")
        resp.body = blob
        await self.send(resp)
        bytes_out = len(blob) if blob else 0

        self.agent.log.info("RECEIVE: gossip-req from {}, kind={} (id={}, ver={}, rid={})", sender, kind, msg_id, msg_version, rid)
        self.agent.log.info("REQUEST_RESPOND: to {} (id={}, rid={}, ver_sent={}, bytes={}, eps={})",
                            sender, f"resp-{rid}", rid, self.agent.current_iteration,
                            bytes_out,
                            f"{float(self.agent.consensus.get_eps()):.6f}" if self.agent.consensus else "n/a")
        self.agent.event.bind(
            event="RECEIVE",
            local_step=self.agent.current_iteration,
            neighbor_id=sender,
            msg_id=msg_id,
            rid=rid,
            msg_version=msg_version,
            pending_length=int(len(self.agent.pending_gossip.values())),
            dropped=False
        ).info("recv")

class WaitResponse(OneShotBehaviour):
    """Small waiter that only resolves the future when a reply arrives or timeout/peer-offline."""

    def __init__(self, fut, peer_jid: str, poll_interval: float = 1.0, timeout: float = 120.0):
        super().__init__()
        self.fut = fut
        self.peer_jid = peer_jid
        self.poll_interval = poll_interval
        self.timeout = timeout

    def _peer_available(self) -> bool:
        active = getattr(self.agent, "active_neighbors", None)
        if isinstance(active, set):
            return self.peer_jid in active
        contact = self.agent.presence.get_contacts().get(self.peer_jid)
        return bool(contact and contact.is_available())

    async def run(self):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout

        if self.timeout == 0:
            if not self._peer_available():
                self.agent.log.warning("WaitResponse: peer {} unavailable (immediate)", self.peer_jid)
                self.fut.set_result(None)
                return
            msg = await self.receive(timeout=0)
            self.fut.set_result(msg)
            return

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self.agent.log.warning("WaitResponse: timeout waiting for {}", self.peer_jid)
                self.fut.set_result(None)
                return

            msg = await self.receive(timeout=min(self.poll_interval, remaining))
            if msg:
                self.fut.set_result(msg)
                return


class BarrierHelloResponder(CyclicBehaviour):
    """Responds to 'barrier-hello' with 'barrier-ack'."""

    async def run(self):
        msg = await self.receive(timeout=0.1)
        if not msg:
            return
        token = msg.get_metadata("token") or ""
        ack = msg.make_reply()
        ack.set_metadata("performative", "inform")
        ack.set_metadata("type", "barrier-ack")
        ack.set_metadata("token", token)
        await self.send(ack)
        self.agent.log.debug("HELLO from {} → ACK(token={})", msg.sender, token)


class BarrierAckRouter(CyclicBehaviour):
    """Routes 'barrier-ack' messages to the queue registered for their token."""

    async def run(self):
        msg = await self.receive(timeout=0.1)
        if not msg:
            return
        token = msg.get_metadata("token") or ""
        q = self.agent.barrier_queues.get(token)
        if q:
            await q.put(PresenceBehaviour.strip_jid(msg.sender))
            self.agent.log.debug("ACK routed token={} from={}", token, msg.sender)
        else:
            self.agent.log.debug("ACK for unknown token={} from={}", token, msg.sender)