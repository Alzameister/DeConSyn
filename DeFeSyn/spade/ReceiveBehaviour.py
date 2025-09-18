import json
import time
import uuid
from asyncio import to_thread
from collections import OrderedDict
from typing import Optional

from spade.behaviour import CyclicBehaviour
from spade.message import Message

class ReceiveBehaviour(CyclicBehaviour):
    """
    Handles inbound gossip messages:
      - Enqueues valid 'inform' + 'gossip' to agent.queue.
      - Replies with 'gossip-reply' when local weights are encodable;
        otherwise defers by pushing the original msg to agent.pending_gossip_replies.
    """

    async def run(self):
        msg = await self.receive(timeout=0.1)
        if not msg:
            return

        msg_id = msg.get_metadata("msg_id") or f"in-{uuid.uuid4().hex[:6]}"
        mtype = (msg.get_metadata("type") or "").strip()
        perf = (msg.get_metadata("performative") or "").strip()
        sender = str(msg.sender)

        if not (perf == "inform" and mtype == "gossip"):
            return

        version_raw = msg.get_metadata("version")
        try:
            msg_version = int(version_raw) if version_raw is not None else -1
        except Exception:
            msg_version = -1

        self.agent.log.info("RECEIVE: gossip from {} (id={}, ver={})", sender, msg_id, msg_version)
        await self.agent.queue.put(msg)
        self._receive_event(sender, msg_id, msg_version, q_after=int(self.agent.queue.qsize()))

        pkg = await self._try_encode_weights()
        if pkg is None:
            self.agent.log.warning(
                "Delaying gossip-reply to {} (weights not ready). Will flush after training.",
                sender
            )
            self.agent.pending_gossip_replies.append(msg)
            return

        eps_i = self._safe_eps()
        version_sent = str(getattr(self.agent, "last_committed_version", self.agent.current_iteration))
        await self._send_gossip_reply(sender, in_msg_id=msg_id, ver=version_sent, pkg=pkg, eps_i=eps_i)

    # ---------- helpers ----------

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

    async def _send_gossip_reply(self, to_jid: str, *, in_msg_id: str, ver: str, pkg: dict, eps_i: Optional[float]):
        resp = Message(to=str(to_jid))
        resp_id = f"resp-{in_msg_id}"
        resp.set_metadata("performative", "inform")
        resp.set_metadata("type", "gossip-reply")
        resp.set_metadata("content-type", "application/x-ctgan-weights")
        resp.set_metadata("content-encoding", "b85+zlib")
        resp.set_metadata("msg_id", resp_id)
        resp.set_metadata("version", ver)
        resp.set_metadata("in_reply_to", in_msg_id)
        if eps_i is not None:
            resp.set_metadata("eps", f"{eps_i:.12f}")

        # pkg_json = await to_thread(json.dumps, pkg)
        resp.body = pkg

        await self.send(resp)

        bytes_out = len(pkg)
        self.agent.log.info(
            "RESPOND_SEND: to {} (id={}, ver_sent={}, eps_i={}, bytes={})",
            to_jid, resp_id, ver, f"{eps_i:.6f}" if eps_i is not None else "n/a", bytes_out
        )
        self.agent.event.bind(
            event="RESPOND_SEND",
            local_step=self.agent.current_iteration,
            neighbor_id=to_jid,
            out_msg_id=resp_id,
            bytes=int(bytes_out),
            eps_self=(float(eps_i) if eps_i is not None else None),
        ).info("send")

    def _receive_event(self, sender, msg_id, msg_version, q_after, dropped: bool = False):
        try:
            v = int(msg_version)
        except Exception:
            v = -1
        self.agent.event.bind(
            event="RECEIVE",
            local_step=self.agent.current_iteration,
            neighbor_id=str(sender),
            msg_id=str(msg_id),
            msg_version=v,
            queue_len_after=int(q_after),
            dropped=bool(dropped),
        ).info("recv")


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
            await q.put(str(msg.sender))
            self.agent.log.debug("ACK routed token={} from={}", token, msg.sender)
        else:
            self.agent.log.debug("ACK for unknown token={} from={}", token, msg.sender)