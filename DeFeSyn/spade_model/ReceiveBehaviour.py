import json
import time
import uuid
from collections import OrderedDict

from spade.behaviour import CyclicBehaviour
from spade.message import Message


def _LRU(maxlen):
    pass


class ReceiveBehaviour(CyclicBehaviour):
    def __init__(self, dedupe_max=2048):
        super().__init__()
        self._seen = _LRU(maxlen=dedupe_max)

    async def run(self):
        msg = await self.receive(timeout=0.1)
        if msg:
            # --- parse metadata safely (do not mutate msg) ---
            msg_id = msg.get_metadata("msg_id") or f"in-{uuid.uuid4().hex[:6]}"
            mtype = (msg.get_metadata("type") or "").strip()
            performative = (msg.get_metadata("performative") or "").strip()
            sender = str(msg.sender)

            # Only handle 'inform'+'gossip'. Other types are ignored.
            if performative == "inform" and mtype == "gossip":
                version_raw = msg.get_metadata("version")
                try:
                    msg_version = int(version_raw) if version_raw is not None else -1
                except Exception:
                    msg_version = -1

                # Log the receipt (we do NOT enqueue)
                self.agent.log.info(
                    "RECEIVE: gossip from {} (id={}, ver={})",
                    sender, msg_id, msg_version
                )
                await self.agent.queue.put(msg)
                self._emit_receive_event(sender, msg_id, msg_version, q_after=int(self.agent.queue.qsize()))

                # --- prepare reply ('gossip-reply') ---
                pkg = None
                try:
                    # load latest local weights onto the model and encode
                    #self.agent.model.load_weights(self.agent.weights)
                    t0 = time.perf_counter()
                    pkg = self.agent.model.encode()
                    enc_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    self.agent.log.warning("RESPOND: encode failed: {}", e)

                if pkg:

                    # eps_i to share; tolerate consensus not ready yet
                    try:
                        eps_i = float(self.agent.consensus.get_eps())
                    except Exception:
                        eps_i = None

                    resp = Message(to=msg.sender)
                    resp.set_metadata("performative", "inform")
                    resp.set_metadata("type", "gossip-reply")
                    resp.set_metadata("content-type", "application/octet-stream+b64")
                    resp_id = f"resp-{msg_id}"
                    resp.set_metadata("msg_id", resp_id)
                    version_sent = str(getattr(self.agent, "last_committed_version", self.agent.current_iteration))
                    resp.set_metadata("version", version_sent)
                    resp.set_metadata("in_reply_to", msg_id)
                    if eps_i is not None:
                        resp.set_metadata("eps", f"{eps_i:.12f}")
                    resp.body = json.dumps(pkg)

                    await self.send(resp)

                    bytes_out = len(resp.body.encode("utf-8"))
                    self.agent.log.info(
                        "RESPOND_SEND: to {} (id={}, ver_sent={}, eps_i={}, bytes={})",
                        sender, resp_id, version_sent,
                        f"{eps_i:.6f}" if eps_i is not None else "n/a",
                        bytes_out
                    )

                    self.agent.event.bind(
                        event="RESPOND_SEND",
                        local_step=self.agent.current_iteration,
                        neighbor_id=sender,
                        out_msg_id=resp_id,
                        bytes=int(bytes_out),
                        eps_self=(float(eps_i) if eps_i is not None else None)
                    ).info("send")

    def _emit_receive_event(self, sender, msg_id, msg_version, q_after, dropped: bool = False):
        self.agent.event.bind(
            event="RECEIVE",
            local_step=self.agent.current_iteration,
            neighbor_id=sender,
            msg_id=msg_id,
            msg_version=int(msg_version) if isinstance(msg_version, int) else -1,
            queue_len_after=int(q_after),
            dropped=bool(dropped),
        ).info("recv")

    class _LRU:
        """Tiny LRU set for deduping msg_ids."""

        def __init__(self, maxlen=2048):
            self._d = OrderedDict()
            self._maxlen = int(maxlen)

        def __contains__(self, k):
            return k in self._d

        def add(self, k):
            if k in self._d:
                self._d.move_to_end(k, last=True)
                return
            self._d[k] = None
            self._d.move_to_end(k, last=True)
            if len(self._d) > self._maxlen:
                self._d.popitem(last=False)

class BarrierHelloResponder(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=0.1)  # template attached below
        if not msg:
            return
        token = msg.get_metadata("token") or ""
        # reply ACK (echo token back)
        ack = msg.make_reply()
        ack.set_metadata("performative", "inform")
        ack.set_metadata("type", "barrier-ack")
        ack.set_metadata("token", token)
        await self.send(ack)
        self.agent.log.debug(f"HELLO from {msg.sender} → ACK(token={token})")

class BarrierAckRouter(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=0.1)  # template filters by type=barrier-ack
        if not msg:
            return
        token = msg.get_metadata("token") or ""
        q = self.agent.barrier_queues.get(token)
        if q:
            await q.put(str(msg.sender))
            self.agent.log.debug(f"ACK routed token={token} from={msg.sender}")
        else:
            self.agent.log.debug(f"ACK for unknown token={token} from={msg.sender}")