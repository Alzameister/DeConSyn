import json
import uuid

from spade.behaviour import CyclicBehaviour, OneShotBehaviour


class ReceiveBehavior(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=0.1)
        if msg:
            msg_id = msg.get_metadata("msg_id") or f"in-{uuid.uuid4().hex[:6]}"
            msg.set_metadata("msg_id", msg_id)
            version = msg.get_metadata("version")

            try:
                msg_version = int(version) if version is not None else -1
            except Exception:
                msg_version = -1

            sender = str(msg.sender)

            # --- NEW: parse neighbor epsilon (keep for logging/analytics; don't mutate local eps here)
            try:
                incoming_eps = float(msg.get_metadata("eps")) if msg.get_metadata("eps") is not None else None
            except (TypeError, ValueError):
                incoming_eps = None

            await self.agent.queue.put(msg)
            q_after = self.agent.queue.qsize()

            self.agent.log.info(
                "RECEIVE: from {} (id={}, ver={}, eps_j={}) → queue={}",
                sender, msg_id, msg_version,
                f"{incoming_eps:.6f}" if incoming_eps is not None else "n/a",
                q_after
            )

            self.agent.event.bind(
                event="RECEIVE",
                local_step=self.agent.current_iteration,
                neighbor_id=sender,
                msg_id=msg_id,
                msg_version=msg_version,
                staleness=(self.agent.current_iteration - msg_version) if msg_version >= 0 else None,
                queue_len_after=int(q_after),
                eps_neighbor=(float(incoming_eps) if incoming_eps is not None else None)
            ).info("enqueue")

            # Respond
            pkg = None
            try:
                self.agent.model.load_weights(self.agent.weights)
                pkg = self.agent.model.encode()
            except Exception as e:
                self.agent.log.warning("RESPOND: encode failed: {}", e)

            if pkg:
                resp = msg.make_reply()
                resp.set_metadata("performative", "inform")
                resp.set_metadata("type", "gossip-reply")
                resp.set_metadata("content-type", "application/octet-stream+b64")
                resp_id = f"resp-{msg_id}"
                resp.set_metadata("msg_id", resp_id)
                version_sent = str(getattr(self.agent, "last_committed_version", self.agent.current_iteration))
                resp.set_metadata("version", version_sent)
                resp.set_metadata("in_reply_to", msg_id)
                # --- NEW: include your current epsilon so the peer can apply min-rule on their side
                try:
                    eps_i = float(self.agent.consensus.get_eps())
                except Exception:
                    eps_i = None
                if eps_i is not None:
                    resp.set_metadata("eps", f"{eps_i:.12f}")  # high precision, string metadata
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
            else:
                self.agent.log.warning("RESPOND: no weights to send yet (train first)")

class BarrierReceiver(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=1.0)
        if not msg:
            return
        if (msg.get_metadata("type") == "barrier"
                and msg.get_metadata("stage") == "start"
                and (msg.body or "").strip() == "ready"):
            self.agent.start_ready_from.add(str(msg.sender))
            if self.agent.start_expected.issubset(self.agent.start_ready_from):
                if not self.agent.start_ready_event.is_set():
                    self.agent.start_ready_event.set()