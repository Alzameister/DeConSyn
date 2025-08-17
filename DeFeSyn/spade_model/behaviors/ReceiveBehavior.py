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
            await self.agent.queue.put(msg)
            q_after = self.agent.queue.qsize()
            self.agent.log.info("RECEIVE: from {} (id={}, ver={}) → queue={}", sender, msg_id, msg_version, q_after)

            # TODO: Structured RECEIVE event

            # Respond
            pkg = None
            try:
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
                resp.body = json.dumps(pkg)
                await self.send(resp)

                bytes_out = len(resp.body.encode("utf-8"))
                self.agent.log.info("RESPOND_SEND: to {} (id={}, ver_sent={}, bytes={})",
                                    sender, resp_id, version_sent, bytes_out)

                # TODO: Structured RESPOND event
            else:
                self.agent.log.warning("RESPOND: no weights to send yet (train first)")