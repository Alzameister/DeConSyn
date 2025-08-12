import json

from spade.behaviour import CyclicBehaviour, OneShotBehaviour


class ReceiveBehavior(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=0.1)
        if msg:
            self.agent.logger.info(f"Received message from {msg.sender}")
            await self.agent.queue.put(msg)
            self.agent.logger.debug("Message added to the queue for future processing.")

            # Send weights back to the sender
            self.agent.logger.info(f"Sending weights back to {msg.sender}...")
            pkg = self.agent.model.encode()
            if not pkg:
                self.agent.logger.warning("No weights to send back. Need to complete training first")
                return

            response_msg = msg.make_reply()
            response_msg.set_metadata("performative", "inform")
            response_msg.set_metadata("type", "gossip-reply")
            response_msg.set_metadata("content-type", "application/octet-stream+b64")
            response_msg.body = json.dumps(pkg)
            await self.send(response_msg)
            self.agent.logger.info(f"Response sent to {msg.sender} with new weights.")

class PushReceiveBehavior(OneShotBehaviour):
    async def run(self):
        msg = await self.receive(timeout=0.1)
        if msg:
            self.agent.logger.info(f"Received message from {msg.sender}")
            await self.agent.push_queue.put(msg)