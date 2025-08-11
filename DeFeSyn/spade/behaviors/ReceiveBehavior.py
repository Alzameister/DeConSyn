from spade.behaviour import CyclicBehaviour


class ReceiveBehavior(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=0.1)
        if msg:
            self.agent.logger.info(f"Received message from {msg.sender}")
            await self.agent.queue.put(msg)
            self.agent.logger.debug("Message added to the queue for future processing.")

