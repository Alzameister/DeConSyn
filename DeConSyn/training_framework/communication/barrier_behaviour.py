from spade.behaviour import CyclicBehaviour

from DeConSyn.training_framework.communication.presence_behaviour import PresenceBehaviour


class BarrierHelloBehaviour(CyclicBehaviour):
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
