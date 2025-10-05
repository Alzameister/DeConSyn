import asyncio

from spade.behaviour import OneShotBehaviour


class WaitResponseBehaviour(OneShotBehaviour):
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
