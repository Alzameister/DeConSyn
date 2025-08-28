import asyncio

from spade.behaviour import CyclicBehaviour


class PresenceBehaviour(CyclicBehaviour):
    """
    PresenceBehavior (ACo-L):
    - Auto-subscribe to declared neighbors on startup.
    - Track active neighbors as XMPP presence changes (available/unavailable).
    - On any change, update consensus degree (-> epsilon_i).
    - Periodically poll presence roster as a safety net.
    """
    def __init__(self, poll_secs: float = 2.0):
        super().__init__()
        self.poll_secs = poll_secs
        self._last_active = set()

    async def on_start(self):
        a = self.agent
        a.log.info("PresenceBehavior: starting; subscribing to declared neighbors...")
        for jid in getattr(a, "neighbors", []):
            try:
                a.presence.subscribe(jid)
            except Exception as e:
                a.log.warning("PresenceBehavior: subscribe({}) failed: {}", jid, e)

        # wire presence callbacks (instant updates)
        a.presence.on_available = lambda jid, *args, **kwargs: asyncio.create_task(self._on_available(jid))
        a.presence.on_unavailable = lambda jid, *args, **kwargs: asyncio.create_task(self._on_unavailable(jid))
        a.presence.on_subscribe = lambda jid, *args, **kwargs: asyncio.create_task(self._on_subscribe(jid))
        a.presence.on_subscribed = lambda jid, *args, **kwargs: asyncio.create_task(self._on_subscribed(jid))
        a.presence.on_unsubscribed = lambda jid, *args, **kwargs: asyncio.create_task(self._on_unsubscribed(jid))

        # initialize active set from current roster
        await self._recompute_from_roster(initial=True)

    async def run(self):
        await asyncio.sleep(self.poll_secs)
        await self._recompute_from_roster()

    # ---------- presence event handlers ----------

    async def _on_subscribe(self, jid):
        a = self.agent
        a.log.info("Presence: subscribe from {}", jid)
        a.presence.approve_subscription(jid)
        a.presence.subscribe(jid)

    async def _on_subscribed(self, jid):
        self.agent.log.info("Presence: subscribed with {}", jid)

    async def _on_unsubscribed(self, jid):
        self.agent.log.info("Presence: unsubscribed by {}", jid)
        await self._mark_change()

    async def _on_available(self, jid):
        self.agent.log.info("Presence: {} is AVAILABLE", jid)
        await self._mark_change()

    async def _on_unavailable(self, jid):
        self.agent.log.info("Presence: {} is UNAVAILABLE", jid)
        await self._mark_change()

    # ---------- internals ----------

    async def _mark_change(self):
        await asyncio.sleep(0.05)
        await self._recompute_from_roster()

    async def _recompute_from_roster(self, initial: bool = False):
        a = self.agent
        contacts = a.presence.get_contacts()
        active = {str(jid) for jid, c in contacts.items() if c.is_available()}

        if active != self._last_active or initial:
            self._last_active = active
            a.active_neighbors = active
            deg = len(active)

            # Update consensus degree
            try:
                a.consensus.set_degree(deg)
                eps = float(a.consensus.get_eps())
            except Exception:
                eps = None

            self.agent.log.info(
                "Presence: active_neighbors={} | degree={} | eps_i={}",
                sorted(list(active)), deg, f"{eps:.6f}" if eps is not None else "n/a"
            )