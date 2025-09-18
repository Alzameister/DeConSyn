import asyncio
import contextlib
from typing import Set, Iterable

from spade.behaviour import CyclicBehaviour


class PresenceBehaviour(CyclicBehaviour):
    """
    PresenceBehavior (ACo-L):
    - Auto-subscribe to declared neighbors on startup.
    - Track active neighbors as XMPP presence changes (available/unavailable).
    - Optionally update consensus degree on any change (toggle with update_degree flag).
    - Periodically poll presence roster as a safety net.
    """
    PRESENCE_DEBOUNCE_SEC = 0.05  # small delay before recompute after events

    def __init__(self, poll_secs: float = 5.0, *, update_degree: bool = False):
        super().__init__()
        self.poll_secs = poll_secs
        self.update_degree = update_degree
        self._last_active: Set[str] = set()

    async def on_start(self):
        a = self.agent
        a.log.info("PresenceBehavior: starting; subscribing to declared neighbors...")

        if not hasattr(a, "active_neighbors"):
            a.active_neighbors = set()

        await self._subscribe_to_neighbors(getattr(a, "neighbors", []))

        with contextlib.suppress(Exception):
            a.presence.set_available()

        self._wire_presence_callbacks()

        await self._recompute_from_roster(initial=True)

    async def on_end(self):
        a = self.agent
        self._clear_presence_callbacks()
        with contextlib.suppress(Exception):
            a.presence.set_unavailable()

    async def run(self):
        await asyncio.sleep(self.poll_secs)
        await self._recompute_from_roster()

    # ---------- presence event handlers ----------

    async def _on_subscribe(self, jid):
        a = self.agent
        a.log.info("Presence: subscribe from {}", jid)
        with contextlib.suppress(Exception):
            a.presence.approve_subscription(jid)
        with contextlib.suppress(Exception):
            a.presence.subscribe(jid)

    async def _on_subscribed(self, jid):
        self.agent.log.info("Presence: subscribed with {}", jid)

    async def _on_unsubscribed(self, jid):
        self.agent.log.info("Presence: unsubscribed by {}", jid)

    async def _on_available(self, jid):
        a = self.agent
        jid_str = self._strip_jid(jid)

        # If neighbors are declared, ignore non-neighbors
        neighbors = set(map(str, getattr(a, "neighbors", [])))
        if neighbors and jid_str not in neighbors:
            a.log.warning("Presence: AVAILABLE from non-neighbor {}; ignoring", jid_str)
            return

        if jid_str not in a.active_neighbors:
            a.active_neighbors.add(jid_str)
            a.log.info("Presence: {} is now AVAILABLE", jid_str)
            await self._report_active()

    async def _on_unavailable(self, jid):
        a = self.agent
        jid_str = self._strip_jid(jid)

        if jid_str in a.active_neighbors:
            a.active_neighbors.remove(jid_str)
            a.log.info("Presence: {} went UNAVAILABLE", jid_str)
            await self._report_active()

    # ---------- internals ----------

    async def _subscribe_to_neighbors(self, neighbors: Iterable[str]):
        if not neighbors:
            return
        a = self.agent
        tasks = []
        for jid in neighbors:
            async def _sub(j=jid):
                with contextlib.suppress(Exception):
                    a.presence.subscribe(j)

            tasks.append(asyncio.create_task(_sub()))
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks)

    def _wire_presence_callbacks(self):
        a = self.agent
        a.presence.on_available = lambda jid, *_, **__: asyncio.create_task(self._on_available(jid))
        a.presence.on_unavailable = lambda jid, *_, **__: asyncio.create_task(self._on_unavailable(jid))
        a.presence.on_subscribe = lambda jid, *_, **__: asyncio.create_task(self._on_subscribe(jid))
        a.presence.on_subscribed = lambda jid, *_, **__: asyncio.create_task(self._on_subscribed(jid))
        a.presence.on_unsubscribed = lambda jid, *_, **__: asyncio.create_task(self._on_unsubscribed(jid))

    def _clear_presence_callbacks(self):
        a = self.agent
        with contextlib.suppress(Exception):
            a.presence.on_available = None
            a.presence.on_unavailable = None
            a.presence.on_subscribe = None
            a.presence.on_subscribed = None
            a.presence.on_unsubscribed = None


    async def _mark_change(self):
        await asyncio.sleep(0.05)
        await self._recompute_from_roster()

    async def _recompute_from_roster(self, initial: bool = False):
        a = self.agent
        contacts = a.presence.get_contacts()

        current_active = {
            self._strip_jid(jid)
            for jid, c in contacts.items()
            if c.is_available()
        }

        # Keep only declared neighbors (if any)
        declared = set(map(str, getattr(a, "neighbors", [])))
        if declared:
            current_active &= declared

        if initial or current_active != self._last_active:
            self._last_active = set(current_active)
            a.active_neighbors = set(current_active)

            if self.update_degree:
                with contextlib.suppress(Exception):
                    a.consensus.set_degree(len(a.active_neighbors))

            eps = a.consensus.get_eps()
            a.log.info(
                "Presence: active_neighbors={} | degree={} | eps_i={}",
                sorted(a.active_neighbors), len(a.active_neighbors),
                f"{eps:.6f}" if eps is not None else "n/a"
            )

    async def _report_active(self):
        await asyncio.sleep(self.PRESENCE_DEBOUNCE_SEC)
        await self._recompute_from_roster()

    # ---------- utils ----------

    @staticmethod
    def _strip_jid(jid) -> str:
        return str(jid).split("/")[0]