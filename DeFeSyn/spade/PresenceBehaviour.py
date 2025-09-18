import asyncio
import contextlib

from spade.behaviour import CyclicBehaviour


class PresenceBehaviour(CyclicBehaviour):
    """
    PresenceBehavior (ACo-L):
    - Auto-subscribe to declared neighbors on startup.
    - Track active neighbors as XMPP presence changes (available/unavailable).
    - On any change, update consensus degree (-> epsilon_i).
    - Periodically poll presence roster as a safety net.
    """
    def __init__(self, poll_secs: float = 5.0):
        super().__init__()
        self.poll_secs = poll_secs
        self._last_active = set()

    async def on_start(self):
        a = self.agent
        a.log.info("PresenceBehavior: starting; subscribing to declared neighbors...")
        if not hasattr(a, "active_neighbors"):
            a.active_neighbors = set()

        for jid in getattr(a, "neighbors", []):
            try:
                a.presence.subscribe(jid)
            except Exception as e:
                a.log.warning("PresenceBehavior: subscribe({}) failed: {}", jid, e)

        try:
            a.presence.set_available()
        except Exception as e:
            a.log.warning("PresenceBehavior: set_available() failed: {}", e)

        # wire presence callbacks (instant updates)
        a.presence.on_available = lambda jid, *args, **kwargs: asyncio.create_task(self._on_available(jid))
        a.presence.on_unavailable = lambda jid, *args, **kwargs: asyncio.create_task(self._on_unavailable(jid))
        a.presence.on_subscribe = lambda jid, *args, **kwargs: asyncio.create_task(self._on_subscribe(jid))
        a.presence.on_subscribed = lambda jid, *args, **kwargs: asyncio.create_task(self._on_subscribed(jid))
        a.presence.on_unsubscribed = lambda jid, *args, **kwargs: asyncio.create_task(self._on_unsubscribed(jid))

        # initialize active set from current roster
        await self._recompute_from_roster(initial=True)

    async def on_end(self):
        a = self.agent
        with contextlib.suppress(Exception):
            a.presence.on_available = None
            a.presence.on_unavailable = None
            a.presence.on_subscribe = None
            a.presence.on_subscribed = None
            a.presence.on_unsubscribed = None
        with contextlib.suppress(Exception):
            a.presence.set_unavailable()

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

    async def _on_available(self, jid):
        a = self.agent
        jid_str = self.strip_jid(jid)

        if hasattr(a, "neighbors") and a.neighbors:
            if jid_str not in set(map(str, a.neighbors)):
                a.log.warning("Presence: AVAILABLE from non-neighbor {}; ignoring", jid_str)
                return

        if jid_str not in a.active_neighbors:
            a.active_neighbors.add(jid_str)
            a.log.info("Presence: {} is now AVAILABLE", jid_str)
            await self._report_active()

    async def _on_unavailable(self, jid):
        a = self.agent
        jid_str = self.strip_jid(jid)

        if jid_str in a.active_neighbors:
            a.active_neighbors.remove(jid_str)
            a.log.info("Presence: {} went UNAVAILABLE", jid_str)
            await self._report_active()

    # ---------- internals ----------

    async def _mark_change(self):
        await asyncio.sleep(0.05)
        await self._recompute_from_roster()

    async def _recompute_from_roster(self, initial: bool = False):
        a = self.agent
        contacts = a.presence.get_contacts()
        active = {
            self.strip_jid(jid)
            for jid, c in contacts.items()
            if c.is_available()
        }

        # keep only declared neighbors
        neighbor_set = set(getattr(a, "neighbors", []))
        new_active = active & neighbor_set
        if new_active != getattr(self, "_last_active", set()) or initial:
            self._last_active = set(new_active)
            a.active_neighbors = new_active
            eps = a.consensus.get_eps()
            a.log.info(
                "Presence: active_neighbors={} | degree={} | eps_i={}",
                sorted(new_active), len(new_active),
                f"{eps:.6f}" if eps is not None else "n/a"
            )


        # active &= neighbor_set
        #
        # # MERGE (do not drop previously active unless explicitly unavailable)
        # merged = set(a.active_neighbors) | active
        #
        # if merged != getattr(self, "_last_active", set()) or initial:
        #     self._last_active = set(merged)
        #     a.active_neighbors = merged
        #     # try:
        #     #     a.consensus.set_degree(len(merged))
        #     #     eps = float(a.consensus.get_eps())
        #     # except Exception:
        #     #     eps = None
        #     eps = a.consensus.get_eps()
        #     a.log.info(
        #         "Presence: active_neighbors={} | degree={} | eps_i={}",
        #         sorted(merged), len(merged),
        #         f"{eps:.6f}" if eps is not None else "n/a"
        #     )

    async def _report_active(self):
        a = self.agent
        deg = len(a.active_neighbors)

        # Update consensus degree
        # try:
        #     a.consensus.set_degree(deg)
        #     eps = float(a.consensus.get_eps())
        # except Exception:
        #     eps = None
        eps = a.consensus.get_eps()
        a.log.info(
            "Presence: active_neighbors={} | degree={} | eps_i={}",
            sorted(list(a.active_neighbors)), deg, f"{eps:.6f}" if eps is not None else "n/a"
        )

    # ---------- utils ----------

    @staticmethod
    def strip_jid(jid) -> str:
        return str(jid).split("/")[0]