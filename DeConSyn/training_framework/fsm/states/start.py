import asyncio
import gc
import time
import uuid
from typing import Iterable

from spade.message import Message

from DeConSyn.training_framework.fsm.fsm_behaviour import MT_INFORM, T_BARRIER_HELLO, BARRIER_TOTAL_TIMEOUT, \
    HELLO_RESEND_SEC, HELLO_WAIT_TIMEOUT, TRAINING_STATE
from DeConSyn.training_framework.fsm.states.state import BaseState


class StartState(BaseState):
    async def run(self):
        await asyncio.sleep(5.0)
        neighbors = list(getattr(self.agent, "neighbors", []))
        token = f"barrier-{self.agent.id}-{uuid.uuid4().hex[:6]}"

        q: asyncio.Queue[str] = asyncio.Queue()
        self.agent.barrier_queues[token] = q

        async def send_hellos(targets: Iterable[str]):
            for jid in targets:
                msg = Message(to=str(jid))
                msg.set_metadata("performative", MT_INFORM)
                msg.set_metadata("type", T_BARRIER_HELLO)
                msg.set_metadata("token", token)
                await self.send(msg)

        await send_hellos(neighbors)
        got = set()
        deadline = time.perf_counter() + BARRIER_TOTAL_TIMEOUT
        resend_at = time.perf_counter() + HELLO_RESEND_SEC

        try:
            while time.perf_counter() < deadline and len(got) < len(neighbors):
                try:
                    sender = await asyncio.wait_for(q.get(), timeout=HELLO_WAIT_TIMEOUT)
                    got.add(sender)
                    self.log.info("START: ACK from {} ({}/{})", sender, len(got), len(neighbors))
                except asyncio.TimeoutError:
                    pass

                now = time.perf_counter()
                if now >= resend_at and len(got) < len(neighbors):
                    missing = [j for j in neighbors if j not in got]
                    await send_hellos(missing)
                    resend_at = now + HELLO_RESEND_SEC
        finally:
            self.agent.barrier_queues.pop(token, None)

        if len(got) < len(neighbors):
            self.log.warning("START: barrier partial {}/{} → proceed anyway", len(got), len(neighbors))
            self.agent.consensus.set_degree(len(got))
        else:
            self.log.info("START: barrier complete")
            self.agent.consensus.set_degree(len(neighbors))

        # Free up memory
        gc.collect()

        self.set_next_state(TRAINING_STATE)
