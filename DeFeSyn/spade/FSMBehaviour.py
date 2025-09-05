import asyncio
import json
import time
import uuid
from abc import ABC
from dataclasses import dataclass
import random
from typing import Optional, Iterable, Dict, Any

from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from DeFeSyn.models.CTGAN.wrapper import CTGANModel, get_gan_snapshot, l2_norm_snapshot, l2_delta_between_snapshots
from DeFeSyn.utils.io import make_path, save_weights_pt, save_model_pickle

START_STATE = "START_STATE"
TRAINING_STATE = "TRAINING_STATE"
PULL_STATE = "PULL_STATE"
PUSH_STATE = "PUSH_STATE"
RECEIVE_STATE = "RECEIVE_STATE"
FINAL_STATE = "FINAL_STATE"

MT_INFORM = "inform"
T_GOSSIP = "gossip"
T_GOSSIP_REPLY = "gossip-reply"
T_BARRIER_HELLO = "barrier-hello"
T_BARRIER_ACK = "barrier-ack"

HELLO_RESEND_SEC = 1.0
HELLO_WAIT_TIMEOUT = 0.2
BARRIER_TOTAL_TIMEOUT = 30.0  # seconds

def discrete_cols_of(df):
    return [c for c in df.columns if getattr(df[c].dtype, "name", "") == "category"]

def parse_float(s: Optional[str], default: float) -> float:
    try:
        return float(s) if s is not None else default
    except (TypeError, ValueError):
        return default

def pick_random_peer(active: Iterable[str]) -> Optional[str]:
    arr = list(active)
    return random.choice(arr) if arr else None

@dataclass
class TrainSnapshot:
    delta_l2: Optional[float] = None
    theta_l2: Optional[float] = None
    rel_delta: Optional[float] = None
    ms: float = 0.0

class NodeFSMBehaviour(FSMBehaviour):
    """
        NodeFSMBehaviour is a finite state machine (FSM) behavior for a NodeAgent in the DeFeSyn framework.
        It manages the states of the agent during the synthetic data generation model training process.
        It includes states for TRAIN, PULL, PUSH
        """
    async def on_start(self):
        self.agent.log.info("FSM starting at initial state {}", self.current_state)

    async def on_end(self):
        self.agent.log.info("FSM finished at state {}", self.current_state)
        await self.agent.stop()

class BaseState(State, ABC):
    @property
    def log(self):
        return self.agent.log

    def _active_neighbors(self) -> set[str]:
        # PresenceBehavior maintains this set; fallback to presence contacts if missing.
        active = getattr(self.agent, "active_neighbors", None)
        if isinstance(active, set):
            print("active", active)
            return active
        print("presence")
        contacts = self.agent.presence.get_contacts()
        return {str(jid) for jid, c in contacts.items() if c.is_available()}

    def _encode_weights(self) -> dict:
        self.agent.model.load_weights(self.agent.weights)
        return self.agent.model.encode()

    def _decode_weights(self, body: str) -> dict:
        return self.agent.model.decode(json.loads(body))

    def _msg_eps(self, msg, fallback: float) -> float:
        return parse_float(msg.get_metadata("eps"), default=fallback)

class StartState(BaseState):
    async def run(self):
        await asyncio.sleep(5.0)  # wait for PresenceBehavior to populate active_neighbors
        neighbors = list(getattr(self.agent, "neighbors", []))
        token = f"barrier-{self.agent.id}-{uuid.uuid4().hex[:6]}"

        # register a queue for this token
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
                # await ACKs
                try:
                    sender = await asyncio.wait_for(q.get(), timeout=HELLO_WAIT_TIMEOUT)
                    got.add(sender)
                    self.log.info("START: ACK from {} ({}/{})", sender, len(got), len(neighbors))
                except asyncio.TimeoutError:
                    pass

                # resend to missing
                now = time.perf_counter()
                if now >= resend_at and len(got) < len(neighbors):
                    missing = [j for j in neighbors if j not in got]
                    await send_hellos(missing)
                    resend_at = now + HELLO_RESEND_SEC
        finally:
            # cleanup even if we break early
            self.agent.barrier_queues.pop(token, None)

        if len(got) < len(neighbors):
            self.log.warning("START: barrier partial {}/{} → proceed anyway", len(got), len(neighbors))
            # Set consensus degree based on actual ACKs
            self.agent.consensus.set_degree(len(got))
        else:
            self.log.info("START: barrier complete")
            # Set consensus degree
            self.agent.consensus.set_degree(len(neighbors))

        self.set_next_state(TRAINING_STATE)

class TrainingState(BaseState):
    def __init__(self):
        super().__init__()
        self._epochs = None
        self._data: Optional[Dict[str, Any]] = None

    async def run(self):
        if self.agent.id == 0:
            print("node 0")
        self.agent.current_iteration += 1
        it = self.agent.current_iteration

        if it > self.agent.max_iterations:
            self.log.info("Max iterations reached. Exiting…")
            self.set_next_state(FINAL_STATE)
            return

        self.log.info("Starting FSM iteration {} → TRAIN", it)
        self._epochs = self._epochs or self.agent.epochs
        self._data = self._data or self.agent.data

        if "train" not in self._data:
            self.log.error("TRAIN: No training split in agent.data; cannot proceed.")
            self.set_next_state(FINAL_STATE)
            return

        data = self._data["train"]
        discrete_cols = discrete_cols_of(data)
        self.log.info("TRAIN: CTGAN epochs={} | discrete_cols={}", self._epochs, discrete_cols)

        if not self.agent.model:
            self.agent.log.info("TRAIN: init CTGAN model (device={})", self.agent.device)
            self.agent.model = CTGANModel(
                full_data=self._data.get("full_train"),
                data=data,
                discrete_columns=discrete_cols,
                epochs=self._epochs,
                verbose=True,
                device=self.agent.device
            )

        if self.agent.weights:
            self.log.info("TRAIN: warm start: loading weights")
            self.agent.model.load_weights(self.agent.weights)
        else:
            self.log.info("TRAIN: cold start (no weights)")

        snap = await self._train_and_snapshot()
        self.agent.loss_values = self.agent.model.model.loss_values
        self.agent.weights = self.agent.model.get_weights()

        # metrics
        G_loss = float(self.agent.loss_values["Generator Loss"].iloc[-1]) if not self.agent.loss_values.empty else None
        D_loss = float(self.agent.loss_values["Discriminator Loss"].iloc[-1]) if not self.agent.loss_values.empty else None

        self.log.info("TRAIN: time={:.1f}ms",
                      snap.ms)

        self.agent.event.bind(
            event="TRAIN",
            local_step=it,
            epochs=int(self._epochs),
            epoch_ms=float(snap.ms),
            G_loss=G_loss,
            D_loss=D_loss
        ).info("ctgan")

        self.agent.consensus.start_consensus_window(self.agent.weights)

        self.log.info("TRAIN: iteration {} completed → transition PULL", it)
        self.set_next_state(PULL_STATE)

    async def _train_and_snapshot(self) -> TrainSnapshot:
        theta_before = get_gan_snapshot(self.agent.model.model)

        t0 = time.perf_counter()
        await asyncio.to_thread(self.agent.model.train)
        ms = (time.perf_counter() - t0) * 1000.0

        theta_after = get_gan_snapshot(self.agent.model.model)

        return TrainSnapshot(ms=ms)

class PullState(BaseState):
    async def run(self):
        if self.agent.id == 0:
            print("node 0")
        if self.agent.queue.empty():
            self.log.info("PULL: queue empty → transition PUSH")
            self.set_next_state(PUSH_STATE)
            return

        it = self.agent.current_iteration

        self.log.info("PULL: processing queue…")

        consumed = []
        q_before = self.agent.queue.qsize()
        self.log.info("PULL: queue size before {}", q_before)
        t0 = time.perf_counter()

        while not self.agent.queue.empty():
            msg = await self.agent.queue.get()
            self.log.info("PULL: processing message from {}, new queue size: {}", msg.sender, self.agent.queue.qsize())
            mtype = msg.get_metadata("type")
            if msg.get_metadata("performative") != MT_INFORM or mtype != T_GOSSIP:
                self.log.warning("PULL: ignoring unexpected message from {}: {}", msg.sender, msg.metadata)
                continue

            self.log.info("PULL: got weights from {}", msg.sender)

            received_weights = self._decode_weights(msg.body)
            eps_j = self._msg_eps(msg, fallback=self.agent.consensus.get_eps())

            self.agent.weights = self.agent.consensus.step_with_neighbor(
                x_i=self.agent.weights,
                x_j=received_weights,
                eps_j=eps_j,
            )
            if self.agent.model:
                self.agent.model.load_weights(self.agent.weights)

            consumed.append({
                "neighbor": str(msg.sender),
                "msg_id": msg.get_metadata("msg_id"),
                "version": int(msg.get_metadata("version") or it),
            })
            self.log.info("PULL: consensus step applied (eps_i→{:.6f}, used eps_j={:.6f})",
                          self.agent.consensus.get_eps(), eps_j)
            self.log.info("PULL: Consumed {} messages so far", len(consumed))

            # cooperative yield
            if (time.perf_counter() - t0) > 0.01:
                await asyncio.sleep(0)

        ms = (time.perf_counter() - t0) * 1000.0
        q_after = self.agent.queue.qsize()
        self.log.info("PULL: averaged {} updates (queue {}→{})", len(consumed), q_before, q_after)

        self.agent.event.bind(
            event="MIX", local_step=self.agent.current_iteration,
            consumed=consumed,
            queue_len_before=int(q_before), queue_len_after=int(q_after),
            mix_time_ms=float(ms)
        ).info("consensus")

        self.log.info("PULL: transition PUSH")
        self.set_next_state(PUSH_STATE)

class PushState(BaseState):
    async def run(self):
        if self.agent.id == 0:
            print("node 0")

        class WaitResponse(OneShotBehaviour):
            def __init__(self, fut, peer_jid: str):
                super().__init__()
                self.fut = fut
                self.peer_jid = peer_jid
                self.poll_interval = 1.0  # seconds
                self.timeout = 1000.0  # seconds

            def _peer_available(self) -> bool:
                active = getattr(self.agent, "active_neighbors", None)
                if isinstance(active, set):
                    return self.peer_jid in active
                contacts = self.agent.presence.get_contacts()
                contact = contacts.get(self.peer_jid)
                return bool(contact and contact.is_available())

            async def run(self):
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self.timeout

                if self.timeout == 0:
                    if not self._peer_available():
                        self.agent.log.warning("WaitResponse: peer {} unavailable (immediate timeout)", self.peer_jid)
                        self.fut.set_result(None)
                        return
                    msg = await self.receive(timeout=0)  # non-blocking
                    self.fut.set_result(msg)
                    return

                while True:
                    if not self._peer_available():
                        self.agent.log.warning("WaitResponse: peer {} went unavailable → abort", self.peer_jid)
                        self.fut.set_result(None)
                        return

                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        self.agent.log.warning("WaitResponse: timeout waiting for {}", self.peer_jid)
                        self.fut.set_result(None)
                        return

                    slice_timeout = min(self.poll_interval, remaining)
                    msg = await self.receive(timeout=slice_timeout)
                    if msg:
                        self.fut.set_result(msg)
                        return

                # msg = await self.receive(timeout=1000.0)  # seconds
                # self.fut.set_result(msg)

        it = self.agent.current_iteration

        # choose an active peer
        active = self._active_neighbors()
        self.log.info("PUSH: active neighbors: {}", active)
        peer = pick_random_peer(active)
        self.log.info("PUSH: peer {}", peer)
        if not peer:
            self.log.warning("PUSH: no available neighbors; skipping this round")
            self.set_next_state(TRAINING_STATE)
            return

        fut = asyncio.get_running_loop().create_future()
        template = Template(metadata={"performative": MT_INFORM, "type": T_GOSSIP_REPLY})
        self.agent.add_behaviour(WaitResponse(fut, peer), template)

        # prepare payload
        pkg = await asyncio.to_thread(self._encode_weights)
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)
        msg = Message(to=str(peer))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_GOSSIP)
        msg.set_metadata("content-type", "application/octet-stream+b64")
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("version", version)
        msg.set_metadata("eps", f"{self.agent.consensus.get_eps():.12f}")
        body = await asyncio.to_thread(json.dumps, pkg)
        msg.body = body

        self.log.info("PUSH: send → {}", peer)
        await self.send(msg)

        payload_bytes = len(msg.body.encode("utf-8"))

        self.agent.event.bind(
            event="PUSH", local_step=self.agent.current_iteration,
            neighbor_id=str(peer), msg_id=msg_id, version=int(version), bytes=int(payload_bytes)
        ).info("send")

        await asyncio.sleep(0.5)

        self.log.info("Waiting for RESPOND from {}", peer)
        reply = await fut
        if reply is None:
            self.log.warning("No RESPOND from {}", peer)
            self.agent.event.bind(
                event="RESPOND_RECV", local_step=self.agent.current_iteration,
                neighbor_id=str(peer), msg_id=None, version=None, timeout=True
            ).info("none")
            self.set_next_state(TRAINING_STATE)
            return

        self.log.info("RESPOND from {}", reply.sender)

        received_weights = self._decode_weights(reply.body)
        eps_j = self._msg_eps(reply, fallback=self.agent.consensus.get_eps())

        self.agent.weights = self.agent.consensus.step_with_neighbor(
            x_i=self.agent.weights,
            x_j=received_weights,
            eps_j=eps_j,
        )
        if self.agent.model:
            self.agent.model.load_weights(self.agent.weights)

        # persist snapshot
        p = make_path(
            run_id=self.agent.run_id,
            node_id=self.agent.id,
            iteration=it,
            phase="weights",
            ext="pt",
            repo_root=self.agent.repo_dir
        )
        save_weights_pt(state_dict=self.agent.weights, path=p)
        p = make_path(
            run_id=self.agent.run_id,
            node_id=self.agent.id,
            iteration=it,
            phase="model",
            ext="pkl",
            repo_root=self.agent.repo_dir
        )
        save_model_pickle(model=self.agent.model.model, path=p)

        self.log.info("RESPOND: consensus step applied (eps_i→{:.6f}, used eps_j={:.6f})",
                      self.agent.consensus.get_eps(), eps_j)

        self.agent.event.bind(
            event="RESPOND_RECV", local_step=self.agent.current_iteration,
            neighbor_id=str(reply.sender),
            msg_id=reply.get_metadata("msg_id"),
            version=int(reply.get_metadata("version") or self.agent.current_iteration),
            timeout=False
        ).info("ok")

        self.log.info("PUSH: transition TRAINING")
        self.set_next_state(TRAINING_STATE)

class FinalState(BaseState):
    async def run(self):
        self.log.info("FSM completed. Stopping agent.")