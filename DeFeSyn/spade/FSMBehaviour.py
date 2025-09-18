import asyncio
import contextlib
import ctypes
import json
import sys
import time
import uuid
from abc import ABC
from dataclasses import dataclass
import random
from typing import Optional, Iterable, Dict, Any

import gc, psutil, torch
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from DeFeSyn.models.CTGAN.wrapper import CTGANModel
from DeFeSyn.utils.io import make_path, save_weights_pt, save_model_pickle

# ----------------------------
# Constants / Types
# ----------------------------
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

@dataclass
class TrainSnapshot:
    delta_l2: Optional[float] = None
    theta_l2: Optional[float] = None
    rel_delta: Optional[float] = None
    ms: float = 0.0

# ----------------------------
# utilities
# ----------------------------
def clear_memory():
    gc.collect()
    try:
        if sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        elif sys.platform.startswith("win"):
            with contextlib.suppress(Exception):
                ctypes.cdll.msvcrt._heapmin()
            with contextlib.suppress(Exception):
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.SetProcessWorkingSetSizeEx(
                    ctypes.c_void_p(-1),
                    ctypes.c_size_t(-1),
                    ctypes.c_size_t(-1),
                    ctypes.c_ulong(0x00000001),
                )
    except Exception:
        pass

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

def _bytes_len(s: Optional[str | bytes]) -> int:
    if s is None:
        return 0
    return len(s if isinstance(s, bytes) else s.encode("utf-8"))

# ----------------------------
# FSM Builder
# ----------------------------
class NodeFSMBehaviour(FSMBehaviour):
    """
    Generic FSMBehaviour that is configured declaratively via:
      - states: dict[name -> State()]
      - transitions: list[(source, dest)]
      - initial: name of the initial state
    """
    def __init__(self, *, states: dict[str, State], transitions: list[tuple[str, str]], initial: str):
        super().__init__()
        if initial not in states:
            raise ValueError(f"Initial state '{initial}' not found in states: {list(states)}")

        # Register states
        for name, state in states.items():
            self.add_state(name=name, state=state, initial=(name == initial))

        # Register transitions
        for source, dest in transitions:
            if source not in states or dest not in states:
                raise ValueError(f"Transition {source} -> {dest} references unknown state.")
            self.add_transition(source=source, dest=dest)

    async def on_start(self):
        self.agent.log.info("FSM starting at initial state {}", self.current_state)

    async def on_end(self):
        self.agent.log.info("FSM finished at state {}", self.current_state)
        self.agent.fsm_done.set()


# ----------------------------
# Base State
# ----------------------------
class BaseState(State, ABC):
    @property
    def log(self):
        return self.agent.log

    def _active_neighbors(self) -> set[str]:
        active = getattr(self.agent, "active_neighbors", None)
        if isinstance(active, set):
            self.agent.log.info("Active neighbors: {}", active)
            return active
        contacts = self.agent.presence.get_contacts()
        return {str(jid) for jid, c in contacts.items() if c.is_available()}

    # ---- Encoding helpers ----
    def _encode_weights(self) -> dict:
        return self.agent.model.encode()

    def _decode_weights(self, body: str) -> dict:
        return self.agent.model.decode(json.loads(body))

    def _msg_eps(self, msg, fallback: float) -> float:
        return parse_float(msg.get_metadata("eps"), default=fallback)

    def report(self, label: str = ""):
        rss_mb = psutil.Process().memory_info().rss / 1024 ** 2
        gpu_mb = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
        self.log.info("{} STATE MEM: rss={:.1f}MB gpu={:.1f}MB behaviours={}",
                      label, rss_mb, gpu_mb, len(self.agent.behaviours))

    # ---- Persistence helpers ----
    def _persist_weights(self, it: int):
        p = make_path(
            run_id=self.agent.run_id,
            node_id=self.agent.id,
            iteration=it,
            phase="weights",
            ext="pt",
            repo_root=self.agent.repo_dir,
        )
        save_weights_pt(state_dict=self.agent.weights, path=p)

    def _persist_model(self, it: int):
        should = (self.agent.current_iteration % 10 == 0) or (self.agent.current_iteration == self.agent.max_iterations)
        if not should:
            return
        p = make_path(
            run_id=self.agent.run_id,
            node_id=self.agent.id,
            iteration=it,
            phase="model",
            ext="pkl",
            repo_root=self.agent.repo_dir,
        )
        save_model_pickle(model=self.agent.model.model, path=p)

    # ---- Barrier helpers ----
    async def _send_barrier_hello(self, to_jid: str, token: str):
        msg = Message(to=str(to_jid))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_BARRIER_HELLO)
        msg.set_metadata("token", token)
        await self.send(msg)

    async def _send_gossip(self, peer: str, it: int, pkg: dict):
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)

        msg = Message(to=str(peer))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_GOSSIP)
        msg.set_metadata("content-type", "application/octet-stream+b64")
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("version", version)
        msg.set_metadata("eps", f"{self.agent.consensus.get_eps():.12f}")
        msg.body = await asyncio.to_thread(json.dumps, pkg)

        await self.send(msg)

        self.agent.event.bind(
            event="PUSH",
            neighbor=str(peer), msg_id=msg_id,
            ver=int(version), bytes=int(_bytes_len(msg.body))
        ).info("send")

        return msg_id, version

    async def _send_gossip_reply(self, req_msg, pkg: dict, eps_i: Optional[float], ver: str) -> str:
        in_id = req_msg.get_metadata("msg_id") or f"in-{uuid.uuid4().hex[:6]}"
        resp_id = f"resp-{in_id}"
        resp = Message(to=req_msg.sender)
        resp.set_metadata("performative", MT_INFORM)
        resp.set_metadata("type", T_GOSSIP_REPLY)
        resp.set_metadata("content-type", "application/octet-stream+b64")
        resp.set_metadata("msg_id", resp_id)
        resp.set_metadata("version", ver)
        resp.set_metadata("in_reply_to", in_id)
        if eps_i is not None:
            resp.set_metadata("eps", f"{eps_i:.12f}")
        resp.body = json.dumps(pkg)
        await self.send(resp)
        self.agent.event.bind(
            event="RESPOND_SEND",
            local_step=self.agent.current_iteration,
            neighbor_id=req_msg.sender,
            out_msg_id=resp_id,
            bytes=int(_bytes_len(resp.body)),
            eps_self=(float(eps_i) if eps_i is not None else None)
        ).info("send")
        return resp_id

# ----------------------------
# Start
# ----------------------------
class StartState(BaseState):
    async def run(self):
        # small delay to let PresenceBehaviour populate
        await asyncio.sleep(5.0)

        neighbors = list(getattr(self.agent, "neighbors", []))
        token = f"barrier-{self.agent.id}-{uuid.uuid4().hex[:6]}"
        q: asyncio.Queue[str] = asyncio.Queue()
        self.agent.barrier_queues[token] = q

        async def ping_all(targets: Iterable[str]):
            await asyncio.gather(*[self._send_barrier_hello(j, token) for j in targets])

        await ping_all(neighbors)
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
                    await ping_all([j for j in neighbors if j not in got])
                    resend_at = now + HELLO_RESEND_SEC
        finally:
            self.agent.barrier_queues.pop(token, None)

        # Set consensus degree based on who actually ACKed
        degree = len(got) if len(got) < len(neighbors) else len(neighbors)
        if len(got) < len(neighbors):
            self.log.warning("START: barrier partial {}/{} → proceed anyway", len(got), len(neighbors))
        else:
            self.log.info("START: barrier complete")
        self.agent.consensus.set_degree(degree)

        clear_memory()
        self.set_next_state(TRAINING_STATE)

# ----------------------------
# Training
# ----------------------------
class TrainingState(BaseState):
    def __init__(self):
        super().__init__()
        self._epochs: Optional[int] = None
        self._data: Optional[Dict[str, Any]] = None

    async def run(self):
        self.agent.current_iteration += 1
        it = self.agent.current_iteration

        if it > self.agent.max_iterations:
            self.log.info("Max iterations reached. Exiting…")
            self.set_next_state(FINAL_STATE)
            return

        self.log.info("Starting FSM iteration {} → TRAIN", it)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self._epochs = self._epochs or self.agent.epochs
        self._data = self._data or self.agent.data
        if "train" not in self._data:
            self.log.error("TRAIN: No training split; cannot proceed.")
            self.set_next_state(FINAL_STATE)
            return

        await self._ensure_model(self._data["train"], self._data.get("full_train"))
        await self._load_weights()

        snap = await self._train()
        self._capture_losses_and_weights()

        self.report("TRAIN")
        await self._flush_pending_gossip_replies()
        self.report("TRAIN AFTER REPLIES")

        self.log.info("TRAIN: time={:.1f}ms", snap.ms)
        self._train_event(it, snap.ms)

        self.agent.consensus.start_consensus_window(self.agent.weights)
        clear_memory()
        self.log.info("TRAIN: iteration {} completed → transition PULL", it)
        self.set_next_state(PULL_STATE)

    async def _ensure_model(self, part_train, full_train):
        if self.agent.model:
            return
        dcols = discrete_cols_of(part_train)
        self.agent.log.info("TRAIN: init CTGAN (epochs={}, device={}, discrete={})",
                            self._epochs, self.agent.device, dcols)
        self.agent.model = CTGANModel(
            full_data=full_train,
            data=part_train,
            discrete_columns=dcols,
            epochs=self._epochs,
            verbose=True,
            device=self.agent.device,
        )

    async def _load_weights(self):
        if not self.agent.weights:
            self.log.info("TRAIN: cold start (no weights)")
            return
        self.log.info("TRAIN: warm start → loading weights")
        self.agent.model.load_weights(self.agent.weights)
        debug_check_single_weight(self.agent, which="generator")
        debug_check_single_weight(self.agent, which="discriminator")

    async def _train(self) -> TrainSnapshot:
        t0 = time.perf_counter()
        await asyncio.to_thread(self.agent.model.train)
        return TrainSnapshot(ms=(time.perf_counter() - t0) * 1000.0)

    def _capture_losses_and_weights(self):
        self.agent.loss_values = self.agent.model.model.loss_values
        self.agent.model.model.loss_values = None
        self.agent.weights = self.agent.model.get_weights()

    async def _flush_pending_gossip_replies(self):
        if not self.agent.pending_gossip_replies or not self.agent.model:
            return
        pkg = self.agent.model.encode()
        if not pkg:
            return
        pending = self.agent.pending_gossip_replies
        self.agent.pending_gossip_replies = []
        self.agent.log.info("Flushing {} pending gossip replies...", len(pending))

        with contextlib.suppress(Exception):
            eps_i = float(self.agent.consensus.get_eps())
        version_sent = str(getattr(self.agent, "last_committed_version", self.agent.current_iteration))

        for msg in pending:
            try:
                await self._send_gossip_reply(msg, pkg, eps_i, version_sent)
            except Exception as e:
                self.agent.log.warning("Failed sending deferred reply to {}: {}", msg.sender, e)
            finally:
                del msg
                gc.collect()

    def _train_event(self, it: int, ms: float):
        lv = self.agent.loss_values
        g = float(lv["Generator Loss"].iloc[-1]) if lv is not None and not lv.empty else None
        d = float(lv["Discriminator Loss"].iloc[-1]) if lv is not None and not lv.empty else None
        self.agent.event.bind(
            event="TRAIN",
            local_step=it,
            epochs=int(self._epochs or 0),
            epoch_ms=float(ms),
            G_loss=g,
            D_loss=d,
        ).info("ctgan")

# ----------------------------
# Pull
# ----------------------------
class PullState(BaseState):
    async def run(self):
        if self.agent.queue.empty():
            self.log.info("PULL: queue empty → transition PUSH")
            self.set_next_state(PUSH_STATE)
            return

        self.log.info("PULL: processing queue…")

        consumed = 0
        q_before = self.agent.queue.qsize()
        self.log.info("PULL: queue size before {}", q_before)
        self.report("PULL before Consume")
        t0 = time.perf_counter()

        while not self.agent.queue.empty():
            msg = await self.agent.queue.get()
            self.log.info("PULL: processing message from {}, new size: {}", msg.sender, self.agent.queue.qsize())
            if msg.get_metadata("performative") != MT_INFORM or msg.get_metadata("type") != T_GOSSIP:
                self.log.warning("PULL: ignoring unexpected message: {}", msg.metadata)
                continue

            received = self._decode_weights(msg.body)
            eps_j = self._msg_eps(msg, fallback=self.agent.consensus.get_eps())

            self.agent.weights = self.agent.consensus.step_with_neighbor(
                x_i=self.agent.weights,
                x_j=received,
                eps_j=eps_j,
            )
            if self.agent.model:
                self.agent.model.load_weights(self.agent.weights)

            consumed += 1
            self.log.info(
                "PULL: consensus step (eps_i: {:.6f} → {:.6f}, used eps_j={:.6f})",
                self.agent.consensus.prev_eps,
                self.agent.consensus.get_eps(),
                eps_j,
            )

            # drop references early
            msg.body = None
            del msg, received
            gc.collect()

            # cooperative yield
            if (time.perf_counter() - t0) > 0.01:
                await asyncio.sleep(0)

        ms = (time.perf_counter() - t0) * 1000.0
        q_after = self.agent.queue.qsize()
        self.log.info("PULL: averaged {} updates (queue {}→{})", consumed, q_before, q_after)

        self.agent.event.bind(
            event="MIX",
            consumed_count=consumed,
            queue_before=int(q_before),
            queue_after=int(q_after),
            mix_ms=float(ms),
        ).info("consensus")

        self.report("PULL after Consume")
        clear_memory()
        self.set_next_state(PUSH_STATE)


# ----------------------------
# Push
# ----------------------------
class _WaitResponse(OneShotBehaviour):
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
            if not self._peer_available():
                self.agent.log.warning("WaitResponse: peer {} went unavailable → abort", self.peer_jid)
                self.fut.set_result(None)
                return

            remaining = deadline - loop.time()
            if remaining <= 0:
                self.agent.log.warning("WaitResponse: timeout waiting for {}", self.peer_jid)
                self.fut.set_result(None)
                return

            msg = await self.receive(timeout=min(self.poll_interval, remaining))
            if msg:
                self.fut.set_result(msg)
                return

class PushState(BaseState):
    async def run(self):
        it = self.agent.current_iteration

        # choose an active peer
        active = self._active_neighbors()
        self.log.info("PUSH: active neighbors: {}", active)
        peer = pick_random_peer(active)
        self.log.info("PUSH: peer {}", peer)

        if not peer:
            self.log.warning("PUSH: no available neighbors; skipping this round")
            self._persist_weights(it)
            self._persist_model(it)
            self.set_next_state(TRAINING_STATE)
            return

        # install waiter
        fut = asyncio.get_running_loop().create_future()
        waiter = _WaitResponse(fut, peer)
        self.agent.add_behaviour(waiter, Template(metadata={"performative": MT_INFORM, "type": T_GOSSIP_REPLY}))

        self.report("PUSH before Encode")
        pkg = self._encode_weights()
        self.report("PUSH after Encode")
        msg_id, version = await self._send_gossip(peer, it, pkg)
        self.report("PUSH after Send")

        del pkg
        gc.collect()
        self.report("PUSH after Del")

        await asyncio.sleep(0.5)

        # wait for response
        self.log.info("Waiting for RESPOND from {}", peer)
        reply = await fut
        if reply is None:
            self.log.warning("No RESPOND from {}", peer)
            self.agent.event.bind(
                event="RESPOND_RECV", local_step=self.agent.current_iteration,
                neighbor_id=str(peer), msg_id=None, version=None, timeout=True
            ).info("none")

            self._persist_weights(it)
            self._persist_model(it)

            with contextlib.suppress(Exception):
                waiter.kill()

            self.set_next_state(TRAINING_STATE)
            return

        # handle response
        self.log.info("RESPOND from {}", reply.sender)
        self.report("PUSH after RESPOND")

        received = self._decode_weights(reply.body)
        self.report("PUSH after Decode")
        eps_j = self._msg_eps(reply, fallback=self.agent.consensus.get_eps())

        self.agent.weights = self.agent.consensus.step_with_neighbor(
            x_i=self.agent.weights,
            x_j=received,
            eps_j=eps_j,
        )
        if self.agent.model:
            self.agent.model.load_weights(self.agent.weights)
        self.report("PUSH after Consensus")

        self._persist_weights(it)
        self._persist_model(it)
        self.report("PUSH after Save")

        self.log.info(
            "PULL: consensus step applied (eps_i: {:.6f} → {:.6f}, used eps_j={:.6f})",
            self.agent.consensus.prev_eps,
            self.agent.consensus.get_eps(),
            eps_j,
        )

        self.agent.event.bind(
            event="RESPOND_RECV", local_step=self.agent.current_iteration,
            neighbor_id=str(reply.sender),
            msg_id=reply.get_metadata("msg_id"),
            version=int(reply.get_metadata("version") or self.agent.current_iteration),
            timeout=False
        ).info("ok")

        with contextlib.suppress(Exception):
            waiter.kill()

        del waiter, fut, reply, received
        gc.collect()
        clear_memory()

        self.report("PUSH")
        self.log.info("PUSH: transition TRAINING")
        self.set_next_state(TRAINING_STATE)

# ----------------------------
# Final
# ----------------------------
class FinalState(BaseState):
    async def run(self):
        self.log.info("FSM completed. Stopping agent.")
        with contextlib.suppress(Exception):
            self.agent.presence.set_unavailable()

def debug_check_single_weight(agent, which="generator"):
    """
    Verify one parameter is identical between the CPU snapshot (agent.weights)
    and the live module on GPU.
    """
    assert agent.model is not None, "Model not initialized yet."

    cpu_block = agent.weights[which]
    param_key = next(k for k, v in cpu_block.items() if torch.is_tensor(v))

    cpu_t = cpu_block[param_key]
    module = getattr(agent.model.model, f"_{which}")
    gpu_t = module.state_dict()[param_key]

    cpu_val = cpu_t.view(-1)[0].item()
    gpu_val = gpu_t.detach().view(-1)[0].item()
    diff = abs(cpu_val - gpu_val)

    agent.log.debug(
        f"[WEIGHT-CHECK] {which}.{param_key} "
        f"device={gpu_t.device} cpu_val={cpu_val:.8f} gpu_val={gpu_val:.8f} diff={diff:.3e}"
    )

    # Strict numeric assertion
    if not torch.allclose(cpu_t, gpu_t.detach().cpu(), atol=1e-7, rtol=1e-7):
        agent.log.error(f"[WEIGHT-CHECK] MISMATCH in {which}.{param_key}")
        raise RuntimeError(f"Mismatch in {which}.{param_key}")