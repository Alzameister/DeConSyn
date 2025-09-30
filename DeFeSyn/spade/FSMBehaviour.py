import asyncio
import contextlib
import ctypes
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

from DeFeSyn.models.models import CTGANModel, Model
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

T_GOSSIP_REQ = "gossip-req"
T_GOSSIP_ACK = "gossip-ack"
T_GOSSIP_WEIGHTS = "gossip-weights"

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

    def ev(self, event: str, msg: str = "info", **fields):
        """Safe wrapper around agent.event.bind(...).info(...)."""
        try:
            self.agent.event.bind(event=event, **fields).info(msg)
        except Exception as e:
            self.agent.log.debug("ev() failed for event='{}': {}", event, e)

    def _active_neighbors(self) -> set[str]:
        active = getattr(self.agent, "active_neighbors", None)
        if isinstance(active, set):
            self.agent.log.info("Active neighbors: {}", active)
            return active
        contacts = self.agent.presence.get_contacts()
        return {str(jid) for jid, c in contacts.items() if c.is_available()}

    # ---- Encoding helpers ----
    def _encode_weights(self) -> str:
        if not getattr(self.agent, "model", None) or not self.agent.model.is_trained():
            raise RuntimeError("Model not trained; cannot encode weights.")
        blob = self.agent.model.encode()  # returns base85+zlib string now
        if not blob:
            raise RuntimeError("Encoding returned empty payload; model snapshot missing.")
        return blob

    def _decode_weights(self, body: str) -> dict:
        return self.agent.model.decode(body)

    def _msg_eps(self, msg, fallback: float) -> float:
        return parse_float(msg.get_metadata("eps"), default=fallback)

    def report(self, label: str = ""):
        rss_mb = psutil.Process().memory_info().rss / 1024 ** 2
        gpu_mb = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
        behaviours = getattr(self.agent, "behaviours", None)
        num_behaviours = len(behaviours) if behaviours is not None else 0
        self.log.info("{} STATE MEM: rss={:.1f}MB gpu={:.1f}MB behaviours={}",
                      label, rss_mb, gpu_mb, num_behaviours)

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
        should = (self.agent.current_iteration % 25 == 0) or (self.agent.current_iteration == self.agent.max_iterations)
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

    async def _send_gossip_request(self, peer: str, it: int, kind: str):
        rid = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)

        msg = Message(to=str(peer))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_GOSSIP_REQ)
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("rid", rid)
        msg.set_metadata("version", version)
        msg.set_metadata("kind", kind)
        await self.send(msg)
        self.ev("REQUEST", "send", neighbor=str(peer), msg_id=rid, ver=int(version), kind=str(kind))

        return rid, version

    async def _send_gossip_weights(self, peer: str, it: int, blob: str, rid: str):
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)
        msg = Message(to=str(peer))
        msg.set_metadata("performative", MT_INFORM)
        msg.set_metadata("type", T_GOSSIP_WEIGHTS)
        msg.set_metadata("content-type", "application/x-ctgan-weights")
        msg.set_metadata("content-encoding", "b85+zlib")
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("rid", rid)
        msg.set_metadata("version", version)
        msg.set_metadata("eps", f"{self.agent.consensus.get_eps():.12f}")
        msg.body = blob
        await self.send(msg)
        self.ev("WEIGHTS", "send", neighbor=str(peer), msg_id=msg_id, ver=int(version),
                bytes=int(_bytes_len(msg.body or b"")), rid=rid)
        return msg_id, version

# ----------------------------
# Start
# ----------------------------
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

        await self._flush_pending_gossip_replies()

        self.log.info("TRAIN: time={:.1f}ms", snap.ms)
        self._train_event(it, snap.ms)

        self.agent.consensus.start_consensus_window(self.agent.weights)
        clear_memory()
        self.report("TRAIN")
        self.log.info("TRAIN: iteration {} completed → transition PULL", it)
        self.set_next_state(PULL_STATE)

    async def _ensure_model(self, part_train, full_train):
        if self.agent.model:
            return
        dcols = discrete_cols_of(part_train)
        self.agent.log.info("TRAIN: init CTGAN (epochs={}, device={}, discrete={})",
                            self._epochs, self.agent.device, dcols)
        self.agent.model: Model = CTGANModel(
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
        self.agent.model.set_weights(self.agent.weights)
        debug_check_single_weight(self.agent, which="generator")
        debug_check_single_weight(self.agent, which="discriminator")

    async def _train(self) -> TrainSnapshot:
        t0 = time.perf_counter()
        await asyncio.to_thread(self.agent.model.fit)
        return TrainSnapshot(ms=(time.perf_counter() - t0) * 1000.0)

    def _capture_losses_and_weights(self):
        self.agent.loss_values = self.agent.model.get_loss_values()
        self.agent.model.clear_loss_values()
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

        for msg in pending:
            try:
                rid = msg.get_metadata("rid")
                await self._send_gossip_weights(
                    peer=msg.sender,
                    it=self.agent.current_iteration,
                    blob=pkg,
                    rid=rid
                )
                self.agent.log.info("Sent deferred gossip-reply to {}, rid={}", msg.sender, rid)
                self.ev("WEIGHTS", "deferred-reply", neighbor=str(msg.sender),
                        rid=msg.get_metadata("rid"), ver=int(msg.get_metadata("version") or -1))
            except Exception as e:
                self.agent.log.warning("Failed sending deferred reply to {}: {}", msg.sender, e)
            finally:
                del msg
                gc.collect()

    def _train_event(self, it: int, ms: float):
        lv = self.agent.loss_values
        g = float(lv["Generator Loss"].iloc[-1]) if lv is not None and not lv.empty else None
        d = float(lv["Discriminator Loss"].iloc[-1]) if lv is not None and not lv.empty else None
        self.ev(
            "TRAIN", "ctgan",
            local_step=it,
            epochs=int(self._epochs or 0),
            epoch_ms=float(ms),
            G_loss=g,
            D_loss=d,
        )

# ----------------------------
# Pull
# ----------------------------
class PullState(BaseState):
    def get_request_agents(self) -> list[str]:
        """
        Returns a list of neighbor keys where 'want_pull' == True in their value dict.
        """
        return [
            k for k, v in self.agent.pending_gossip.items()
            if isinstance(v, dict) and v.get("want_pull") is True
        ]

    async def run(self):
        neighbors = self.get_request_agents()
        if not neighbors:
            self.log.info("PULL: no pending pulls → transition PUSH")
            self.set_next_state(PUSH_STATE)
            return

        self.log.info("requests: {}", neighbors)
        self.log.info("PULL: processing {} reciprocal pulls …", len(neighbors))
        self.report("PULL before Consume")

        consumed = 0
        dict_before = len(neighbors)
        t0 = time.perf_counter()


        for neighbor in neighbors:
            rid, _ = await self._send_gossip_request(neighbor, self.agent.current_iteration, kind="pull")

            fut = asyncio.get_running_loop().create_future()
            waiter = _WaitResponse(fut, neighbor, timeout=180.0)
            self.agent.add_behaviour(
                waiter,
                Template(
                    metadata={"performative": MT_INFORM, "type": T_GOSSIP_WEIGHTS, "rid": rid}                )
            )

            self.log.info("PULL: waiting for weights from {} (rid'={})", neighbor, rid)
            reply = await fut
            with contextlib.suppress(Exception):
                waiter.kill()

            if not reply:
                self.log.warning("PULL: no weights from {} (rid={})", neighbor, rid)
                self.agent.pending_gossip.pop(neighbor, None)
                continue

            try:
                received = self._decode_weights(reply.body)
            finally:
                reply.body = None

            eps_j = self._msg_eps(reply, fallback=self.agent.consensus.get_eps())
            with torch.no_grad():
                self.agent.weights = self.agent.consensus.step_with_neighbor(
                    x_i=self.agent.weights,
                    x_j=received,
                    eps_j=eps_j,
                )
                if self.agent.model:
                    self.agent.model.set_weights(self.agent.weights)

            consumed += 1
            self.log.info(
                "PULL: consensus step with {} (eps_i: {:.6f} → {:.6f}, used eps_j={:.6f})",
                neighbor, self.agent.consensus.prev_eps, self.agent.consensus.get_eps(), eps_j,
            )
            self.ev(
                "PULL_RECV", "ok",
                local_step=self.agent.current_iteration,
                neighbor_id=str(reply.sender),
                msg_id=reply.get_metadata("msg_id"),
                version=int(reply.get_metadata("version") or self.agent.current_iteration),
                timeout=False,
            )
            self.agent.pending_gossip.pop(neighbor, None)

        ms = (time.perf_counter() - t0) * 1000.0
        dict_after = len(self.get_request_agents())

        self.log.info("PULL: averaged {} updates (dict size {}→{})", consumed, dict_before, dict_after)
        self.ev(
            "MIX", "consensus",
            consumed_count=int(consumed),
            queue_before=int(dict_before),
            queue_after=int(dict_after),
            mix_ms=float(ms),
        )

        clear_memory()
        self.report("PULL")
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
        active = self._active_neighbors()
        peer = pick_random_peer(active)
        self.log.info("PUSH: active neighbors: {}", active)
        self.log.info("PUSH: peer {}", peer)

        if not peer:
            self.log.warning("PUSH: no available neighbors; skipping this round")
            self._persist_weights(it)
            self._persist_model(it)
            self.set_next_state(TRAINING_STATE)
            return

        rid, _ver = await self._send_gossip_request(peer, it, kind="push")

        fut = asyncio.get_running_loop().create_future()
        waiter = _WaitResponse(fut, peer)
        self.agent.add_behaviour(
            waiter,
            Template(metadata={"performative": MT_INFORM, "type": T_GOSSIP_WEIGHTS, "rid": rid})
        )

        self.log.info("Waiting for weights from {} (rid={})", peer, rid)
        reply = await fut
        with contextlib.suppress(Exception):
            waiter.kill()

        if not reply:
            self.log.warning("No weights from {}", peer)
            self._persist_weights(it)
            self._persist_model(it)
            self.set_next_state(TRAINING_STATE)
            return

        self.log.info("Received weights from {}", peer)
        try:
            payload = reply.body
            if payload is None:
                self.log.error("PUSH: missing payload from %s (rid=%s)", peer, rid)
                # decide: retry, pick another neighbor, or skip consensus this round
                return
            received = self._decode_weights(reply.body)
        finally:
            reply.body = None

        eps_j = self._msg_eps(reply, fallback=self.agent.consensus.get_eps())
        with torch.no_grad():
            new_w = self.agent.consensus.step_with_neighbor(
                x_i=self.agent.weights, x_j=received, eps_j=eps_j
            )

        if new_w is not None:
            self.agent.weights = new_w
            if self.agent.model:
                self.agent.model.set_weights(self.agent.weights)
        else:
            self.log.warning("PUSH: consensus step returned None → skipping weight update")

        def _valid_weights(w) -> bool:
            return (
                    isinstance(w, dict)
                    and isinstance(w.get("generator"), dict)
                    and isinstance(w.get("discriminator"), dict)
                    and all(v is not None for v in w["generator"].values())
                    and all(v is not None for v in w["discriminator"].values())
            )

        if not _valid_weights(self.agent.weights):
            self.log.warning(
                "PUSH: consensus produced invalid weights → skipping persist and forcing cold start next round")

        self.log.info(
            "PUSH: consensus step applied (eps_i: {:.6f} → {:.6f}, used eps_j={:.6f})",
            self.agent.consensus.prev_eps, self.agent.consensus.get_eps(), eps_j,
        )

        try:
            v = int(reply.get_metadata("version"))
        except Exception:
            v = -1
        self.ev(
            "PUSH_RECV", "ok",
            local_step=self.agent.current_iteration,
            neighbor_id=str(reply.sender),
            msg_id=str(reply.get_metadata("msg_id")),
            version=v,
            timeout=False,
        )

        self._persist_weights(it)
        self._persist_model(it)

        with contextlib.suppress(Exception):
            waiter.kill()

        # del waiter, fut, reply, received
        gc.collect()
        clear_memory()
        self.report("PUSH")
        self.set_next_state(TRAINING_STATE)

    def _push_event(self, it: int, ms: float):
        self.agent.event.bind(
            event="PUSH",
            local_step=it,
            epoch_ms=float(ms),
        ).info("push")

    def _receive_event(self, sender, msg_id, msg_version, timeout: bool = False):
        try:
            v = int(msg_version)
        except Exception:
            v = -1
        self.agent.event.bind(
            event="PUSH_RECV",
            local_step=self.agent.current_iteration,
            neighbor_id=str(sender),
            msg_id=str(msg_id),
            version=v,
            timeout=timeout
        ).info("ok" if not timeout else "timeout")

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