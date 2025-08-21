import asyncio
import json
import time
import uuid

import random

from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from DeFeSyn.models.CTGAN.wrapper import CTGANModel, gan_snapshot, l2_delta_between_snapshots, l2_norm_snapshot, \
    try_gan_snapshot

START_STATE = "START_STATE"
TRAINING_STATE = "TRAINING_STATE"
PULL_STATE = "PULL_STATE"
PUSH_STATE = "PUSH_STATE"
RECEIVE_STATE = "RECEIVE_STATE"
FINAL_STATE = "FINAL_STATE"

class NodeFSMBehaviour(FSMBehaviour):
    """
    NodeFSMBehaviour is a finite state machine (FSM) behavior for a NodeAgent in the DeFeSyn framework.
    It manages the states of the agent during the synthetic data generation model training process.
    It includes states for training, pulling, pushing, and receiving data.
    """
    async def on_start(self):
        self.agent.log.info(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        self.agent.log.info(f"FSM finished at state {self.current_state}")
        await self.agent.stop()

class StartState(State):
    async def run(self):
        self.agent.presence.set_available()

        for jid in self.agent.neighbors:
            self.agent.log.info(f"START: subscribing to {jid}")
            self.agent.presence.subscribe(jid)

        # Wait until all neighbors are available (or timeout)
        target = set(map(str, self.agent.neighbors))
        timeout = asyncio.get_event_loop().time() + 120.0  # 2 min timeout
        while True:
            contacts = self.agent.presence.get_contacts()
            avail = {str(j) for j, c in contacts.items() if c.is_available()}
            if target.issubset(avail):
                self.agent.log.info(f"START: neighbors ready: {sorted(avail)}")
                break
            if asyncio.get_event_loop().time() > timeout:
                self.agent.log.warning(f"START: timeout waiting for neighbors; continuing with {sorted(avail)}")
                break
            await asyncio.sleep(0.2)  # yield; don't block the loop

        # 3) Barrier: everyone must say "ready" for stage=start
        async def send_ready():
            for peer in self.agent.start_expected:
                if peer == str(self.agent.jid):
                    continue
                m = Message(to=peer)
                m.set_metadata("performative", "inform")
                m.set_metadata("type", "barrier")
                m.set_metadata("stage", "start")
                m.body = "ready"
                await self.send(m)

        await send_ready()

        # wait with rebroadcasts
        deadline = asyncio.get_event_loop().time() + 120.0
        rebroadcast_at = asyncio.get_event_loop().time() + 3.0
        while not self.agent.start_ready_event.is_set():
            now = asyncio.get_event_loop().time()
            if now >= rebroadcast_at:
                await send_ready()
                rebroadcast_at = now + 3.0
            if now > deadline:
                missing = sorted(self.agent.start_expected - self.agent.start_ready_from)
                self.agent.log.warning("START: barrier timeout; missing={}. Proceeding.", missing)
                break
            await asyncio.sleep(0.2)

        self.agent.log.info("START: barrier passed with {}", sorted(list(self.agent.start_ready_from)))
        self.agent.log.info("START_STATE transition to TRAINING_STATE")
        self.set_next_state(TRAINING_STATE)

class TrainingState(State):
    """
    TrainingState is a state in the FSM that handles the training of a Synthetic Data Generation model.
    """
    def __init__(self):
        super().__init__()
        self.epochs = None
        self.data: dict = None

    async def run(self):
        self.agent.current_iteration += 1
        it = self.agent.current_iteration
        if self.agent.current_iteration > self.agent.max_iterations:
            self.agent.log.info("Max iterations reached. Exiting…")
            self.set_next_state(FINAL_STATE)
        else:
            self.agent.log.info(f"Starting FSM iteration {it} → TRAIN")
            self.epochs = self.epochs or self.agent.epochs
            self.data = self.data or self.agent.data

            if 'train' not in self.data:
                self.agent.log.error("TRAIN: No training split in agent.data; Cannot proceed with training.")
                self.set_next_state(FINAL_STATE)

            data = self.data['train']
            discrete_cols = [c for c in data.columns if data[c].dtype.name == "category"]

            self.agent.log.info(f"TRAIN: CTGAN epochs={self.epochs} | discrete_cols={discrete_cols}")

            if not self.agent.model:
                self.agent.log.info("TRAIN: Init CTGAN model")
                self.agent.model = CTGANModel(
                    full_data=self.agent.full_train_data,
                    data=data,
                    discrete_columns=discrete_cols,
                    epochs=self.epochs
                )

            if self.agent.weights:
                self.agent.log.info("TRAIN: Warm start: loading weights")
                self.agent.model.load_weights(self.agent.weights)
            else:
                self.agent.log.info("TRAIN: Cold start (no weights)")

            theta_before = try_gan_snapshot(self.agent.model.model)
            t0 = time.perf_counter()
            self.agent.model.train()
            ms = (time.perf_counter() - t0) * 1000.0

            theta_after = try_gan_snapshot(self.agent.model.model)
            theta_norm = l2_norm_snapshot(theta_after)

            if theta_before is not None:
                delta_theta = l2_delta_between_snapshots(theta_before, theta_after)
                theta_rel = delta_theta / (theta_norm + 1e-12)
            else:
                delta_theta = None
                theta_rel = None

            self.agent.loss_values = self.agent.model.model.loss_values
            self.agent.weights = self.agent.model.get_weights()

            # --- initialize ε from local degree & start window snapshot (x_i^0) ---
            contacts = self.agent.presence.get_contacts()
            neighbors = [jid for jid, c in contacts.items() if c.is_available()]
            self.agent.consensus.set_degree(len(neighbors))
            self.agent.consensus.start_consensus_window(self.agent.weights)

            self.agent.log.info(
                "CONSENSUS: window started | degree={} | eps={:.6f}",
                len(neighbors), self.agent.consensus.get_eps()
            )

            G_loss = float(self.agent.loss_values["Generator Loss"].iloc[-1]) if not self.agent.loss_values.empty else None
            D_loss = float(self.agent.loss_values["Discriminator Loss"].iloc[-1]) if not self.agent.loss_values.empty else None

            delta_str = f"{delta_theta:.4f}" if delta_theta is not None else "n/a"
            rel_str = f"{theta_rel:.3e}" if theta_rel is not None else "n/a"
            norm_str = f"{theta_norm:.4f}" if theta_norm is not None else "n/a"

            self.agent.log.info(
                "TRAIN: Δθ={} (rel {}) | ||θ||={} | time={:.1f}ms",
                delta_str, rel_str, norm_str, ms
            )

            self.agent.event.bind(
                event="TRAIN",
                local_step=it,
                epochs=int(self.epochs),
                epoch_ms=float(ms),
                G_loss=G_loss,
                D_loss=D_loss,
                delta_theta_l2=(float(delta_theta) if delta_theta is not None else None),  # ||θ_after − θ_before||₂
                theta_l2=(float(theta_norm) if theta_norm is not None else None),  # ||θ_after||₂
                rel_delta_theta=(float(theta_rel) if theta_rel is not None else None) # scale-free update
            ).info("ctgan")

            self.agent.log.info(f"TRAIN: iteration {it} completed -> transition PULL")
            self.set_next_state(PULL_STATE)

class PullState(State):
    async def run(self):
        if self.agent.queue.empty():
            self.agent.log.info("PULL: queue empty → transition PUSH")
        else:
            it = self.agent.current_iteration
            self.agent.log.info("PULL: processing queue…")
            consumed = []
            q_before = self.agent.queue.qsize()
            t0 = time.perf_counter()

            while not self.agent.queue.empty():
                msg = await self.agent.queue.get()
                if msg.get_metadata("performative") == "inform" and msg.get_metadata("type") == "gossip":
                    self.agent.log.info(f"PULL: got weights from {msg.sender}")

                    received_weights = self.agent.model.decode(json.loads(msg.body))
                    # --- NEW: parse neighbor ε (backward-compat fallback) ---
                    try:
                        eps_j = float(msg.get_metadata("eps"))
                    except (TypeError, ValueError):
                        eps_j = self.agent.consensus.get_eps()

                    # --- NEW: apply dynamic-ε consensus step (with correction term) ---
                    self.agent.weights = self.agent.consensus.step_with_neighbor(
                        x_i=self.agent.weights,
                        x_j=received_weights,
                        eps_j=eps_j,
                    )

                    consumed.append({
                        "neighbor": str(msg.sender),
                        "msg_id": msg.get_metadata("msg_id"),
                        "version": int(msg.get_metadata("version") or it)
                    })
            ms = (time.perf_counter() - t0) * 1000.0
            q_after = self.agent.queue.qsize()
            self.agent.log.info(f"PULL: averaged {len(consumed)} updates (queue {q_before}→{q_after})")

            self.agent.event.bind(
                event="MIX", local_step=self.agent.current_iteration,
                consumed=consumed,
                queue_len_before=int(q_before), queue_len_after=int(q_after),
                mix_time_ms=float(ms)
            ).info("consensus")

        self.agent.log.info(f"PULL transition PUSH")
        self.set_next_state(PUSH_STATE)

class PushState(State):
    async def run(self):
        class WaitResponse(OneShotBehaviour):
            def __init__(self, fut):
                super().__init__()
                self.fut = fut

            async def run(self):
                future = await self.receive(timeout=30.0)
                self.fut.set_result(future)

        it = self.agent.current_iteration

        fut = asyncio.get_running_loop().create_future()
        template = Template(metadata={"performative": "inform", "type": "gossip-reply"})
        self.agent.add_behaviour(WaitResponse(fut), template)

        contacts = self.agent.presence.get_contacts()
        neighbors = [jid for jid, c in contacts.items() if c.is_available()]
        peer = random.choice(neighbors)
        await asyncio.sleep(random.uniform(0.5, 1.5))

        self.agent.model.load_weights(self.agent.weights)
        pkg = self.agent.model.encode()
        msg_id = f"{self.agent.id}-{it}-{uuid.uuid4().hex[:6]}"
        version = str(it)

        msg = Message(to=str(peer))
        msg.set_metadata("performative", "inform")
        msg.set_metadata("type", "gossip")
        msg.set_metadata("content-type", "application/octet-stream+b64")
        msg.set_metadata("msg_id", msg_id)
        msg.set_metadata("version", version)
        # --- NEW: ship your ε so peers can do min-rule ---
        msg.set_metadata("eps", str(self.agent.consensus.get_eps()))
        msg.body = json.dumps(pkg)

        self.agent.log.info(f"PUSH: send → {peer}")
        await self.send(msg)
        payload_bytes = len(msg.body.encode("utf-8"))

        self.agent.event.bind(
            event="PUSH", local_step=self.agent.current_iteration,
            neighbor_id=str(peer), msg_id=msg_id, version=int(version), bytes=int(payload_bytes)
        ).info("send")

        self.agent.log.info(f"Waiting for RESPOND from {peer}")
        reply = await fut
        if reply is None:
            # TODO: Handle timeout or no response on other side?
            self.agent.log.warning(f"No RESPOND from {peer}")
            self.agent.event.bind(
                event="RESPOND_RECV", local_step=self.agent.current_iteration,
                neighbor_id=str(peer), msg_id=None, version=None, timeout=True
            ).info("none")
        else:
            self.agent.log.info(f"RESPOND from {reply.sender}")

            received_weights = self.agent.model.decode(json.loads(reply.body))
            try:
                eps_j = float(reply.get_metadata("eps"))
            except (TypeError, ValueError):
                eps_j = self.agent.consensus.get_eps()

            self.agent.weights = self.agent.consensus.step_with_neighbor(
                x_i=self.agent.weights,
                x_j=received_weights,
                eps_j=eps_j,
            )

            self.agent.log.info(
                "RESPOND: consensus step applied (eps_i→{:.6f}, used eps_j={:.6f})",
                self.agent.consensus.get_eps(), eps_j
            )

            self.agent.event.bind(
                event="RESPOND_RECV", local_step=self.agent.current_iteration,
                neighbor_id=str(reply.sender),
                msg_id=reply.get_metadata("msg_id"),
                version=int(reply.get_metadata("version") or self.agent.current_iteration),
                timeout=False
            ).info("ok")

        self.agent.log.info(f"PUSH: transition TRAINING")
        self.set_next_state(TRAINING_STATE)

class FinalState(State):
    async def run(self):
        # TODO: Cleanup / Final Reporting?
        self.agent.log.info("FSM completed. Stopping agent.")

        # TODO: Structured FINAL event