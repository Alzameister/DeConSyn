import asyncio
import gc
import os
import time
from typing import Optional, Dict, Any

import torch

from DeFeSyn.io.io import get_run_dir
from DeFeSyn.models.models import Model, TabDDPMModel, CTGANModel
from DeFeSyn.training_framework.fsm.fsm_behaviour import FINAL_STATE, clear_memory, PULL_STATE, discrete_cols_of, \
    TrainSnapshot
from DeFeSyn.training_framework.fsm.states.state import BaseState


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
        self.agent.log.info("TRAIN: init Model (epochs={}, device={}, discrete={})",
                            self._epochs, self.agent.device, dcols)
        if self.agent.model_type.lower() == "tabddpm":
            try:
                self.agent.model: Model = TabDDPMModel(
                    data=part_train,
                    discrete_columns=dcols,
                    epochs=self._epochs,
                    verbose=True,
                    device=self.agent.device,
                    real_data_path=self.agent.real_data_path,
                    target=self.agent.target,
                    encoder=self.agent.encoder
                )
            except Exception as e:
                raise e
        elif self.agent.model_type.lower() == "ctgan":
            self.agent.model: Model = CTGANModel(
                data_transformer=self.agent.data_transformer,
                data=part_train,
                discrete_columns=dcols,
                epochs=self._epochs,
                verbose=True,
                device=self.agent.device,
            )
        else:
            self.agent.log.error("No Model found for type '{}'", self.agent.model_type)

    async def _load_weights(self):
        if not self.agent.weights:
            self.log.info("TRAIN: cold start (no weights)")
            return
        self.log.info("TRAIN: warm start → loading weights")
        self.agent.model.set_weights(self.agent.weights)

    async def _train(self) -> TrainSnapshot:
        t0 = time.perf_counter()
        await asyncio.to_thread(self.agent.model.fit)
        return TrainSnapshot(ms=(time.perf_counter() - t0) * 1000.0)

    def _capture_losses_and_weights(self):
        self.agent.loss_values = self.agent.model.get_loss_values()
        # Log loss to console
        if self.agent.loss_values is not None and not self.agent.loss_values.empty:
            # Print all losses
            self.agent.log.info("Loss values: {}", self.agent.loss_values.to_dict(orient="records"))
            # Save to CSV
            run_dir = get_run_dir(
                run_id=self.agent.run_id,
                node_id=self.agent.id,
                repo_root=self.agent.repo_dir,
            )
            p = os.path.join(run_dir, "loss.csv")
            self.agent.loss_values.to_csv(p, index=False)
            self.agent.log.info("Saved loss history to {}", p)
        # self.agent.model.clear_loss_values()
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
        if lv is not None and not lv.empty:
            if "Generator Loss" in lv and "Discriminator Loss" in lv:
                g = float(lv["Generator Loss"].iloc[-1])
                d = float(lv["Discriminator Loss"].iloc[-1])
                self.ev(
                    "TRAIN", "ctgan",
                    local_step=it,
                    epochs=int(self._epochs or 0),
                    epoch_ms=float(ms),
                    G_loss=g,
                    D_loss=d,
                )
            elif "mloss" in lv and "gloss" in lv and "loss" in lv:
                m = float(lv["mloss"].iloc[-1])
                g = float(lv["gloss"].iloc[-1])
                s = float(lv["loss"].iloc[-1])
                self.ev(
                    "TRAIN", "tabddpm",
                    local_step=it,
                    epochs=int(self._epochs or 0),
                    epoch_ms=float(ms),
                    M_loss=m,
                    G_loss=g,
                    loss=s,
                )
