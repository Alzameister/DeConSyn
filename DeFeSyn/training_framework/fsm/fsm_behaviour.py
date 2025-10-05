import contextlib
import ctypes
import sys
from dataclasses import dataclass
import random
from typing import Optional, Iterable

import gc, torch
from spade.behaviour import FSMBehaviour, State

# ----------------------------
# Constants / Types
# ----------------------------
START_STATE = "START_STATE"
TRAINING_STATE = "TRAINING_STATE"
PULL_STATE = "PULL_STATE"
PUSH_STATE = "PUSH_STATE"
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

def debug_check_single_weight(agent, which="generator"):
    """
    Verify one parameter is identical between the CPU snapshot (agent.weights)
    and the live module on GPU.
    """
    assert agent.model is not None, "Model not initialized yet."
    weights = agent.weights

    # CTGAN: nested dict
    if isinstance(weights, dict) and which in ("generator", "discriminator") and which in weights:
        cpu_block = weights[which]
        param_key = next(k for k, v in cpu_block.items() if torch.is_tensor(v))
        cpu_t = cpu_block[param_key]
        module = getattr(agent.model.model, f"_{which}")
        gpu_t = module.state_dict()[param_key]
    # TabDDPM: flat dict
    elif isinstance(weights, dict):
        param_key = next(k for k, v in weights.items() if torch.is_tensor(v))
        cpu_t = weights[param_key]
        gpu_t = agent.model.model.state_dict()[param_key]
        which = "tabddpm"

    cpu_val = cpu_t.view(-1)[0].item()
    gpu_val = gpu_t.detach().view(-1)[0].item()
    diff = abs(cpu_val - gpu_val)

    agent.log.debug(
        f"[WEIGHT-CHECK] {which}.{param_key} "
        f"device={gpu_t.device} cpu_val={cpu_val:.8f} gpu_val={gpu_val:.8f} diff={diff:.3e}"
    )

    if not torch.allclose(cpu_t, gpu_t.detach().cpu(), atol=1e-7, rtol=1e-7):
        agent.log.error(f"[WEIGHT-CHECK] MISMATCH in {which}.{param_key}")
        raise RuntimeError(f"Mismatch in {which}.{param_key}")

def pick_random_peer(active: Iterable[str]) -> Optional[str]:
    arr = list(active)
    return random.choice(arr) if arr else None

def _bytes_len(s: Optional[str | bytes]) -> int:
    if s is None:
        return 0
    return len(s if isinstance(s, bytes) else s.encode("utf-8"))

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

