import re
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from DeConSyn.io.io import get_repo_root

repo_root = get_repo_root()

run_dir = repo_root / 'exp' / 'adult' / 'runs' / 'tabddpm' / '10A-30E-1000R-Ring' / 'run-20251018-215127-10Agents-30Epochs-1000Iterations-ring-tabddpm'
log_dir = repo_root / 'exp' / 'adult' / 'logs' / 'tabddpm' / '10E-30E-1000R-Ring' / 'run-20251018-215127-10Agents-30Epochs-1000Iterations-ring-tabddpm'
event_files = sorted(log_dir.glob("events*"))
received_by_agent = Counter()
by_sender = {}
seen = set()
rows = []
RECV_EVENTS = {"PULL_RECV", "PUSH_RECV"}

for event_file in event_files:
    with event_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rec = obj.get("record", {})
            extra = rec.get("extra", {})
            event = extra.get("event")
            if event not in {"PULL_RECV", "PUSH_RECV"}:
                continue
            agent = extra.get("jid") or "unknown"
            step = extra.get("local_step")
            try:
                step = int(step)
            except Exception:
                step = None
            rows.append({"agent": agent, "event": event, "step": step})
df = pd.DataFrame(rows).dropna(subset=["step"]).sort_values("step").reset_index(drop=True)

# 1) Cumulative receives over rounds (uniformity shows as ~straight lines)
plt.figure()
for agent, g in df.groupby("agent"):
    steps = g["step"].values
    y = np.arange(1, len(steps) + 1)
    plt.plot(steps, y, label=agent)
plt.xlabel("Round (local_step)")
plt.ylabel("Cumulative received messages")
plt.title("Cumulative receives over rounds (by agent)")
plt.legend(loc="best", fontsize="small")
cum_path = run_dir / 'results' / 'messages' / 'cumulative_receives.png'
cum_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(cum_path, bbox_inches="tight")
plt.show()

# 2) Per-round receive rate (rolling window)
# Count receives per round per agent, then smooth
per_round = (
    df.groupby(["agent", "step"])
    .size()
    .rename("count")
    .reset_index()
)
# Create a complete step index to avoid gaps in rolling
all_steps = pd.Index(sorted(df["step"].unique()), name="step")
smoothed = []
for agent, g in per_round.groupby("agent"):
    g2 = g.set_index("step").reindex(all_steps, fill_value=0)
    g2["agent"] = agent
    g2["roll_50"] = g2["count"].rolling(50, min_periods=1).mean()
    smoothed.append(g2.reset_index())
smoothed = pd.concat(smoothed, ignore_index=True)

plt.figure()
for agent, g in smoothed.groupby("agent"):
    plt.plot(g["step"], g["roll_50"], label=agent)
plt.xlabel("Round (local_step)")
plt.ylabel("Receives per round (rolling mean, w=50)")
plt.title("Rolling receive rate (window=50 rounds)")
plt.legend(loc="best", fontsize="small")
rate_path = run_dir / 'results' / 'messages' / 'rolling_receive_rate.png'
rate_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(rate_path, bbox_inches="tight")
plt.show()


# log_files = list(run_dir.rglob('console.log'))
#
# RE_PREFIX     = re.compile(r"\bn=(?P<n>\d{2})\s+(?P<jid>\S+)\b")
# RE_PUSH_RECV  = re.compile(r"Received weights from\s+(\S+)")
# RE_PULL_RECV  = re.compile(r"PULL:\s+consensus step with\s+(\S+)")
# RE_FINAL_RECV = re.compile(r"Received updated weights from\s+(\S+)")
#
#
# per_agent = defaultdict(Counter)
# for log_file in log_files:
#     with log_file.open('r', encoding='utf-8', errors='replace') as f:
#         for raw in f:
#             line = raw.rstrip("\n")
#             m = RE_PREFIX.search(line)
#             if not m:
#                 continue
#             receiver  = m.group("jid")
#
#             if RE_PUSH_RECV.search(line):
#                 per_agent[receiver]["PUSH_console"] += 1
#                 per_agent[receiver]["TOTAL_console"] += 1
#                 continue
#
#             if RE_PULL_RECV.search(line):
#                 per_agent[receiver]["PULL_console"] += 1
#                 per_agent[receiver]["TOTAL_console"] += 1
#                 continue
#
#             if RE_FINAL_RECV.search(line):
#                 per_agent[receiver]["FINAL_console"] += 1
#                 per_agent[receiver]["TOTAL_console"] += 1
#                 continue
#         for agent in sorted(per_agent):
#             c = per_agent[agent]
#             total = c.get("TOTAL_console", 0)
#             print(
#                 f"{agent}\tPUSH={c.get('PUSH_console', 0)}\tPULL={c.get('PULL_console', 0)}\tFINAL={c.get('FINAL_console', 0)}\tTOTAL={total}")
