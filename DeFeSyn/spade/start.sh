#!/bin/bash

#PROJECT_ROOT="/home/ubuntu/FeDeSyn"
#PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"   # poetry’s venv
#SCRIPT="$PROJECT_ROOT/DeFeSyn/spade/start.py"
#
#DATA_ROOT="$HOME/data/adult/"

# Usage:
#   ./start.sh "7 4 10" "1 1 1" "500 500 500" "small-world full"
# Positional args:
#   $1 = AGENTS_LIST (space-separated)
#   $2 = EPOCHS_LIST (space-separated; zipped with ITERATIONS_LIST)
#   $3 = ITERATIONS_LIST (space-separated; zipped with EPOCHS_LIST)
#   $4 = TOPOLOGY (space-separated, e.g., "small-world full")

if [[ $# -lt 4 ]]; then
  echo "Usage: $(basename "$0") \"AGENTS_LIST\" \"EPOCHS_LIST\" \"ITERATIONS_LIST\" \"TOPOLOGY\""
  echo "Example: $(basename "$0") \"7 4\" \"1 1\" \"500 500\" \"small-world full\""
  exit 1
fi

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT="/home/ubuntu/FeDeSyn"
PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
SCRIPT="$PROJECT_ROOT/DeFeSyn/spade/start.py"

DATA_ROOT="$HOME/data/adult/"
MANIFEST="manifest.yaml"
SEED=42
N_JOBS=1
LOG_LEVEL="INFO"
SLEEP_SECS=10

ALPHA=1.0
K=4
P=0.1

# ----------------------------
# Inputs
# ----------------------------
read -r -a AGENTS_LIST      <<< "$1"
read -r -a EPOCHS_LIST      <<< "$2"
read -r -a ITERATIONS_LIST  <<< "$3"
read -r -a TOPOLOGY_LIST    <<< "$4"

if [[ ${#EPOCHS_LIST[@]} -ne ${#ITERATIONS_LIST[@]} ]]; then
  echo "ERROR: EPOCHS_LIST and ITERATIONS_LIST must have the same length."
  exit 1
fi

# ----------------------------
# Runner
# ----------------------------
run_once() {
  local agents="$1" epochs="$2" iterations="$3" topology="$4"

  echo ">>> Running: agents=$agents, epochs=$epochs, iterations=$iterations, topology=$topology"

  (
    cd "$PROJECT_ROOT"
    sudo chrt -r 10 "$PYTHON_EXEC" -m DeFeSyn.spade.start run \
      --agents "$agents" \
      --epochs "$epochs" \
      --iterations "$iterations" \
      --alpha "$ALPHA" \
      --data-root "$DATA_ROOT" \
      --manifest "$MANIFEST" \
      --topology "$topology" \
      --k "$K" \
      --p "$P" \
      --seed "$SEED" \
      --n-jobs "$N_JOBS" \
      --log-level "$LOG_LEVEL"
  )

  echo ">>> Finished: agents=$agents, epochs=$epochs, iterations=$iterations, topology=$topology"
  echo ">>> Sleeping ${SLEEP_SECS}s to let XMPP server cleanup..."
  sleep "$SLEEP_SECS"
  echo "------------------------------------------------------------"
}

# ----------------------------
# Main loop
# ----------------------------
for topology in "${TOPOLOGY_LIST[@]}"; do
  for agents in "${AGENTS_LIST[@]}"; do
    for i in "${!EPOCHS_LIST[@]}"; do
      run_once "$agents" "${EPOCHS_LIST[$i]}" "${ITERATIONS_LIST[$i]}" "$topology"
    done
  done
done