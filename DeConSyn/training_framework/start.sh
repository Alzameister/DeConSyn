#!/bin/bash

# Usage:
#   ./start.sh "7 4 10" "1 1 1" "500 500 500" "small-world full"
# Positional args:
#   $1 = AGENTS_LIST (space-separated)
#   $2 = EPOCHS_LIST (space-separated; zipped with ITERATIONS_LIST)
#   $3 = ITERATIONS_LIST (space-separated; zipped with EPOCHS_LIST)
#   $4 = TOPOLOGY (space-separated, e.g., "small-world full")

if [[ $# -lt 5 ]]; then
  echo "Usage: $(basename "$0") \"AGENTS_LIST\" \"EPOCHS_LIST\" \"ITERATIONS_LIST\" \"TOPOLOGY\" \"MODEL-TYPE\""
  echo "Example: $(basename "$0") \"7 4\" \"1 1\" \"500 500\" \"small-world full\" \"tabddpm\""
  exit 1
fi

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT="/home/ubuntu/DeConSyn"
PYTHON_EXEC="/home/ubuntu/.cache/pypoetry/virtualenvs/defesyn-diyj7ln9-py3.11/bin/python"

DATA_ROOT="$HOME/DeConSyn/data/cardio"
SEED=42
N_JOBS=1
LOG_LEVEL="INFO"
SLEEP_SECS=10

ALPHA=1.0
K=4
P=0.1
CONFIG_PATH="$HOME/DeConSyn/exp/cardio/tabddpm_config.toml"

# ----------------------------
# Inputs
# ----------------------------
read -r -a AGENTS_LIST      <<< "$1"
read -r -a EPOCHS_LIST      <<< "$2"
read -r -a ITERATIONS_LIST  <<< "$3"
read -r -a TOPOLOGY_LIST    <<< "$4"
read -r -a MODEL_TYPE_LIST  <<< "$5"
read -r K <<< "${6:-4}"

if [[ ${#EPOCHS_LIST[@]} -ne ${#ITERATIONS_LIST[@]} ]]; then
  echo "ERROR: EPOCHS_LIST and ITERATIONS_LIST must have the same length."
  exit 1
fi


# ----------------------------
# Runner
# ----------------------------
run_once() {
  local agents="$1" epochs="$2" iterations="$3" topology="$4" model_type="$5"

  echo ">>> Running: agents=$agents, epochs=$epochs, iterations=$iterations, topology=$topology, model_type=$model_type"

  (
    cd "$PROJECT_ROOT"
    chrt -r 10 "$PYTHON_EXEC" -m DeConSyn.training_framework.start run \
      --agents "$agents" \
      --epochs "$epochs" \
      --iterations "$iterations" \
      --alpha "$ALPHA" \
      --data-root "$DATA_ROOT" \
      --topology "$topology" \
      --k "$K" \
      --p "$P" \
      --seed "$SEED" \
      --n-jobs "$N_JOBS" \
      --log-level "$LOG_LEVEL" \
      --model-type "$model_type" \
      --config "$CONFIG_PATH"
  )

  echo ">>> Finished: agents=$agents, epochs=$epochs, iterations=$iterations, topology=$topology, model_type=$model_type"
  echo ">>> Sleeping ${SLEEP_SECS}s to let XMPP server cleanup..."
  sleep "$SLEEP_SECS"
  echo "------------------------------------------------------------"
}

# ----------------------------
# Main loop
# ----------------------------
for model_type in "${MODEL_TYPE_LIST[@]}"; do
for topology in "${TOPOLOGY_LIST[@]}"; do
  for agents in "${AGENTS_LIST[@]}"; do
    for i in "${!EPOCHS_LIST[@]}"; do
      run_once "$agents" "${EPOCHS_LIST[$i]}" "${ITERATIONS_LIST[$i]}" "$topology" "$model_type"
      sleep "$SLEEP_SECS"
    done
  done
done
done