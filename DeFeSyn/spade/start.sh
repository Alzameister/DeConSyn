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
LAPTOP_USER="trist"
LAPTOP_HOST="localhost"
LAPTOP_PORT=2222
LAPTOP_KEY="$HOME/.ssh/id_vm_to_laptop"
DEST_RUNS="/mnt/c/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/runs"
DEST_LOGS="/mnt/c/Users/trist/OneDrive/Dokumente/UZH/BA/06_Code/DeFeSyn/logs"
SSH_OPTS="-p $LAPTOP_PORT -o IdentitiesOnly=yes -i $LAPTOP_KEY -o StrictHostKeyChecking=accept-new"
RSYNC_FLAGS="-a --no-perms --no-owner --no-group -h --info=progress2 --partial"

PROJECT_ROOT="/home/ubuntu/FeDeSyn"
PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
SCRIPT="$PROJECT_ROOT/DeFeSyn/spade/start.py"

RUNS_DIR="$PROJECT_ROOT/runs"
LOGS_DIR="$PROJECT_ROOT/logs"

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
# Helpers
# ----------------------------
ensure_remote_dirs() {
  ssh $SSH_OPTS "$LAPTOP_USER@$LAPTOP_HOST" \
    "mkdir -p \"$DEST_RUNS\" \"$DEST_LOGS\""
}

rsync_with_retries() {
  # $1=src_dir  $2=dest_dir
  local src="$1" dest="$2"
  local tries=5 delay=5

  if [[ ! -d "$src" ]]; then
    echo ">>> WARN: $src does not exist, skipping."
    return 0
  fi

  # Ensure destination exists on laptop
  ssh $SSH_OPTS "$LAPTOP_USER@$LAPTOP_HOST" "mkdir -p \"$dest\""

  for ((i=1;i<=tries;i++)); do
    # NOTE: fixed the stray `" "` here; source ends with a single trailing slash
    rsync $RSYNC_FLAGS -e "ssh $SSH_OPTS" \
      "$src/" "$LAPTOP_USER@$LAPTOP_HOST:$dest/" && return 0
    echo ">>> rsync $src -> $dest attempt $i failed; retrying in ${delay}s..."
    sleep "$delay"; delay=$((delay*2))
  done

  echo ">>> ERROR: rsync $src -> $dest failed after $tries attempts."
  return 1
}

sync_runs_and_logs() {
  ensure_remote_dirs
  rsync_with_retries "$RUNS_DIR" "$DEST_RUNS"
  rsync_with_retries "$LOGS_DIR" "$DEST_LOGS"
}

delete_runs_and_logs() {
  rm -rf "${RUNS_DIR:?}"/* "${LOGS_DIR:?}"/*
}


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
      echo ">>> Syncing runs and logs to laptop via reverse SSH..."
      if sync_runs_and_logs; then
        echo ">>> Sync OK. Deleting local runs/logs to free space."
        delete_runs_and_logs
      else
        echo ">>> WARNING: Sync failed. Keeping local runs/logs."
      fi
    done
  done
done