#!/bin/bash

PYTHON_EXEC=python
SCRIPT="$HOME/FeDeSyn/DeFeSyn/spade/start.py"

DATA_ROOT="$HOME/data/adult/"
MANIFEST="manifest.yaml"
ALPHA=1.0
TOPOLOGY="ring"
SEED=42
N_JOBS=1
LOG_LEVEL="INFO"

AGENTS_LIST=(4 7 10)
EPOCHS_LIST=(1 2 5 10 15 20)
ITERATIONS_LIST=(500 250 100 50 34 25)

for agents in "${AGENTS_LIST[@]}"; do
  for idx in "${!EPOCHS_LIST[@]}"; do
    epochs=${EPOCHS_LIST[$idx]}
    iterations=${ITERATIONS_LIST[$idx]}

    echo ">>> Running: agents=$agents, epochs=$epochs, iterations=$iterations"

    $PYTHON_EXEC "$SCRIPT" run \
      --agents "$agents" \
      --epochs "$epochs" \
      --iterations "$iterations" \
      --alpha "$ALPHA" \
      --data-root "$DATA_ROOT" \
      --manifest "$MANIFEST" \
      --topology "$TOPOLOGY" \
      --seed "$SEED" \
      --n-jobs "$N_JOBS" \
      --log-level "$LOG_LEVEL"

    echo ">>> Finished: agents=$agents, epochs=$epochs, iterations=$iterations"
    echo ">>> Sleeping 60s to let XMPP server cleanup..."
    sleep 60
    echo "------------------------------------------------------------"
  done
done
