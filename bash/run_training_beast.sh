#!/bin/bash
set -e

# settings
REPO_DIR="/home/ies/Repositories/DynaPlex"
IO_DIR="/home/ies/Repositories/DynaPlex/IO_DynaPlex"
EXECUTABLE="./out/LinRel/bin/dual_sourcing_gca"
DCL_CONFIG="${1:-dcl_config_test.json}"
MDP_CONFIG="${2:-mdp_config_flat_joined.json}"
NTFY_TOPIC="ies_beast_621349"
RUN_ID="DCL=${DCL_CONFIG%.json}_MDP=${MDP_CONFIG%.json}"
START_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

notify() {
    curl -s -d "$1" "https://ntfy.sh/${NTFY_TOPIC}" > /dev/null
}

cd "$REPO_DIR"
echo "Working directory: $REPO_DIR"
echo "Pull code"
git pull

notify "START - Beast Run: [$START_DATETIME] $RUN_ID" 

echo "Pull complete"
echo "Copying config files to IO directory"
mkdir -p "$IO_DIR/mdp_config_examples/dual_sourcing_backlog/configs/"
cp src/lib/models/models/dual_sourcing_backlog/configs/*.json "$IO_DIR/mdp_config_examples/dual_sourcing_backlog/configs/"
echo "Config files copied"
echo "Building"
export CXX=g++
export CC=gcc
cmake --preset LinRel
cmake --build out/LinRel --target dual_sourcing_gca -j 128
echo "Build complete"

echo "Starting training"

set +e
$EXECUTABLE "$DCL_CONFIG" "$MDP_CONFIG"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training complete"
    echo "Start pushing..."
    git add "$IO_DIR/dual_sourcing/"
    git commit -m "Beast Run: $START_DATETIME $RUN_ID"
    git push 
    notify "FINISH - Beast Run: [$START_DATETIME] $RUN_ID"
else 
    echo "Training failed"
    notify "ERROR - Beast Run: [$START_DATETIME] $RUN_ID"
fi 

echo "Done. Exit code: $EXIT_CODE" 
