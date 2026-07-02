#!/bin/bash
set -e

REPO_DIR="/home/ies/Repositories/DynaPlex"
IO_DIR="/home/ies/DynaPlex_IO/IO_DynaPlex"
EXECUTABLE="./out/LinRel/bin/dual_sourcing_gca"
DCL_CONFIG="${1:-dcl_config_0.json}"
MDP_CONFIG="${2:-mdp_config_0.json}"
NTFY_TOPIC="ies_beast_621349"

notify() {
    curl -s -d "$1" "https://ntfy.sh/${NTFY_TOPIC}" > /dev/null
}

cd "$REPO_DIR"
echo "Working directory: $REPO_DIR"
echo "Pull code"
git fetch origin
git reset --hard origin/feature/dual-sourcing-mdp
chmod +x bash/run_training_beast.sh
echo "Pull complete"
echo "Copying config files to IO directory"
cp src/lib/models/models/dual_sourcing_backlog/*.json "$IO_DIR/mdp_config_examples/dual_sourcing_backlog/"
echo "Config files copied"
echo "Building"
export CXX=g++
export CC=gcc
cmake --preset LinRel
cmake --build out/LinRel --target dual_sourcing_gca -j 128
echo "Build complete"
notify "Beast: Setup complete, starting training..."

echo "Starting training"
$EXECUTABLE "$DCL_CONFIG" "$MDP_CONFIG"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training complete"
    notify "Beast: Training complete!"
else 
    notify "Beast: Training failed :("
fi 

echo "Done. Exit code: $EXIT_CODE" 