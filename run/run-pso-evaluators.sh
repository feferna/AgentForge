#!/bin/bash

# run-pso-evaluators.sh

# Usage: ./run-pso-evaluators.sh <total_gpus> [<start_process_id>]
# Example: ./run-pso-evaluators.sh 4 1  # Start process_id from 1

# Description:
# This script is used to run multiple independent evaluator processes (`evaluator.py`) 
# for the PSO algorithm. Each evaluator runs on a separate GPU.

# Arguments:
# - <total_gpus>: Total number of GPUs available for running evaluators in the current machine.
# - [<start_process_id>]: Optional starting process ID for evaluators. Defaults to 0 if not provided.

# Directory Setup:
# - Creates a `logs` directory under `./results` if it doesn't exist, where logs from evaluators will be saved.

# Execution Steps:
# 1. Sets up the environment and configuration:
#    - Defines `CONFIG_FILE` which points to the configuration file (`config.yaml`).
# 2. Defines a function `run_evaluator()` to launch `pso-individual-evaluator.py` for each GPU:
#    - Sets `CUDA_VISIBLE_DEVICES` to the specific GPU ID.
#    - Redirects stdout and stderr to `./results/logs/evaluator_output_${process_id}.txt`.
# 3. Iterates over the range of `<total_gpus>` and launches `pso-individual-evaluator.py` instances:
#    - Calculates the process ID based on the provided or default starting ID.
#    - Prints messages indicating the GPU and process ID for each instance.
# 4. Prints a message indicating the successful launch of all `pso-individual-evaluator.py` instances.

# Example Usage:
# - `./run-pso-evaluators.sh 4 1` starts 4 `evaluator.py` instances with process IDs starting from 1 instead of 0.

# Get the absolute path of the current script
SCRIPT_DIR=$(realpath $(dirname "$0"))

# Set the results and logs directories using absolute paths
RESULTS_DIR="$SCRIPT_DIR/../results"
LOGS_DIR="$RESULTS_DIR/logs"

# Ensure OMP_NUM_THREADS is set
export OMP_NUM_THREADS=1

TOTAL_GPUS=$1
START_PROCESS_ID=${2:-0}  # Default to 0 if not provided

# Create the logs directory if it doesn't exist (using absolute paths)
mkdir -p "$LOGS_DIR"

# Function to launch evaluator.py with specific GPU and process ID
run_evaluator() {
    local gpu_id=$1
    local process_id=$2
    local process_name="AgentForge-PSO-$process_id"
    
    AGENT_NAME="$process_name" PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=$gpu_id nohup python "$SCRIPT_DIR/pso-individual-evaluator.py" "$process_id" \
        > "$LOGS_DIR/AgentForge-pso-evaluator_output_${process_id}.txt" 2>&1 &
}

# Iterate over TOTAL_GPUS and launch evaluator.py instances
for (( gpu_id=0; gpu_id<$TOTAL_GPUS; gpu_id++ ))
do
    process_id=$((START_PROCESS_ID + gpu_id))  # Calculate process ID

    echo "Launching evaluator.py for GPU $gpu_id with process ID $process_id and TOTAL_GPUS $TOTAL_GPUS"
    run_evaluator $gpu_id $process_id
done

echo "All pso-individual-evaluator.py instances launched."