#!/bin/bash

# run-optuna.sh

# Usage: ./run-optuna.sh <total_gpus>
# Example: ./run-optuna.sh 4  # Runs optuna-main.py on 4 GPUs

# Description:
# This script is used to run multiple independent Optuna hyperparameter optimization processes 
# (`optuna-main.py`). Each instance runs on a separate GPU.

# Arguments:
# - <total_gpus>: Total number of GPUs available for running Optuna optimization instances.

# Directory Setup:
# - Creates a `logs` directory under `./results` if it doesn't exist, where logs from each instance will be saved.

# Execution Steps:
# 1. Sets up the environment and configuration:
#    - Defines `CONFIG_FILE` which points to the configuration file (`config.yaml`).
# 2. Defines a function `run_optuna()` to launch `optuna-main.py` for each GPU:
#    - Sets `CUDA_VISIBLE_DEVICES` to the specific GPU ID.
#    - Redirects stdout and stderr to `./results/logs/optuna_output_${gpu_id}.txt`.
# 3. Iterates over the range of `<total_gpus>` and launches `optuna-main.py` instances:
#    - Prints messages indicating the GPU and the script being run for each instance.
# 4. Prints a message indicating the successful launch of all `optuna-main.py` instances.

# Example Usage:
# - `./run-optuna.sh 4` starts 4 `optuna-main.py` instances, each on a different GPU.

# Get the absolute path of the current script
SCRIPT_DIR=$(realpath $(dirname "$0"))

# Set the results directory using an absolute path
RESULTS_DIR="$SCRIPT_DIR/../results"

# Set the logs directory using an absolute path
LOGS_DIR="$RESULTS_DIR/logs"

# Ensure OMP_NUM_THREADS is set
export OMP_NUM_THREADS=1

# Get the total number of GPUs from the first argument
TOTAL_GPUS=$1

# Create the logs directory if it doesn't exist (using absolute path)
mkdir -p "$LOGS_DIR"

# Function to launch optuna-main.py with specific GPU
run_optuna() {
    local gpu_id=$1
    local process_name="AgentForge-$gpu_id"
    
    # Set the AGENT_NAME environment variable for the Python process
    AGENT_NAME="$process_name" CUDA_VISIBLE_DEVICES=$gpu_id PYTHONWARNINGS="ignore" \
    nohup python "$SCRIPT_DIR/optuna-main.py" > "$LOGS_DIR/AgentForge_output_${gpu_id}.txt" 2>&1 &
    
    echo "Started $process_name on GPU $gpu_id."
}

# Iterate over TOTAL_GPUS and launch optuna-main.py instances
for (( gpu_id=0; gpu_id<$TOTAL_GPUS; gpu_id++ ))
do
    echo "Launching optuna-main.py for GPU $gpu_id with TOTAL_GPUS $TOTAL_GPUS"
    run_optuna $gpu_id
    # Wait for 10 seconds between launching processes
    sleep 10
done

echo "All optuna-main.py instances launched."