#!/bin/bash

# run-optuna.sh

# Usage: ./run-optuna.sh <total_processes> [total_gpus]
# Example: ./run-optuna.sh 12 3  # Runs 12 processes on 3 GPUs
# If <total_gpus> is not specified, all processes run on GPU 0.

# Description:
# This script runs multiple independent Optuna hyperparameter optimization processes (`optuna-main.py`).
# It distributes the specified number of processes across the specified GPUs.
# If no GPU count is provided, all processes are run on GPU 0.

# Arguments:
# - <total_processes>: Total number of Optuna processes to run.
# - [total_gpus]: (Optional) Total number of GPUs available for distributing the processes.
#                 If not provided, defaults to using GPU 0 for all processes.

# Directory Setup:
# - Creates a `logs` directory under `./results` if it doesn't exist, where logs from each instance will be saved.

# Get the absolute path of the current script
SCRIPT_DIR=$(realpath $(dirname "$0"))

# Set the results directory using an absolute path
RESULTS_DIR="$SCRIPT_DIR/../results"

# Set the logs directory using an absolute path
LOGS_DIR="$RESULTS_DIR/logs"

# Ensure OMP_NUM_THREADS is set
export OMP_NUM_THREADS=1

# Get total number of processes and GPUs from arguments
TOTAL_PROCESSES=$1
TOTAL_GPUS=${2:-1}  # Default to 1 GPU if not provided

# Create the logs directory if it doesn't exist (using absolute path)
mkdir -p "$LOGS_DIR"

# Function to launch optuna-main.py with specific GPU
run_optuna() {
    local gpu_id=$1
    local process_id=$2
    local process_name="AgentForge-GPU${gpu_id}-Proc${process_id}"
    
    # Set the AGENT_NAME environment variable for the Python process
    AGENT_NAME="$process_name" CUDA_VISIBLE_DEVICES=$gpu_id PYTHONWARNINGS="ignore" \
    nohup python "$SCRIPT_DIR/optuna-main.py" > "$LOGS_DIR/${process_name}_output.txt" 2>&1 &
    
    echo "Started $process_name on GPU $gpu_id."
}

# Calculate the number of processes per GPU
processes_per_gpu=$((TOTAL_PROCESSES / TOTAL_GPUS))
remaining_processes=$((TOTAL_PROCESSES % TOTAL_GPUS))

# Launch processes across GPUs
process_count=0
for (( gpu_id=0; gpu_id<TOTAL_GPUS; gpu_id++ )); do
    # Determine the number of processes to launch on this GPU
    num_processes_on_gpu=$processes_per_gpu
    if (( gpu_id < remaining_processes )); then
        ((num_processes_on_gpu++))  # Distribute remaining processes
    fi

    # Launch the determined number of processes for the current GPU
    for (( i=0; i<num_processes_on_gpu; i++ )); do
        echo "Launching optuna-main.py for GPU $gpu_id (Process $((process_count + 1)) of $TOTAL_PROCESSES)"
        run_optuna $gpu_id $process_count
        ((process_count++))
        # Wait for 10 seconds between launching processes
        sleep 10
    done
done

echo "All $TOTAL_PROCESSES optuna-main.py instances launched across $TOTAL_GPUS GPUs."