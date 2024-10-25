#!/bin/bash

# run-pso-main.sh

# Usage: ./run-pso-main.sh <gpus_on_main>
# Example: ./run-pso-main.sh 4

# Description:
# This script initializes the main process (`main.py`) for the PSO algorithm.
# The PSO algorithm uses multiple independent Python processes for optimization.
# It also optionally starts evaluator processes (`evaluator.py`) based on the number of GPUs allocated for the main process.

# Arguments:
# - <gpus_on_main>: Number of GPUs allocated to run in the same machine that the main process is running.

# Directory Setup:
# - Creates a `logs` directory under `./results` if it doesn't exist, where logs from the main process will be saved.

# Execution Steps:
# 1. Runs `pso-main-loop.py` in the background and redirects stdout and stderr to `./results/logs/main_output.txt`.
# 2. If `<gpus_on_main>` is greater than 0:
#    - Waits for 5 seconds to ensure `pso-main-loop.py` has started.
#    - Executes `run-pso-evaluators.sh` script to start `evaluator.py` instances with the specified number of GPUs.
# 4. Prints status messages indicating that the main loop and the initialization of evaluator scripts are running.

# Example Usage:
# - `./run-pso-main.sh 4` starts `pso-main-loop.py`, and initiates 4 evaluator processes.
# - `./run-pso-main.sh 0` starts `pso-main-loop.py` without any evaluator processes (in case you want to run the evaluators in a different machine).

# Get the absolute path of the current script
SCRIPT_DIR=$(realpath $(dirname "$0"))

# Set the results and logs directories using absolute paths
RESULTS_DIR="$SCRIPT_DIR/../results"
LOGS_DIR="$RESULTS_DIR/logs"

# Ensure OMP_NUM_THREADS is set
export OMP_NUM_THREADS=1

GPUS_ON_MAIN=$1

# Create the logs directory if it doesn't exist (using absolute path)
mkdir -p "$LOGS_DIR"

process_name="AgentForge-PSO-MAIN"

# Run pso-main-loop.py and save output to $LOGS_DIR/main_output.txt
AGENT_NAME="$process_name" PYTHONWARNINGS="ignore" nohup python "$SCRIPT_DIR/pso-main-loop.py" > "$LOGS_DIR/AgentForge_pso_main_output.txt" 2>&1 &

# If GPUS_ON_MAIN is greater than 0, run the second script to start evaluators
if [ "$GPUS_ON_MAIN" -gt 0 ]; then
    # Wait for a moment to ensure pso-main-loop.py has started
    sleep 5
    
    # Run the second script to start evaluator.py instances, using an absolute path
    "$SCRIPT_DIR/run-pso-evaluators.sh" $GPUS_ON_MAIN

    echo "Main PSO script is running."
    echo "Evaluator scripts (evaluator.py) are initialized."
else
    echo "Main PSO script is running."
    echo "No evaluator scripts are initialized since GPUS_ON_MAIN is 0."
fi