#!/bin/bash

# Config file path
CONFIG_FILE="./config_files/config.yaml"

# Extract the optimization algorithm and number of GPUs from the YAML config file
OPTIMIZATION_ALGORITHM=$(yq -r '.parameter_optimization_config.optimization_algorithm' $CONFIG_FILE)
N_PROCESSES=$(yq -r '.parallelization.n_processes' $CONFIG_FILE)

# Check if N_PROCESSES is a valid integer
if ! [[ "$N_PROCESSES" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid number of processes (n_processes) specified in the config file: $N_PROCESSES"
    exit 1
fi

# Main logic based on the selected optimization algorithm
if [[ "$OPTIMIZATION_ALGORITHM" == "random_search" || "$OPTIMIZATION_ALGORITHM" == "bayesian_optimization" ]]; then
    echo "Selected Optimization Algorithm: $OPTIMIZATION_ALGORITHM (Using Optuna)"
    echo "Launching Optuna with $N_PROCESSES GPUs"
    ./run/run-optuna.sh $N_PROCESSES

elif [[ "$OPTIMIZATION_ALGORITHM" == "PSO" ]]; then
    echo "Selected Optimization Algorithm: PSO"
    echo "Launching PSO with $N_PROCESSES GPUs"
    ./run/run-pso-main.sh $N_PROCESSES

else
    echo "Error: Invalid optimization algorithm '$OPTIMIZATION_ALGORITHM' in the config file."
    exit 1
fi
