"""
pso-main-loop.py

This script serves as the entry point for running the Particle Swarm Optimization (PSO) algorithm
for hyperparameter optimization using the parameters from a configuration file and 
a custom Reinforcement Learning (RL) environment.

Usage:
    Ensure you have set up a custom RL environment and implemented the required methods 
    in the user-specified environment file.

    Modify the configuration file `config.yaml` in the `config_files/` directory to customize
    PSO parameters and settings, including specifying the environment setup file.

    Run this script to initiate the PSO optimization process.

Example:
    $ python pso-main-loop.py

Prerequisites:
    - Python environment with necessary dependencies installed (see conda-environment.yml).
    - A configuration file `config.yaml` with appropriate settings.
    - The following functions must be implemented in the user-specified file (indicated by 
      `name_setup_user_file` in `config.yaml`):
        - `user_train_environment()`: for setting up training and evaluation environments.
        - `user_evaluate_policy()`: for evaluating the RL policy.
        - `user_record_video()`: for recording a video of the policy.

The user does not need to modify this script. Instead, they only need to set the correct parameters 
in the configuration file and supply the required functions in the environment script.

Dependencies:
    - optimizers.pso_optimizer.PSOOptimizer
    - user-supplied methods: `user_train_environment`, `user_evaluate_policy`, and `user_record_video`.

This script loads the PSO configuration from `config.yaml`, dynamically imports the environment script 
containing user-supplied methods, and runs the optimization process. At the end, it prints the best hyperparameters 
found by the PSO algorithm.
"""

import os
import setproctitle
import sys
import importlib
import yaml

# Get the agent name from the environment variable
agent_name = os.getenv('AGENT_NAME', 'AgentForge')

# Set the process title
setproctitle.setproctitle(agent_name)

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up one directory (to the root project directory)
prj_root_dir = os.path.abspath(os.path.join(current_dir, "../"))

# Add the root directory to sys.path
sys.path.append(prj_root_dir)

from optimizers.pso_optimizer import PSOOptimizer

def load_config(config_file):
    """Load the configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Determine the absolute path to the configuration file
    config_file = os.path.abspath(os.path.join(prj_root_dir, './config_files/config.yaml'))

    # Load the configuration file
    config = load_config(config_file)

    # Extract the relative path of the environment script from the configuration
    try:
        env_script_relative = config['configuration']['name_setup_user_file']
    except KeyError:
        print("Error: 'name_setup_user_file' not found in configuration.")
        sys.exit(1)

    # Convert the relative path to an absolute path
    env_script_absolute = os.path.abspath(os.path.join(prj_root_dir, env_script_relative))

    # Extract the module name and directory path
    env_module_dir, env_module_name = os.path.split(env_script_absolute)
    env_module_name = os.path.splitext(env_module_name)[0]  # Remove the .py extension

    # Add the directory containing the environment script to sys.path
    sys.path.append(env_module_dir)

    # Dynamically import the specified environment script
    try:
        env_module = importlib.import_module(env_module_name)
    except ModuleNotFoundError:
        print(f"Error: The module '{env_module_name}' could not be found in the directory '{env_module_dir}'.")
        sys.exit(1)

    # Extract the required functions from the module
    try:
        user_train_environment = getattr(env_module, "user_train_environment")
        user_evaluate_policy = getattr(env_module, "user_evaluate_policy")
        user_record_video = getattr(env_module, "user_record_video")
    except AttributeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize PSO optimizer with configuration and environment setup function
    pso_optimizer = PSOOptimizer(config_file, user_train_environment, user_evaluate_policy, user_record_video)

    # Run PSO optimization
    pso_optimizer.fit()

    # Print the best parameters found by PSO
    print("Best parameters found by PSO:")
    print(pso_optimizer.best())