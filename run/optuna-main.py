"""
optuna-main.py

This script is used to run hyperparameter optimization using the Optuna framework, 
leveraging the user-defined environment and policy evaluation functions.

Usage:
    The user must ensure the configuration file `config.yaml` is properly set up, including 
    specifying the environment setup file containing the necessary functions.

    Example command to run the optimizer:
        $ python optuna-main.py

Prerequisites:
    - Python environment with necessary dependencies installed (see conda-environment.yml).
    - A configuration file `config.yaml` with appropriate settings.
    - The following functions must be implemented in the user-specified file (indicated by 
      `name_setup_user_file` in `config.yaml`):
        - `user_train_environment()`: for setting up training and evaluation environments.
        - `user_evaluate_policy()`: for evaluating the RL policy.
        - `user_record_video()`: for recording a video of the policy.

Dependencies:
    - optimizers.optuna_optimizer.OptunaOptimizer
    - user-supplied methods: `user_train_environment`, `user_evaluate_policy`, and `user_record_video`.

This script loads the Optuna hyperparameter optimization configuration from `config.yaml`, dynamically imports 
the environment script containing the user-supplied methods, and runs the optimization process. It performs 
hyperparameter optimization using Optuna and the specified RL environment and evaluation methods.
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

# Navigate up two directories (to the root project directory)
prj_root_dir = os.path.abspath(os.path.join(current_dir, "../"))

# Add the root directory to sys.path
sys.path.append(prj_root_dir)

from optimizers.optuna_optimizer import OptunaOptimizer

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
    except Exception as e:
        print(f"Error while importing '{env_module_name}' module: {e}.")
        sys.exit(1)

    # Extract the required functions from the module
    try:
        user_train_environment = getattr(env_module, "user_train_environment")
        user_evaluate_policy = getattr(env_module, "user_evaluate_policy")
        user_record_video = getattr(env_module, "user_record_video")
    except AttributeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize OptunaOptimizer with configuration and environment setup functions
    optuna_optimizer = OptunaOptimizer(config_file, user_train_environment, user_evaluate_policy, user_record_video)

    # Perform hyperparameter optimization
    optuna_optimizer.fit()
