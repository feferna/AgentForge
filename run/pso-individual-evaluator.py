"""
pso-individual-evaluator.py

This script is used to evaluate individual solutions assigned to <process_id> by the PSO main loop.
It leverages the `Evaluator` class from `optimizers.individual_evaluator` to evaluate the performance of a given solution using the 
user-defined environment and policy evaluation functions.

Usage:
    This script should be executed with one command-line argument:
    1. The process ID for the current evaluator instance.

    Example command to run the evaluator:
        $ python pso-evaluators.py <process_id>

    Example:
        $ python pso-evaluators.py 4

Arguments:
    process_id (int): The process ID for the current evaluator instance.

Prerequisites:
    - Python environment with necessary dependencies installed (see conda-environment.yml).
    - Custom RL environment module implemented in environments/ and set up for Stable Baselines3.
    - The configuration file config.yaml with appropriate settings.
    - The following functions must be implemented in the user-specified file (indicated by 
      `name_setup_user_file` in `config.yaml`):
        - `user_train_environment()`: for setting up training and evaluation environments.
        - `user_evaluate_policy()`: for evaluating the RL policy.
        - `user_record_video()`: for recording a video of the policy.

Dependencies:
    - optimizers.individual_evaluator.Evaluator
    - user-supplied methods: `user_train_environment`, `user_evaluate_policy`, and `user_record_video`.
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

from optimizers.individual_evaluator import Evaluator


def load_config(config_file):
    """Load the configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pso-individual-evaluator.py <process_id>")
        sys.exit(1)

    process_id = int(sys.argv[1])

    print(f"Process ID: {process_id}")

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

    evaluator = Evaluator(process_id, config_file, user_train_environment, user_evaluate_policy, user_record_video)
    evaluator.evaluate_solution()
