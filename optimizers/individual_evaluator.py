"""

individual_evaluator.py

This script implements the Evaluator class, which is responsible for evaluating solutions in a distributed manner
using the Particle Swarm Optimization (PSO) algorithm and reinforcement learning environments.

Dependencies:
    - time
    - yaml
    - torch
    - sqlalchemy (create_engine, MetaData, Table, sessionmaker)
    - stable_baselines3
    - sb3_contrib
    - stable_baselines3.common.evaluation (evaluate_policy)
    - torch.nn (nn)
    - optimizers.tools (generate_hash)

Classes:
    Evaluator

Methods:
    - __init__(self, process_id, config_file, set_environment)
    - get_assigned_solution_id(self, session)
    - get_current_generation(self, session)
    - evaluate_solution(self)
    - objective_function(self, solution_id, current_gen, environment_kwargs, **kwargs)

"""

import os
import sys
import json
import numpy as np
import random
import time
import yaml
import torch
import subprocess
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

class Evaluator:
    """
    Evaluator class is responsible for evaluating individual solutions within a distributed PSO framework.
    
    Attributes:
        process_id (int): Unique identifier for the evaluator process.
        config_file (str): Path to the configuration file.
        user_environment (func): Function to set up the training and evaluation environments.
        user_evaluate_policy (func): Function to evaluate policy performance.
        environment_file (str): Path to the environment file specified in the configuration.
        parameter_bounds (dict): Dictionary of parameter bounds for optimization.
        total_generations (int): Total number of generations for the PSO.
        max_n_train_timesteps (int): Maximum number of training timesteps.
        eval_every_n_batches (int): Frequency of evaluation during training.
        max_episode_steps (int): Maximum number of steps per episode.
        n_train_envs (int): Number of training environments.
        algorithm_name (str): Name of the stable-baselines algorithm.
        policy_name (str): Name of the stable-baselines policy.
        sb3_contrib (bool): Flag to use sb3_contrib algorithms.
        use_training_reward_for_optimization (bool): Flag to use training reward for optimization.
        engine (SQLAlchemy Engine): SQLAlchemy engine for database connection.
        Session (SQLAlchemy Session): SQLAlchemy session for database operations.
        metadata (SQLAlchemy MetaData): SQLAlchemy metadata for reflecting database tables.
        solutions_table (SQLAlchemy Table): SQLAlchemy table for solutions.
        evaluation_history_table (SQLAlchemy Table): SQLAlchemy table for evaluation history.
        
    Methods:
        __init__(self, process_id, config_file, user_environment, user_evaluate_policy):
            Initializes the Evaluator with the given process ID, config file, and environment setup function.
        
        get_assigned_solution_id(self, session):
            Retrieves the solution ID assigned to the current process for evaluation.
        
        get_current_generation(self, session):
            Retrieves the current generation number for the assigned solution.
        
        evaluate_solution(self):
            Main loop for evaluating assigned solutions. This method continually checks for assigned solutions,
            evaluates them using the objective function, and updates the database with the results.
        
        objective_function(self, solution_id, current_gen, environment_kwargs, **kwargs):
            Objective function for training and evaluating the reinforcement learning model.
    """

    def __init__(self, process_id, config_file, user_train_environment, user_evaluate_policy, user_record_video):
        """
        Initializes the Evaluator with the given process ID, config file, and environment setup function.
        
        Args:
            process_id (int): Unique identifier for the evaluator process.
            config_file (str): Path to the configuration file.
            user_environment (func): Function to set up the training and evaluation environments.
            user_evaluate_policy (func): Function to evaluate policy performance.
        """

        self.config_file = config_file
        self.user_train_environment = user_train_environment
        self.user_evaluate_policy = user_evaluate_policy
        self.user_recorder_func = user_record_video

        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.save_videos = config["configuration"]["save_videos"]

        self.parameter_bounds = config['parameters_bounds']
        self.total_generations = config["parameter_optimization_config"]['particle_swarm_optimization']['num_generations']
        self.db_url = config["configuration"]["database_url"]

        self.max_n_train_timesteps = config["agent_training_configuration"]["number_training_timesteps"]
        self.eval_every_n_batches = config["agent_training_configuration"]["eval_every_n_batches"]
        self.max_episode_steps = config["agent_training_configuration"]["max_number_steps_per_episode"]
        self.n_train_envs = config["agent_training_configuration"]["number_environments_for_training"]
        self.batch_size = config["agent_training_configuration"]["training_batch_size"]

        self.algorithm_name = config["agent_training_configuration"]["stable-baselines-algorithm"]
        self.policy_name = config["agent_training_configuration"]["stable-baselines-policy"]
        self.sb3_contrib = config["agent_training_configuration"]["sb3-contrib"]
        self.random_seed = config["agent_training_configuration"]["random-seed"]
        self.device_name = config["agent_training_configuration"]["device"]

        self.howto_eval_rewards = config["parameter_optimization_config"]["single_objective_evaluation_function"]

        self.set_random_seed()

        self.process_id = process_id
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        self.config_file = config_file

        self.solutions_table = Table('solutions', self.metadata, autoload_with=self.engine)
        self.evaluation_history_table = Table('evaluation_history', self.metadata, autoload_with=self.engine)


    def set_random_seed(self):
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)


    def get_assigned_solution_id(self, session):
        """
        Retrieves the solution ID assigned to the current process for evaluation.
        
        Args:
            session (SQLAlchemy Session): Active SQLAlchemy session.
            
        Returns:
            int: Solution ID assigned to the current process, or None if no solution is assigned.
        """

        result = session.execute(
            self.solutions_table.select()
            .where(self.solutions_table.c.assigned_to == self.process_id)
            .where(self.solutions_table.c.evaluated == 0)
            .limit(1)
        ).fetchone()
        return result.id if result else None
    
    def get_current_generation(self, session):
        """
        Retrieves the current generation number for the assigned solution.
        
        Args:
            session (SQLAlchemy Session): Active SQLAlchemy session.
            
        Returns:
            int: Current generation number.
        """

        result = session.execute(
            self.solutions_table.select()
            .where(self.solutions_table.c.assigned_to == self.process_id)
            .limit(1)
        ).fetchone()
        return result.generation

    def evaluate_solution(self):
        """
        Main loop for evaluating assigned solutions. This method continually checks for assigned solutions,
        evaluates them using the objective function, and updates the database with the results.
        """

        while True:
            with self.Session() as session:
                solution_id = self.get_assigned_solution_id(session)
                if solution_id:
                    solution = session.execute(
                        self.solutions_table.select().where(self.solutions_table.c.id == solution_id)
                    ).fetchone()

                    current_gen = getattr(solution, 'generation')

                    # Extract parameters
                    params = {param: getattr(solution, param) for param in self.parameter_bounds.keys()}

                    # Construct the dictionary for environment_kwargs separately
                    environment_kwargs = {
                        param: getattr(solution, param) for param, config in self.parameter_bounds.items() if config['type'] == 'environment_kwargs'
                    }

                    # Construct the dictionary for policy_kwargs separately
                    policy_params = {
                        param: getattr(solution, param) for param, config in self.parameter_bounds.items() if config['type'] == 'policy_kwargs'
                    }

                    policy_kwargs = {}
                    # Handle specific transformations
                    if 'activation_fn' in policy_params:
                        policy_kwargs['activation_fn'] = "ReLU" if policy_params.pop('activation_fn') > 0.5 else "Tanh"

                    # Construct the network architecture for both policy and value functions
                    if 'policy_arch_num_layers' in policy_params and 'policy_arch_num_neurons' in policy_params:
                        num_layers = int(policy_params.pop('policy_arch_num_layers'))
                        num_neurons = int(policy_params.pop('policy_arch_num_neurons'))
                        policy_kwargs['net_arch'] = dict(pi=[num_neurons] * num_layers, vf=[num_neurons] * num_layers)

                    # Handle value function architecture if distinct from policy
                    if 'value_arch_num_layers' in policy_params and 'value_arch_num_neurons' in policy_params:
                        value_layers = int(policy_params.pop('value_arch_num_layers'))
                        value_neurons = int(policy_params.pop('value_arch_num_neurons'))
                        policy_kwargs['net_arch']['vf'] = [value_neurons] * value_layers

                    # Construct the final objective parameters
                    objective_params = {
                        param: value for param, value in params.items() if self.parameter_bounds[param]['type'] == 'default'
                    }
                    objective_params['policy_kwargs'] = policy_kwargs
                    objective_params['verbose'] = True
                    objective_params['device'] = self.device_name
            
            if solution_id:
                # Perform the evaluation outside of the session
                print(f"Process {self.process_id} evaluating solution {solution_id} with parameters {params}", flush=True)

                # Convert to JSON
                env_kwargs_json = json.dumps(environment_kwargs)
                model_kwargs_json = json.dumps(objective_params)

                objective_function_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './objective_function_pso.py'))

                # Prepare command-line arguments for the script
                command = [
                    "python", objective_function_path,
                    "--solution_id", str(solution_id),
                    "--current_gen", str(current_gen),
                    "--environment_kwargs", env_kwargs_json,
                    "--model_kwargs", model_kwargs_json,
                    "--config_file", self.config_file,
                    "--db_url", self.db_url
                ]

                # Run the standalone objective function script
                subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)

                print(f"Solution {solution_id} evaluation completed and recorded by process {self.process_id}.", flush=True)
            else:
                print(f"Process {self.process_id}: No solutions to evaluate.", flush=True)
                time.sleep(30)

                with self.Session() as session:
                    if self.get_current_generation(session) >= (self.total_generations):
                        print("Total number of generation reached. No more solutions to evaluate.", flush=True)
                        break

