"""
optuna_optimizer.py

This script defines the OptunaOptimizer class which uses Optuna for hyperparameter optimization
of reinforcement learning algorithms using Stable Baselines3.

Dependencies:
- Python 3.6 or higher
- Numpy
- PyYaml
- PyTorch
- Optuna
- Stable Baselines3
- Stable Baselines3 Contrib

Usage:
Instantiate the OptunaOptimizer class with a configuration file and user-supplied functions,
then call the fit() method to start the optimization process.

Example:
    optimizer = OptunaOptimizer('config.yml', user_environment, user_evaluate_policy, user_video_recorder_optuna)
    optimizer.fit()

"""

from typing import Any
from typing import Dict

import os
import gc
import yaml
import random
import optuna
import json
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.samplers import RandomSampler

import stable_baselines3
import sb3_contrib

import torch
import torch.nn as nn

import numpy as np


def increased_exploration_gamma(x: int) -> int:
    return min(int(np.ceil(0.2 * x)), 50)


class OptunaOptimizer:
    """
    OptunaOptimizer class performs hyperparameter optimization using Optuna.

    Attributes:
    - optim_alg (str): Optimization algorithm ('bayesian_optimization' or 'random_search').
    - num_trials (int): Total number of trials for optimization.
    - num_startup_trials (int): Number of initial trials for exploration.
    - sampler (optuna.samplers.BaseSampler): Sampler object for parameter sampling.
    - num_training_timesteps (int): Total number of training timesteps per trial.
    - eval_every_n_batches (int): Evaluation frequency in training batches.
    - max_number_steps_per_episode (int): Maximum steps per episode in the environment.
    - number_envs_for_train (int): Number of environments for training.
    - algorithm_name (str): Name of the reinforcement learning algorithm.
    - policy_name (str): Name of the policy used in the algorithm.
    - sb3_contrib (bool): Whether to use stable-baselines3-contrib.
    - parameter_bounds (dict): Dictionary defining the bounds and type of each parameter.
    - db_url (str): URL for storing optimization results.
    - environment_file (str): File path for the environment configuration.
    - user_environment (function): Function to set up the RL environment.
    - pruner (optuna.pruners.BasePruner): Pruner object for early stopping of trials.
    - study (optuna.study.Study): Optuna study object for managing trials.
    - user_evaluate_policy (function): Function to evaluate the RL policy.
    - user_video_recorder_optuna (function): Function to record videos of the agent's performance.

    Methods:
    - __init__(self, config_file, user_environment, user_evaluate_policy, user_video_recorder_optuna): Initializes the optimizer with configuration.
    - fit(self): Performs the optimization process using Optuna.
    - set_random_seed(self, seed): Sets random seeds for reproducibility.
    - sample_params(self, trial: optuna.Trial) -> Dict[str, Any]: Samples parameters for a trial.
    - objective(self, trial: optuna.Trial) -> float: Objective function to optimize.
    """
    
    def __init__(self, config_file, user_train_environment, user_evaluate_policy, user_video_recorder_optuna):
        """
        Initialize OptunaOptimizer with configuration file and user-supplied functions.

        Args:
        - config_file (str): Path to YAML configuration file.
        - user_train_environment (callable): Function to set up RL environment.
        - user_evaluate_policy (callable): Function to evaluate RL policy.
        - user_video_recorder_optuna (callable): Function to record videos of the agent's performance.

        Reads configuration settings, initializes sampler and pruner, and creates Optuna study.
        """
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.optim_alg = config["parameter_optimization_config"]["optimization_algorithm"]
        self.continue_from_existing_database = config["agent_training_configuration"]["continue_from_existing_database"]
        self.random_seed = config["agent_training_configuration"]["random-seed"]

        self.num_trials = config["parameter_optimization_config"]["optuna"]["num_trials"]
        self.num_startup_trials = config["parameter_optimization_config"]["optuna"]["num_startup_trials"]
        self.n_ei_candidates = config["parameter_optimization_config"]["optuna"]["num_ei_candidates"]
        self.multivariate = config["parameter_optimization_config"]["optuna"]["multivariate"]

        if self.optim_alg == "bayesian_optimization":
            self.sampler = TPESampler(n_startup_trials=self.num_startup_trials,
                                      n_ei_candidates=self.n_ei_candidates,
                                      gamma=increased_exploration_gamma,
                                      multivariate=self.multivariate,
                                      seed=self.random_seed,
                                      constant_liar=True)
        elif self.optim_alg == "random_search":
            self.sampler = RandomSampler(seed=self.random_seed)


        self.num_training_timesteps = config["agent_training_configuration"]["number_training_timesteps"]
        self.eval_every_n_batches = config["agent_training_configuration"]["eval_every_n_batches"]
        self.max_number_steps_per_episode = config["agent_training_configuration"]["max_number_steps_per_episode"]
        self.number_envs_for_train = config["agent_training_configuration"]["number_environments_for_training"]
        self.batch_size = config["agent_training_configuration"]["training_batch_size"]

        self.algorithm_name = config["agent_training_configuration"]["stable-baselines-algorithm"]
        self.policy_name = config["agent_training_configuration"]["stable-baselines-policy"]
        self.sb3_contrib = config["agent_training_configuration"]["sb3-contrib"]
        self.device_name = config["agent_training_configuration"]["device"]

        self.parameter_bounds = config["parameters_bounds"]

        self.db_url = config["configuration"]["database_url"]
        self.save_videos = config["configuration"]["save_videos"]

        self.user_train_environment = user_train_environment
        self.user_video_recorder_optuna = user_video_recorder_optuna

        self.howto_eval_rewards = config["parameter_optimization_config"]["single_objective_evaluation_function"]
        self.user_evaluate_policy = user_evaluate_policy

        # Set random seed for reproducibility
        self.set_random_seed()

        # Calculate total evaluations for pruner warmup
        total_n_evals = self.num_training_timesteps // (self.max_number_steps_per_episode * self.number_envs_for_train * self.eval_every_n_batches)
        
        self.pruner = MedianPruner(n_startup_trials=self.num_startup_trials, n_warmup_steps=total_n_evals)

        # Create Optuna study
        self.study = optuna.create_study(storage=self.db_url,
                                    sampler=self.sampler,
                                    pruner=self.pruner,
                                    study_name=self.optim_alg,
                                    direction="maximize",
                                    load_if_exists=self.continue_from_existing_database)
    
    
    def fit(self):
        """
        Perform hyperparameter optimization using Optuna.

        Optimizes the objective function over a defined number of trials,
        handling interruptions and exceptions gracefully.
        """

        # Set pytorch num threads to 1 for faster training.
        torch.set_num_threads(1)
        try:
            self.study.optimize(self.objective,
                        n_trials=self.num_trials,
                        timeout=None,
                        n_jobs=1,
                        # callbacks=[trial_callback],
                        show_progress_bar=False)
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        except Exception as e:
            print(f"An error occurred during optimization: {e}")

        print("Number of finished trials: ", len(self.study.trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))


    def set_random_seed(self):
        """
        Set random seeds for reproducibility in Python, NumPy, random, and PyTorch.

        Args:
        - seed (int): Random seed value.
        """

        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)


    def sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample parameters for a given Optuna trial.

        Args:
        - trial (optuna.Trial): Optuna trial object.

        Returns:
        - params (dict): Dictionary of sampled parameters.
        - environment_kwargs (dict): Dictionary of environment-specific parameters.
        """

        params = {}
        policy_params = {}

        policy_kwargs = {}
        environment_kwargs = {}

        for param, config in self.parameter_bounds.items():
            if config['searchable']:
                if config['integer']:
                    sampled_value = trial.suggest_int(name=param, low=config['start'], high=config['stop'])
                else:
                    log = config.get('log', False)
                    sampled_value = trial.suggest_float(name=param, low=config['start'], high=config['stop'], log=log)
            else:
                sampled_value = config['user_preference']

            if config['type'] == 'policy_kwargs':
                policy_params[param] = sampled_value
            elif config['type'] == 'environment_kwargs':
                environment_kwargs[param] = sampled_value
            else:
                params[param] = sampled_value

        # Handle specific transformations
        if 'activation_fn' in policy_params:
            policy_kwargs['activation_fn'] = nn.ReLU if policy_params.pop('activation_fn') > 0.5 else nn.Tanh

        # Construct the network architecture for both policy and value functions
        if 'policy_arch_num_layers' in policy_params and 'policy_arch_num_neurons' in policy_params:
            num_layers = policy_params.pop('policy_arch_num_layers')
            num_neurons = policy_params.pop('policy_arch_num_neurons')
            policy_kwargs['net_arch'] = dict(pi=[num_neurons] * num_layers, vf=[num_neurons] * num_layers)

        # Handle value function architecture if distinct from policy
        if 'value_arch_num_layers' in policy_params and 'value_arch_num_neurons' in policy_params:
            value_layers = policy_params.pop('value_arch_num_layers')
            value_neurons = policy_params.pop('value_arch_num_neurons')
            policy_kwargs['net_arch']['vf'] = [value_neurons] * value_layers

        params.update({
            "n_steps": self.max_number_steps_per_episode,
            "batch_size": self.batch_size,
            "policy_kwargs": policy_kwargs,
            "verbose": True,
            "device": self.device_name
        })

        return params, environment_kwargs


    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna to optimize.

        Args:
        - trial (optuna.Trial): Optuna trial object.

        Returns:
        - float: Mean reward obtained from the optimization process.
        """
        self.set_random_seed()

        torch.set_num_threads(1)

        kwargs, env_kwargs = self.sample_params(trial)

        train_env = self.user_train_environment(self.random_seed, self.number_envs_for_train, **env_kwargs)

        if self.sb3_contrib:
            algorithm_class = getattr(sb3_contrib, self.algorithm_name)
        else:
            algorithm_class = getattr(stable_baselines3, self.algorithm_name)

        model = algorithm_class(self.policy_name, train_env, seed=self.random_seed, **kwargs)

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path by going one level up and then down to the 'results' folder
        model_path = os.path.join(script_dir, "../results", f"{trial.study.study_name}", "models_saved/")
        model_path = os.path.abspath(model_path)  # Resolve the relative path to an absolute path

        #model_path = f"../../results/{trial.study.study_name}/models_saved/"
        os.makedirs(model_path, exist_ok=True)
        model_filename = model_path + f"/trial_{trial.number}_model.zip"
        save_extra_info_filename = model_path + f"/trial_{trial.number}_extra_info.json"

        eval_freq = self.max_number_steps_per_episode * self.number_envs_for_train * self.eval_every_n_batches
        try:
            nan_encountered = False
            should_prune = False
            max_objective_observed = float("-inf")
            obj_value = 0.0
            for timesteps in range(0, self.num_training_timesteps, eval_freq):
                model.learn(eval_freq, callback=None, progress_bar=False, reset_num_timesteps=False)
                print(f"Training is pausing at timestep: {timesteps + eval_freq} for evaluation of the Objective Function.")

                with torch.no_grad():
                    obj_value, extra_info = self.user_evaluate_policy(self.random_seed, model, **env_kwargs)

                print(f"Objective Value: {obj_value}.")

                # Save model based on the evaluation method
                if self.howto_eval_rewards == "max":
                    if obj_value > max_objective_observed:
                        max_objective_observed = obj_value
                        model.save(model_filename)
                        print(f"New best model saved with objective value: {obj_value}.")

                        if extra_info is not None:
                            print(f"Saving extra info: {extra_info}")

                            with open(save_extra_info_filename, 'w') as file:
                                json.dump(extra_info, file, indent=4) 
                    
                    trial.report(max_objective_observed, step=timesteps)
                else:
                    trial.report(obj_value, step=timesteps)
                    # Save model after the training is done
                    model.save(model_filename)

                    if extra_info is not None:
                        print(f"Saving extra info: {extra_info}")

                        with open(save_extra_info_filename, 'w') as file:
                            json.dump(extra_info, file, indent=4) 

                if trial.should_prune():
                    should_prune = True
                    raise optuna.exceptions.TrialPruned()
            
            if self.save_videos:
                video_path = os.path.join(script_dir, "../results", f"{trial.study.study_name}", "videos", f"trial_{trial.number}/")
                video_path = os.path.abspath(video_path)
                os.makedirs(video_path, exist_ok=True)
                self.user_video_recorder_optuna(self.random_seed, model_filename, video_path, env_kwargs)
        except optuna.exceptions.TrialPruned:
            pass
        except Exception as e:
            print("An error ocurred: ", e)
            nan_encountered = True
        finally:
            model.env.close()
            train_env.close()

            del model
            del train_env

            # Free up memory
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

        if nan_encountered:
            return float("nan")
            
        if should_prune:
            raise optuna.exceptions.TrialPruned()
        
    
        if self.howto_eval_rewards == "max":
            return max_objective_observed
        else:
            return obj_value
