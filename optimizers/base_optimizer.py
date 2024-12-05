"""
base_optimizer.py

This module defines an abstract base class `BaseOptimizer` for implementing custom optimization algorithms.
It includes methods for initializing configuration from a YAML file, assigning solutions to parallel processes,
and defines abstract methods `best()`, `fit()`, and `initialize_database()` that must be implemented by subclasses.

Usage:
    Extend the `BaseOptimizer` class to create custom optimization algorithms.
    Implement the abstract methods `best()`, `fit()`, and `initialize_database()` according to the specific algorithm.

Example:
    - Define a subclass of `BaseOptimizer` for Particle Swarm Optimization (PSO) or any other algorithm.
    - Implement methods `best()`, `fit()`, and `initialize_database()` based on the algorithm requirements.

Prerequisites:
    - Python environment with required dependencies (yaml, sqlalchemy).
    - Configuration file (`optimizer_config.yaml`) defining database URL and parallelization settings.

"""

import yaml
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Abstract base class for defining optimization algorithms.

    Attributes:
        db_url (str): URL to the database configured in the YAML configuration file.
        NUM_PARALLEL_PROCESSES (int): Number of parallel processes for execution.
        user_env_func (function): Function reference to set the custom environment.
        user_eval_policy_func (function): Function reference to evaluate the policy.
        user_recorder_func (function): Function reference to record videos of agent's performance.

    Methods:
        __init__(self, config_file, user_environment_func, user_evaluate_policy_func, user_recorder_func):
            Initializes the optimizer with configuration from a YAML file.

        assign_solutions(self):
            Assigns unsolved solutions from the database to parallel processes.

        best(self):
            Abstract method to be implemented by subclasses to return the best solution.

        fit(self):
            Abstract method to be implemented by subclasses to fit/train the optimizer.

        initialize_database(self):
            Abstract method to be implemented by subclasses to set up the database schema.

    """

    def __init__(self, config_file, user_train_environment_func, user_evaluate_policy_func, user_recorder_func):
        """
        Initializes the optimizer with configuration from a YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.
            user_train_environment_func (function): Function to set the custom environment for optimization.
            user_evaluate_policy_func (function): Function to evaluate the policy.
            user_recorder_func (function): Function to record videos of the agent's performance.

        Raises:
            FileNotFoundError: If the `config_file` is not found.

        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.db_url = config["configuration"]["database_url"]
        self.NUM_PARALLEL_PROCESSES = config['parallelization']['n_processes']
        self.user_train_env_func = user_train_environment_func
        self.user_eval_policy_func = user_evaluate_policy_func
        self.user_recorder_func = user_recorder_func

    def assign_solutions(self, session):
        """
        Assigns unsolved solutions from the database to parallel processes.

        This method divides unsolved solutions evenly among parallel processes based on `NUM_PARALLEL_PROCESSES`.

        """
        
        solutions = session.execute(self.solutions_table.select().where(self.solutions_table.c.evaluated == 0)).fetchall()
        
        for i, solution in enumerate(solutions):
            assigned_to = i % self.NUM_PARALLEL_PROCESSES
            session.execute(self.solutions_table.update().where(self.solutions_table.c.id == solution.id).values(assigned_to=assigned_to))
        
        session.commit()

    @abstractmethod
    def best(self):
        """
        Abstract method to be implemented by subclasses.

        Returns:
            Best solution found by the optimization algorithm.

        """
        pass

    @abstractmethod
    def fit(self):
        """
        Abstract method to be implemented by subclasses.

        Performs the optimization process.

        """
        pass

    @abstractmethod
    def initialize_database(self):
        """
        Abstract method to be implemented by subclasses.

        Initializes the database schema or prepares it for optimization.

        """
        pass
