"""
pso_optimizer.py

This module implements the Particle Swarm Optimization (PSO) algorithm as a subclass of `BaseOptimizer`.
It includes methods for initializing the optimization process, updating particles, and executing the PSO algorithm.

Usage:
    Extend the `BaseOptimizer` class to implement PSO-specific methods such as `initialize_swarm()`, `update_particle()`,
    `update_swarm()`, `fit()`, and `best()`, following the abstract method definitions in `BaseOptimizer`.

Example:
    - Define a `PSOOptimizer` class as a subclass of `BaseOptimizer`.
    - Implement PSO-specific initialization (`initialize_swarm()`), particle update (`update_particle()`),
      swarm update (`update_swarm()`), optimization execution (`fit()`), and best solution retrieval (`best()`).

Prerequisites:
    - Requires Python environment with dependencies specified in `conda-environment.yml`.
    - Configuration file (`particle_swarm_optimization.yaml`) defining PSO parameters, parameter bounds, and database settings.

"""

import os
import traceback
import shutil
import numpy as np
import torch
import random
import time
import yaml
from optimizers.base_optimizer import BaseOptimizer
from sqlalchemy import create_engine, Column, Integer, Float, MetaData, Table
from sqlalchemy.orm import sessionmaker


class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) algorithm implementation.

    Attributes:
        current_iteration (int): Current iteration number of the optimization process.
        NUM_ITERATIONS (int): Total number of iterations/generations for PSO.
        NUM_SOLUTIONS (int): Population size (number of solutions/particles).
        w (float): Inertia weight for PSO.
        c1 (float): Cognitive parameter (learning rate) for PSO.
        c2 (float): Social parameter (learning rate) for PSO.
        continue_from_existing_db (bool): Flag indicating whether to continue from an existing database.
        parameter_bounds (dict): Dictionary defining bounds and configuration for optimization parameters.
        global_best (dict): Dictionary storing the best global solution found during optimization.
        engine (SQLAlchemy Engine): Engine for database operations.
        session (SQLAlchemy Session): Database session for executing queries.
        metadata (SQLAlchemy MetaData): Metadata for database schema operations.
        solutions_table (SQLAlchemy Table): Table to store solutions and their states.
        solution_history_table (SQLAlchemy Table): Table to store history of solution states across generations.
        global_best_history_table (SQLAlchemy Table): Table to store history of global best solutions across generations.
        personal_best_history_table (SQLAlchemy Table): Table to store history of personal best solutions across generations.
        velocity_history_table (SQLAlchemy Table): Table to store history of particle velocities across generations.
        evaluation_history_table (SQLAlchemy Table): Table to store history of evaluations for solutions.

    Methods:
        __init__(self, config_file, set_environment_func):
            Initializes the PSO optimizer with configuration from a YAML file.
        
        initialize_database(self, continue_from_existing_db):
            Initializes the database schema for PSO-specific tables and optionally continues from an existing database.
        
        load_existing_state(self):
            Loads existing optimization state from the database to resume or restart the current iteration.
        
        initialize_swarm(self):
            Initializes the swarm of particles with random positions and velocities within defined parameter bounds.
        
        update_particle(self, particle):
            Updates the position and velocity of a particle based on PSO rules.
        
        update_swarm(self):
            Updates the entire swarm of particles by evaluating fitness, updating global and personal bests,
            and recording history of particle states.
        
        fit(self):
            Executes the PSO optimization process over a specified number of iterations by assigning solutions,
            evaluating fitness, updating the swarm, and checking termination conditions.
        
        best(self):
            Retrieves the best solution found during PSO optimization based on `best_fitness`.

    """

    def __init__(self, config_file, user_train_environment_func, user_evaluate_policy_func, user_recorder_func):
        """
        Initializes the PSO optimizer with configuration from a YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.
            user_train_environment_func (function): Function to set the custom environment for optimization.
            user_evaluate_policy_func (function): Function to evaluate the policy.
            user_recorder_func (function): Function to record videos of the agent's performance.

        Raises:
            FileNotFoundError: If the `config_file` is not found.

        """

        super().__init__(config_file, user_train_environment_func, user_evaluate_policy_func, user_recorder_func)

        self.current_generation = 0

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.random_seed = config["agent_training_configuration"]["random-seed"]
        
        # Initialize particle swarm optimization configuration
        self.NUM_ITERATIONS = config["parameter_optimization_config"]['particle_swarm_optimization']['num_generations']
        self.NUM_SOLUTIONS = config["parameter_optimization_config"]['particle_swarm_optimization']['population_size']
        self.w = config["parameter_optimization_config"]['particle_swarm_optimization']['w']
        self.c1 = config["parameter_optimization_config"]['particle_swarm_optimization']['c1']
        self.c2 = config["parameter_optimization_config"]['particle_swarm_optimization']['c2']

        self.continue_from_existing_db = config["agent_training_configuration"]["continue_from_existing_database"]

        # Initialize parameter bounds configuration
        self.parameter_bounds = {}
        for param, param_config in config['parameters_bounds'].items():
            self.parameter_bounds[param] = {
                'type': param_config['type'],
                'searchable': param_config['searchable'],
                'user_preference': param_config['user_preference'],
                'integer': param_config['integer'],
                'start': param_config['start'],
                'stop': param_config['stop']
            }

        self.global_best = {'fitness': float('-inf')}
        for param in self.parameter_bounds.keys():
            self.global_best[param] = None

        self.set_random_seed()

        self.engine = create_engine(self.db_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.metadata = MetaData()

        if self.continue_from_existing_db:
            print("Continuing optimization from existing database...")
        
        self.initialize_database(self.continue_from_existing_db)


    def set_random_seed(self):
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)


    def initialize_database(self, continue_from_existing_db):
        """
        Initializes the database schema for PSO-specific tables and optionally continues from an existing database.

        Args:
            continue_from_existing_db (bool): Flag indicating whether to continue from an existing database.

        """

        def create_columns(velocity_best_params=True):
            """
            Creates column definitions for SQLAlchemy tables based on parameter bounds configuration.

            Args:
                velocity_best_params (bool): Flag to include velocity and best parameter columns.

            Returns:
                list: List of `Column` objects defining table columns.

            """

            columns = [
                Column('id', Integer, primary_key=True),
                Column('generation', Integer),
                Column('fitness', Float),
                Column('evaluated', Integer, default=0),
                Column('assigned_to', Integer),
                Column('timestep', Integer),
                Column('best_fitness', Float),
            ]
            for param, param_config in self.parameter_bounds.items():
                if param_config['integer']:
                    columns.append(Column(param, Integer))
                    if velocity_best_params:
                        columns.append(Column(f'velocity_{param}', Integer))
                        columns.append(Column(f'best_{param}', Integer))
                else:
                    columns.append(Column(param, Float))
                    if velocity_best_params:
                        columns.append(Column(f'velocity_{param}', Float))
                        columns.append(Column(f'best_{param}', Float))
            return columns

        # Define columns for solutions table
        self.solutions_table = Table('solutions', self.metadata, *create_columns())
        
        # Define columns for solution history table
        self.solution_history_table = Table('solution_history', self.metadata,
                                            Column('generation', Integer),
                                            Column('solution_id', Integer),
                                            Column('fitness', Float),
                                            *create_columns(velocity_best_params=False)[7:])  # Reuse the columns except 'id', fitness, evaluated, assigned_to
        
        # Define columns for global best history table
        self.global_best_history_table = Table('global_best_history', self.metadata,
                                               Column('generation', Integer),
                                               Column('fitness', Float),
                                               *create_columns(velocity_best_params=False)[7:])  # Reuse the columns except 'id', fitness, evaluated, assigned_to

        # Define columns for personal best history table
        self.personal_best_history_table = Table('personal_best_history', self.metadata,
                                                 Column('generation', Integer),
                                                 Column('solution_id', Integer),
                                                 Column('fitness', Float),
                                                 *create_columns(velocity_best_params=False)[7:])  # Reuse the columns except 'id', fitness, evaluated, assigned_to
        
        # Define columns for personal best history table
        self.velocity_history_table = Table('velocity_history', self.metadata,
                                                 Column('generation', Integer),
                                                 Column('solution_id', Integer),
                                                 Column('fitness', Float),
                                                 *create_columns(velocity_best_params=False)[7:])  # Reuse the columns except 'id', fitness, evaluated, assigned_to
        
        self.evaluation_history_table = Table('evaluation_history', self.metadata,
                                             Column('generation', Integer),
                                             Column('solution_id', Integer),
                                             Column('timestep', Integer),
                                             Column('fitness', Float)
                                            )

        if not continue_from_existing_db:
            self.metadata.drop_all(self.engine)
            self.metadata.create_all(self.engine)
            self.initialize_swarm()
        else:
            self.metadata.create_all(self.engine, checkfirst=True)
            self.load_existing_state()

    def load_existing_state(self):
        """
        Loads existing optimization state from the database to resume or restart the current iteration.

        - Loads the current iteration number.
        - Deletes existing global best for the current generation to restart the iteration.
        - Loads the global best from the existing database.
        - Erases evaluation history of the current generation.
        - Overwrites 'best_fitness' column in `solutions_table` with 'fitness' values from `personal_best_history_table`
          of the previous generation.

        """

        # Load the current iteration from the existing database
        latest_generation = self.session.query(self.solutions_table).order_by(self.solutions_table.c.generation.desc()).first()
        if latest_generation:
            self.current_generation = latest_generation.generation

        # Load the global best from the existing database
        global_best = self.session.query(self.global_best_history_table).order_by(self.global_best_history_table.c.generation.desc()).first()
        if global_best:
            # Convert the result to a dictionary and exclude the generation field
            global_best_dict = {col: val for col, val in global_best._asdict().items() if col != 'generation'}
            self.global_best.update(global_best_dict)

        # Query solutions that have not been evaluated in the current generation
        solutions_not_evaluated = self.session.query(
            self.solutions_table.c.id
        ).filter(
            self.solutions_table.c.generation == self.current_generation,
            self.solutions_table.c.evaluated == 0
        ).all()

        if solutions_not_evaluated:
            solution_ids_not_evaluated = [solution.id for solution in solutions_not_evaluated]

            # Erase the evaluation history only for solutions that haven't been evaluated
            self.session.execute(
                self.evaluation_history_table.delete().where(
                    self.evaluation_history_table.c.generation == self.current_generation,
                    self.evaluation_history_table.c.solution_id.in_(solution_ids_not_evaluated)
                )
            )
            self.session.commit()

            # Update the 'best_fitness' for solutions that haven't been evaluated
            latest_personal_bests = self.session.query(
                self.personal_best_history_table.c.solution_id,
                self.personal_best_history_table.c.fitness
            ).filter_by(generation=(self.current_generation - 1)).all()

            for personal_best in latest_personal_bests:
                if personal_best.solution_id in solution_ids_not_evaluated:
                    self.session.execute(
                        self.solutions_table.update()
                        .where(self.solutions_table.c.id == personal_best.solution_id)
                        .values(best_fitness=personal_best.fitness)
                    )

            self.session.commit()


    def initialize_swarm(self):
        """
        Initializes the swarm of particles with random positions and velocities within defined parameter bounds.

        """

        for _ in range(self.NUM_SOLUTIONS):
            values = {
                'generation': self.current_generation,
                'fitness': float('-inf'),
                'evaluated': 0,
                'assigned_to': None,
                'best_fitness': float('-inf'),
            }
            for param, param_config in self.parameter_bounds.items():
                if param_config['searchable']:
                    if param_config['integer']:
                        initial_value = random.randint(param_config['start'], param_config['stop'])
                        initial_velocity = round(random.uniform(0, abs(param_config['stop'] - param_config['start']) / 2))
                    else:
                        initial_value = round(random.uniform(param_config['start'], param_config['stop']), 4)
                        initial_velocity = round(random.uniform(0, abs(param_config['stop'] - param_config['start']) / 2), 4)
                else:
                    initial_value = param_config['user_preference']
                    initial_velocity = 0
                
                values[param] = initial_value
                values[f'velocity_{param}'] = initial_velocity
                values[f'best_{param}'] = initial_value

            self.session.execute(self.solutions_table.insert().values(**values))
        
        self.session.commit()

    def update_particle(self, particle, session):
        """
        Updates the position and velocity of a particle based on PSO rules.

        Args:
            particle (SQLAlchemy Row): Particle to be updated.

        """

        inertia_weight = self.w
        cognitive_weight = self.c1
        social_weight = self.c2

        for param in self.parameter_bounds.keys():
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

            current_position = getattr(particle, param)
            current_velocity = getattr(particle, f'velocity_{param}')
            personal_best_position = getattr(particle, f'best_{param}')
            global_best_position = self.global_best[param]

            new_velocity = (
                inertia_weight * current_velocity +
                cognitive_weight * r1 * (personal_best_position - current_position) +
                social_weight * r2 * (global_best_position - current_position)
            )

            new_position = current_position + new_velocity

            # Apply bounds to positions
            bounds = self.parameter_bounds[param]
            if bounds['integer']:
                new_position = max(bounds['start'], min(bounds['stop'], round(new_position)))
            else:
                new_position = max(bounds['start'], min(bounds['stop'], new_position))

            session.execute(
                                self.solutions_table.update()
                                .where(self.solutions_table.c.id == particle.id)
                                .values(**{param: new_position})
                            )

            session.execute(
                                self.solutions_table.update()
                                .where(self.solutions_table.c.id == particle.id)
                                .values(**{f'velocity_{param}': new_velocity})
                            )
        
        # Reset the particle's evaluated counter
        session.execute(self.solutions_table.update()
                             .where(self.solutions_table.c.id == particle.id)
                             .values(generation=self.current_generation+1, evaluated=0)
        )
        
        session.commit()

    def update_swarm(self, session):
        """
        Updates the entire swarm of particles by evaluating fitness, updating global and personal bests,
        and recording history of particle states.

        """

        evaluated_particles = session.query(self.solutions_table).filter_by(evaluated=1).all()

        for particle in evaluated_particles:
            if particle.fitness > particle.best_fitness:
                for param in self.parameter_bounds.keys():
                    session.execute(
                        self.solutions_table.update()
                        .where(self.solutions_table.c.id == particle.id)
                        .values(**{f'best_{param}': getattr(particle, param)})
                    )

                session.execute(
                        self.solutions_table.update()
                        .where(self.solutions_table.c.id == particle.id)
                        .values(**{f'best_fitness': particle.fitness})
                    )

                if particle.fitness > self.global_best['fitness']:
                    for param in self.parameter_bounds.keys():
                        self.global_best[param] = getattr(particle, param)
                    self.global_best['fitness'] = particle.fitness

                    # When a new global best solution is found, show it and move files.
                    print("New global best found: ", flush=True)
                    print(f"\tFitness: {self.global_best['fitness']}", flush=True)
                    print(f"\tParameters: {self.global_best}", flush=True)

                    model_path = f"../../results/pso/gen_{self.current_generation}/id_{particle.id}/"
                    
                    gbest_path = f"../../results/pso/global_best/gen_{self.current_generation}/"
                    os.makedirs(gbest_path, exist_ok=True)

                    # Copy all files from model_path to gbest_path
                    for filename in os.listdir(model_path):
                        # Full path to the file
                        src = os.path.join(model_path, filename)
                        dst = os.path.join(gbest_path, filename)
                        
                        # Copy the file
                        shutil.copy(src, dst)

        session.commit()

        # Save global best to history
        session.execute(self.global_best_history_table.insert().values(
            generation=self.current_generation,
            **self.global_best
        ))
        session.commit()

        # Update particle positions and velocities
        particles = self.session.query(self.solutions_table).all()
        for particle in particles:
            # Save particle's state to history
            session.execute(self.solution_history_table.insert().values(
                generation=self.current_generation,
                solution_id=particle.id,
                fitness=particle.fitness,
                **{param: getattr(particle, param) for param in self.parameter_bounds.keys()}
            ))

            # Save particle's personal best to history
            session.execute(self.personal_best_history_table.insert().values(
                generation=self.current_generation,
                solution_id=particle.id,
                fitness=particle.best_fitness,
                **{param: getattr(particle, f'best_{param}') for param in self.parameter_bounds.keys()}
            ))

            # Save particle's velocity to history
            session.execute(self.velocity_history_table.insert().values(
                generation=self.current_generation,
                solution_id=particle.id,
                fitness=particle.best_fitness,
                **{param: getattr(particle, f'velocity_{param}') for param in self.parameter_bounds.keys()}
            ))

            # First save history, then update particle
            self.update_particle(particle, session)


        session.commit()

    def fit(self):
        """
        Executes the PSO optimization process over a specified number of iterations by assigning solutions,
        evaluating fitness, updating the swarm, and checking termination conditions.

        Returns:
            SQLAlchemy Row: Best solution found during PSO optimization based on 'best_fitness'.
        """
        retry_delay = 30  # seconds
        Session = sessionmaker(bind=self.engine)

        for iter in range(self.current_generation, self.NUM_ITERATIONS):
            print(f"Generation {iter}/{self.NUM_ITERATIONS-1}", flush=True)

            # Infinite retry logic for assigning solutions
            while True:
                try:
                    with Session() as session:
                        self.assign_solutions(session=session)
                    print("Assigned solutions to parallel programs.", flush=True)
                    break
                except Exception as e:
                    print(f"Error assigning solutions: {e}. Retrying in {retry_delay} seconds...", flush=True)
                    traceback.print_exc()
                    time.sleep(retry_delay)

            all_evaluated = False
            while not all_evaluated:
                print("Waiting for all solutions to be evaluated...", flush=True)
                time.sleep(30)
                
                while True:
                    try:
                        with Session() as session:
                            not_evaluated_count = session.query(self.solutions_table).filter_by(evaluated=0).count()
                        if not_evaluated_count == 0:
                            all_evaluated = True
                        break
                    except Exception as e:
                        print(f"Error querying evaluation status: {e}. Retrying in {retry_delay} seconds...", flush=True)
                        traceback.print_exc()
                        time.sleep(retry_delay)

            print("All solutions have been evaluated.", flush=True)
            self.current_generation = iter

            # Infinite retry logic for updating swarm
            while True:
                try:
                    with Session() as session:
                        self.update_swarm(session=session)
                    break
                except Exception as e:
                    print(f"Error updating swarm: {e}. Retrying in {retry_delay} seconds...", flush=True)
                    traceback.print_exc()
                    time.sleep(retry_delay)

        print("Optimization complete.", flush=True)
        return self.best()


    def best(self):
        """
        Retrieves the best solution found during PSO optimization based on 'best_fitness'.

        Returns:
            SQLAlchemy Row: Best solution found during PSO optimization based on 'best_fitness'.

        """
        
        best_solution = self.session.execute(self.solutions_table.select().where(self.solutions_table.c.best_fitness != None).order_by(self.solutions_table.c.best_fitness.asc()).limit(1)).fetchone()
        return best_solution
