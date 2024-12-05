import argparse
import json
import yaml
import importlib
import random
import os
import sys
import gc
import numpy as np
import time
import torch
import stable_baselines3
import sb3_contrib
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import torch.nn as nn

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two directories
prj_root_dir = os.path.abspath(os.path.join(current_dir, "../"))

# Add the root directory to sys.path
sys.path.append(prj_root_dir)

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_id', type=int, required=True)
    parser.add_argument('--current_gen', type=int, required=True)
    parser.add_argument('--environment_kwargs', type=str, required=True)
    parser.add_argument('--model_kwargs', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--db_url', type=str, required=True)
    args = parser.parse_args()

    current_gen = args.current_gen
    solution_id = args.solution_id

    # Parse environment_kwargs and kwargs
    environment_kwargs = json.loads(args.environment_kwargs)
    model_kwargs = json.loads(args.model_kwargs)

    # Load configuration
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # environment_file = config["configuration"]["environment_file"]
    env_script_relative = config['configuration']['name_setup_user_file']

    max_n_train_timesteps = config["agent_training_configuration"]["number_training_timesteps"]
    max_episode_steps = config["agent_training_configuration"]["max_number_steps_per_episode"]
    n_train_envs = config["agent_training_configuration"]["number_environments_for_training"]
    use_sb3_contrib = config["agent_training_configuration"]["sb3-contrib"]
    eval_every_n_batches = config["agent_training_configuration"]["eval_every_n_batches"]
    batch_size = config["agent_training_configuration"]["training_batch_size"]

    algorithm_name = config["agent_training_configuration"]["stable-baselines-algorithm"]
    policy_name = config["agent_training_configuration"]["stable-baselines-policy"]
    random_seed = config["agent_training_configuration"]["random-seed"]

    howto_eval_rewards = config["parameter_optimization_config"]["single_objective_evaluation_function"]
    save_videos = config["configuration"]["save_videos"]

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

    engine = create_engine(args.db_url)
    Session = sessionmaker(bind=engine)
    metadata = MetaData()

    solutions_table = Table('solutions', metadata, autoload_with=engine)
    evaluation_history_table = Table('evaluation_history', metadata, autoload_with=engine)

    set_random_seed(seed=random_seed)
    torch.set_num_threads(1)

    model_kwargs["n_steps"] = max_episode_steps
    model_kwargs["batch_size"] = batch_size

    activation_fn_str = model_kwargs["policy_kwargs"]["activation_fn"]
    if activation_fn_str == 'ReLU':
        model_kwargs["policy_kwargs"]["activation_fn"] = nn.ReLU
    elif activation_fn_str == 'Tanh':
        model_kwargs["policy_kwargs"]["activation_fn"] = nn.Tanh

    train_env = user_train_environment(random_seed, n_train_envs, **environment_kwargs)

    if use_sb3_contrib:
        algorithm_class = getattr(sb3_contrib, algorithm_name)
    else:
        algorithm_class = getattr(stable_baselines3, algorithm_name)

    model = algorithm_class(policy_name, train_env, seed=random_seed, **model_kwargs)

    # Construct the absolute path to save stuff
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../results/pso/gen_{current_gen}/id_{solution_id}"))
    os.makedirs(model_path, exist_ok=True)
    model_filename = model_path + f"/model_gen_{current_gen}_id_{solution_id}.zip"
    save_extra_info_filename = model_path + f"/gen_{current_gen}_id_{solution_id}_extra_info.json"

    nan_encountered = False
    max_objective_observed = float("-inf")

    obj_vec = []
    extra_info_vec = []

    eval_freq = max_episode_steps * n_train_envs * eval_every_n_batches
    try:
        obj_value = 0.0
        for timesteps in range(0, max_n_train_timesteps, eval_freq):
            model.learn(eval_freq, callback=None, progress_bar=False, reset_num_timesteps=False)
            print(f"Training is pausing at timestep: {timesteps + eval_freq} for evaluation of the Objective Function.")

            obj_value, extra_info = user_evaluate_policy(random_seed, model, **environment_kwargs)

            obj_vec.append(obj_value)
            extra_info_vec.append(extra_info)

            # Save mean and std reward to the database
            while True:
                try:
                    with Session() as session:
                        session.execute(evaluation_history_table.insert().values(
                                                generation=current_gen,
                                                solution_id=solution_id,
                                                timestep=(timesteps+eval_freq),
                                                fitness=obj_value)
                                        )
                        session.commit()
                    break  # Break the loop if the operation is successful
                except Exception as db_error:
                    print("An error occurred during database operation: ", db_error)
                    print("Retrying in 60 seconds...")
                    time.sleep(60)  # Wait for 1 minute before retrying

            print(f"Objective Value: {obj_value}.")

            # Save model based on the evaluation method
            if howto_eval_rewards == "max":
                if obj_value > max_objective_observed:
                    max_objective_observed = obj_value
                    model.save(model_filename)

                    if extra_info is not None:
                        print(f"Saving extra info: {extra_info}")

                        with open(save_extra_info_filename, 'w') as file:
                            json.dump(extra_info, file, indent=4) 
            else:
                # Save model after the training is done
                model.save(model_filename)

                if extra_info is not None:
                    print(f"Saving extra info: {extra_info}")

                    with open(save_extra_info_filename, 'w') as file:
                        json.dump(extra_info, file, indent=4)

        if save_videos:
            video_filename = model_path
            user_record_video(random_seed, model_filename, video_filename, environment_kwargs)

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

    timestep = timesteps+eval_freq

    if howto_eval_rewards == "max":
        obj = np.max(obj_vec)
    else:
        obj = obj_value

    if nan_encountered:
        obj = float("-inf")
    
    print(f"Solution {solution_id} fitness: {obj}")

    # Reopen a session to update the database
    with Session() as update_session:
        update_session.execute(
            solutions_table.update()
            .where(solutions_table.c.id == solution_id)
            .values(fitness=obj, evaluated=1)
        )
        update_session.commit()


if __name__ == "__main__":
    main()
