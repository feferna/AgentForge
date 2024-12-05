"""
User-defined functions for setting up and interacting with the custom RL environment. These functions 
are used in conjunction with hyperparameter optimization techniques like PSO or Optuna, to train, 
evaluate, and record videos of RL agents.

Each function in this script is intended to be plugged into the optimization process. The user 
must define the logic for setting up training environments, evaluating policy performance, and 
recording videos of the agent's behavior in the environment.

Functions:
----------
1. user_train_environment(seed, n_train_envs, **environment_kwargs)
    - Sets up the training environment for reinforcement learning using user-specified parameters.
    - Should return a vectorized training environment with common wrappers for observation normalization, 
      stacking frames, and monitoring.

2. user_evaluate_policy(seed, model, **environment_kwargs)
    - Evaluates the policy of the trained model in a separate evaluation environment.
    - Should return a mean objective value (such as average reward) and additional user-defined metrics.

3. user_record_video(seed, model_path, video_folder, environment_kwargs, num_episodes=20, video_length=10000)
    - Optionally records video of the agent's performance in the environment.
    - Videos can be recorded for a specified number of episodes, with each video lasting a defined number of frames.
"""

# Import whatever you need for your RL agent.

def user_train_environment(seed, n_train_envs, **environment_kwargs):
    """
    Set up reinforcement learning environments for training.

    Args:
    - seed (int): Random seed for reproducibility.
    - n_train_envs (int): Number of training environments to create.
    - **environment_kwargs: Additional keyword arguments for environment creation.

    Returns:
    - train_env (VecEnv): Configured training environment with VecFrameStack, VecNormalize, and VecMonitor.
    """
    # USER SUPPLIED CODE
    pass
    


def user_evaluate_policy(seed, model, **environment_kwargs):
    """
    Evaluate the policy of a given model on a separate evaluation environment.

    Args:
    - seed (int): Random seed for reproducibility.
    - model (BaseAlgorithm): The RL model to evaluate.
    - **environment_kwargs: Additional keyword arguments for environment creation.

    Returns:
    - mean_objective_value (float): Mean objective value obtained during evaluation.
    - extra_info (dict): Dictionary with additional metrics defined by the user. Some examples:
      - 'std_reward' (float): Standard deviation of the reward.
      - 'landing_success_rate' (float): Rate of successful landings.
    """
    # USER SUPPLIED CODE
    pass

def user_record_video(seed, model_path, video_folder, environment_kwargs, num_episodes=20, video_length=10000):
    """
    Optional: Record videos of the agent's performance in the environment.

    Args:
    - seed (int): Random seed for reproducibility.
    - model_path (str): Path to the saved RL model.
    - video_folder (str): Directory to save recorded videos.
    - environment_kwargs: Additional keyword arguments for environment creation.
    - num_episodes (int, optional): Number of episodes to record. Default is 20.
    - video_length (int, optional): Length of each video in frames. Default is 10,000.

    Returns: None
    """
    # USER SUPPLIED CODE
    pass