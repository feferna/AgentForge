import os
import numpy as np
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder

from environments.PixelLunarLander import make_vec_envs


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def user_train_environment(seed, n_train_envs, **environment_kwargs):
    torch.set_num_threads(1)

    set_random_seed(seed=seed)
    
    # Configure training environment
    train_env = make_vec_envs(n_train_envs, seed=seed, **environment_kwargs)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

    return train_env


def user_evaluate_policy(seed, model, **environment_kwargs):
    num_eval_episodes = 300
    set_random_seed(seed=seed)
    model.set_random_seed(seed=seed)
 
    torch.set_num_threads(1)

    # Configure evaluation environment
    # Make random seed of the eval environment different from any trained environment
    eval_env = make_vec_envs(1, seed=(seed+100), **environment_kwargs)

    # Use an alternative reward to evaluate the performance of the agent
    landings_count = 0
    rewards = []

    #eval_env.seed(seed=seed)
    obs = eval_env.reset()
    for _ in range(num_eval_episodes):
        eval_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = eval_env.step(action)
            
            eval_reward += rew

            if 'landed' in info[0] and info[0]['landed']:
                landings_count += 1
        
        rewards.append(eval_reward)

    extra_info = {"landing_success_rate": landings_count / num_eval_episodes}

    eval_env.close()
    del eval_env
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    extra_info['std_reward'] = std_reward

    return mean_reward, extra_info

def user_record_video(seed, model_path, video_folder, environment_kwargs, num_episodes=20, video_length=10000):
    set_random_seed(seed=seed)
    torch.set_num_threads(1)

    eval_env = make_vec_envs(1, seed=(seed+100), **environment_kwargs)
    eval_env.reset()

    model = PPO.load(model_path)
    model.set_random_seed(seed=seed)

    video_recorder = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length
    )

    current_episode = 0

    video_recorder.seed(seed=seed)
    obs = video_recorder.reset()
    while current_episode < num_episodes:
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = video_recorder.step(action)
            
        current_episode += 1

    eval_env.close()
    video_recorder.close()