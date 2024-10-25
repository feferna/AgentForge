import os
import cv2
import numpy as np
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

from environments.FovealLunarLander import make_vec_envs


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
    useTrainingRewardForOptimization = False
    # Flag to determine if the training reward should be used for parameter optimization
    # Only has effect when using Single Objective Optimizations.
    # Turning off this flag only makes sense if you using a Vision POMDP agent.
    # For MDP Agents, you only have a single reward, instead of a reward shaping. 
    # For example:
    # - True: (external_reward + w * internal_reward) is used for both agent's training and parameter optimization
    # - False: (external_reward + w * internal_reward) is used only for agent's training
    #          (external_reward) is used for parameter optimization

    num_eval_episodes = 200
    set_random_seed(seed=seed)
    model.set_random_seed(seed=seed)
 
    torch.set_num_threads(1)

    # Configure evaluation environment
    # Make random seed of the eval environment different from any trained environment
    eval_env = make_vec_envs(1, seed=(seed+100), **environment_kwargs)

    # Use an alternative reward to evaluate the performance of the agent
    landings_count = 0
    rewards = []
    alternative_rewards = []

    #eval_env.seed(seed=seed)
    obs = eval_env.reset()
    for _ in range(num_eval_episodes):
        eval_alternative_reward = 0.0
        eval_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = eval_env.step(action)
            
            eval_reward += rew

            if 'landed' in info[0] and info[0]['landed']:
                landings_count += 1

            if not useTrainingRewardForOptimization:
                eval_alternative_reward += info[0]['envr']
        
        rewards.append(eval_reward)

        if not useTrainingRewardForOptimization:
            alternative_rewards.append(eval_alternative_reward)

    extra_info = {"landing_success_rate": landings_count / num_eval_episodes}

    eval_env.close()
    del eval_env
    
    if useTrainingRewardForOptimization:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        extra_info['std_reward'] = std_reward

        return mean_reward, extra_info
    else:
        mean_alternative_reward = np.mean(alternative_rewards)
        std_alternative_reward = np.std(alternative_rewards)
        extra_info['std_alternative_reward'] = std_alternative_reward

        return mean_alternative_reward, extra_info
    

def user_record_video(seed, model_path, video_folder, environment_kwargs, num_episodes=20, video_length=10000):
    torch.set_num_threads(1)
    set_random_seed(seed=seed)

    num_eval_episodes = 20

    eval_env = make_vec_envs(1, seed=(seed+100), **environment_kwargs)
    eval_env.env_method("saveFrames", save_img_location=video_folder)

    model = PPO.load(model_path)

    # Generate some episodes to render images for the video
    obs = eval_env.reset()
    for _ in range(num_eval_episodes):
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = eval_env.step(action)

    eval_env.close()
    del eval_env

    video_from_images(video_folder, "gaze_tracked")
    video_from_images(video_folder, "fused")


def video_from_images(image_folder, startswith, fps=10, multiplier=1):
    images = [img for img in os.listdir(image_folder) if img.startswith(startswith)]
    images.sort()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(image_folder + '/' + "video_" + startswith + ".avi", fourcc, fps, (multiplier*500,500), )

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        img = cv2.resize(img, (multiplier*500, 500))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()