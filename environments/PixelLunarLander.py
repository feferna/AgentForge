# custom_env.py
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import cv2
import numpy as np
import os
import torch
import random
from collections import deque

class PixelLunarLander(gym.Wrapper):
    def __init__(self, env, downsampling_size=40, skip=4):
        super(PixelLunarLander, self).__init__(env)
        self.resize_shape = (downsampling_size, downsampling_size)
        self.skip = skip
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.process_observation(obs), info

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, done, truncated, info_ = self.env.step(action)
            
            if reward == 100:
                landed = True
            else:
                landed = False

            total_reward += reward
            if done or truncated:
                break
        
        info['landed'] = landed
        info['info'] = info_
        
        return self.process_observation(obs), total_reward, done, truncated, info

    def process_observation(self, obs):
        pixels = obs['pixels']

        resized_obs = cv2.resize(pixels, self.resize_shape)
        gray_obs = cv2.cvtColor(resized_obs, cv2.COLOR_RGB2GRAY)
        gray_obs = np.expand_dims(gray_obs, axis=0)

        return gray_obs
    

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_stack):
        super(FrameStack, self).__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(n_stack, obs_shape[1], obs_shape[2]), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=0)


def make_vec_envs(n_envs, seed=None, **environment_kwargs):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def set_random_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    if seed is not None:
        set_random_seed(seed)

    def make_env(rank, **environment_kwargs):
        def _init():
            env = gym.make("LunarLander-v2", continuous=True, enable_wind=False, wind_power=0.0, turbulence_power=0.0, render_mode='rgb_array')
            env = PixelObservationWrapper(env, pixels_only=True)
            env = PixelLunarLander(env, **environment_kwargs)
            env = FrameStack(env, n_stack=4)
            if seed is not None:
                env.reset(seed=seed + rank)
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            return env
        return _init
    return SubprocVecEnv([make_env(i, **environment_kwargs) for i in range(n_envs)])
