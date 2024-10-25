import os, sys, random, math
from math import sqrt,pow
import numpy as np
import cv2, yaml
import gymnasium as gym
from gymnasium import spaces

class POMDPLunarLander(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(self, ffov=99, obs=36, par_scale=1.5, gray_coef=0.5, gaze_bonus=2, tr1=0.8, w1=0.7):
        super(POMDPLunarLander, self).__init__()

        # Load lunar lander
        self.vmenv = gym.make(
            "LunarLander-v2",
            continuous = True,
            render_mode = "rgb_array",
            gravity = -10.0,
            enable_wind = False,
            wind_power = 0,             # original 15
            turbulence_power = 0.2,     # original 1.5
        )

        self.par_scale = par_scale
        self.gray_coef = gray_coef
        self.gaze_bonus = gaze_bonus
        self.tr1 = tr1
        self.w1 = w1

        # Set internal parameters
        self.ffov_w, self.ffov_h = ffov, ffov
        self.obs_w, self.obs_h = obs, obs

        self.max_dist = 2.82 # For reward-scaling

        # Define Gymnasium spaces
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0]))
        self.observation_space = spaces.Box(0, 255, (1,3 * self.obs_h, self.obs_w), np.uint8)

        # Observation and state variables
        self.obs, self.vs_image, self.vs_prvs_image, self.info = {}, {}, {}, {}         # Observation, VSEnv image, self.info
        self.prvs_percept, self.percept = {}, {}                # Previous and current state of VSENV
        self.fused = None
        self.t, self.lifetime = 0, 0                            # Timestep, Lifetime of this env

        # Toggle to save frames to files when testing/evaluating
        self.save_frames = False

    # Take an image rendered by Lunar Lander. Resize and grayscale
    def obtain_pixelimage(self):
        vs_image = self.vmenv.render()
        vs_image  = cv2.resize(vs_image, (self.ffov_w, self.ffov_h), interpolation=cv2.INTER_LINEAR)
        vs_image = cv2.cvtColor(vs_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return vs_image
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.t, self.info["t"] = 0, 0

        # Reset Lunar Lander
        envstate, self.info = self.vmenv.reset()

        # Skip first states to make the lander visible
        for i in range (0, 20):
            envstate, envreward, envdone, envtruncated, _ = self.vmenv.step([0,0])

        # Obtain image of the display
        self.vs_image = self.obtain_pixelimage()
        self.vs_prvs_image = self.vs_image.copy()

        # Start position ax,ay for gaze in the center + noise
        a_x, a_y = self.np_random.random() / 5 + 0.4, self.np_random.random() / 5 + 0.4

        self.percept = self.attend(a_x, a_y)
        self.prvs_percept = self.percept.copy()
        self.obs = self.fuse_to_obs(self.percept, self.prvs_percept)
        self.info["dist_r"], self.info["reward"], self.info["envr"] = 0,0,0
        if self.save_frames: self.saveFrame(self.lifetime)

        return self.obs, self.info

    def step(self, action):
        self.t += 1
        self.info["t"] = self.t
        self.lifetime += 1
        self.info["lifetime"] = self.lifetime
        self.info["landed"] = False

        # Step action and obtain image from vmenv
        envstate, envreward, envdone, envtruncated, _ = self.vmenv.step(action[2:4])

        if envreward == 100:
            self.info["landed"] = True
        else:
            self.info["landed"] = False

        self.vs_image = self.obtain_pixelimage()
        self.vs_prvs_image = self.vs_image

        # Track lunar lander state for eye movement reward (dist_r)
        lunar_x = np.clip(envstate[0], -1, 1)
        lunar_y = np.clip(envstate[1], -1, 1)

        # Attend the display
        self.prvs_percept = self.percept.copy()
        a_x1, a_x2 = (action[0] + 1.0) / 2.0, (action[1] + 1.0) / 2.0
        self.percept = self.attend(a_x1, a_x2)
        self.obs = self.fuse_to_obs(self.percept, self.prvs_percept)

        # Distance reward
        dist_r = 1.0 - math.sqrt((lunar_x - action[0]) ** 2 + (- lunar_y - action[1]) ** 2) / (self.max_dist)
        if dist_r > self.tr1:
            dist_r *= self.gaze_bonus

        reward = dist_r + envreward * self.w1

        self.info["dist_r"], self.info["reward"], self.info["envr"] = dist_r, reward, envreward
        if self.save_frames: self.saveFrame(self.lifetime)

        done, truncated = envdone, envtruncated

        return self.obs, reward, done, truncated, self.info

    # Generates a sensory frame (percept) from pixelimage
    def attend (self, action_x, action_y):

        # map fovea coordinates to ffov coordinates (grid)
        ffov_y = int(action_x * self.ffov_w)
        ffov_x = int(action_y * self.ffov_h)
        ffov_x_start = max([0, int(ffov_x - self.obs_h/2)])
        ffov_y_start = max([0, int(ffov_y -self.obs_w/2)])
        ffov_x_end =  min([int(ffov_x+self.obs_h/2),self.ffov_h])
        ffov_y_end =  min([int(ffov_y+self.obs_w/2),self.ffov_w])
        fov = self.vs_image[ffov_x_start:ffov_x_end, ffov_y_start:ffov_y_end]
        fov = cv2.resize(fov, (self.obs_w, self.obs_h), interpolation=cv2.INTER_LINEAR)

        par_scale = self.par_scale
        par_x_start = max([0, int(ffov_x - self.obs_h * self.par_scale)])
        par_y_start = max([0, int(ffov_y -self.obs_w * self.par_scale)])
        par_x_end =  min([int(ffov_x+self.obs_h * self.par_scale),self.ffov_h])
        par_y_end =  min([int(ffov_y+self.obs_w * self.par_scale),self.ffov_w])
        par = self.vs_image[par_x_start:par_x_end, par_y_start:par_y_end]
        par = cv2.resize(par, (self.obs_w, self.obs_h), interpolation=cv2.INTER_LINEAR)
        #par = cv2.blur(par,(4,4))

        if self.t == 0:
            ior = np.zeros((self.ffov_w, self.ffov_h)).astype(np.uint8) # reset IOR map
        else:
            ior = self.percept["ior"]
        ior = ior * self.gray_coef # gray out
        ior[ffov_x_start:ffov_x_end, ffov_y_start:ffov_y_end] = 255 # latest patch
        ior = ior.astype(np.uint8)

        per = cv2.resize(self.vs_image, (self.obs_w, self.obs_h), interpolation=cv2.INTER_LINEAR)
        #per = cv2.blur(per,(5,5))


        new_sensory_frame = {
            "fov": fov,
            "par": par,
            "per": per,
            "ior": ior,
            "vec": np.array([action_x, action_y]),
            "fov_cx": np.array([ffov_x, ffov_y]),
        }
        return new_sensory_frame

    # Takes percepts and fuses them in to one observation vector
    def fuse_to_obs (self, obs, prev_obs):
        prvs_action_x, prvs_action_y = prev_obs["vec"][0], prev_obs["vec"][1]
        action_x, action_y = obs["vec"][0], obs["vec"][1]
        fov, prvs_fov, per = obs["fov"], prev_obs["fov"], obs["per"]
        par, prvs_par = obs["par"], prev_obs["par"]
        fov_T, prvs_fov_T, per_T  = np.transpose(fov), np.transpose(prvs_fov), np.transpose(per)
        par_T, prvs_par_T = np.transpose(par), np.transpose(prvs_par)
        par_S = par_T.reshape(1, self.obs_w, self.obs_h)
        prvs_par_S = prvs_par_T.reshape(1, self.obs_w, self.obs_h)
        per_S = per_T.reshape(1, self.obs_w, self.obs_h)  # Menikö h ja w oikeinpäin?
        ior = obs["ior"]

        ior_resized = cv2.resize(ior, (self.obs_w, self.obs_h),  interpolation=cv2.INTER_LINEAR)
        ior_resized = ior_resized[:,:,np.newaxis]
        ior_resized_T = np.transpose(ior_resized)

        fov_S = fov_T.reshape(1,self.obs_w, self.obs_h)
        prvs_fov_S = prvs_fov_T.reshape(1,self.obs_w, self.obs_h)

        # Define the observation vector
        fused = np.concatenate ( (fov_S, prvs_fov_S, ior_resized_T), axis = 1)
        self.fused = fused.T

        return fused.astype(np.uint8)

    def saveFrames(self, save_img_location):
        self.save_img_location = save_img_location
        self.save_frames = True

    def saveFrame(self, id = None): # Saves visual stack to files in /output
        if id == None:
            framenumber = '{:0>5}'.format(self.lifetime)
        else:
            framenumber = '{:0>5}'.format(id)
        if self.info["reward"] == 0:
            framenumber = framenumber +"r"
        #print('{0:6s} distr {1:.2f} envr {2:.2f} r_total {3:.2f}'.format(framenumber,  self.info["dist_r"], self.info["envr"], self.info["reward"]))

        save_path = self.save_img_location

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Display
        display_img = self.vs_image.copy()
        #cv2.imwrite('output/display' + framenumber + '.png', display_img)

        # Fused percept
        cv2.imwrite(f'{save_path}/fused{framenumber}.png', self.fused)

        # Gaze tracker
        display_x = self.percept['fov_cx'][0]
        display_y = self.percept['fov_cx'][1]

        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        display_img = cv2.circle(display_img, (display_y, display_x), int(self.obs_w/2), (0, 0, 255), 1)
        cv2.imwrite(f'{save_path}/gaze_tracked{framenumber}.png', display_img)

    def render(self):
        return self._render_frame()
    def _get_obs(self):
        return self.obs
    def _get_info(self):
        return self.info
    def _render_frame(self):
        print ("Error rendering frame")
        return None
    def close(self):
        self.vmenv.close()
        pass


def make_vec_envs(n_envs, seed=None, **environment_kwargs):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make_env(rank, **environment_kwargs):
        def _init():
            env = POMDPLunarLander(**environment_kwargs)
            if seed is not None:
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
                env.reset(seed=seed + rank)
            return env
        return _init
    return SubprocVecEnv([make_env(i, **environment_kwargs) for i in range(n_envs)])