from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

class LIBEROImageWrapper(gym.Env):  
    def __init__(self, 
        env: BDDLBaseDomain,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1    
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def set_init_state(self, init_state):
        obs = self.env.set_init_state(init_state)
        return obs
    
    def reset(self):
        if self.init_state is not None:
            raise NotImplementedError('init_state not supported yet')
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            np.random.seed(seed=seed)
            raw_obs = self.env.reset()
            state = self.env.get_sim_state()
            self.seed_state_map[seed] = state
            self._seed = None
        else:
            raw_obs = self.env.reset()
        
        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def close(self): 
        self.env.close()
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        # import pdb; pdb.set_trace()
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = self.render_cache
        img = img[::-1]
        return img
