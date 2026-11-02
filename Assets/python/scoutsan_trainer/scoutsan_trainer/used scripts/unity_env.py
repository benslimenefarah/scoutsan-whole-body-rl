

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class UnityGymnasiumEnv(gym.Env):
    def __init__(self, file_name=None, no_graphics=False, time_scale=20.0):
        self.channel = EngineConfigurationChannel()
        self._env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics,
                                     side_channels=[self.channel])
        self.channel.set_configuration_parameters(time_scale=time_scale)
        self._env.reset()
        self.behavior_name = list(self._env.behavior_specs)[0]
        spec = self._env.behavior_specs[self.behavior_name]

        obs_shape = spec.observation_specs[0].shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        n_cont = spec.action_spec.continuous_size
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_cont,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset()
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        while len(decision_steps) == 0:
            self._env.step()
            decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        return decision_steps.obs[0][0].astype(np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        self._env.set_actions(self.behavior_name, ActionTuple(continuous=action))
        self._env.step()
        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)

        while len(decision_steps) == 0 and len(terminal_steps) == 0:
            self._env.step()
            decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            obs = terminal_steps.obs[0][0].astype(np.float32)
            reward = float(terminal_steps.reward[0])
            interrupted = bool(terminal_steps.interrupted[0])
            return obs, reward, not interrupted, interrupted, {}
        else:
            obs = decision_steps.obs[0][0].astype(np.float32)
            return obs, float(decision_steps.reward[0]), False, False, {}

    def close(self):
        self._env.close()
