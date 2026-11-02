

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from gymnasium import spaces

class UnityMultiAgentVecEnv(VecEnv):
    def __init__(self, file_name=None, no_graphics=False, time_scale=20.0, worker_id=0,
                 log_folder=None):
        self.channel = EngineConfigurationChannel()

        self._env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics,
                                     worker_id=worker_id, side_channels=[self.channel],
                                     log_folder=log_folder)
        self.channel.set_configuration_parameters(time_scale=time_scale)
        self._env.reset()

        self.behavior_name = list(self._env.behavior_specs)[0]
        spec = self._env.behavior_specs[self.behavior_name]
        obs_shape = spec.observation_specs[0].shape
        n_cont = spec.action_spec.continuous_size
        self._obs_dim = int(obs_shape[0])
        self._act_dim = int(n_cont)

        obs_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(n_cont,), dtype=np.float32)

        decision_steps, _ = self._env.get_steps(self.behavior_name)
        self._agent_ids = sorted(int(a) for a in decision_steps.agent_id)
        num_envs = len(self._agent_ids)
        if num_envs == 0:
            raise RuntimeError(
                "No agents found in the scene. Did you press Play, and do all agents share "
                "one Behavior Name? (This is the N-agents-in-one-scene trainer.)")
        self._id_to_idx = {aid: i for i, aid in enumerate(self._agent_ids)}

        super().__init__(num_envs, obs_space, act_space)

        self._obs = np.zeros((num_envs, self._obs_dim), dtype=np.float32)
        self._fill_from_decision(decision_steps)
        self._actions = np.zeros((num_envs, self._act_dim), dtype=np.float32)

    def _fill_from_decision(self, decision_steps):
        obs0 = decision_steps.obs[0]
        for row, aid in enumerate(decision_steps.agent_id):
            i = self._id_to_idx.get(int(aid))
            if i is not None:
                self._obs[i] = obs0[row].astype(np.float32)

    def reset(self):
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.behavior_name)

        while len(decision_steps) < self.num_envs:
            self._env.step()
            decision_steps, _ = self._env.get_steps(self.behavior_name)
        self._fill_from_decision(decision_steps)
        return self._obs.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self._act_dim)

    def step_wait(self):
        decision_steps, _ = self._env.get_steps(self.behavior_name)
        if len(decision_steps) > 0:
            act = np.zeros((len(decision_steps), self._act_dim), dtype=np.float32)
            for row, aid in enumerate(decision_steps.agent_id):
                act[row] = self._actions[self._id_to_idx[int(aid)]]
            self._env.set_actions(self.behavior_name, ActionTuple(continuous=act))

        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        tobs0 = terminal_steps.obs[0] if len(terminal_steps) > 0 else None
        for row, aid in enumerate(terminal_steps.agent_id):
            i = self._id_to_idx[int(aid)]
            term = tobs0[row].astype(np.float32)
            rewards[i] = float(terminal_steps.reward[row])
            dones[i] = True
            infos[i]["terminal_observation"] = term

            infos[i]["TimeLimit.truncated"] = bool(terminal_steps.interrupted[row])
            self._obs[i] = term

        dobs0 = decision_steps.obs[0] if len(decision_steps) > 0 else None
        for row, aid in enumerate(decision_steps.agent_id):
            i = self._id_to_idx[int(aid)]
            rewards[i] = float(decision_steps.reward[row])
            self._obs[i] = dobs0[row].astype(np.float32)

        return self._obs.copy(), rewards, dones, infos

    def close(self):
        self._env.close()

    def _targets(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def get_attr(self, attr_name, indices=None):
        val = getattr(self, attr_name, None)
        return [val for _ in self._targets(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("env_method is not supported on the Unity multi-agent VecEnv.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._targets(indices)]
