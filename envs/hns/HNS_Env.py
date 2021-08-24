import numpy as np
from utils.multi_discrete import MultiDiscrete
from .envs.box_locking import BoxLockingEnv
from .envs.blueprint_construction import BlueprintConstructionEnv
from .envs.hide_and_seek import HideAndSeekEnv


class HNSEnv:
    def __init__(self, map_name, env_config):
        self.max_n_agents = env_config['max_n_agents'] + 1  # max others + self
        if map_name == "BoxLocking":
            self.num_agents = env_config['n_agents']
            self.env = BoxLockingEnv(**env_config)
            self.ordered_obs_keys = ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'observation_self']
            self.ordered_obs_mask_keys = ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', None]
        elif map_name == "BlueprintConstruction":
            self.num_agents = env_config['n_agents']
            self.env = BlueprintConstructionEnv(**env_config)
            self.ordered_obs_keys = [
                'agent_qpos_qvel', 'box_obs', 'ramp_obs', 'construction_site_obs', 'observation_self'
            ]
            self.ordered_obs_mask_keys = [None, None, None, None, None]
        elif map_name == "HideAndSeek":
            self.num_seekers = env_config['n_seekers']
            self.num_hiders = env_config['n_hiders']
            self.env = HideAndSeekEnv(**env_config)
            self.num_agents = env_config['n_seekers'] + env_config['n_hiders']
            self.ordered_obs_keys = [
                'agent_qpos_qvel', 'box_obs', 'ramp_obs', 'foodict_obsbs', 'observation_self', 'lidar'
            ]
            self.ordered_obs_mask_keys = ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', 'mask_af_obs', None, None]
        else:
            raise NotImplementedError

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.action_movement_dim = []

        for agent_id in range(self.num_agents):
            # deal with dict action space
            action_vec = []

            action_movement = self.env.action_space['action_movement'][agent_id].nvec
            action_vec += list(action_movement)
            self.action_movement_dim.append(len(action_movement))

            # aka lock
            glueall_dim = self.env.action_space['action_glueall'][agent_id].n
            action_vec.append(glueall_dim)

            # aka grab
            if 'action_pull' in self.env.action_space.spaces.keys():
                pull_dim = self.env.action_space['action_pull'][agent_id].n
                action_vec.append(pull_dim)
            action_space = MultiDiscrete([[0, vec - 1] for vec in action_vec])
            self.action_space.append(action_space)

            # deal with dict obs space
            obs_space = {}
            for key in self.ordered_obs_keys:
                if key in self.env.observation_space.spaces.keys():
                    space = tuple(self.env.observation_space[key].shape)
                    obs_space[key] = (1, *space) if len(space) < 2 else tuple(space)
            self.observation_space.append(obs_space)

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def _pad_dict_obs(self, dict_obs):
        if self.max_n_agents == self.num_agents:
            return dict_obs
        else:
            padded_obs = {}
            for k, v in dict_obs.items():
                pad = [(self.max_n_agents - v.shape[0], 0)] + [(0, 0) for _ in range(len(v.shape) - 1)]
                padded_obs[k] = np.pad(v, pad)
            return padded_obs

    def reset(self):
        dict_obs = self.env.reset()
        return self._pad_dict_obs(dict_obs)

    def step(self, actions):
        action_movement = []
        action_pull = []
        action_glueall = []
        for agent_id in range(self.num_agents):
            action_movement.append(actions[agent_id][:self.action_movement_dim[agent_id]])
            action_glueall.append(int(actions[agent_id][self.action_movement_dim[agent_id]]))
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull.append(int(actions[agent_id][-1]))
        action_movement = np.stack(action_movement, axis=0)
        action_glueall = np.stack(action_glueall, axis=0)
        if 'action_pull' in self.env.action_space.spaces.keys():
            action_pull = np.stack(action_pull, axis=0)
        actions_env = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}

        dict_obs, rewards, done, info = self.env.step(actions_env)

        info['force_termination'] = info.get('discard_episode', False)

        return self._pad_dict_obs(dict_obs), rewards, done, info

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
