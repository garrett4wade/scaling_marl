#!/usr/bin/env python
import sys
# import setproctitle
import numpy as np
import yaml

import torch
from config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from worker_node import WorkerNode
from envs.starcraft2.smac_maps import get_map_params
"""Train script for SMAC."""


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m', help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)

    cfg = parser.parse_known_args(args)[0]

    return cfg


def build_actor_env(rank, cfg):
    if cfg.env_name == "StarCraft2":
        env = StarCraft2Env(cfg)
    else:
        print("Can not support the " + cfg.env_name + "environment.")
        raise NotImplementedError
    env.seed(cfg.seed + rank * 10000)
    return env


def make_example_env(cfg):
    def get_env_fn(rank):
        def init_env():
            if cfg.env_name == "StarCraft2":
                env = StarCraft2Env(cfg)
            else:
                print("Can not support the " + cfg.env_name + "environment.")
                raise NotImplementedError
            env.seed(cfg.seed + rank * 10000)
            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(trainer_id, cfg):
    def get_env_fn(rank):
        def init_env():
            if cfg.env_name == "StarCraft2":
                env = StarCraft2Env(cfg)
            else:
                print("Can not support the " + cfg.env_name + "environment.")
                raise NotImplementedError
            env.seed(cfg.seed * 50000 + rank * 10000 + 12345 * trainer_id)
            return env

        return init_env

    if cfg.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(cfg.n_eval_rollout_threads)])


def main():
    parser = get_config()
    cfg = parse_args(sys.argv[1:], parser)
    # overwrite default configuration using yaml file
    if cfg.config is not None:
        with open(cfg.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)

    if cfg.algorithm_name == "rmappo":
        cfg.use_recurrent_policy = True
    elif cfg.algorithm_name == 'mappo':
        cfg.use_recurrent_policy = False
    else:
        raise NotImplementedError

    # NOTE: this line may incur a bug
    # torch.set_num_threads(cfg.n_training_threads)
    if cfg.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    example_env = make_example_env(cfg)
    cfg.share_observation_space = example_env.share_observation_space[
        0] if cfg.use_centralized_V else example_env.observation_space[0]
    cfg.observation_space = example_env.observation_space[0]
    cfg.action_space = example_env.action_space[0]
    example_env.close()
    del example_env

    cfg.num_agents = get_map_params(cfg.map_name)["n_agents"]

    node = WorkerNode(cfg, build_actor_env)
    node.run()


if __name__ == "__main__":
    main()
