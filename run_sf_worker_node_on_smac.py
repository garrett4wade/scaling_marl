#!/usr/bin/env python
import sys
# import setproctitle
import numpy as np

import torch
from config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from sf_woker_node import SFWorkerNode
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

    all_args = parser.parse_known_args(args)[0]

    return all_args


def build_actor_env(rank, all_args):
    if all_args.env_name == "StarCraft2":
        env = StarCraft2Env(all_args)
    else:
        print("Can not support the " + all_args.env_name + "environment.")
        raise NotImplementedError
    env.seed(all_args.seed + rank * 10000)
    return env


def make_example_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 10000)
            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(trainer_id, all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000 + 12345 * trainer_id)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def main():
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
    elif all_args.algorithm_name == 'mappo':
        all_args.use_recurrent_policy = False
    else:
        raise NotImplementedError

    # NOTE: this line may incur a bug
    # torch.set_num_threads(all_args.n_training_threads)
    if all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    example_env = make_example_env(all_args)
    all_args.share_observation_space = example_env.share_observation_space[
        0] if all_args.use_centralized_V else example_env.observation_space[0]
    all_args.observation_space = example_env.observation_space[0]
    all_args.action_space = example_env.action_space[0]

    all_args.num_agents = get_map_params(all_args.map_name)["n_agents"]

    node = SFWorkerNode(all_args, build_actor_env)
    node.run()


if __name__ == "__main__":
    main()
