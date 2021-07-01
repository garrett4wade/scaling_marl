#!/usr/bin/env python
import os
import sys
import numpy as np
import pathlib
import yaml
import torch
import multiprocessing as mp
from config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from envs.starcraft2.smac_maps import get_map_params
from utils.buffer import LearnerBuffer
from system.receiver import Receiver
from torch.multiprocessing import JoinableQueue as TorchJoinableQueue
from system.trainer import Trainer
from utils.utils import log
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
    # overwrite default configuration using yaml file
    if all_args.config is not None:
        with open(all_args.config) as f:
            all_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in all_args_dict.items():
            setattr(all_args, k, v)
    if all_args.trainer_indices is None:
        all_args.trainer_indices = list(range(all_args.num_trainers))
    else:
        all_args.trainer_indices = [int(ind) for ind in all_args.trainer_indices.split(',')]
    # TODO: we need to have some assertion about trainer indices and num_trainer_nodes

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

    run_dir = pathlib.Path('./log') / all_args.env_name
    if all_args.env_name == 'StarCraft2':
        run_dir /= all_args.map_name
    run_dir = run_dir / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    example_env = make_example_env(all_args)
    all_args.share_observation_space = example_env.share_observation_space[
        0] if all_args.use_centralized_V else example_env.observation_space[0]
    all_args.observation_space = example_env.observation_space[0]
    all_args.action_space = example_env.action_space[0]

    example_env.close()
    del example_env

    all_args.num_agents = get_map_params(all_args.map_name)["n_agents"]

    buffer = LearnerBuffer(all_args, all_args.observation_space, all_args.share_observation_space,
                           all_args.action_space)

    num_worker_nodes = len(all_args.seg_addrs)
    num_learner_nodes = len(all_args.model_weights_addrs)
    assert num_worker_nodes % num_learner_nodes == 0, ('currently worker nodes must be '
                                                                'statically distributed among learner nodes')
    worker_nodes_per_learner = num_worker_nodes // num_learner_nodes

    nodes_ready_events = [mp.Event() for _ in range(worker_nodes_per_learner)]

    recievers = [
        Receiver(all_args, receiver_idx, TorchJoinableQueue(), buffer, nodes_ready_events[i])
        for i, receiver_idx in enumerate(range(all_args.learner_node_idx * worker_nodes_per_learner, (all_args.learner_node_idx + 1) *
                       worker_nodes_per_learner))
    ]
    for r in recievers:
        r.init()

    trainers = [
        Trainer(rank, gpu_rank, buffer, all_args, nodes_ready_events, run_dir=run_dir) for gpu_rank, rank in enumerate(all_args.trainer_indices)
    ]
    for trainer in trainers:
        trainer.process.start()

    for trainer in trainers:
        trainer.process.join()
    log.info('Trainers joined!')

    for r in recievers:
        r.close()
    log.info('Receivers joined!')

    log.info('Done!')


if __name__ == "__main__":
    main()
