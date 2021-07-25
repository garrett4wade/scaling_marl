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

    cfg = parser.parse_known_args(args)[0]

    return cfg


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
    # TODO: we need to have some assertion about trainer indices and num_trainer_nodes

    # learner config has the following strucuture:
    # {learner_node_idx: {gpu_idx: policy_learner_idx, ...}, ...}
    task_rank_offsets, task_rank_cnts, task_gpu_ranks = {}, {}, {}
    local_learner_config = cfg.learner_config[str(cfg.learner_node_idx)]
    for gpu_rank, v in local_learner_config.items():
        task_rank_offsets[v] = 0
        if v not in task_rank_cnts:
            task_rank_cnts[v] = 1
        else:
            task_rank_cnts[v] += 1
        if v not in task_gpu_ranks:
            task_gpu_ranks[v] = [int(gpu_rank)]
        else:
            task_gpu_ranks[v].append(int(gpu_rank))

    all_policy_learner_idxes = []
    for node_idx, local_config in cfg.learner_config.items():
        for _, v in local_config.items():
            all_policy_learner_idxes.append(v)
            for task_v in task_rank_offsets.keys():
                if int(node_idx) < cfg.learner_node_idx and v == task_v:
                    task_rank_offsets[task_v] += 1
    # sanity checks
    assert cfg.learner_node_idx < len(cfg.learner_config)
    assert list(range(cfg.num_policies)) == list(np.unique(sorted(all_policy_learner_idxes)))

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

    run_dir = pathlib.Path('./log') / cfg.env_name
    if cfg.env_name == 'StarCraft2':
        run_dir /= cfg.map_name
    run_dir = run_dir / cfg.algorithm_name / cfg.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

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
    # TODO: num_trainers is different for different buffers
    cfg.num_trainers = len(local_learner_config)

    if cfg.learner_node_idx == 0:
        from system.task_dispatcher import TaskDispatcher
        from meta_controllers.naive import NaiveMetaController
        task_dispatcher = TaskDispatcher(cfg, NaiveMetaController(cfg))
        task_dispatcher.start_process()

    # TODO: currently we only support training a single policy (policy sharing)
    assert cfg.num_policies == 1
    buffer = LearnerBuffer(cfg, cfg.observation_space, cfg.share_observation_space,
                           cfg.action_space)

    num_worker_nodes = len(cfg.seg_addrs[0])
    nodes_ready_events = [mp.Event() for _ in range(num_worker_nodes)]

    # (num_worker_nodes * num_learner_nodes) receivers in total, (num_worker_nodes) receivers for each learner node
    recievers = [
        Receiver(cfg, num_worker_nodes * cfg.learner_node_idx + i, TorchJoinableQueue(), buffer,
                 nodes_ready_events[i]) for i in range(num_worker_nodes)
    ]
    for r in recievers:
        r.init()

    all_trainers = []
    # NOTE: gpu_rank may not necessarily be the same as local_rank
    for policy_id, offset in task_rank_offsets.items():
        cur_num_trainers = task_rank_cnts[policy_id]
        for local_rank, gpu_rank in zip(range(cur_num_trainers), task_gpu_ranks[policy_id]):
            global_rank = local_rank + offset

            # TODO: some logic about ranks in trainer may be not correct, check them
            trainer = Trainer(global_rank, local_rank, gpu_rank, buffer, cfg, nodes_ready_events, run_dir=run_dir)
            trainer.process.start()

            all_trainers.append(trainer)

    for trainer in all_trainers:
        trainer.process.join()
    log.info('Trainers joined!')

    for r in recievers:
        r.close()
    log.info('Receivers joined!')

    if cfg.learner_node_idx == 0:
        task_dispatcher.close()
        log.info('Task Dispatcher joined!')

    log.info('Done!')


if __name__ == "__main__":
    main()
