#!/usr/bin/env python
import sys
# import setproctitle
import numpy as np
import yaml

import torch
from copy import deepcopy
from config import get_config
from envs.hns.HNS_Env import HNSEnv
from system.worker_node import WorkerNode


def build_actor_env(rank, seed_idx, cfg):
    assert cfg.env_name == "HideAndSeek"
    env_config = deepcopy(yaml.load(open('./envs/hns/configs/hide_and_seek_paper.yaml', 'r'), Loader=yaml.FullLoader))
    np.random.seed(seed_idx)
    env_config['n_hiders'] = np.random.randint(1, 4)
    env_config['n_seekers'] = np.random.randint(1, 4)
    # print(f'seed idx {seed_idx}, hiders {env_config["n_hiders"]}, seekers {env_config["n_seekers"]}')
    env = HNSEnv('HideAndSeek', env_config)
    env.seed(cfg.seed + rank * 10000)
    np.random.seed(cfg.seed)
    return env


def make_example_env(cfg):
    assert cfg.env_name == "HideAndSeek"
    env_config = deepcopy(yaml.load(open('./envs/hns/configs/hide_and_seek_paper.yaml', 'r'), Loader=yaml.FullLoader))
    env_config['n_hiders'] = 3
    env_config['n_seekers'] = 3
    return HNSEnv('HideAndSeek', env_config)


def main():
    parser = get_config()
    cfg = parser.parse_known_args(sys.argv[1:])[0]
    # overwrite default configuration using yaml file
    if cfg.config is not None:
        with open(cfg.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)

    assert len(cfg.ddp_init_methods) == cfg.num_policies
    num_worker_nodes = len(cfg.seg_addrs[0])
    if num_worker_nodes * cfg.num_tasks_per_node % cfg.num_policies != 0:
        from utils.utils import log
        log.warning(
            "All worker tasks can not be equally distributed for different policies! "
            "Try to revise the configuration to make (num_worker_nodes * num_tasks_per_node % num_policies == 0)")

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
    cfg.observation_space = example_env.observation_space
    cfg.action_space = example_env.action_space
    example_env.close()
    del example_env

    cfg.num_agents = 6
    assert len(cfg.policy2agents) == cfg.num_policies
    assert sum([len(v) for v in cfg.policy2agents.values()]) == cfg.num_agents

    node = WorkerNode(cfg, build_actor_env)
    node.run()


if __name__ == "__main__":
    main()
