#!/usr/bin/env python
import sys
import numpy as np
import yaml
import torch
import itertools
import multiprocessing as mp
from config import get_config
from envs.hns.HNS_Env import HNSEnv
from copy import deepcopy
from system.receiver import Receiver
from system.trainer import Trainer
from system.reanalyzer import Reanalyzer
from system.value_tracer import ValueTracer
from utils.utils import log


def make_example_env(cfg):
    assert cfg.env_name == "HideAndSeek"
    env_config = deepcopy(yaml.load(open('./envs/hns/configs/quadrant.yaml', 'r'), Loader=yaml.FullLoader))
    env_config['n_hiders'] = 2
    env_config['n_seekers'] = 2
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

    all_policy_learner_idxes = []
    for node_idx, local_config in cfg.learner_config.items():
        for _, v in local_config.items():
            all_policy_learner_idxes.append(v)
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

    # seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    example_env = make_example_env(cfg)
    cfg.share_observation_space = (example_env.share_observation_space
                                   if cfg.use_centralized_V else example_env.observation_space)
    cfg.observation_space = example_env.observation_space
    cfg.action_space = example_env.action_space

    example_env.close()
    del example_env

    cfg.num_agents = 4

    if cfg.learner_node_idx == 0:
        from system.task_dispatcher import TaskDispatcher
        from meta_controllers.naive import NaiveMetaController
        task_dispatcher = TaskDispatcher(cfg, NaiveMetaController(cfg))
        task_dispatcher.start_process()

    local_learner_config = cfg.learner_config[str(cfg.learner_node_idx)]
    num_worker_nodes = len(cfg.seg_addrs[0])
    nodes_ready_events = [mp.Event() for _ in range(num_worker_nodes)]
    trainer_ready_events = [mp.Event() for _ in range(len(local_learner_config))]

    trainers = []
    shm_state_dicts = []
    for i, gpu_idx in enumerate(local_learner_config.keys()):
        # initialize a example policy to get a example state dict
        policy_id = cfg.learner_config[str(cfg.learner_node_idx)][str(gpu_idx)]
        example_agent = cfg.policy2agents[str(policy_id)][0]

        obs_space = cfg.observation_space[example_agent]
        act_space = cfg.action_space[example_agent]
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        example_policy = Policy('cpu', cfg, obs_space, act_space, is_training=False)
        shm_state_dict = {k: v.detach().cpu().share_memory_() for k, v in example_policy.state_dict().items()}
        del example_policy

        tn = Trainer(cfg, int(gpu_idx), nodes_ready_events, trainer_ready_events[i], shm_state_dict)
        tn.start_process()
        trainers.append(tn)
        shm_state_dicts.append(shm_state_dict)

    for e in trainer_ready_events:
        e.wait()

    recievers = [Receiver(cfg, i, trainers, nodes_ready_events[i]) for i in range(num_worker_nodes)]
    for r in recievers:
        r.start_proess()

    all_reanalyzers = []
    all_value_tracers = []
    for t_i, trainer in enumerate(trainers):
        reanalyzers = []
        value_tracers = []
        for i in range(cfg.num_reanalyzers_per_trainer):
            # TODO: use different GPU rank
            r_a = Reanalyzer(
                cfg,
                trainer.trainer_idx,
                i,
                trainer.gpu_rank,
                trainer.buffer,
                trainer.value_tracer_queue,
                trainer.batch_queue,
                trainer.reanalyzer_task_queues[i],
                shm_state_dicts[t_i],
                trainer.policy_version,
                trainer.param_lock,
            )
            r_a.start_process()
            reanalyzers.append(r_a)

        for i in range(cfg.num_value_tracers_per_trainer):
            v_t = ValueTracer(
                cfg,
                trainer.trainer_idx,
                i,
                trainer.buffer,
                trainer.value_tracer_queue,
                trainer.batch_queue,
                trainer.value_tracer_task_queues[i],
                shm_state_dicts[t_i],
                trainer.policy_version,
                trainer.param_lock,
            )
            v_t.start_process()
            value_tracers.append(v_t)
        all_reanalyzers.append(reanalyzers)
        all_value_tracers.append(value_tracers)

    for trainer in trainers:
        trainer.process.join()
    log.info('Trainers joined!')

    for r_a in itertools.chain(*all_reanalyzers):
        r_a.close()

    for v_t in itertools.chain(*all_value_tracers):
        v_t.close()

    for r in recievers:
        r.close()
    log.info('Receivers joined!')

    if cfg.learner_node_idx == 0:
        task_dispatcher.process.join()
        log.info('Task Dispatcher joined!')

    log.info('Done!')


if __name__ == "__main__":
    main()
