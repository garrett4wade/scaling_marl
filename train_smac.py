#!/usr/bin/env python
import sys
import os
import wandb
import socket
# import setproctitle
import numpy as np
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import rpc

from system.trainer.smac_trainer import SMACTrainer
from system.server.smac_server import SMACServer
from utils.buffer import SharedReplayBuffer
from config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.starcraft2.smac_maps import get_map_params
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
"""Train script for SMAC."""


class SMACBuffer(SharedReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        summary_shp = (self.num_servers, self.num_actors, self.num_split)
        self._btwon_shm, self.battles_won = self.shm_array(summary_shp, np.float32)
        self._btgm_shm, self.battles_game = self.shm_array(summary_shp, np.float32)
        self.summary_lock = mp.RLock()


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


def init_summary(config, all_args):
    run_dir = config['run_dir']
    import datetime
    algo = all_args.algorithm_name
    network_cls = 'rnn' if algo == 'rmappo' else 'mlp' if algo == 'mappo' else None
    batch_size_factor = (all_args.episode_length * all_args.num_actors * all_args.env_per_actor *
                         all_args.num_trainers / all_args.num_split / 8 / 400)
    replay = str(all_args.ppo_epoch)
    postfix = '_{:.2f}x_r{}_'.format(batch_size_factor, replay) + network_cls
    exp_name = str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed)
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=exp_name,
                         group=all_args.map_name + postfix,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
        return run
    else:
        curr_run = exp_name + postfix + '_' + str(datetime.datetime.now()).replace(' ', '_')
        config['run_dir'] = run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        return None


def run(rank, world_size, weights_queue, buffer, config):
    all_args = config['all_args']
    rpc_init_method = 'file:///dev/shm/smac_rpc'
    ddp_init_method = 'file:///dev/shm/smac_ddp'
    if rank < all_args.num_trainers:
        dist.init_process_group('nccl', init_method=ddp_init_method, rank=rank, world_size=all_args.num_trainers)

        rpc_opt = rpc.TensorPipeRpcBackendOptions(init_method=rpc_init_method, rpc_timeout=300)

        rpc.init_rpc('trainer_' + str(rank), rank=rank, world_size=world_size, rpc_backend_options=rpc_opt)

        if rank == 0:
            run = init_summary(config, all_args)

        trainer = SMACTrainer(rank, weights_queue, buffer, config)
        trainer.run()

        if rank == 0:
            if run is not None:
                run.finish()
            else:
                trainer.writter.export_scalars_to_json(str(trainer.log_dir + '/summary.json'))
                trainer.writter.close()

        dist.destroy_process_group()

    elif rank >= all_args.num_trainers and rank < all_args.num_trainers + all_args.num_servers:
        offset = all_args.num_trainers

        rpc_opt = rpc.TensorPipeRpcBackendOptions(init_method=rpc_init_method,
                                                  num_worker_threads=max(16, all_args.num_actors),
                                                  rpc_timeout=300)

        rpc.init_rpc('agent_' + str(rank - offset), rank=rank, world_size=world_size, rpc_backend_options=rpc_opt)

        server_gpu_ranks = all_args.server_gpu_ranks
        if len(server_gpu_ranks) == 1:
            gpu_rank = server_gpu_ranks[0]
        elif len(server_gpu_ranks) == all_args.num_servers:
            gpu_rank = server_gpu_ranks[rank - offset]
        else:
            raise RuntimeError('server_gpu_ranks needs to either have length 1 or length #num_servers.')

        server = SMACServer(rank, gpu_rank, weights_queue, buffer, config)
        server.setup_actors()

    else:
        offset = all_args.num_servers + all_args.num_trainers

        rpc_opt = rpc.TensorPipeRpcBackendOptions(init_method=rpc_init_method, rpc_timeout=300)

        rpc.init_rpc('actor_' + str(rank - offset), rank=rank, world_size=world_size, rpc_backend_options=rpc_opt)

    rpc.shutdown()


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

    run_dir = Path('/summary') / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    # env
    example_env = make_example_env(all_args)
    num_agents = get_map_params(all_args.map_name)["n_agents"]

    config = {
        "all_args": all_args,
        "env_fn": build_actor_env,
        "example_env": example_env,
        "eval_envs_fn": make_eval_env,
        "num_agents": num_agents,
        "run_dir": run_dir
    }

    share_obs_space = example_env.share_observation_space[
        0] if all_args.use_centralized_V else example_env.observation_space[0]
    obs_space = example_env.observation_space[0]
    act_space = example_env.action_space[0]

    buffer = SMACBuffer(all_args, num_agents, obs_space, share_obs_space, act_space)
    weights_queue = mp.Queue(maxsize=8)

    world_size = all_args.num_servers * (all_args.num_actors + 1) + all_args.num_trainers
    procs = []
    for i in range(world_size):
        p = mp.Process(target=run, args=(i, world_size, weights_queue, buffer, config))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
