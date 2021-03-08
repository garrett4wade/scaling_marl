#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import torch.multiprocessing as mp
from torch.distributed import rpc
from config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.starcraft2.smac_maps import get_map_params
from system.smac_agent import SMACAgent as Agent
from envs.env_wrappers import ShareDummyVecEnv
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


def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    if rank == 0:

        rpc.init_rpc('agent',
                     rank=rank,
                     world_size=world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=300))

        if all_args.algorithm_name == "rmappo":
            assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
        else:
            raise NotImplementedError

        # cuda
        if all_args.cuda and torch.cuda.is_available():
            print("choose to use gpu...")
            device = torch.device("cuda:0")
            if torch.cuda.device_count() > 1:
                rollout_device = torch.device("cuda:1")
            else:
                rollout_device = device
            torch.set_num_threads(all_args.n_training_threads)
            if all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        else:
            print("choose to use cpu...")
            device = torch.device("cpu")
            torch.set_num_threads(all_args.n_training_threads)

        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results"
                       ) / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        if all_args.use_wandb:
            run = wandb.init(config=all_args,
                             project=all_args.env_name,
                             entity=all_args.user_name,
                             notes=socket.gethostname(),
                             name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" +
                             str(all_args.seed),
                             group=all_args.map_name,
                             dir=str(run_dir),
                             job_type="training",
                             reinit=True)
        else:
            if not run_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [
                    int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()
                    if str(folder.name).startswith('run')
                ]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))

        setproctitle.setproctitle(
            str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" +
            str(all_args.user_name))

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
            # "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "rollout_device": rollout_device,
            "run_dir": run_dir
        }
        agent = Agent(config)
        agent.run()
        if all_args.use_wandb:
            run.finish()
    else:
        rpc.init_rpc('actor_' + str(rank - 1),
                     rank=rank,
                     world_size=world_size,
                     rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=300))
    rpc.shutdown()


if __name__ == "__main__":
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)
    procs = []
    for i in range(all_args.num_actors + 1):
        p = mp.Process(target=main, args=(i, all_args.num_actors + 1))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
