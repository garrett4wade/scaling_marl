#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from torch.distributed import rpc
import torch.multiprocessing as mp
from system.hanabi_agent import HanabiAgent as Agent
from onpolicy.config import get_config
from onpolicy.envs.hanabi.Hanabi_Env import HanabiEnv
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ChooseDummyVecEnv, ChooseSubprocVecEnv
"""Train script for Hanabi."""


def parse_args(args, parser):
    parser.add_argument('--hanabi_name', type=str, default='Hanabi-Very-Small', help="Which env to run on")
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def build_actor_env(rank, all_args):
    if all_args.env_name == "Hanabi":
        assert all_args.num_agents > 1 and all_args.num_agents < 6, ("num_agents can be only between 2-5.")
        env = HanabiEnv(all_args, (all_args.seed + rank * 1000))
    else:
        print("Can not support the " + all_args.env_name + "environment.")
        raise NotImplementedError
    env.seed(all_args.seed + rank * 10000)
    return env


def make_example_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert all_args.num_agents > 1 and all_args.num_agents < 6, ("num_agents can be only between 2-5.")
                env = HanabiEnv(all_args, (all_args.seed * 50000 + rank * 10000))
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert all_args.num_agents > 1 and all_args.num_agents < 6, ("num_agents can be only between 2-5.")
                env = HanabiEnv(all_args, (all_args.seed * 50000 + rank * 10000))
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23457'
    if rank == 0:
        rpc_opt = rpc.TensorPipeRpcBackendOptions(num_worker_threads=max(16, all_args.num_actors), rpc_timeout=300)
        rpc.init_rpc('agent', rank=rank, world_size=world_size, rpc_backend_options=rpc_opt)

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

        # run dir
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results"
                       ) / all_args.env_name / all_args.hanabi_name / all_args.algorithm_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        # wandb
        if all_args.use_wandb:
            run = wandb.init(config=all_args,
                             project=all_args.env_name,
                             entity=all_args.user_name,
                             notes=socket.gethostname(),
                             name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" +
                             str(all_args.seed),
                             group=all_args.hanabi_name,
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

        # env init
        example_env = make_example_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        num_agents = all_args.num_agents

        config = {
            "all_args": all_args,
            "env_fn": build_actor_env,
            "example_env": example_env,
            "eval_envs": eval_envs,
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
            agent.writter.export_scalars_to_json(str(agent.log_dir + '/summary.json'))
            agent.writter.close()
    else:
        rpc_opt = rpc.TensorPipeRpcBackendOptions(rpc_timeout=300)
        rpc.init_rpc('actor_' + str(rank - 1), rank=rank, world_size=world_size, rpc_backend_options=rpc_opt)
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
