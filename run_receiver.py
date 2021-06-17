#!/usr/bin/env python
import sys
# import setproctitle
import numpy as np
import multiprocessing as mp

import yaml
import torch
import time
import zmq
import blosc
import wandb
from config import get_config
from envs.starcraft2.StarCraft2_Env import StarCraft2Env
from envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from envs.starcraft2.smac_maps import get_map_params
from utils.buffer import LearnerBuffer

import torch.distributed as dist
from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
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

    example_env.close()
    del example_env

    all_args.num_agents = get_map_params(all_args.map_name)["n_agents"]

    # run = wandb.init(config=all_args,
    #                 project=all_args.env_name + '_distributed_nodes',
    #                 entity=all_args.user_name,
    #                 name='throughput_test',
    #                 group=all_args.map_name,
    #                 reinit=True)

    buffer = LearnerBuffer(all_args, all_args.observation_space, all_args.share_observation_space,
                           all_args.action_space)

    def receive_data(idx, buffer, all_args):
        socket = zmq.Context().socket(zmq.ROUTER)
        seg_port = all_args.seg_addrs[idx].split(':')[-1]
        socket.bind('tcp://*:' + seg_port)

        step = 0
        while True:
            tik = time.time()
            msg = socket.recv_multipart()
            print('receiver {} receive some message!'.format(idx), time.time() - tik)

            socket.send_multipart([msg[0], msg[1], b'ok'])

            if len(msg) > 3:
                msg = msg[2:]
                tik = time.time()
                assert len(msg) % 2 == 0
                seg_dict = {}
                decompression_time = 0
                for i in range(len(msg) // 2):
                    k, v = msg[2 * i].decode('ascii'), msg[2 * i + 1]
                    shape, dtype = buffer.shapes_and_dtypes[k]
                    tik = time.time()
                    decompressed = blosc.decompress(v)
                    decompression_time += time.time() - tik
                    array = np.frombuffer(decompressed, dtype=np.float32).reshape(*shape)
                    seg_dict[k] = array

                tik = time.time()
                buffer.put(seg_dict)
                buffer_put_time = time.time() - tik
                step += 1
                print('receiver {} decompression time: {:.2f}, buffer put time: {:.2f}'.format(idx, decompression_time, buffer_put_time))

    receivers = [mp.Process(target=receive_data, args=(idx, buffer, all_args)) for idx in range(len(all_args.seg_addrs))]
    for receiver in receivers:
        receiver.daemon = True
        receiver.start()

    dist.init_process_group('nccl', rank=0, world_size=1, init_method='file:///dev/shm/smac_ddp')
    policy = Policy(0, all_args, all_args.observation_space, all_args.share_observation_space, all_args.action_space, is_training=True)
    model_weights_socket = zmq.Context().socket(zmq.PUB)
    model_port = all_args.model_weights_addr.split(':')[-1]
    model_weights_socket.bind('tcp://*:' + model_port)

    step = 1
    while True:
        time.sleep(2)
        numpy_state_dict = {k.replace('module.', ''): v.cpu().numpy() for k, v in policy.state_dict().items()}
        msg = []
        for k, v in numpy_state_dict.items():
            msg.extend([k.encode('ascii'), v])
        msg.append(str(step).encode('ascii'))
        model_weights_socket.send_multipart(msg)

        # wandb.log({'total_timesteps': buffer.total_timesteps.item()}, step=step)
        step += 1



if __name__ == "__main__":
    main()
