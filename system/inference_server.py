import numpy as np
import torch
import threading
from system.actor import Actor
from torch.distributed import rpc
from queue import Empty
import zmq


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class InferenceServer:
    def __init__(self, rpc_rank, gpu_rank, weights_queue, buffer, config):
        self.rpc_rank = rpc_rank
        self.gpu_rank = gpu_rank
        self.all_args = config['all_args']
        self.server_id = self.rpc_rank - self.all_args.num_trainers
        assert self.all_args.cuda and torch.cuda.is_available(), 'cpu training currently not supported'
        # torch.cuda.set_device(gpu_rank)

        self.env_fn = config['env_fn']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # -------- parameters --------
        # tricks
        self.use_centralized_V = self.all_args.use_centralized_V
        # system dataflow
        # NOTE: #actors = #clients per inference server
        self.num_actors = self.all_args.num_actors
        self.env_per_actor = self.all_args.env_per_actor
        self.num_split = self.all_args.num_split
        self.env_per_split = self.env_per_actor // self.num_split
        assert self.env_per_actor % self.num_split == 0

        self.episode_length = self.all_args.episode_length
        self.num_env_steps = self.all_args.num_env_steps
        self.rollout_bs = self.all_args.rollout_bs
        self.use_render = self.all_args.use_render

        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        example_env = config['example_env']
        share_observation_space = example_env.share_observation_space[
            0] if self.use_centralized_V else example_env.observation_space[0]
        observation_space = example_env.observation_space[0]
        action_space = example_env.action_space[0]

        # policy network
        self.rollout_policy = Policy(gpu_rank,
                                     self.all_args,
                                     observation_space,
                                     share_observation_space,
                                     action_space,
                                     is_training=False)
        self.rollout_policy.eval_mode()

        # synchronization utilities
        self.buffer = buffer
        self.weights_queue = weights_queue

        # actors
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.identity = "Server-{}".format(self.server_id).encode("ascii")
        self.socket.connect(self.all_args.backend_addr)

        # tell broker (rounter) that server is prepared to conduct inference
        self.socket.send(b'READY')

    def load_weights(self, block=False):
        # TODO: use pub/sub instead of queue
        try:
            state_dict = self.weights_queue.get(block)
        except Empty:
            # for debug
            # print("queue empty, load weights failed")
            return

        self.rollout_policy.load_state_dict(state_dict)

    def select_action(self, *args, **kwargs):
        raise NotImplementedError

    def serve(self):
        while True:
            self.load_weights(block=False)

            msg = self.socket.recv_multipart()
            clients = list(filter(lambda x: x != b'', msg))[:-1]

            policy_inputs = self.buffer.get_policy_inputs(clients)

            policy_outputs = self.select_action(policy_inputs)

            self.buffer.insert_after_inference(clients, *policy_outputs)

            # directly send received message back such that broker can fetch actions from buffer
            self.socket.send_multipart(msg)


class HanabiServer(InferenceServer):
    @torch.no_grad()
    def select_action(self, policy_inputs):
        policy_outputs = self.rollout_policy.get_actions(*map(
            lambda x: x.reshape(self.rollout_bs * self.env_per_split, *x.shape[2:]), policy_inputs))
        policy_outputs = map(lambda x: _t2n(x).reshape(self.rollout_bs, self.env_per_split, *x.shape[1:]), policy_outputs)

        return policy_outputs


class SMACServer(InferenceServer):
    @torch.no_grad()
    def select_action(self, policy_inputs):
        policy_outputs = self.rollout_policy.get_actions(*map(
            lambda x: x.reshape(self.rollout_bs * self.num_agents * self.env_per_split, *x.shape[3:]), policy_inputs))

        policy_outputs = map(lambda x: _t2n(x).reshape(self.rollout_bs, self.env_per_split, self.num_agents, *x.shape[1:]), policy_outputs)

        return policy_outputs