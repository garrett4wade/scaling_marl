import numpy as np
import torch
import threading
from system.actor import Actor
from torch.distributed import rpc
from queue import Empty


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class InferenceServer:
    def __init__(self, rpc_rank, gpu_rank, weights_queue, buffer, config):
        # NOTE: inference servers rpc_ranks come after trainers rpc_ranks
        self.rpc_rank = rpc_rank
        self.gpu_rank = gpu_rank
        self.all_args = config['all_args']
        self.server_id = self.rpc_rank - self.all_args.num_trainers
        torch.cuda.set_device(gpu_rank)

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
        self.rollout_batch_size = self.num_actors * self.env_per_split
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

        # actors
        self.rref = rpc.RRef(self)
        self.actor_rrefs = None
        self.actor_job_rrefs = None

        self.locks = [threading.Lock() for _ in range(self.num_split)]
        self.future_outputs = [torch.futures.Future() for _ in range(self.num_split)]
        self.queued_cnt = np.zeros((self.num_split, ), dtype=np.float32)

        # synchronization utilities
        self.buffer = buffer
        self.weights_queue = weights_queue

    def load_weights(self, block=False):
        try:
            state_dict = self.weights_queue.get(block)
        except Empty:
            # for debug
            print("queue empty, load weights failed")
            return
        for lock in self.locks:
            lock.acquire()

        # for debug
        for k, v in state_dict[0].items():
            assert not torch.any(v == self.rollout_policy.actor.state_dict()[k])
        for k, v in state_dict[1].items():
            assert not torch.any(v == self.rollout_policy.critic.state_dict()[k])

        self.rollout_policy.load_state_dict(state_dict)

        for lock in self.locks:
            lock.release()

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
        raise NotImplementedError

    def setup_actors(self):
        self.load_weights(block=True)
        self.actor_rrefs = []
        self.actor_job_rrefs = []
        for i in range(self.num_actors):
            # NOTE: actor ids of any server is [0, 1, ..., #actors-1]
            name = 'actor_' + str(i)
            actor_rref = rpc.remote(name, Actor, args=(i, self.env_fn, self.rref, self.all_args))
            self.actor_job_rrefs.append(actor_rref.remote().run())
            self.actor_rrefs.append(actor_rref)
