import numpy as np
import torch
import threading
from system.actor import Actor
from torch.distributed import rpc


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class InferenceServer:
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, rank, config):
        # inference servers occupy first #num_servers GPUs
        self.rank = self.server_id = rank
        self.all_args = config['all_args']

        self.env_fn = config['env_fn']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # -------- parameters --------
        # tricks
        self.use_centralized_V = self.all_args.use_centralized_V
        # system dataflow
        assert self.all_args.num_actors % self.all_args.num_servers == 0
        self.num_actors = self.all_args.num_actors // self.all_args.num_servers
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
        self.rollout_policy = Policy(rank, self.all_args, observation_space, share_observation_space, action_space)
        self.rollout_policy.actor.eval()
        self.rollout_policy.critic.eval()

        # actors
        self.rref = rpc.RRef(self)
        self.actor_rrefs = None
        self.actor_job_rrefs = None

        self.locks = [threading.Lock() for _ in range(self.num_split)]
        self.future_outputs = [torch.futures.Future() for _ in range(self.num_split)]
        self.queued_cnt = np.zeros((self.num_split, ), dtype=np.float32)

    def load_weights(self, state_dict):
        for lock in self.locks:
            lock.acquire()
        self.rollout_policy.load_state_dict(state_dict)
        for lock in self.locks:
            lock.release()

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
        raise NotImplementedError

    def setup_actors(self):
        self.actor_rrefs = []
        self.actor_job_rrefs = []
        for i in range(self.num_actors):
            name = 'actor_' + str(i + self.num_actors * self.server_id)
            actor_rref = rpc.remote(name, Actor, args=(i, self.env_fn, self.rref, self.all_args))
            self.actor_job_rrefs.append(actor_rref.remote().run())
            self.actor_rrefs.append(actor_rref)
