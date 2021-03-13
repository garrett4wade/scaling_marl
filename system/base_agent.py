import wandb
import os
import numpy as np
import torch
import threading
# from queue import Queue
from system.actor import Actor
from torch.distributed import rpc
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Agent:
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):
        self.all_args = config['all_args']

        self.env_fn = config['env_fn']
        self.example_env = config['example_env']
        # self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.rollout_device = config['rollout_device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # -------- parameters --------
        # names
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        # tricks
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        # model
        self.recurrent_N = self.all_args.recurrent_N
        self.hidden_size = self.all_args.hidden_size
        # system dataflow
        self.num_actors = self.all_args.num_actors
        self.env_per_actor = self.all_args.env_per_actor
        self.num_split = self.all_args.num_split
        self.env_per_split = self.env_per_actor // self.num_split
        assert self.env_per_actor % self.num_split == 0
        self.episode_length = self.all_args.episode_length
        self.num_env_steps = self.all_args.num_env_steps
        # TODO: support arbitrary rollout batch size
        self.rollout_batch_size = self.num_actors * self.env_per_split
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        # dir
        self.model_dir = self.all_args.model_dir
        # summay & render
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.example_env.share_observation_space[
            0] if self.use_centralized_V else self.example_env.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                             self.example_env.observation_space[0],
                             share_observation_space,
                             self.example_env.action_space[0],
                             device=self.device)
        if self.rollout_device != self.device:
            self.rollout_policy = Policy(self.all_args,
                                         self.example_env.observation_space[0],
                                         share_observation_space,
                                         self.example_env.action_space[0],
                                         device=self.rollout_device)
            self.rollout_policy.load_weights(self.policy)
        else:
            self.rollout_policy = self.policy

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.rollout_policy, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.example_env.observation_space[0],
                                         share_observation_space, self.example_env.action_space[0],
                                         self.trainer.value_normalizer)

        # actors
        self.rref = rpc.RRef(self)
        self.actor_rrefs = None
        self.actor_job_rrefs = None

        self.locks = [threading.Lock() for _ in range(self.num_split)]
        self.future_outputs = [torch.futures.Future() for _ in range(self.num_split)]
        self.queued_cnt = np.zeros((self.num_split, ), dtype=np.float32)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
        raise NotImplementedError

    def setup_actors(self):
        self.actor_rrefs = []
        self.actor_job_rrefs = []
        for i in range(self.num_actors):
            # id+1 because rpc_id=0 is the agent process
            name = 'actor_' + str(i)
            actor_rref = rpc.remote(name, Actor, args=(i, self.env_fn, self.rref, self.all_args))
            self.actor_job_rrefs.append(actor_rref.remote().run())
            self.actor_rrefs.append(actor_rref)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        if self.policy is not self.rollout_policy:
            self.rollout_policy.load_weights(self.policy)
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
