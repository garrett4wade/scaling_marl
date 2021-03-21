import wandb
import os
import torch
from tensorboardX import SummaryWriter


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Trainer:
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, rpc_rank, weights_queue, buffer, config):
        self.rpc_rank = rpc_rank
        self.all_args = config['all_args']
        # NOTE: trainers occupy last #num_trainers GPUs
        self.trainer_id = self.dpp_rank = rpc_rank + self.all_args.num_servers
        self.num_trainers = self.all_args.num_trainers
        torch.cuda.set_device(self.rpc_rank)

        self.eval_envs = config['eval_envs']
        self.num_agents = config['num_agents']

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
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
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

        example_env = config['example_env']
        share_observation_space = example_env.share_observation_space[
            0] if self.use_centralized_V else example_env.observation_space[0]
        observation_space = example_env.observation_space[0]
        action_space = example_env.action_space[0]

        # policy network
        self.policy = Policy(rpc_rank, self.all_args, observation_space, share_observation_space, action_space)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.algorithm = TrainAlgo(self.all_args, self.policy)

        self.weights_queue = weights_queue
        self.buffer = buffer

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def pack_off_weights(self):
        if self.ddp_rank == 0:
            # send weights to rollout policy
            actor_state_dict, critic_state_dict = self.policy.state_dict()
            actor_state_dict = {k.replace('module.', ''): v for k, v in actor_state_dict.items()}
            critic_state_dict = {k.replace('module.', ''): v for k, v in critic_state_dict.items()}
            self.weights_queue.put((actor_state_dict, critic_state_dict))

    def training_step(self):
        """Train policies with data in buffer. """
        self.policy.train_mode()
        train_infos = self.algorithm.step(self.buffer)
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_info(self, infos, total_num_steps):
        assert isinstance(infos)
        for k, v in infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
