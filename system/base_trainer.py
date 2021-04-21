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
        # NOTE: trainers occupy first #num_trainers GPUs, rpc_ranks come ahead of inference servers'
        self.rpc_rank = self.trainer_id = self.ddp_rank = rpc_rank
        self.all_args = config['all_args']
        self.num_trainers = self.all_args.num_trainers
        assert self.all_args.cuda and torch.cuda.is_available(), 'cpu training currently not supported'
        torch.cuda.set_device(self.ddp_rank)

        self.eval_envs = config['eval_envs_fn'](self.trainer_id, self.all_args)
        self.num_agents = config['num_agents']

        # -------- parameters --------
        # names
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        # tricks
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        # system dataflow
        self.num_actors = self.all_args.num_actors
        self.env_per_actor = self.all_args.env_per_actor
        self.num_split = self.all_args.num_split
        self.env_per_split = self.env_per_actor // self.num_split
        assert self.env_per_actor % self.num_split == 0
        self.episode_length = self.all_args.episode_length
        self.num_env_steps = self.all_args.num_env_steps
        self.slots_per_update = self.all_args.slots_per_update
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
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

        if self.ddp_rank == 0:
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
            0] if self.all_args.use_centralized_V else example_env.observation_space[0]
        observation_space = example_env.observation_space[0]
        action_space = example_env.action_space[0]

        # policy network
        self.policy = Policy(self.ddp_rank,
                             self.all_args,
                             observation_space,
                             share_observation_space,
                             action_space,
                             is_training=True)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.algorithm = TrainAlgo(self.all_args, self.policy)

        # synchronization utilities
        self.weights_queue = weights_queue
        self.buffer = buffer

    def run(self):
        raise NotImplementedError

    def eval(self):
        # TODO: conduct evaluation using inference server rather than trainer
        raise NotImplementedError

    def pack_off_weights(self):
        if self.ddp_rank == 0:
            # remove prefix 'module.' of DDP models
            self.weights_queue.put({k.replace('module.', ''): v.cpu() for k, v in self.policy.state_dict().items()})

    def training_step(self):
        """Train policies with data in buffer. """
        self.policy.train_mode()
        train_infos = self.algorithm.step(self.buffer)
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        torch.save(self.policy.state_dict(), str(self.save_dir) + "/mdoel.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        self.policy.actor_critic.load_state_dict(torch.load(str(self.model_dir) + '/model.pt'))

    def log_info(self, infos, total_num_steps):
        for k, v in infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
