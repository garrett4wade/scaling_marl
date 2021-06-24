import wandb
import os
import torch
from tensorboardX import SummaryWriter


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Trainer:
    """ Base class for training. """
    def __init__(self, rank, task_queue, buffer, config):
        self.rank = rank
        self.all_args = config['all_args']
        self.num_trainers = self.all_args.num_trainers
        assert self.all_args.cuda and torch.cuda.is_available(), 'cpu training currently not supported'
        torch.cuda.set_device(self.rank)

        self.eval_envs = config['eval_envs_fn'](self.rank, self.all_args)
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
        self.envs_per_actor = self.all_args.envs_per_actor
        self.num_splits = self.all_args.num_splits
        self.envs_per_split = self.envs_per_actor // self.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.episode_length = self.all_args.episode_length
        self.train_for_env_steps = self.all_args.train_for_env_steps
        self.train_for_seconds = self.all_args.train_for_seconds
        self.slots_per_update = self.all_args.slots_per_update
        # interval
        self.use_eval = self.all_args.use_eval
        self.save_interval = self.all_args.save_interval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        # dir
        self.model_dir = self.all_args.model_dir
        # summay & render
        self.no_summary = self.all_args.no_summary
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render

        if self.rank == 0:
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
        self.policy = Policy(self.rank,
                             self.all_args,
                             observation_space,
                             share_observation_space,
                             action_space,
                             is_training=True)
        self.policy_version = 0

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.algorithm = TrainAlgo(self.all_args, self.policy)
        self.buffer = buffer

        self.task_queue = task_queue
        
        self.initialized = False
        self.terminate = False
        
        self.model_weights_socket = None

    def _init(self):
        self.model_weights_socket = zmq.Context().socket(zmq.PUB)
        model_port = self.all_args.model_weights_addr.split(':')[-1]
        self.model_weights_socket.bind('tcp://*:' + model_port)

        self.initialized = True

    def run(self):
        raise NotImplementedError

    def eval(self):
        # TODO: conduct evaluation using inference server rather than trainer
        raise NotImplementedError

    def pack_off_weights(self):
        if self.rank == 0:
            # remove prefix 'module.' of DDP models
            numpy_state_dict = {k.replace('module.', ''): v.cpu().numpy() for k, v in self.policy.state_dict().items()}
            msg = []
            for k, v in numpy_state_dict.items():
                msg.extend([k.encode('ascii'), v])
            msg.append(str(step).encode('ascii'))
            self.model_weights_socket.send_multipart(msg)

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
        if not self.no_summary:
            for k, v in infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    def init(self):
        self.task_queue.put(TaskType.INIT)

    def terminate(self):
        self.task_queue.put(TaskType.TERMINATE)
