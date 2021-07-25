import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.

    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies.
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases.
            or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time.
            If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)

    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate

    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something,
            the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--worker_node_idx', type=int, help='index of worker nodes')
    parser.add_argument('--learner_node_idx', type=int, help='index of learner nodes')
    parser.add_argument('--num_tasks_per_node', type=int, default=1, help='how many tasks a worker node will manage')
    parser.add_argument('--task_dispatcher_addr', type=str, help='address of meta controller to distributed tasks')
    parser.add_argument('--learner_config', help='learner gpu configuration')
    parser.add_argument('--num_policies', type=int, default=1, help='number of policies to be trained')

    parser.add_argument('--config', type=str)
    parser.add_argument('--project_name', type=str, default='test')
    parser.add_argument('--group_name', type=str, default='test')
    parser.add_argument('--no_summary', action='store_true')

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default='mappo', choices=["rmappo", "mappo"])

    parser.add_argument("--experiment_name",
                        type=str,
                        default="check",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda",
                        action='store_false',
                        default=True,
                        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false',
                        default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=16, help="Number of torch threads for training")
    parser.add_argument("--n_eval_rollout_threads",
                        type=int,
                        default=4,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--num_env_steps",
                        type=int,
                        default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name",
                        type=str,
                        default='garrett4wade',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_true', help="whether to use [Weights & Biases] for logging")

    # system parameters
    parser.add_argument("--num_actors", type=int, default=4)
    parser.add_argument("--actor_group_size", type=int, default=1, help='trade off between static and dynamic rollout')
    parser.add_argument("--envs_per_actor", type=int, default=2)
    parser.add_argument("--num_splits", type=int, default=2)
    parser.add_argument("--qsize", type=int, default=8)
    parser.add_argument("--stats_avg", type=int, default=100)
    parser.add_argument("--num_transmitters", type=int, default=1)

    parser.add_argument("--force_envs_single_thread", default=True, action='store_false')
    parser.add_argument("--set_workers_cpu_affinity", default=True, action='store_false')

    parser.add_argument("--model_weights_addrs", type=str, nargs='+')
    parser.add_argument("--seg_addrs", type=str, nargs='+')

    parser.add_argument('--benchmark', default=False, action='store_true', help='Benchmark mode')

    parser.add_argument("--default_niceness", default=0, type=int)
    parser.add_argument("--actor_worker_gpus", default=0, type=int)

    parser.add_argument("--num_policy_workers", type=int, default=1)

    parser.add_argument("--slots_per_update", type=int, default=1)
    parser.add_argument("--min_num_requests", type=int, default=-1)

    parser.add_argument("--reset_timeout_seconds", type=int, default=60)
    parser.add_argument('--train_for_env_steps',
                        default=int(1e10),
                        type=int,
                        help='Stop after all policies are trained for this many env steps')
    parser.add_argument('--train_for_seconds',
                        default=int(1e10),
                        type=int,
                        help='Stop training after this many seconds')
    parser.add_argument('--train_in_background', action='store_true')
    parser.add_argument('--ddp_init_method', type=str, default='file:///dev/shm/ddp_marl')
    parser.add_argument('--learner_num_threads', type=int, default=8)
    parser.add_argument('--broadcast_interval', default=1, type=int)

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state",
                        action='store_true',
                        default=False,
                        help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=400, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy",
                        action='store_false',
                        default=True,
                        help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V",
                        action='store_false',
                        default=True,
                        help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames",
                        type=int,
                        default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames",
                        action='store_true',
                        default=False,
                        help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size",
                        type=int,
                        default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false', default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart",
                        action='store_false',
                        default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization",
                        action='store_false',
                        default=True,
                        help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal",
                        action='store_false',
                        default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--rec_n", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length",
                        type=int,
                        default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--sample_reuse", type=int, default=5, help='number of ppo epochs (default: 5)')
    parser.add_argument("--use_reanalyze",
                        action='store_true',
                        help="whether to re-compute values after each udpate iteration")
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false',
                        default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--use_advantage_normalization",
                        action='store_false',
                        default=True,
                        help="by default, use advantage normalization.")
    parser.add_argument("--clip_param", type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_coef", type=float, default=1.0, help='value term coefficient (default: 1.0)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false',
                        default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits",
                        action='store_true',
                        default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss",
                        action='store_false',
                        default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_active_masks",
                        action='store_false',
                        default=True,
                        help="by default True, whether to mask useless data in losses.")
    parser.add_argument("--use_fct_masks",
                        action='store_false',
                        default=True,
                        help="by default True, whether to mask the last frame of force terminated episodes")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay",
                        action='store_true',
                        default=False,
                        help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval",
                        type=int,
                        default=1,
                        help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval",
                        type=int,
                        default=20,
                        help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval",
                        action='store_true',
                        default=False,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval",
                        type=int,
                        default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs",
                        action='store_true',
                        default=False,
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument(
        "--use_render",
        action='store_true',
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, "
        "the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi",
                        type=float,
                        default=0.1,
                        help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help="by default None. set the path to pretrained model.")

    return parser
