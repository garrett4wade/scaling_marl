import torch
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor_Critic
from utils.utils import update_linear_schedule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler


def get_actions_from_dist(action_dists, action_reduce_fn, log_prob_reduce_fn, deterministic=False):
    actions = []
    action_log_probs = []
    for action_dist in action_dists:
        action = action_dist.mode() if deterministic else action_dist.sample()
        action_log_prob = action_dist.log_probs(action)
        actions.append(action.float())
        action_log_probs.append(action_log_prob)
    actions = action_reduce_fn(actions)
    action_log_probs = log_prob_reduce_fn(action_log_probs)
    return actions, action_log_probs


def evaluate_actions_from_dist(action_dists,
                               actions,
                               log_prob_reduce_fn,
                               action_preprocess_fn,
                               entropy_fn,
                               entropy_reduce_fn,
                               active_masks=None):
    actions = action_preprocess_fn(actions)
    action_log_probs = []
    dist_entropy = []
    for action_dist, act in zip(action_dists, actions):
        action_log_probs.append(action_dist.log_probs(act))
        dist_entropy.append(entropy_fn(action_dist, active_masks))

    action_log_probs = log_prob_reduce_fn(action_log_probs)
    dist_entropy = entropy_reduce_fn(dist_entropy)
    return action_log_probs, dist_entropy


class R_MAPPOPolicy:
    def __init__(self, rank, args, obs_space, act_space, is_training=True):
        self.device = torch.device(rank)
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor_critic = R_Actor_Critic(args, self.obs_space, self.act_space).to(rank)
        if is_training:
            self.actor_critic = DDP(self.actor_critic, device_ids=[rank], output_device=rank)

            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),
                                              lr=self.lr,
                                              eps=self.opti_eps,
                                              weight_decay=self.weight_decay)
            self.scaler = GradScaler(init_scale=100.0)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, rnn_states, rnn_states_critic, masks, deterministic=False, **obs):
        (action_dists, rnn_states, values, rnn_states_critic, _) = self.actor_critic(obs, rnn_states, masks, rnn_states_critic, train_normalization=False, use_ckpt=False)
        actions = [action_dist.sample() if deterministic else action_dist.mode() for action_dist in action_dists]
        action_log_probs = [action_dist.log_probs(action) for action_dist, action in zip(action_dists, actions)]

        return {
            'values': values,
            'actions': torch.cat(actions, dim=-1),
            'action_log_probs': torch.cat(action_log_probs, dim=-1),
            'rnn_states': rnn_states,
            'rnn_states_critic': rnn_states_critic
        }

    def evaluate_actions(self, rnn_states, rnn_states_critic, actions, masks, v_target, **obs):
        (action_dists, _, values, _, v_target) = self.actor_critic(obs, rnn_states, masks, rnn_states_critic, v_target, train_normalization=True, use_ckpt=True)
        actions = torch.split(actions, 1, dim=-1)
        action_log_probs = [action_dist.log_probs(action) for action_dist, action in zip(action_dists, actions)]
        dist_entropy = [action_dist.entropy().mean() for action_dist in action_dists]

        return values, torch.cat(action_log_probs, -1), sum(dist_entropy) / len(dist_entropy), v_target

    def state_dict(self):
        return self.actor_critic.state_dict()

    def load_state_dict(self, state_dict):
        self.actor_critic.load_state_dict(state_dict)

    def train_mode(self):
        self.actor_critic.train()

    def eval_mode(self):
        self.actor_critic.eval()
