import torch
import itertools
from algorithms.r_mappo.algorithm.actor_critic import Actor, Critic
from utils.utils import update_linear_schedule
from torch.nn.parallel import DistributedDataParallel as DDP


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
                               action,
                               log_prob_reduce_fn,
                               action_preprocess_fn,
                               entropy_fn,
                               entropy_reduce_fn,
                               active_masks=None):
    action = action_preprocess_fn(action)
    action_log_probs = []
    dist_entropy = []
    for action_dist, act in zip(action_dists, action):
        action_log_probs.append(action_dist.log_probs(act))
        dist_entropy.append(entropy_fn(action_dist, active_masks))

    action_log_probs = log_prob_reduce_fn(action_log_probs)
    dist_entropy = entropy_reduce_fn(dist_entropy)
    return action_log_probs, dist_entropy


class R_MAPPOPolicy:
    def __init__(self, rank, args, obs_space, cent_obs_space, act_space, is_training=True):
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        '''
        We decompose the model into the actor and the critic instead of a single actor-critic
        because in specific scenarios only one of the two is used (for acting or value computation).
        The decomposition can reduce computation in these scenarios.

        An actor-critic model with a common network backbone is further decomposed into
        base, actor head and critic head.
        '''
        self.actor = Actor(args, obs_space, act_space).to(rank)
        self.critic = Critic(args, cent_obs_space).to(rank)
        if is_training:
            self.actor = DDP(self.actor, device_ids=[rank], output_device=rank)
            self.critic = DDP(self.critic, device_ids=[rank], output_device=rank)

            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.lr,
                                              eps=self.opti_eps,
                                              weight_decay=self.weight_decay)
        else:
            for p in self.parameters():
                p.requires_grad = False  # we don't train anything here

        self.actor_state_dict_keys = list(self.actor.state_dict().keys())
        self.critic_state_dict_keys = list(self.critic.state_dict().keys())
        assert all(k not in self.critic_state_dict_keys
                   for k in self.actor_state_dict_keys), ("actor and critic model can not have same parameter names")

    def parameters(self):
        return itertools.chain(self.actor.parameters(), self.critic.parameters())

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.policy_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.value_optimizer, episode, episodes, self.lr)

    def get_actions(self,
                    share_obs,
                    obs,
                    rnn_states=None,
                    rnn_states_critic=None,
                    masks=None,
                    available_actions=None,
                    deterministic=False):
        # used during batched rollout, utilizing both actor and critic
        (action_dists, action_reduce_fn, log_prob_reduce_fn, _, _, _,
         rnn_states) = self.actor(obs, rnn_states, masks, available_actions)
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks)
        actions, action_log_probs = get_actions_from_dist(action_dists, action_reduce_fn, log_prob_reduce_fn,
                                                          deterministic)

        return {
            'values': values,
            'actions': actions,
            'action_log_probs': action_log_probs,
            'rnn_states': rnn_states,
            'rnn_states_critic': rnn_states_critic
        }

    def get_values(self, share_obs, rnn_states_critic=None, masks=None):
        # used when reanalyzing values, only utilizing the critic
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self,
                         share_obs,
                         obs,
                         actions,
                         rnn_states=None,
                         rnn_states_critic=None,
                         masks=None,
                         available_actions=None,
                         active_masks=None,
                         **kwargs):
        (action_dists, _, log_prob_reduce_fn, action_preprocess_fn, entropy_fn, entropy_reduce_fn,
         _) = self.actor(obs, rnn_states, masks, available_actions)
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        action_log_probs, dist_entropy = evaluate_actions_from_dist(action_dists, actions, log_prob_reduce_fn,
                                                                    action_preprocess_fn, entropy_fn, entropy_reduce_fn,
                                                                    active_masks)

        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=True):
        # used during evaluation, only utilizing the actor
        (action_dists, action_reduce_fn, log_prob_reduce_fn, _, _, _,
         rnn_states) = self.actor(obs, rnn_states, masks, available_actions)

        actions, _ = get_actions_from_dist(action_dists, action_reduce_fn, log_prob_reduce_fn, deterministic)

        return actions, rnn_states

    def state_dict(self):
        return {**self.actor.state_dict(), **self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        actor_state_dict, critic_state_dict = {}, {}
        for k, v in state_dict.items():
            if k in self.actor_state_dict_keys:
                actor_state_dict[k] = v
            elif k in self.critic_state_dict_keys:
                critic_state_dict[k] = v
            else:
                raise NotImplementedError
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()
