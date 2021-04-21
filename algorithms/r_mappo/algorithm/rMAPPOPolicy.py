import torch
from algorithms.utils.util import check
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor_Critic
from utils.util import update_linear_schedule
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
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, rank, args, obs_space, cent_obs_space, act_space, is_training=True):
        self.device = torch.device(rank)
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor_critic = R_Actor_Critic(args, self.obs_space, self.share_obs_space, self.act_space).to(rank)
        if is_training:
            self.actor_critic = DDP(self.actor_critic, device_ids=[rank], output_device=rank)

            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),
                                                lr=self.lr,
                                                eps=self.opti_eps,
                                                weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self,
                    cent_obs,
                    obs,
                    rnn_states_actor,
                    rnn_states_critic,
                    masks,
                    available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        obs = check(obs).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)

        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        (action_dists, action_reduce_fn, log_prob_reduce_fn, _, _, _,
         rnn_states_actor, values, rnn_states_critic) = self.actor_critic(obs, rnn_states_actor, masks, available_actions, cent_obs, rnn_states_critic)
        actions, action_log_probs = get_actions_from_dist(action_dists, action_reduce_fn, log_prob_reduce_fn,
                                                          deterministic)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def evaluate_actions(self,
                         cent_obs,
                         obs,
                         rnn_states_actor,
                         rnn_states_critic,
                         action,
                         masks,
                         available_actions=None,
                         active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        cent_obs = check(cent_obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        (action_dists, _, log_prob_reduce_fn, action_preprocess_fn, entropy_fn, entropy_reduce_fn,
         _, values, _ ) = self.actor_critic(obs, rnn_states_actor, masks, available_actions, cent_obs, rnn_states_critic)
        action_log_probs, dist_entropy = evaluate_actions_from_dist(action_dists, action, log_prob_reduce_fn,
                                                                    action_preprocess_fn, entropy_fn, entropy_reduce_fn,
                                                                    active_masks)

        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        (action_dists, action_reduce_fn, log_prob_reduce_fn, _, _, _,
         rnn_states_actor, _, _) = self.actor_critic(obs, rnn_states_actor, masks, available_actions)
        actions, _ = get_actions_from_dist(action_dists, action_reduce_fn, log_prob_reduce_fn, deterministic)

        return actions, rnn_states_actor

    def state_dict(self):
        return self.actor_critic.state_dict()

    def load_state_dict(self, state_dict):
        self.actor_critic.load_state_dict(state_dict)

    def train_mode(self):
        self.actor_critic.train()

    def eval_mode(self):
        self.actor_critic.eval()
