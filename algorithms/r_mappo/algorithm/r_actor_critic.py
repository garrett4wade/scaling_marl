import torch.nn as nn
from algorithms.utils.util import init
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.utils import get_shape_from_obs_space


class R_Actor_Critic(nn.Module):
    def __init__(self, args, obs_space, cent_obs_space, action_space):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        obs_shape = get_shape_from_obs_space(obs_space)
        self.actor_base = MLPBase(args, obs_shape)
        cent_obs_space = get_shape_from_obs_space(cent_obs_space)
        self.critic_base = MLPBase(args, cent_obs_space)

        if self._use_recurrent_policy:
            self.actor_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.critic_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.v_out = init_(nn.Linear(self.hidden_size, 1))

    def forward(self, obs, actor_rnn_states, masks, available_actions=None, cent_obs=None, critic_rnn_states=None):
        compute_critic = cent_obs is not None
        values = None

        actor_features = self.actor_base(obs)
        if compute_critic:
            critic_features = self.critic_base(cent_obs)

        if self._use_recurrent_policy:
            actor_features, actor_rnn_states = self.actor_rnn(actor_features, actor_rnn_states, masks)
            if compute_critic:
                critic_features, critic_rnn_states = self.critic_rnn(critic_features, critic_rnn_states, masks)

        (action_dists, action_reduce_fn, log_prob_reduce_fn, action_preprocess_fn, entropy_fn,
         entropy_reduce_fn) = self.act(actor_features, available_actions)

        if compute_critic:
            values = self.v_out(critic_features)

        return (action_dists, action_reduce_fn, log_prob_reduce_fn, action_preprocess_fn, entropy_fn, entropy_reduce_fn,
                actor_rnn_states, values, critic_rnn_states)
