import torch.nn as nn
from algorithms.utils.util import init
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

    def forward(self, obs, rnn_states, masks, available_actions=None):
        actor_features = self.base(obs)

        if self._use_recurrent_policy:
            rnn_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        (action_dists, action_reduce_fn, log_prob_reduce_fn, action_preprocess_fn, entropy_fn,
         entropy_reduce_fn) = self.act(actor_features, available_actions)

        return (action_dists, action_reduce_fn, log_prob_reduce_fn, action_preprocess_fn, entropy_fn, entropy_reduce_fn,
                rnn_states)


class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_space = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_space) == 3 else MLPBase
        self.base = base(args, cent_obs_space)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

    def forward(self, cent_obs, rnn_states, masks):
        critic_features = self.base(cent_obs)

        if self._use_recurrent_policy:
            rnn_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)
        return values, rnn_states
