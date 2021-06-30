import torch.nn as nn
from algorithms.utils.util import init
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.utils import get_shape_from_obs_space
'''
We decompose the model into the actor and the critic instead of a single actor-critic
because in specific scenarios only one of the two is used (for acting or value computation).
The decomposition can reduce computation in these scenarios.

An actor-critic model with a common network backbone is further decomposed into
base, actor head and critic head.
'''


class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._rec_n = args.rec_n

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        obs_shape = get_shape_from_obs_space(obs_space)
        self.actor_base = MLPBase(args, obs_shape)

        if self._use_recurrent_policy:
            self.actor_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._rec_n, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, args.gain)

    def forward(self, obs, actor_rnn_states=None, masks=None, available_actions=None):
        actor_features = self.actor_base(obs)

        if self._use_recurrent_policy:
            actor_features, actor_rnn_states = self.actor_rnn(actor_features, actor_rnn_states, masks)

        (action_dists, action_reduce_fn, log_prob_reduce_fn, action_preprocess_fn, entropy_fn,
         entropy_reduce_fn) = self.act(actor_features, available_actions)

        return (action_dists, action_reduce_fn, log_prob_reduce_fn, action_preprocess_fn, entropy_fn, entropy_reduce_fn,
                actor_rnn_states)


class Critic(nn.Module):
    def __init__(self, args, cent_obs_space):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._rec_n = args.rec_n

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        cent_obs_space = get_shape_from_obs_space(cent_obs_space)
        self.critic_base = MLPBase(args, cent_obs_space)

        if self._use_recurrent_policy:
            self.critic_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._rec_n, self._use_orthogonal)

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

    def forward(self, cent_obs, critic_rnn_states=None, masks=None):
        critic_features = self.critic_base(cent_obs)

        if self._use_recurrent_policy:
            critic_features, critic_rnn_states = self.critic_rnn(critic_features, critic_rnn_states, masks)

        values = self.v_out(critic_features)

        return values, critic_rnn_states
