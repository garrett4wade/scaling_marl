import torch.nn as nn
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.value_head import ValueHead
from .hns_encoder import HNSEncoder


class R_Actor_Critic(nn.Module):
    def __init__(self, args, obs_space, act_space):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._rec_n = args.rec_n

        self.actor_base = HNSEncoder(obs_space, omniscient=False, use_orthogonal=self._use_orthogonal)
        self.critic_base = HNSEncoder(obs_space, omniscient=True, use_orthogonal=self._use_orthogonal)

        if self._use_recurrent_policy:
            self.actor_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._rec_n, self._use_orthogonal)
            self.critic_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._rec_n, self._use_orthogonal)

        self.act = ACTLayer(act_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.v_out = ValueHead(self.hidden_size, 1, self._use_orthogonal, args.use_popart)

    def forward(self, obs, actor_rnn_states, masks, critic_rnn_states=None, unnormalized_v_target=None, train_normalization=False, use_ckpt=False):
        compute_critic = critic_rnn_states is not None
        values = None

        actor_features = self.actor_base(obs, train_normalization, use_ckpt)
        if compute_critic:
            critic_features = self.critic_base(obs, train_normalization, use_ckpt)

        if self._use_recurrent_policy:
            actor_features, actor_rnn_states = self.actor_rnn(actor_features, actor_rnn_states, masks)
            if compute_critic:
                critic_features, critic_rnn_states = self.critic_rnn(critic_features, critic_rnn_states, masks)

        action_dists = self.act(actor_features, None)

        if compute_critic:
            values, v_target = self.v_out(critic_features, unnormalized_v_target)

        return (action_dists, actor_rnn_states, values, critic_rnn_states, v_target)
