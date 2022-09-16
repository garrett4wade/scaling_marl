import torch.nn as nn
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.value_head import ValueHead
from .hns_encoder import HNSEncoder
from algorithms.utils.running_normalization import RunningNormalization
import torch


class R_Actor_Critic(nn.Module):
    def __init__(self, args, obs_space, act_space):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_recurrent_policy = args.use_recurrent_policy
        self._rec_n = args.rec_n
        self.num_critic = args.num_critic
        
        self.normalize_obs_keys = ['observation_self', 'lidar', 'agent_qpos_qvel', 'box_obs', 'ramp_obs']
        for key in self.normalize_obs_keys:
            setattr(self, key + '_normalization', RunningNormalization((obs_space[key][-1], ), beta=1 - 1e-5))

        # fully observable
        self.actor_base = HNSEncoder(obs_space, omniscient=True, use_orthogonal=self._use_orthogonal)
        self.critic_base = nn.ModuleList()
        for _ in range(self.num_critic):
            self.critic_base.append(HNSEncoder(obs_space, omniscient=True, use_orthogonal=self._use_orthogonal))

        if self._use_recurrent_policy:
            self.actor_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._rec_n, self._use_orthogonal)
            self.critic_rnn = RNNLayer(self.hidden_size, self.hidden_size, self._rec_n, self._use_orthogonal)

        self.act = ACTLayer(act_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.v_out = ValueHead(self.hidden_size, 1, self._use_orthogonal, args.use_popart)

    def forward(self,
                obs,
                actor_rnn_states,
                masks,
                critic_rnn_states=None,
                unnormalized_v_target=None,
                train_normalization=False,
                use_ckpt=False):
        for k in self.normalize_obs_keys:
            # print(k,obs[k].shape)
            obs[k] = getattr(self, k + '_normalization')(obs[k], train_normalization)

        compute_critic = critic_rnn_states is not None
        values = None

        # critic_rnn_states : torch.Size([:, 1, num_critic, 512])
        actor_features = self.actor_base(obs, use_ckpt)
        if compute_critic:
            critic_features = []
            for critic_id in range(self.num_critic):
                critic_features.append(self.critic_base[critic_id](obs, use_ckpt))

        if self._use_recurrent_policy:
            actor_features, actor_rnn_states = self.actor_rnn(actor_features, actor_rnn_states, masks)
            if compute_critic:
                for critic_id in range(self.num_critic):
                    # critic_rnn_states_one = critic_rnn_states[]
                    critic_features_one, critic_rnn_states_one = self.critic_rnn(critic_features[critic_id], critic_rnn_states[:,:,critic_id], masks)
                    critic_features[critic_id] = critic_features_one
                    critic_rnn_states[:,:,critic_id] = critic_rnn_states_one

        action_dists = self.act(actor_features, None)

        if compute_critic:
            values = []
            v_target = []
            for critic_id in range(self.num_critic):
                if unnormalized_v_target is not None:
                    if self.num_critic == 1:
                        real_unnormalized_v_target = unnormalized_v_target
                    else:
                        real_unnormalized_v_target = unnormalized_v_target[:,:,critic_id].unsqueeze(-1)
                    values_one, v_target_one = self.v_out(critic_features[critic_id], real_unnormalized_v_target)
                else:
                    values_one, v_target_one = self.v_out(critic_features[critic_id], unnormalized_v_target)
                values.append(values_one)
                v_target.append(v_target_one)
            values = torch.stack(values,axis=-1).squeeze(1)
            if None in v_target:
                v_target = None
            else:
                v_target = torch.stack(v_target,axis=-1).squeeze(1)

        else:
            v_target = None

        return (action_dists, actor_rnn_states, values, critic_rnn_states, v_target)

    # def forward(self,
    #             obs,
    #             actor_rnn_states,
    #             masks,
    #             critic_rnn_states=None,
    #             unnormalized_v_target=None,
    #             train_normalization=False,
    #             use_ckpt=False):
    #     for k in self.normalize_obs_keys:
    #         # print(k,obs[k].shape)
    #         obs[k] = getattr(self, k + '_normalization')(obs[k], train_normalization)
    #     # print('actor_rnn_states', actor_rnn_states.shape)

    #     compute_critic = critic_rnn_states is not None
    #     values = None

    #     actor_features = self.actor_base(obs, use_ckpt)
    #     if compute_critic:
    #         critic_features = self.critic_base(obs, use_ckpt)

    #     if self._use_recurrent_policy:
    #         actor_features, actor_rnn_states = self.actor_rnn(actor_features, actor_rnn_states, masks)
    #         if compute_critic:
    #             critic_features, critic_rnn_states = self.critic_rnn(critic_features, critic_rnn_states, masks)

    #     action_dists = self.act(actor_features, None)

    #     if compute_critic:
    #         values, v_target = self.v_out(critic_features, unnormalized_v_target)
    #     else:
    #         v_target = None

    #     return (action_dists, actor_rnn_states, values, critic_rnn_states, v_target)
