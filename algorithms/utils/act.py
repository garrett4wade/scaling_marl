from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList([
                DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain),
                Categorical(inputs_dim, discrete_dim, use_orthogonal, gain)
            ])

    def forward(self, x, available_actions=None):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if self.mixed_action:
            action_dists = [action_out(x) for action_out in self.action_outs]

            def action_reduce_fn(x):
                return torch.cat(x, -1)

            def log_prob_reduce_fn(x):
                return torch.sum(torch.cat(x, -1), -1, keepdim=True)

            def action_preprocess_fn(x):
                a, b = x.split((2, 1), -1)
                b = b.long()
                return [a, b]

            def entropy_fn(action_dist, active_masks=None):
                if active_masks is not None:
                    if len(action_dist.entropy().shape) == len(active_masks.shape):
                        x = (action_dist.entropy() * active_masks).sum() / active_masks.sum()
                    else:
                        x = (action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    x = action_dist.entropy().mean()
                return x

            def entropy_reduce_fn(dist_entropy):
                return dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98  # ! dosen't make sense

        elif self.multi_discrete:
            action_dists = [action_out(x) for action_out in self.action_outs]

            def action_reduce_fn(x):
                return torch.cat(x, -1)

            log_prob_reduce_fn = action_reduce_fn

            def action_preprocess_fn(x):
                return torch.transpose(x, 0, 1)

            def entropy_fn(action_dist, active_masks=None):
                if active_masks is not None:
                    x = (action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    x = action_dist.entropy().mean()
                return x

            def entropy_reduce_fn(dist_entropy):
                return torch.tensor(dist_entropy).mean()

        else:
            action_dists = [self.action_out(x, available_actions)]

            def action_reduce_fn(x):
                return x[0]

            def action_preprocess_fn(x):
                return [x]

            def entropy_fn(action_dist, active_masks=None):
                if active_masks is not None:
                    x = (action_dist.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
                else:
                    x = action_dist.entropy().mean()
                return x

            entropy_reduce_fn = log_prob_reduce_fn = action_reduce_fn

        return action_dists, action_reduce_fn, log_prob_reduce_fn, action_preprocess_fn, entropy_fn, entropy_reduce_fn
