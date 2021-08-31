import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from algorithms.utils.util import init


class ValueHead(nn.Module):
    def __init__(self, input_dim, output_dim, use_orthogonal, use_popart, beta=1-1e-5, epsilon=1e-5):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

        self.input_dim = input_dim
        self.output_dim = output_dim

        self._use_popart = use_popart
        self._use_orthogonal = use_orthogonal

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        self.stddev = nn.Parameter(torch.ones(output_dim), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.mean_sq = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        init_(self)
        self.stddev.zero_().add_(1)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, feature, unnormalized_v_target=None):
        value = F.linear(feature, self.weight, self.bias)
        if unnormalized_v_target is not None:
            # during training
            if self._use_popart:
                self.update(unnormalized_v_target)
                v_target = self.normalize(unnormalized_v_target)
            else:
                v_target = unnormalized_v_target
        else:
            # during rollout
            v_target = None
        return value, v_target

    @torch.no_grad()
    def update(self, x):
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        reduce_axes = len(x.shape) - 1

        old_mean, old_stddev = self.mean.data.clone(), self.stddev.data.clone()

        batch_mean = x.mean(dim=tuple(range(reduce_axes))) / dist.get_world_size()
        batch_sq_mean = x.square().mean(dim=tuple(range(reduce_axes))) / dist.get_world_size()
        dist.all_reduce(batch_mean)
        dist.all_reduce(batch_sq_mean)

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev.data[:] = (self.mean_sq - self.mean**2).sqrt().clamp(min=1e-4)

        # self.weight.data[:] = self.weight * old_stddev / self.stddev
        # self.bias.data[:] = (old_stddev * self.bias + old_mean - self.mean) / self.stddev

    @torch.no_grad()
    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def normalize(self, x):
        np_input = isinstance(x, np.ndarray)
        x = torch.from_numpy(x) if np_input else x
        assert x.shape[-1:] == self.mean.shape, (
            "trailing dimensions of the input vector " + "are expected to be {} ".format(self.mean.shape) +
            "while the input vector has shape {}".format(x.shape[-1:]))

        mean, var = self.debiased_mean_var()
        out = ((x - mean) / torch.sqrt(var)).clip(-5.0, 5.0)

        return out.numpy() if np_input else out

    @torch.no_grad()
    def denormalize(self, x):
        # there exists some problem when computation is conducted on torch.Tensor
        # since denormalization is only invoked on CPU, we use NumPy array here
        np_input = isinstance(x, np.ndarray)
        x = x if np_input else x.numpy()
        assert x.shape[-1:] == self.mean.shape, (
            "trailing dimensions of the input vector " + "are expected to be {} ".format(self.mean.shape) +
            "while the input vector has shape {}".format(x.shape[-1:]))

        mean, var = self.debiased_mean_var()
        out = x * np.sqrt(var.numpy()) + mean.numpy()

        return out if np_input else torch.from_numpy(out)
