import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp


class PopArt(nn.Module):
    def __init__(self, input_shape, num_trainers, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
        super(PopArt, self).__init__()
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.barrier = mp.Barrier(num_trainers)

        # make PopArt accessible for every training process
        self.share_memory()
        for p in self.parameters():
            assert p.is_shared()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector, train=True):
        # TODO: when computing loss, the same batch is used for #ppo_epoch times!
        # called before computing loss, to normalize target values and to update running mean/std
        np_input = isinstance(input_vector, np.ndarray)
        input_vector = torch.from_numpy(input_vector) if np_input else input_vector

        if train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            detached_input = input_vector.detach()
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (detached_input**2).mean(dim=tuple(range(self.norm_axes)))

            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta**batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        self.barrier.wait()
        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None, ) * self.norm_axes]) / torch.sqrt(var)[(None, ) * self.norm_axes]

        return out.numpy() if np_input else out

    def denormalize(self, input_vector):
        # called when closing an episode and computing returns,
        # to denormalize values during rollout inference
        np_input = isinstance(input_vector, np.ndarray)
        input_vector = torch.from_numpy(input_vector) if np_input else input_vector

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None, ) * self.norm_axes] + mean[(None, ) * self.norm_axes]

        return out.numpy() if np_input else out
