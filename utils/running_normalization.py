import numpy as np
import torch
import torch.distributed as dist


class RunningNormalization:
    def __init__(self, input_shape, beta=0.999, epsilon=1e-5, device=torch.device('cpu')):
        self.epsilon = epsilon
        self.beta = beta

        self.remaining_axes = len(input_shape)
        self.input_shape = input_shape

        # make PopArt accessible for every training process
        self.running_mean = torch.zeros(input_shape, dtype=torch.float32, device=device)
        self.running_mean_sq = torch.zeros(input_shape, dtype=torch.float32, device=device)
        self.debiasing_term = torch.zeros(1, dtype=torch.float32, device=device)

    def reset_parameters(self):
        self.running_mean[:] = 0
        self.running_mean_sq[:] = 0
        self.debiasing_term[:] = 0

    def debiased_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clip(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clip(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clip(min=1e-2)
        return debiased_mean, debiased_var

    def __call__(self, x, train=True):
        # called before computing loss, to normalize target values and to update running mean/std
        np_input = isinstance(x, np.ndarray)
        x = torch.from_numpy(x) if np_input else x
        x = x.to(self.running_mean)
        assert x.shape[-self.remaining_axes:] == self.input_shape, (
            "trailing dimensions of the input vector " + "are expected to be {} ".format(self.input_shape) +
            "while the input vector has shape {}".format(x.shape[-self.remaining_axes:]))
        norm_axes = len(x.shape) - self.remaining_axes

        if train:
            batch_mean = x.mean(axis=tuple(range(norm_axes))) / dist.get_world_size()
            batch_sq_mean = (x**2).mean(axis=tuple(range(norm_axes))) / dist.get_world_size()
            dist.all_reduce(batch_mean)
            dist.all_reduce(batch_sq_mean)

            self.running_mean = self.beta * self.running_mean + batch_mean * (1.0 - self.beta)
            self.running_mean_sq = self.beta * self.running_mean_sq + batch_sq_mean * (1.0 - self.beta)
            self.debiasing_term = self.beta * self.debiasing_term + 1.0 - self.beta

        mean, var = self.debiased_mean_var()
        out = (x - mean) / var.sqrt()

        return out.cpu().numpy() if np_input else out

    def denormalize(self, x):
        # called when closing an episode and computing returns,
        # to denormalize values during rollout inference
        np_input = isinstance(x, np.ndarray)
        x = torch.from_numpy(x) if np_input else x
        x = x.to(self.running_mean)
        assert x.shape[-self.remaining_axes:] == self.input_shape, (
            "trailing dimensions of the input vector " + "are expected to be {} ".format(self.input_shape) +
            "while the input vector has shape {}".format(x.shape[-self.remaining_axes:]))

        mean, var = self.debiased_mean_var()
        out = x * var.sqrt() + mean

        return out.cpu().numpy() if np_input else out
