import numpy as np
import torch
import torch.multiprocessing as mp


class PopArt:
    def __init__(self, input_shape, num_trainers, beta=0.999, per_element_update=False, epsilon=1e-5):
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.remaining_axes = len(input_shape)
        self.input_shape = input_shape

        # make PopArt accessible for every training process
        self._running_mean = torch.zeros(input_shape, dtype=torch.float32).share_memory_()
        self._running_mean_sq = torch.zeros(input_shape, dtype=torch.float32).share_memory_()
        self._debiasing_term = torch.tensor(0.0, dtype=torch.float32).share_memory_()

        self.running_mean = self._running_mean.numpy()
        self.running_mean_sq = self._running_mean_sq.numpy()
        self.debiasing_term = self._debiasing_term.numpy()

        # TODO: use pytorch ddp to synchronize multiple trainers
        self.barrier = mp.Barrier(num_trainers)

    def reset_parameters(self):
        self.running_mean[:] = 0
        self.running_mean_sq[:] = 0
        self.debiasing_term[:] = 0

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clip(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clip(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clip(min=1e-2)
        return debiased_mean, debiased_var

    def __call__(self, input_vector, train=True):
        # called before computing loss, to normalize target values and to update running mean/std
        np_input = isinstance(input_vector, np.ndarray)
        input_vector = input_vector if np_input else input_vector.numpy()
        assert input_vector.shape[-self.remaining_axes:] == self.input_shape, (
            "trailing dimensions of the input vector " + "are expected to be {} ".format(self.input_shape) +
            "while the input vector has shape {}".format(input_vector.shape[-self.remaining_axes:]))
        norm_axes = len(input_vector.shape) - self.remaining_axes

        if train:
            batch_mean = input_vector.mean(axis=tuple(range(norm_axes)))
            batch_sq_mean = (input_vector**2).mean(axis=tuple(range(norm_axes)))

            if self.per_element_update:
                batch_size = np.prod(input_vector.shape[:norm_axes])
                weight = self.beta**batch_size
            else:
                weight = self.beta

            self.running_mean = weight * self.running_mean + batch_mean * (1.0 - weight)
            self.running_mean_sq = weight * self.running_mean_sq + batch_sq_mean * (1.0 - weight)
            self.debiasing_term = weight * self.debiasing_term + 1.0 - weight

        self.barrier.wait()
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / np.sqrt(var)

        return out if np_input else torch.from_numpy(out)

    def denormalize(self, input_vector):
        # called when closing an episode and computing returns,
        # to denormalize values during rollout inference
        np_input = isinstance(input_vector, np.ndarray)
        input_vector = input_vector if np_input else input_vector.numpy()
        assert input_vector.shape[-self.remaining_axes:] == self.input_shape, (
            "trailing dimensions of the input vector " + "are expected to be {} ".format(self.input_shape) +
            "while the input vector has shape {}".format(input_vector.shape[-self.remaining_axes:]))

        mean, var = self.running_mean_var()
        out = input_vector * np.sqrt(var) + mean

        return out if np_input else torch.from_numpy(out)
