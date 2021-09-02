import torch
from utils.utils import store_args


class BaseAlgorithm:
    @store_args
    def __init__(self, cfg, policy, optimizer='adam'):
        """ Initialize an algorithm given a policy.

        Args:
            cfg (argparse.Namespace): Configurations.
            policy (BasePolicy): Training policy, whose evaluate_action method will be called.
            optimizer (str): Name of optimizer to use. Defaults to 'adam'.
        """
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=cfg.lr,
                                              eps=cfg.opti_eps,
                                              weight_decay=cfg.weight_decay)
        else:
            self.optimizer = None
            # TODO: add more optimizer options
            raise NotImplementedError

    def step(self, sample):
        """ Advance one training step given samples collected by actor workers.

        Example code:

        |-------------------------------------------------------|
        | ...                                                   |
        |                                                       |
        | some_data = self.policy.evaluate_actions(**sample)    |
        |                                                       |
        | loss = loss_fn(some_data, sample)                     |
        |                                                       |
        | self.optimizer.zero_grad()                            |
        | loss.backward()                                       |
        |                                                       |
        | ...                                                   |
        |                                                       |
        | self.optimizer.step()                                 |
        |                                                       |
        | ...                                                   |
        |-------------------------------------------------------|

        Args:
            sample (dict): All data required for training.

        Returns:
            dict: Detached training infos, e.g. losses and gradient norms.
        """
        raise NotImplementedError
