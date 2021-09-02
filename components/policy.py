import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class BasePolicy:
    def __init__(self, rank, cfg, model_fn, is_training=True):
        """ Initialize a policy given a model function.

        Args:
            rank (int): GPU rank to use.
            cfg (argparse.Namespace): Configurations.
            model_fn (function): Constructor of the neural net model.
            is_training (bool, optional): Whether to train this policy. If this policy will be trained, the neural net
                                          model will be wrapped by PyTorch DistributedDataParallel. Defaults to True.
        """
        self.device = torch.device(rank)
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        self.neral_net = model_fn().to(self.device)
        if is_training:
            self.neral_net = DDP(self.neral_net, device_ids=[rank], output_device=rank)

    @torch.no_grad()
    def get_actions(self, *args, deterministic=False, **kwargs):
        """ Generate actions and other inference ouputs during rollout, e.g. action log-probs.

        Args:
            deterministic (bool, optional): Whether to use a stochastic policy or conduct exploration,
                                            e.g. whether to use eps-greedy. Defaults to False.

        Returns:
            dict: Actions and other inference ouputs.
        """
        raise NotImplementedError

    def evaluate_actions(self, *args, **kwargs):
        """ Generate outputs required for loss computation during training step,
            e.g. value target and action distribution entropies.

        Returns:
            tuple: Data required for loss computation.
        """
        raise NotImplementedError

    @torch.no_grad()
    def act(self, *args, deterministic=False, **kwargs):
        """ Generate actions (and rnn hidden states) during evaluation.

        Args:
            deterministic (bool, optional): Whether to use a stochastic policy or conduct exploration,
                                            e.g. whether to use eps-greedy. Defaults to False.

        Returns:
            tuple: Actions (and rnn hidden states).
        """
        raise NotImplementedError

    def parameters(self):
        return self.neral_net.parameters()

    def state_dict(self):
        return self.neual_net.state_dict()

    def load_state_dict(self, state_dict):
        self.neual_net.load_state_dict(state_dict)

    def train_mode(self):
        self.neual_net.train()

    def eval_mode(self):
        self.neual_net.eval()
