import numpy as np
# import tensorflow as tf
import torch


def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return torch.constant(out)
        # return tf.constant(out)

    return _initializer


def listdict2dictnp(lis, keepdims=False):
    '''
        Convert a list of dicts of numpy arrays to a dict of numpy arrays.
        If keepdims is False the new outer dimension in each dict element will be
            the length of the list
        If keepdims is True, then the new outdimension in each dict will be the sum of the
            outer dimensions of each item in the list
    '''
    if keepdims:
        return {k: np.concatenate([d[k] for d in lis]) for k in lis[0]}
    else:
        return {k: np.array([d[k] for d in lis]) for k in lis[0]}


def shape_list(x):
    '''
        deal with dynamic shape in tensorflow cleanly
    '''
    ps = x.get_shape().as_list()
    ts = torch.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def l2_loss(pred, label, std, mask):
    '''
        Masked L2 loss with a scaling paramter (std). We made the choice that
            the loss would scale with the number of unmasked data points rather
            than have the same magnitude regardless of how many samples came in.
            TODO: Revisit whether this is the right choice.
    '''
    if mask is None:
        return 0.5 * torch.reduce_mean(torch.square((pred - label) / std))
    else:
        return 0.5 * torch.reduce_mean(mask * torch.square((pred - label) / std))
