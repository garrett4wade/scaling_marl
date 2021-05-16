import numpy as np


def flatten(x, ndim):
    return x.reshape(-1, *x.shape[ndim:])


def to_chunk(x, num_chunks, t_dim=0, bs_dim=1):
    # split along 'time dimension' then concatenate along 'batch dimension'
    # then merge 'batch dimension' and 'agent dimension'
    return np.concatenate(np.split(x, num_chunks, axis=t_dim), axis=bs_dim)


def select(h, num_to_select, dim=0):
    assert h.shape[dim] % num_to_select == 0
    inds = np.arange(h.shape[dim], step=h.shape[dim] // num_to_select)
    return h[inds]
