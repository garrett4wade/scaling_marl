import numpy as np

# from numba import njit, prange


# @njit
# njit does not help much if env batch size and number of agents is large
def compute_gae(buffer, slot_id, values, gamma=np.float32(0.99), gae_lambda=np.float32(0.97)):
    rewards = buffer.rewards[slot_id]
    masks = buffer.masks[slot_id]
    fct_masks = buffer.fct_masks[slot_id]
    episode_length = int(rewards.shape[0])

    delta = rewards + gamma * values[1:] * masks[1:] - values[:-1]
    gae = np.zeros(rewards.shape[1:], dtype=np.float32)
    m = gamma * gae_lambda * masks[1:]
    step = episode_length - 1
    while step >= 0:
        # if env is terminated compulsively, then abandon the finnal step
        # i.e. advantage of final step is 0, value target of final step is predicted value
        gae = (delta[step] + m[step] * gae) * fct_masks[step + 1]
        buffer.advantages[slot_id, step] = gae
        step -= 1
    buffer.v_target[slot_id] = buffer.advantages[slot_id] + values[:-1]


def masked_adv_normalization(buffer, slot_id, eps=np.float32(1e-5)):
    if hasattr(buffer, 'active_masks'):
        adv = buffer.advantages[slot_id].copy()
        adv[buffer.active_masks[slot_id, :-1] == 0] = np.nan
    buffer.advantages[slot_id] = (buffer.advantages[slot_id] - np.nanmean(adv)) / (np.nanstd(adv) + eps)
