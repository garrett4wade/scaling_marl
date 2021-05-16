import numpy as np
import time
from collections import namedtuple
# from numba import njit, prange

GAE_return = namedtuple('GAE_return', ['v_target', 'advantage'])


# @njit
# njit does not help much if env batch size and number of agents is large
def compute_gae(rewards, values, masks, fct_masks, gamma=np.float32(0.99), gae_lambda=np.float32(0.97)):
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    episode_length = int(rewards.shape[0])

    delta = rewards + gamma * values[1:] * masks[1:] - values[:-1]
    gae = np.zeros(rewards.shape[1:], dtype=np.float32)
    m = gamma * gae_lambda * masks[1:]
    step = episode_length - 1
    while step >= 0:
        # if env is terminated compulsively, then abandon the finnal step
        # i.e. advantage of final step is 0, value target of final step is predicted value
        gae = (delta[step] + m[step] * gae) * fct_masks[step + 1]
        advantages[step] = gae
        step -= 1
    returns = advantages + values[:-1]
    return GAE_return(returns, advantages)


def masked_normalization(x, active_masks=None, eps=np.float32(1e-5)):
    if active_masks:
        x_ = x.copy()
        x[x_ == 0] = np.nan
    return (x - np.nanmean(x)) / (np.nanstd(x) + eps)


if __name__ == "__main__":
    num_slots = 50
    ep_l = 400
    bs = 60
    num_agents = 8
    shape = (num_slots, ep_l, bs, num_agents)

    rewards = np.random.randn(num_slots, ep_l, bs, num_agents).astype(np.float32)
    values = np.random.randn(num_slots, ep_l + 1, bs, num_agents).astype(np.float32)
    masks = np.random.randint(0, 2, (num_slots, ep_l + 1, bs, num_agents)).astype(np.float32)
    active_masks = np.random.randint(0, 2, (num_slots, ep_l + 1, bs, num_agents)).astype(np.float32)
    fct_masks = np.random.randint(0, 2, (num_slots, ep_l + 1, bs, num_agents)).astype(np.float32)

    x, y = compute_gae(rewards[0], values[0], masks[0], active_masks[0], fct_masks[0])
    # assert np.all(x == x2)
    # assert np.all(y == y2)
    # print(returns[0])

    tik = time.time()
    for i in range(1, num_slots):
        x, y = compute_gae(rewards[i], values[i], masks[i], active_masks[i], fct_masks[i])
    print(time.time() - tik, (time.time() - tik) / (num_slots - 1))

    # tik = time.time()
    # for i in range(num_slots):
    #     norm_r = normalize(rewards[i])
    # print((time.time() - tik) / num_slots)
