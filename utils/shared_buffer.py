import torch
import time
import numpy as np
import torch.multiprocessing as mp

from utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(x, ndim):
    return x.reshape(-1, *x.shape[ndim:])


def _to_chunk(x, num_chunks):
    # split along 'time dimension' then concatenate along 'batch dimension'
    # then merge 'batch dimension' and 'agent dimension'
    x = np.concatenate(np.split(x, num_chunks), axis=1)
    return x.reshape(x.shape[0], -1, *x.shape[3:])


def _select_rnn_states(h, num_chunks):
    episode_length = h.shape[0]
    assert episode_length % num_chunks == 0
    chunk_len = h.shape[0] // num_chunks
    inds = np.arange(episode_length, step=chunk_len)
    return h[inds].reshape(-1, *h.shape[3:]).swapaxes(0, 1)


class SharedReplayBuffer:
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space, value_normalizer):
        """buffer for multi-agent environments with shared observation

        Structure is shown below. Because of asynchronous rollout execution,
        buffer has multiple queue positions such that actor returning fast
        does not need to wait others, instead it takes up next queue position.

        Envs in each actor have #env_split splits, each split has multiple envs
        and takes up a section of actor (a minibatch of data). Env splits of all
        actors with the same id compose of a slot in buffer, i.e. composition
        of all actors' minibatch data corresponding to the same env split. A slot
        is also a (normal) batch that will be loaded into GPU for loss computation
        and optimization.

        (suppose num_actors=3, qsize=5, num_split=2)

        +++++++++++ ————————————————
                  +                 | =========================
                  +                 |    section of actor 0
                  +                 | -------------------------
                  +    env split 0  |    section of actor 1
                  +  (slot/batch 0) | -------------------------
                  +                 |    section of actor 2
          queue   +                 | =========================
         position +  ————————————————
            0     +                 | =========================
                  +                 |    section of actor 0
                  +                 | -------------------------
                  +    env split 1  |    section of actor 1
                  +  (slot/batch 1) | -------------------------
                  +                 |    section of actor 2
                  +                 | =========================
        +++++++++++  ————————————————
        ...
        ...
        +++++++++++ ————————————————
                  +                 | =========================
                  +                 |    section of actor 0
                  +                 | -------------------------
                  +    env split 0  |    section of actor 1
                  +  (slot/batch 8) | -------------------------
                  +                 |    section of actor 2
          queue   +                 | =========================
         position +  ————————————————
            4     +                 | =========================
                  +                 |    section of actor 0
                  +                 | -------------------------
                  +    env split 1  |    section of actor 1
                  +  (slot/batch 9) | -------------------------
                  +                 |    section of actor 2
                  +                 | =========================
        +++++++++++  ————————————————
        """
        # storage shape configuration
        self.num_actors = num_actors = args.num_actors
        self.qsize = qsize = args.qsize
        self.num_split = num_split = args.num_split
        self.num_slots = num_slots = qsize * num_split
        self.env_per_split = env_per_split = args.env_per_actor // num_split
        self.batch_size = bs = num_actors * env_per_split
        self.num_agents = num_agents
        assert args.env_per_actor % num_split == 0

        self.episode_length = ep_l = args.episode_length
        self.hidden_size = hs = args.hidden_size
        self.recurrent_N = rec_n = args.recurrent_N

        self.data_chunk_length = args.data_chunk_length
        self.num_mini_batch = args.num_mini_batch

        # RL specific configuration
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.sample_reuse = args.ppo_epoch
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_proper_time_limits = args.use_proper_time_limits

        self.value_normalizer = value_normalizer

        # initialize storage
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((num_slots, ep_l + 1, bs, num_agents, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((num_slots, ep_l + 1, bs, num_agents, *obs_shape), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((num_slots, ep_l + 1, bs, num_agents, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros((num_slots, ep_l, bs, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((num_slots, ep_l, bs, num_agents, act_shape), dtype=np.float32)

        self.rnn_states = np.zeros((num_slots, ep_l + 1, bs, num_agents, rec_n, hs), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.masks = np.ones((num_slots, ep_l + 1, bs, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.rewards = np.zeros((num_slots, ep_l, bs, num_agents, 1), dtype=np.float32)
        self.value_preds = np.zeros((num_slots, ep_l + 1, bs, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros_like(self.rewards)

        # queue index
        self._prev_q_idx = -np.ones((num_split, ), dtype=np.int16)
        self._q_idx = np.zeros((num_split, ), dtype=np.uint8)

        # episode step record
        self._ep_step = np.zeros((num_split, ), dtype=np.int32)

        # buffer indicators
        self._used_times = np.zeros((num_slots, ), dtype=np.uint8)
        self._is_readable = np.zeros((num_slots, ), dtype=np.uint8)
        self._is_being_read = np.zeros((num_slots, ), dtype=np.uint8)
        self._is_writable = np.ones((num_slots, ), dtype=np.uint8)
        assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)

        self._read_ready = mp.Condition(mp.Lock())

        self.total_timesteps = 0

    def get_policy_inputs(self, split_id):
        slot_id = self._q_idx[split_id] * self.num_split + split_id
        ep_step = self._ep_step[split_id]

        return (self.share_obs[slot_id, ep_step], self.obs[slot_id, ep_step], self.rnn_states[slot_id, ep_step],
                self.rnn_states_critic[slot_id, ep_step], self.masks[slot_id, ep_step], self.available_actions[slot_id,
                                                                                                               ep_step])

    def insert_before_inference(self,
                                actor_id,
                                split_id,
                                share_obs,
                                obs,
                                rewards,
                                masks,
                                bad_masks=None,
                                active_masks=None,
                                available_actions=None):
        slot_id = self._q_idx[split_id] * self.num_split + split_id
        ep_step = self._ep_step[split_id]
        env_slice = slice(actor_id * self.env_per_split, (actor_id + 1) * self.env_per_split)

        # env step returns
        self.share_obs[slot_id, ep_step, env_slice] = share_obs
        self.obs[slot_id, ep_step, env_slice] = obs
        self.masks[slot_id, ep_step, env_slice] = masks
        if ep_step >= 1:
            self.rewards[slot_id, ep_step - 1, env_slice] = rewards
        if bad_masks is not None:
            self.bad_masks[slot_id, ep_step, env_slice] = bad_masks
        if active_masks is not None:
            self.active_masks[slot_id, ep_step, env_slice] = active_masks
        if available_actions is not None:
            self.available_actions[slot_id, ep_step, env_slice] = available_actions

        self.total_timesteps += self.env_per_split

    def insert_after_inference(self, split_id, value_preds, actions, action_log_probs, rnn_states, rnn_states_critic):
        slot_id = self._q_idx[split_id] * self.num_split + split_id
        ep_step = self._ep_step[split_id]

        if ep_step == 0 and self._prev_q_idx[split_id] >= 0:
            # fill bootstrap data in previous slot
            self._closure(split_id, value_preds)

        # model inference returns
        self.actions[slot_id, ep_step] = actions
        self.action_log_probs[slot_id, ep_step] = action_log_probs
        self.value_preds[slot_id, ep_step] = value_preds
        self.rnn_states[slot_id, ep_step + 1] = rnn_states
        self.rnn_states_critic[slot_id, ep_step + 1] = rnn_states_critic

        self._ep_step[split_id] += 1

        # section of this actor in current slot is full except for bootstrap step
        if ep_step == self.episode_length - 1:
            self._ep_step[split_id] = 0

            # move global queue pointer to next corresponding slot
            cur_q_idx = self._q_idx[split_id]
            self._q_idx[split_id] += 1
            self._q_idx[split_id] %= self.qsize
            new_slot_id = self._q_idx[split_id] * self.num_split + split_id
            # find next slot which is not busy
            while self._is_being_read[new_slot_id]:
                self._q_idx[split_id] += 1
                self._q_idx[split_id] %= self.qsize
                new_slot_id = self._q_idx[split_id] * self.num_split + split_id
            self._prev_q_idx[split_id] = cur_q_idx

            self._opening(new_slot_id, rnn_states, rnn_states_critic)

    def _opening(self, new_slot_id, rnn_states, rnn_states_critic):
        self.rnn_states[new_slot_id, 0] = rnn_states
        self.rnn_states_critic[new_slot_id, 0] = rnn_states_critic

        # if the next corresponding slot is readable, overwrite it
        if self._is_readable[new_slot_id]:
            # reset indicator for overwriting
            self._is_readable[new_slot_id] = 0
            self._is_writable[new_slot_id] = 1
        assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)

    def _closure(self, split_id, value_preds):
        slot_id = self._prev_q_idx[split_id] * self.num_split + split_id
        cur_slot_id = self._q_idx[split_id] * self.num_split + split_id

        self.share_obs[slot_id, -1] = self.share_obs[cur_slot_id, 0]
        self.obs[slot_id, -1] = self.obs[cur_slot_id, 0]
        self.masks[slot_id, -1] = self.masks[cur_slot_id, 0]
        self.rewards[slot_id, -1] = self.rewards[cur_slot_id, 0]
        if self.bad_masks is not None:
            self.bad_masks[slot_id, -1] = self.bad_masks[cur_slot_id, 0]
        if self.active_masks is not None:
            self.active_masks[slot_id, -1] = self.active_masks[cur_slot_id, 0]
        if self.available_actions is not None:
            self.available_actions[slot_id, -1] = self.available_actions[cur_slot_id, 0]

        self._compute_returns(slot_id, value_preds)

        # update indicator of current slot
        no_availble_before = not np.any(self._is_readable)
        self._is_readable[slot_id] = 1
        self._is_writable[slot_id] = 0
        # if reader is waiting for data, notify it
        if no_availble_before:
            with self._read_ready:
                self._read_ready.notify(1)

    # def chooseinsert(self,
    #                  share_obs,
    #                  obs,
    #                  rnn_states,
    #                  rnn_states_critic,
    #                  actions,
    #                  action_log_probs,
    #                  value_preds,
    #                  rewards,
    #                  masks,
    #                  bad_masks=None,
    #                  active_masks=None,
    #                  available_actions=None):
    #     self.share_obs[self.step] = share_obs
    #     self.obs[self.step] = obs
    #     self.rnn_states[self.step + 1] = rnn_states
    #     self.rnn_states_critic[self.step + 1] = rnn_states_critic
    #     self.actions[self.step] = actions
    #     self.action_log_probs[self.step] = action_log_probs
    #     self.value_preds[self.step] = value_preds
    #     self.rewards[self.step] = rewards
    #     self.masks[self.step + 1] = masks
    #     if bad_masks is not None:
    #         self.bad_masks[self.step + 1] = bad_masks
    #     if active_masks is not None:
    #         self.active_masks[self.step] = active_masks
    #     if available_actions is not None:
    #         self.available_actions[self.step] = available_actions

    #     self.step = (self.step + 1) % self.episode_length

    def get(self, recur=True):
        with self._read_ready:
            self._read_ready.wait_for(lambda: np.any(self._is_readable))
            slot_id = np.nonzero(self._is_readable)[0][0]
            # indicator readable -> being-read
            self._is_readable[slot_id] = 0
            self._is_being_read[slot_id] = 1
            assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)
        return slot_id, self.recurrent_generator(slot_id) if recur else self.feed_forward_generator(slot_id)

    def after_training_step(self, slot_id):
        with self._read_ready:
            # reset indicator, being-read -> writable
            self._is_being_read[slot_id] = 0
            self._used_times[slot_id] += 1
            if self._used_times[slot_id] >= self.sample_reuse:
                self._is_writable[slot_id] = 1
                self._used_times[slot_id] = 0
            else:
                self._is_readable[slot_id] = 1
            assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)

    # def chooseafter_update(self):
    #     self.rnn_states[0] = self.rnn_states[-1]
    #     self.rnn_states_critic[0] = self.rnn_states_critic[-1]
    #     self.masks[0] = self.masks[-1]
    #     self.bad_masks[0] = self.bad_masks[-1]

    def _compute_returns(self, slot_id, next_value, adv_normalization=True):
        tik = time.time()
        value_normalizer = self.value_normalizer
        if self._use_gae:
            self.value_preds[slot_id, -1] = next_value
            gae = 0
            for step in reversed(range(self.episode_length)):
                if self._use_popart:
                    bootstrap_v = value_normalizer.denormalize(self.value_preds[slot_id, step + 1])
                    cur_v = value_normalizer.denormalize(self.value_preds[slot_id, step])
                else:
                    bootstrap_v = self.value_preds[slot_id, step + 1]
                    cur_v = self.value_preds[slot_id, step]

                one_step_td = self.rewards[slot_id, step] + self.gamma * bootstrap_v * self.masks[slot_id, step + 1]
                delta = one_step_td - cur_v
                gae = delta + self.gamma * self.gae_lambda * gae * self.masks[slot_id, step + 1]
                if self._use_proper_time_limits:
                    gae = gae * self.bad_masks[slot_id, step + 1]
                self.returns[slot_id, step] = gae + cur_v
                self.advantages[slot_id, step] = gae
        else:
            self.returns[slot_id, -1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                if self._use_popart:
                    cur_v = value_normalizer.denormalize(self.value_preds[slot_id, step])
                else:
                    cur_v = self.value_preds[slot_id, step]

                if self._use_proper_time_limits:
                    bad_mask = self.bad_masks[slot_id, step + 1]
                else:
                    bad_mask = 1

                discount_r = (self.returns[slot_id, step + 1] * self.gamma * self.masks[slot_id, step + 1] +
                              self.rewards[slot_id, step])
                self.returns[slot_id, step] = discount_r * bad_mask + (1 - bad_mask) * cur_v
                self.advantages[slot_id, step] = self.returns[slot_id, step] - cur_v

        if adv_normalization:
            adv = self.advantages[slot_id, :].copy()
            adv[self.active_masks[slot_id, :-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(adv)
            std_advantages = np.nanstd(adv)
            self.advantages[slot_id, :] = (self.advantages[slot_id, :] - mean_advantages) / (std_advantages + 1e-5)
        print(time.time() - tik)

    def feed_forward_generator(self, slot_id):
        num_mini_batch = self.num_mini_batch
        batch_size = self.batch_size * self.episode_length * self.num_agents

        assert batch_size >= num_mini_batch and batch_size % num_mini_batch == 0
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        def flatten(x):
            # flatten first 3 dims
            # [T, B, A, D] -> [T*B*A, D]
            return _flatten(x, 3)

        share_obs = flatten(self.share_obs[slot_id, :-1])
        obs = flatten(self.obs[slot_id, :-1])
        actions = flatten(self.actions[slot_id])
        if self.available_actions is not None:
            available_actions = flatten(self.available_actions[slot_id, :-1])
            assert available_actions.shape[0] == batch_size
        value_preds = flatten(self.value_preds[slot_id, :-1])
        returns = flatten(self.returns[slot_id, :-1])
        masks = flatten(self.masks[slot_id, :-1])
        active_masks = flatten(self.active_masks[slot_id, :-1])
        action_log_probs = flatten(self.action_log_probs[slot_id])
        advantages = flatten(self.advantages[slot_id])
        assert share_obs[0] == batch_size
        assert obs.shape[0] == batch_size
        assert actions.shape[0] == batch_size
        assert value_preds.shape[0] == batch_size
        assert returns.shape[0] == batch_size
        assert masks.shape[0] == batch_size
        assert active_masks.shape[0] == batch_size
        assert action_log_probs.shape[0] == batch_size
        assert advantages.shape[0] == batch_size

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield (share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch)

    def recurrent_generator(self, slot_id):
        data_chunk_length = self.data_chunk_length
        num_mini_batch = self.num_mini_batch
        assert self.episode_length % data_chunk_length == 0
        assert self.batch_size >= num_mini_batch and self.batch_size % num_mini_batch == 0
        num_chunks = self.episode_length // data_chunk_length
        batch_size = self.batch_size * self.num_agents * num_chunks
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        def _cast(x):
            # B' = T * B * A / L
            # [T, B, A, D] -> [L, T*B/L, A, D] -> [L, B', D]
            return _to_chunk(x, num_chunks)

        share_obs = _cast(self.share_obs[slot_id, :-1])
        obs = _cast(self.obs[slot_id, :-1])
        actions = _cast(self.actions[slot_id])
        action_log_probs = _cast(self.action_log_probs[slot_id])
        value_preds = _cast(self.value_preds[slot_id, :-1])
        returns = _cast(self.returns[slot_id, :-1])
        masks = _cast(self.masks[slot_id, :-1])
        active_masks = _cast(self.active_masks[slot_id, :-1])
        advantages = _cast(self.advantages[slot_id])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[slot_id, :-1])

        def _cast_h(h):
            # B' = T * B * A / L
            # [T, B, A, rN, D] -> [T/L, B, A, rN, D] -> [B', rN, D] -> [rN, B', D]
            return _select_rnn_states(h, num_chunks)

        rnn_states = _cast_h(self.rnn_states[slot_id, :-1])
        rnn_states_critic = _cast_h(self.rnn_states_critic[slot_id, :-1])

        for indices in sampler:
            share_obs_batch = _flatten(share_obs[:, indices], 2)
            obs_batch = _flatten(obs[:, indices], 2)
            rnn_states_batch = np.swapaxes(rnn_states[:, indices], 0, 1)
            rnn_states_critic_batch = np.swapaxes(rnn_states_critic[:, indices], 0, 1)
            actions_batch = _flatten(actions[:, indices], 2)
            available_actions_batch = None if self.available_actions is None else _flatten(
                available_actions[:, indices], 2)
            value_preds_batch = _flatten(value_preds[:, indices], 2)
            return_batch = _flatten(returns[:, indices], 2)
            masks_batch = _flatten(masks[:, indices], 2)
            active_masks_batch = _flatten(active_masks[:, indices], 2)
            old_action_log_probs_batch = _flatten(action_log_probs[:, indices], 2)
            adv_targ = _flatten(advantages[:, indices], 2)

            outputs = (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
                       value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
                       adv_targ, available_actions_batch)
            for i, item in enumerate(outputs):
                assert np.all(1 - np.isnan(item)) and np.all(1 - np.isinf(item)), i
            yield outputs
