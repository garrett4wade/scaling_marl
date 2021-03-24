import torch
import numpy as np
from multiprocessing import Lock, Condition
from multiprocessing.managers import SharedMemoryManager
from utils.popart import PopArt
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


def byte_of_dtype(dtype):
    if '16' in str(dtype):
        byte_per_digit = 2
    elif '32' in str(dtype):
        byte_per_digit = 4
    elif '64' in str(dtype):
        byte_per_digit = 8
    elif 'bool' in str(dtype) or '8' in str(dtype):
        byte_per_digit = 1
    else:
        raise NotImplementedError
    return byte_per_digit


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


class ReplayBuffer:
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        """base buffer, can be integrated with different rollout collection mechanism

        Structure is shown below. Because of asynchronous rollout execution,
        buffer has multiple queue positions such that actor and inference server
        does not need to wait for the completion of optimization, instead it
        continues rollout and just takes up next queue position.

        Envs in each actor have #env_split splits, each split has multiple envs
        and takes up a section of actor (a minibatch of data). Env splits of all
        actors with the same id compose of a slot in buffer, i.e. composition
        of all actors' minibatch data corresponding to the same env split. A slot
        is also a (normal) batch that will be loaded into GPU for loss computation
        and optimization.

        If buffer is shared among multiple inference servers,
        then this buffer is just the concatenation of single-server buffers,
        e.g. first half slots are prepared for server 0 and second half
        slots are prepared for server 1, such that there is no read/write
        interleaving.

        (suppose num_actors=3, qsize=5, num_split=2, single-server)

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
        # NOTE: #actors = #clients per inference server
        # NOTE: actor ids of any server is [0, 1, ..., #actors-1]
        self.num_actors = num_actors = args.num_actors
        self.num_split = num_split = args.num_split
        self.num_servers = num_servers = args.num_servers
        self.qsize = qsize = args.qsize

        self.num_slots = num_slots = qsize * num_split * num_servers
        self.slots_per_server = qsize * num_split
        self.env_per_split = env_per_split = args.env_per_actor // num_split
        self.batch_size = bs = num_actors * env_per_split

        self.num_agents = num_agents
        self.num_trainers = args.num_trainers
        assert args.env_per_actor % num_split == 0
        self.slots_per_update = args.slots_per_update
        self.target_available_slots = self.slots_per_update * self.num_trainers

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

        if self._use_popart:
            self.value_normalizer = PopArt(input_shape=(1, ), num_trainers=args.num_trainers)
        else:
            self.value_normalizer = None

        self._smm = SharedMemoryManager()
        self._smm.start()

        # initialize storage
        def shape_prefix(bootstrap):
            t = ep_l + 1 if bootstrap else ep_l
            return (num_slots, t, bs, num_agents)

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self._share_obs_shm, self.share_obs = self.shm_array((*shape_prefix(True), *share_obs_shape), np.float32)
        self._obs_shm, self.obs = self.shm_array((*shape_prefix(True), *obs_shape), np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self._avail_shm, self.available_actions = self.shm_array((*shape_prefix(True), act_space.n), np.float32)
            self.available_actions[:] = 1
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self._act_shm, self.actions = self.shm_array((*shape_prefix(False), act_shape), np.float32)
        self._logp_shm, self.action_log_probs = self.shm_array((*shape_prefix(False), act_shape), np.float32)

        self._h_shm, self.rnn_states = self.shm_array((*shape_prefix(True), rec_n, hs), np.float32)
        self._hc_shm, self.rnn_states_critic = self.shm_array((*shape_prefix(True), rec_n, hs), np.float32)

        self._msk_shm, self.masks = self.shm_array((*shape_prefix(True), 1), np.float32)
        self.masks[:] = 1
        self._amsk_shm, self.active_masks = self.shm_array((*shape_prefix(True), 1), np.float32)
        self.active_masks[:] = 1

        self._r_shm, self.rewards = self.shm_array((*shape_prefix(False), 1), np.float32)
        self._v_shm, self.value_preds = self.shm_array((*shape_prefix(True), 1), np.float32)
        self._rt_shm, self.returns = self.shm_array((*shape_prefix(True), 1), np.float32)
        self._adv_shm, self.advantages = self.shm_array((*shape_prefix(False), 1), np.float32)

        # queue index
        self._prev_q_shm, self._prev_q_idx = self.shm_array((num_servers, num_split), np.int16)
        self._prev_q_idx[:] = -1
        self._q_shm, self._q_idx = self.shm_array((num_servers, num_split), np.uint8)

        # episode step record
        self._step_shm, self._ep_step = self.shm_array((num_servers, num_split), np.int32)

        # buffer indicators
        self._ut_shm, self._used_times = self.shm_array((num_slots, ), np.uint8)
        self._read_shm, self._is_readable = self.shm_array((num_slots, ), np.uint8)
        self._being_read_shm, self._is_being_read = self.shm_array((num_slots, ), np.uint8)
        self._wrt_shm, self._is_writable = self.shm_array((num_slots, ), np.uint8)
        self._is_writable[:] = 1
        self._rt_ready_shm, self._is_return_ready = self.shm_array((num_slots, ), np.uint8)
        assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)

        self._read_ready = Condition(Lock())

        self._timestep_shm, self.total_timesteps = self.shm_array((), np.int64)

        # to read/write env-specific summary info, e.g. winning rate, scores
        self.summary_lock = Lock()

    def get_utilization(self):
        with self._read_ready:
            available_slots = np.sum(self._is_readable)
        return available_slots / self.num_slots

    def shm_array(self, shape, dtype):
        byte_per_digit = byte_of_dtype(dtype)
        shm = self._smm.SharedMemory(size=byte_per_digit * np.prod(shape))
        shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return shm, shm_array

    def _locate(self, server_id, split_id):
        return server_id * self.slots_per_server + self._q_idx[server_id, split_id] * self.num_split + split_id

    def _locate_prev(self, server_id, split_id):
        assert self._prev_q_idx[server_id, split_id] >= 0
        return server_id * self.slots_per_server + self._prev_q_idx[server_id, split_id] * self.num_split + split_id

    def _opening(self, old_slot_id, new_slot_id):
        # start up a new slot
        self.rnn_states[new_slot_id, 0] = self.rnn_states[old_slot_id, -1]
        self.rnn_states_critic[new_slot_id, 0] = self.rnn_states_critic[old_slot_id, -1]

        with self._read_ready:
            # if the next corresponding slot is readable, overwrite it
            if self._is_readable[new_slot_id]:
                # reset indicator for overwriting
                self._is_return_ready[new_slot_id] = 0
                self._is_readable[new_slot_id] = 0
                self._is_writable[new_slot_id] = 1
            assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)

    def _closure(self, old_slot_id, new_slot_id):
        # complete bootstrap data & indicator of a written slot
        self.share_obs[old_slot_id, -1] = self.share_obs[new_slot_id, 0]
        self.obs[old_slot_id, -1] = self.obs[new_slot_id, 0]
        self.masks[old_slot_id, -1] = self.masks[new_slot_id, 0]
        self.rewards[old_slot_id, -1] = self.rewards[new_slot_id, 0]
        self.active_masks[old_slot_id, -1] = self.active_masks[new_slot_id, 0]
        if hasattr(self, 'available_actions'):
            self.available_actions[old_slot_id, -1] = self.available_actions[new_slot_id, 0]
        self.value_preds[old_slot_id, -1] = self.value_preds[new_slot_id, 0]

        with self._read_ready:
            # update indicator of current slot
            no_availble_before = np.sum(self._is_readable) < self.target_available_slots
            self._is_readable[old_slot_id] = 1
            self._is_writable[old_slot_id] = 0
            assert not self._is_return_ready[old_slot_id]
            # if reader is waiting for data, notify it
            if no_availble_before and np.sum(self._is_readable) >= self.target_available_slots:
                self._read_ready.notify(self.num_trainers)

    def get(self, train_popart=True, recur=True):
        with self._read_ready:
            self._read_ready.wait_for(lambda: np.sum(self._is_readable) >= self.target_available_slots)
            slots = np.nonzero(self._is_readable)[0][:self.slots_per_update]
            # indicator readable -> being-read
            self._is_readable[slots] = 0
            self._is_being_read[slots] = 1
            assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)
        return slots, self.recurrent_generator(slots, train_popart) if recur else self.feed_forward_generator(
            slots, train_popart)

    def after_training_step(self, slots):
        with self._read_ready:
            # reset indicator, being-read -> writable
            self._is_being_read[slots] = 0
            self._used_times[slots] += 1
            assert np.all(self._is_return_ready[slots])
            for slot_id in slots:
                if self._used_times[slot_id] >= self.sample_reuse:
                    self._is_return_ready[slot_id] = 0
                    self._is_writable[slot_id] = 1
                    self._used_times[slot_id] = 0
                else:
                    self._is_readable[slot_id] = 1
            assert np.all(self._is_readable + self._is_being_read + self._is_writable == 1)

    def _compute_returns(self, slots, adv_normalization=True):
        value_normalizer = self.value_normalizer
        if self._use_gae:
            gae = 0
            for step in reversed(range(self.episode_length)):
                if self._use_popart:
                    bootstrap_v = value_normalizer.denormalize(self.value_preds[slots, step + 1])
                    cur_v = value_normalizer.denormalize(self.value_preds[slots, step])
                else:
                    bootstrap_v = self.value_preds[slots, step + 1]
                    cur_v = self.value_preds[slots, step]

                one_step_return = self.rewards[slots, step] + self.gamma * bootstrap_v * self.masks[slots, step + 1]
                delta = one_step_return - cur_v
                gae = delta + self.gamma * self.gae_lambda * gae * self.masks[slots, step + 1]
                self.returns[slots, step] = gae + cur_v
                self.advantages[slots, step] = gae
        else:
            for step in reversed(range(self.rewards.shape[0])):
                if self._use_popart:
                    cur_v = value_normalizer.denormalize(self.value_preds[slots, step])
                else:
                    cur_v = self.value_preds[slots, step]

                discount_r = (self.returns[slots, step + 1] * self.gamma * self.masks[slots, step + 1] +
                              self.rewards[slots, step])
                self.returns[slots, step] = discount_r
                self.advantages[slots, step] = self.returns[slots, step] - cur_v

        if adv_normalization:
            adv = self.advantages[slots, :].copy()
            adv[self.active_masks[slots, :-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(adv)
            std_advantages = np.nanstd(adv)
            self.advantages[slots, :] = (self.advantages[slots, :] - mean_advantages) / (std_advantages + 1e-5)

    def feed_forward_generator(self, slots, train_popart):
        # once we ensure different trainers get different slot_ids,
        # the following few lines do not need lock
        non_ready_slot_ids = []
        for slot_id in slots:
            if not self._is_return_ready[slot_id]:
                non_ready_slot_ids.append(slot_id)
        self._compute_returns(non_ready_slot_ids)
        self._is_return_ready[non_ready_slot_ids] = 1

        num_mini_batch = self.num_mini_batch
        batch_size = self.batch_size * self.episode_length * self.num_agents

        assert batch_size >= num_mini_batch and batch_size % num_mini_batch == 0
        mini_batch_size = batch_size // num_mini_batch

        # flatten first 4 dims
        # [N, T, B, A, D] -> [N * T * B * A, D]
        share_obs = _flatten(self.share_obs[slots, :-1], 4)
        obs = _flatten(self.obs[slots, :-1], 4)
        actions = _flatten(self.actions[slots], 4)
        if hasattr(self, 'available_actions'):
            available_actions = _flatten(self.available_actions[slots, :-1], 4)
        value_preds = _flatten(self.value_preds[slots, :-1], 4)
        returns = _flatten(self.returns[slots, :-1], 4)
        v_target = self.value_normalizer(returns, train_popart) if self._use_popart else returns
        masks = _flatten(self.masks[slots, :-1], 4)
        active_masks = _flatten(self.active_masks[slots, :-1], 4)
        action_log_probs = _flatten(self.action_log_probs[slots], 4)
        advantages = _flatten(self.advantages[slots], 4)

        # TODO: although rnn_state is useless when using MLP,
        # we must return it to ensure the correctness of model dataflow ...
        rnn_states = _flatten(self.rnn_states[slots, :-1], 4)
        rnn_states_critic = _flatten(self.rnn_states_critic[slots, :-1], 4)

        if num_mini_batch == 1:
            outputs = (share_obs, obs, rnn_states, rnn_states_critic, actions, value_preds, v_target, masks,
                       active_masks, action_log_probs, advantages, available_actions)
            for i, item in enumerate(outputs):
                assert np.all(1 - np.isnan(item)) and np.all(1 - np.isinf(item)), i
            yield outputs
        else:
            rand = torch.randperm(batch_size).numpy()
            sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
            for indices in sampler:
                share_obs_batch = share_obs[indices]
                obs_batch = obs[indices]
                rnn_states_batch = rnn_states[indices]
                rnn_states_critic_batch = rnn_states_critic[indices]
                actions_batch = actions[indices]
                if hasattr(self, 'available_actions'):
                    available_actions_batch = available_actions[indices]
                else:
                    available_actions_batch = None
                value_preds_batch = value_preds[indices]
                v_target_batch = v_target[indices]
                masks_batch = masks[indices]
                active_masks_batch = active_masks[indices]
                old_action_log_probs_batch = action_log_probs[indices]
                if advantages is None:
                    adv_targ = None
                else:
                    adv_targ = advantages[indices]

                outputs = (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
                           value_preds_batch, v_target_batch, masks_batch, active_masks_batch,
                           old_action_log_probs_batch, adv_targ, available_actions_batch)
                for i, item in enumerate(outputs):
                    assert np.all(1 - np.isnan(item)) and np.all(1 - np.isinf(item)), i
                yield outputs

    def recurrent_generator(self, slots, train_popart):
        # once we ensure different trainers get different slot_ids,
        # the following few lines do not need lock
        non_ready_slot_ids = []
        for slot_id in slots:
            if not self._is_return_ready[slot_id]:
                non_ready_slot_ids.append(slot_id)
        self._compute_returns(non_ready_slot_ids)
        self._is_return_ready[non_ready_slot_ids] = 1

        data_chunk_length = self.data_chunk_length
        num_mini_batch = self.num_mini_batch
        assert self.episode_length % data_chunk_length == 0
        assert self.batch_size >= num_mini_batch and self.batch_size % num_mini_batch == 0
        num_chunks = self.episode_length // data_chunk_length
        batch_size = self.batch_size * self.num_agents * num_chunks
        mini_batch_size = batch_size // num_mini_batch

        def _cast(x):
            # [N, T, B, A, D] -> [T, N, B, A, D] -> [T, N * B, A, D]
            x = x.swapaxes(0, 1)
            x = x.reshape(x.shape[0], -1, *x.shape[3:])
            # B <- N * B, B' = T * B * A / L
            # [T, B, A, D] -> [L, T*B/L, A, D] -> [L, B', D]
            return _to_chunk(x, num_chunks)

        share_obs = _cast(self.share_obs[slots, :-1])
        obs = _cast(self.obs[slots, :-1])
        actions = _cast(self.actions[slots])
        action_log_probs = _cast(self.action_log_probs[slots])
        value_preds = _cast(self.value_preds[slots, :-1])
        returns = _cast(self.returns[slots, :-1])
        v_target = self.value_normalizer(returns.reshape(-1, 1), train_popart).reshape(
            *returns.shape) if self._use_popart else returns
        masks = _cast(self.masks[slots, :-1])
        active_masks = _cast(self.active_masks[slots, :-1])
        advantages = _cast(self.advantages[slots])

        if hasattr(self, 'available_actions'):
            available_actions = _cast(self.available_actions[slots, :-1])

        def _cast_h(h):
            # [N, T, B, A, rN, D] -> [T, N, B, A, rN, D] -> [T, N * B, A, rN, D]
            h = h.swapaxes(0, 1)
            h = h.reshape(h.shape[0], -1, *h.shape[3:])
            # B <- N * B, B' = T * B * A / L
            # [T, B, A, rN, D] -> [T/L, B, A, rN, D] -> [B', rN, D] -> [rN, B', D]
            return _select_rnn_states(h, num_chunks)

        rnn_states = _cast_h(self.rnn_states[slots, :-1])
        rnn_states_critic = _cast_h(self.rnn_states_critic[slots, :-1])

        if num_mini_batch == 1:
            share_obs_batch = share_obs
            obs_batch = obs
            rnn_states_batch = rnn_states
            rnn_states_critic_batch = rnn_states_critic
            actions_batch = actions
            available_actions_batch = None if self.available_actions is None else available_actions
            value_preds_batch = value_preds
            v_target_batch = v_target
            masks_batch = masks
            active_masks_batch = active_masks
            old_action_log_probs_batch = action_log_probs
            adv_targ = advantages

            outputs = (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
                       value_preds_batch, v_target_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
                       adv_targ, available_actions_batch)
            for i, item in enumerate(outputs):
                assert np.all(1 - np.isnan(item)) and np.all(1 - np.isinf(item)), i
            yield outputs
        else:
            rand = torch.randperm(batch_size).numpy()
            sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

            for indices in sampler:
                share_obs_batch = share_obs[:, indices]
                obs_batch = obs[:, indices]
                rnn_states_batch = np.swapaxes(rnn_states[:, indices], 0, 1)
                rnn_states_critic_batch = np.swapaxes(rnn_states_critic[:, indices], 0, 1)
                actions_batch = actions[:, indices]
                available_actions_batch = None if self.available_actions is None else available_actions[:, indices]
                value_preds_batch = value_preds[:, indices]
                v_target_batch = v_target[:, indices]
                masks_batch = masks[:, indices]
                active_masks_batch = active_masks[:, indices]
                old_action_log_probs_batch = action_log_probs[:, indices]
                adv_targ = advantages[:, indices]

                outputs = (share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
                           value_preds_batch, v_target_batch, masks_batch, active_masks_batch,
                           old_action_log_probs_batch, adv_targ, available_actions_batch)
                for i, item in enumerate(outputs):
                    assert np.all(1 - np.isnan(item)) and np.all(1 - np.isinf(item)), i
                yield outputs


class PolicyMixin:
    def get_policy_inputs(self, *args, **kwargs):
        ''' fetch data from buffer as rollout policy input. '''
        raise NotImplementedError

    def insert_before_inference(self, *args, **kwargs):
        ''' insert data returned by env.step.

        after all actors issue a inference request and invoke this method,
        a rollout data batch is ready in buffer.
        '''
        raise NotImplementedError

    def insert_after_inference(self, *args, **kwargs):
        ''' insert data returned by policy inference.

        invocation of this method indicates the termination of a
        inference request.
        '''
        raise NotImplementedError


class SharedPolicyMixin(PolicyMixin):
    def get_policy_inputs(self, server_id, split_id):
        slot_id = self._locate(server_id, split_id)
        ep_step = self._ep_step[server_id, split_id]

        return (self.share_obs[slot_id, ep_step], self.obs[slot_id, ep_step], self.rnn_states[slot_id, ep_step],
                self.rnn_states_critic[slot_id, ep_step], self.masks[slot_id, ep_step], self.available_actions[slot_id,
                                                                                                               ep_step])

    def insert_before_inference(self,
                                server_id,
                                actor_id,
                                split_id,
                                share_obs,
                                obs,
                                rewards,
                                dones,
                                available_actions=None):
        slot_id = self._locate(server_id, split_id)
        ep_step = self._ep_step[server_id, split_id]
        env_slice = slice(actor_id * self.env_per_split, (actor_id + 1) * self.env_per_split)

        # env step returns
        self.share_obs[slot_id, ep_step, env_slice] = share_obs
        self.obs[slot_id, ep_step, env_slice] = obs
        if ep_step >= 1:
            self.rewards[slot_id, ep_step - 1, env_slice] = rewards
        if available_actions is not None:
            self.available_actions[slot_id, ep_step, env_slice] = available_actions

        self.masks[slot_id, ep_step, env_slice] = 1 - np.all(dones, axis=1, keepdims=True)
        dones[np.all(dones, axis=1).squeeze(-1)] = 0
        self.active_masks[slot_id, ep_step, env_slice] = 1 - dones

        self.total_timesteps += self.env_per_split

    def insert_after_inference(self, server_id, split_id, value_preds, actions, action_log_probs, rnn_states,
                               rnn_states_critic):
        slot_id = self._locate(server_id, split_id)
        ep_step = self._ep_step[server_id, split_id]

        self.value_preds[slot_id, ep_step] = value_preds

        if ep_step == 0 and self._prev_q_idx[server_id, split_id] >= 0:
            old_slot_id = self._locate_prev(server_id, split_id)
            # fill bootstrap data in previous slot
            self._closure(old_slot_id, slot_id)

        # model inference returns
        self.actions[slot_id, ep_step] = actions
        self.action_log_probs[slot_id, ep_step] = action_log_probs

        rnn_mask = np.expand_dims(self.masks[slot_id, ep_step], -1)
        self.rnn_states[slot_id, ep_step + 1] = rnn_states * rnn_mask
        self.rnn_states_critic[slot_id, ep_step + 1] = rnn_states_critic * rnn_mask

        self._ep_step[server_id, split_id] += 1

        # section of this actor in current slot is full except for bootstrap step
        if ep_step == self.episode_length - 1:
            self._ep_step[server_id, split_id] = 0

            # move global queue pointer to next corresponding slot
            cur_q_idx = self._q_idx[server_id, split_id]
            self._q_idx[server_id, split_id] += 1
            self._q_idx[server_id, split_id] %= self.qsize
            new_slot_id = self._locate(server_id, split_id)
            # find next slot which is not busy
            while self._is_being_read[new_slot_id]:
                self._q_idx[server_id, split_id] += 1
                self._q_idx[server_id, split_id] %= self.qsize
                new_slot_id = self._locate(server_id, split_id)
            self._prev_q_idx[server_id, split_id] = cur_q_idx

            self._opening(slot_id, new_slot_id)


class SharedReplayBuffer(ReplayBuffer, SharedPolicyMixin):
    pass


class SequentialPolicyMixin(PolicyMixin):
    def get_policy_inputs(self, server_id, split_id):
        slot_id = self._locate(server_id, split_id)
        ep_step = self._ep_step[server_id, split_id]
        agent_id = self._agent_ids[server_id, split_id]

        return (self.share_obs[slot_id, ep_step, :, agent_id], self.obs[slot_id, ep_step, :, agent_id],
                self.rnn_states[slot_id, ep_step, :, agent_id], self.rnn_states_critic[slot_id, ep_step, :, agent_id],
                self.masks[slot_id, ep_step, :, agent_id], self.available_actions[slot_id, ep_step, :, agent_id])

    def insert_before_inference(self,
                                server_id,
                                actor_id,
                                split_id,
                                share_obs,
                                obs,
                                rewards,
                                dones,
                                available_actions=None):
        assert share_obs.shape == (self.env_per_split, *self.share_obs.shape[4:]), (share_obs.shape,
                                                                                    (self.env_per_split,
                                                                                     *self.share_obs.shape[4:]))
        assert obs.shape == (self.env_per_split, *self.obs.shape[4:]), (obs.shape, (self.env_per_split,
                                                                                    *self.obs.shape[4:]))

        slot_id = self._locate(server_id, split_id)
        ep_step = self._ep_step[server_id, split_id]
        agent_id = self._agent_ids[server_id, split_id]
        env_slice = slice(actor_id * self.env_per_split, (actor_id + 1) * self.env_per_split)

        # env step returns
        self.share_obs[slot_id, ep_step, env_slice, agent_id] = share_obs
        self.obs[slot_id, ep_step, env_slice, agent_id] = obs

        assert rewards.shape == (self.env_per_split, self.num_agents, 1), (rewards.shape, (self.env_per_split,
                                                                                           self.num_agents, 1))
        # accumulate reward first, and then record final accumulated reward when env terminates,
        # because if current step is 'done', reported reward is from previous transition, which
        # belongs to the previous episode (current step is the opening of the next episode)
        self._reward_since_last_action[server_id, split_id, env_slice] += rewards
        if ep_step >= 1:
            is_done_before = self._env_done_trigger[server_id, split_id, env_slice] > 0
            not_done_yet = (1 - is_done_before).astype(np.bool)

            accumulated_reward = self._reward_since_last_action[server_id, split_id, env_slice][not_done_yet, agent_id]
            self.rewards[slot_id, ep_step - 1, env_slice][not_done_yet, agent_id] = accumulated_reward
            self._reward_since_last_action[server_id, split_id, env_slice][not_done_yet, agent_id] = 0

            saved_reward = self._reward_when_env_done[server_id, split_id, env_slice][is_done_before, agent_id]
            self.rewards[slot_id, ep_step - 1, env_slice][is_done_before, agent_id] = saved_reward

        # record final accumulated reward when env terminates
        self._reward_when_env_done[server_id, split_id, env_slice][dones.squeeze(-1)] = self._reward_since_last_action[
            server_id, split_id, env_slice][dones.squeeze(-1)]
        self._reward_since_last_action[server_id, split_id, env_slice][dones.squeeze(-1)] = 0

        if available_actions is not None:
            assert available_actions.shape == (self.env_per_split,
                                               *self.available_actions.shape[4:]), (available_actions.shape,
                                                                                    (self.env_per_split,
                                                                                     *self.available_actions.shape[4:]))
            self.available_actions[slot_id, ep_step, env_slice, agent_id] = available_actions

        assert dones.shape == (self.env_per_split, 1), (dones.shape, (self.env_per_split, 1))
        assert dones.dtype == np.bool, dones.dtype
        # once env is done, fill the next #agents timestep with 0 to mask bootstrap values
        # env_done_trigger records remaining timesteps to be filled with 0
        trigger = self._env_done_trigger[server_id, split_id, env_slice]
        trigger[dones.squeeze(-1)] = self.num_agents
        # NOTE: mask is initialized as all 1, hence we only care about filling 0
        self.masks[slot_id, ep_step, env_slice, agent_id][trigger > 0] = 0
        self._env_done_trigger[server_id, split_id, env_slice] = np.maximum(trigger - 1, 0)
        assert np.all(self._env_done_trigger >= 0) and np.all(self._env_done_trigger <= self.num_agents)

        # active_mask is always 1 because env automatically resets when any agent induces termination

        if agent_id == self.num_agents - 1:
            self.total_timesteps += self.env_per_split

    def insert_after_inference(self, server_id, split_id, value_preds, actions, action_log_probs, rnn_states,
                               rnn_states_critic):
        slot_id = self._locate(server_id, split_id)
        ep_step = self._ep_step[server_id, split_id]
        agent_id = self._agent_ids[server_id, split_id]

        self.value_preds[slot_id, ep_step, :, agent_id] = value_preds

        if ep_step == 0 and agent_id == self.num_agents - 1 and self._prev_q_idx[server_id, split_id] >= 0:
            old_slot_id = self._locate_prev(server_id, split_id)
            # fill bootstrap data in previous slot
            self._closure(old_slot_id, slot_id)

        # model inference returns
        self.actions[slot_id, ep_step, :, agent_id] = actions
        self.action_log_probs[slot_id, ep_step, :, agent_id] = action_log_probs

        rnn_mask = np.expand_dims(self.masks[slot_id, ep_step, :, agent_id], -1)
        self.rnn_states[slot_id, ep_step + 1, :, agent_id] = rnn_states * rnn_mask
        self.rnn_states_critic[slot_id, ep_step + 1, :, agent_id] = rnn_states_critic * rnn_mask

        if agent_id == self.num_agents - 1:
            self._agent_ids[server_id, split_id] = 0
            self._ep_step[server_id, split_id] += 1
            # section of this actor in current slot is full except for bootstrap step
            if ep_step == self.episode_length - 1:
                self._ep_step[server_id, split_id] = 0

                # move global queue pointer to next corresponding slot
                cur_q_idx = self._q_idx[server_id, split_id]
                self._q_idx[server_id, split_id] += 1
                self._q_idx[server_id, split_id] %= self.qsize
                new_slot_id = self._locate(server_id, split_id)
                # find next slot which is not busy
                while self._is_being_read[new_slot_id]:
                    self._q_idx[server_id, split_id] += 1
                    self._q_idx[server_id, split_id] %= self.qsize
                    new_slot_id = self._locate(server_id, split_id)
                self._prev_q_idx[server_id, split_id] = cur_q_idx

                self._opening(slot_id, new_slot_id)
        else:
            self._agent_ids[server_id, split_id] += 1


class SequentialReplayBuffer(ReplayBuffer, SequentialPolicyMixin):
    def __init__(self, *args):
        super().__init__(*args)
        self._agent_ids = np.zeros((self.num_servers, self.num_split), dtype=np.uint8)
        self._reward_since_last_action = np.zeros(
            (self.num_servers, self.num_split, self.batch_size, self.num_agents, 1), dtype=np.float32)
        self._reward_when_env_done = np.zeros_like(self._reward_since_last_action)
        self._env_done_trigger = np.zeros((self.num_servers, self.num_split, self.batch_size), dtype=np.int16)
