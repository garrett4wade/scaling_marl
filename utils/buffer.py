import torch
import time
import numpy as np
import multiprocessing as mp
from queue import Empty
from algorithms.storage_registries import get_ppo_storage_specs, to_numpy_type, SUMMARY_KEYS
from algorithms.utils.modules import compute_gae, masked_normalization
from utils.utils import log
from algorithms.utils.transforms import flatten, to_chunk, select
from utils.popart import PopArt


class ReplayBuffer:
    def __init__(self, args, obs_space, share_obs_space, act_space):
        self.args = args

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        # system configuration
        self.num_actors = args.num_actors
        self.num_actor_groups = args.num_actor_groups
        self.num_splits = args.num_splits
        self.qsize = args.qsize

        self.envs_per_actor = args.envs_per_actor
        self.envs_per_split = args.envs_per_actor // args.num_splits
        self.actors_per_group = self.num_actors // self.num_actor_groups
        self.data_chunk_length = args.data_chunk_length

        assert args.envs_per_actor % self.num_splits == 0
        assert self.num_actors % self.num_actor_groups == 0

        # storage shape configuration
        self.num_agents = args.num_agents
        self.episode_length = args.episode_length

        # TODO: support n-step bootstrap
        # self.bootstrap_step = bootstrap_step = args.bootstrap_step

    def _init_storage(self):
        # initialize storage
        # TODO: replace get_ppo_storage_specs with get_${algorithm}_storage_specs
        self.storage_specs, self.policy_input_keys, self.policy_output_keys = get_ppo_storage_specs(
            self.args, self.obs_space, self.share_obs_space, self.act_space)
        self.storage_keys = [storage_spec.name for storage_spec in self.storage_specs]

        self.shapes_and_dtypes = {}

        def shape_prefix(bootstrap, select=False, per_seg=False):
            t = self.episode_length if not select else self.episode_length // self.data_chunk_length
            t = t + 1 if bootstrap else t
            bs = self.envs_per_seg if per_seg else self.envs_per_slot
            return (self.num_slots, t, bs, self.num_agents)

        for storage_spec in self.storage_specs:
            name, shape, dtype, bootstrap, init_value = storage_spec
            assert init_value == 0 or init_value == 1
            init_method = torch.zeros if init_value == 0 else torch.ones

            # only store rnn_states at the beginning of a chunk
            real_shape = (*shape_prefix(bootstrap, ('rnn_states' in name)), *shape)

            setattr(self, '_' + name, init_method(real_shape, dtype=dtype).share_memory_())
            setattr(self, name, getattr(self, '_' + name).numpy())

            # saved shape need to remove slot dim
            seg_shape = (*shape_prefix(bootstrap, ('rnn_states' in name), per_seg=True), *shape)
            self.shapes_and_dtypes[name] = (seg_shape[1:], to_numpy_type(dtype))

        # self._storage is torch.Tensor handle while self.storage is numpy.array handle
        # these 2 handles point to the same block of memory
        self._storage = {k: getattr(self, '_' + k) for k in self.storage_keys}
        self.storage = {k: getattr(self, k) for k in self.storage_keys}

        # to specify recorded summary infos
        self.summary_keys = SUMMARY_KEYS[self.args.env_name]
        self.summary_lock = mp.Lock()

        # buffer indicators
        self._is_readable = torch.zeros((self.num_slots, ), dtype=torch.uint8).share_memory_().numpy()
        self._is_busy = torch.zeros((self.num_slots, ), dtype=torch.uint8).share_memory_().numpy()
        self._is_writable = torch.ones((self.num_slots, ), dtype=torch.uint8).share_memory_().numpy()
        assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)

        self._time_stamp = torch.zeros((self.num_slots, ), dtype=torch.float32).share_memory_().numpy()

        self._read_ready = mp.Condition(mp.RLock())

        self.total_timesteps = torch.zeros((), dtype=torch.int64).share_memory_()

    def _allocate(self):
        with self._read_ready:
            writable_slots = np.nonzero(self._is_writable)[0]
            if len(writable_slots) > 0:
                slot_id = writable_slots[0]
                # writable -> busy
                self._is_writable[slot_id] = 0
            else:
                readable_slots = np.nonzero(self._is_readable)[0]
                assert len(readable_slots) > 0, 'please increase qsize!'
                # replace the oldest readable slot, in a FIFO pattern
                slot_id = readable_slots[np.argsort(self._time_stamp[readable_slots])[0]]
                # readable -> busy
                self._is_readable[slot_id] = 0
                self._time_stamp[slot_id] = 0
            self._is_busy[slot_id] = 1
            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)
        return slot_id

    def _mark_as_readable(self, slots):
        with self._read_ready:
            # update indicator of current slot
            no_availble_before = np.sum(self._is_readable) < self.target_num_slots
            # readable -> busy (being written)
            self._is_readable[slots] = 1
            self._is_busy[slots] = 0
            self._time_stamp[slots] = time.time()
            # if reader is waiting for data, notify it
            if no_availble_before and np.sum(self._is_readable) >= self.target_num_slots:
                self._read_ready.notify(self.num_consumers_to_notify)

    @property
    def utilization(self):
        with self._read_ready:
            available_slots = np.sum(self._is_readable).item()
        return available_slots / self.num_slots

    def get(self, block=True, timeout=None, reduce_fn=lambda x: x[0]):
        with self._read_ready:
            if np.sum(self._is_readable) == 0 and not block and not timeout:
                raise Empty
            ready = self._read_ready.wait_for(lambda: np.sum(self._is_readable) >= self.target_num_slots,
                                              timeout=timeout)
            if not ready:
                return None

            available_slots = np.nonzero(self._is_readable)[0]
            # use reduce fn to select required slots from sorted timestamps
            # select the oldest one as default, as workers send data to the learner in a FIFO pattern
            slot = available_slots[reduce_fn(np.argsort(self._time_stamp[available_slots]))]

            # readable -> busy (being-read)
            self._is_readable[slot] = 0
            self._is_busy[slot] = 1

            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)
        return slot

    def get_many(self, timeout=None):
        with self._read_ready:
            # if timeout is reached, get_many will return an empty list, which has no effect on transmitter
            self._read_ready.wait_for(lambda: np.sum(self._is_readable) >= self.target_num_slots, timeout=timeout)

            available_slots = np.nonzero(self._is_readable)[0]

            # readable -> busy (being-read)
            self._is_readable[available_slots] = 0
            self._is_busy[available_slots] = 1

            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)
        return available_slots

    def close_out(self, slot_id):
        with self._read_ready:
            # reset indicator, busy (being-read) -> writable
            self._is_busy[slot_id] = 0
            self._is_writable[slot_id] = 1
            self._time_stamp[slot_id] = 0
            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)


class WorkerBuffer(ReplayBuffer):
    def __init__(self, args, obs_space, share_obs_space, act_space):
        # NOTE: value target computation is deferred to centralized trainer in consistent with off-policy correction
        # e.g. V-trace and Retrace
        super().__init__(args, obs_space, share_obs_space, act_space)
        self.target_num_slots = 1
        self.num_consumers_to_notify = 1

        self.num_slots = self.qsize * self.num_actor_groups
        self.envs_per_seg = self.envs_per_slot = self.actors_per_group * self.envs_per_split

        super()._init_storage()

        # indices mapping client identity to slot id
        self._slot_indices = (-1) * torch.ones((self.num_splits * self.num_actor_groups, ), dtype=torch.int32).share_memory_().numpy()
        self._prev_slot_indices = (-1) * torch.ones((self.num_splits * self.num_actor_groups, ), dtype=torch.int32).share_memory_().numpy()

        # episode step record
        self._ep_step = torch.zeros((self.num_slots, ), dtype=torch.int32).share_memory_().numpy()
        # Buffer insertion is invoked after copying actions to small shared memory for actors.
        # Since buffer insertion is slow, it is very possible that before the buffer insertion on policy worker #1
        # finishes, policy worker #2 starts buffer insertion of the same client (slot),
        # which causes confusion on ep_step and slot_id. One solution is adding a multiprocessing lock.
        self._insertion_idx_lock = mp.Lock()

        # summary block
        self.summary_block = torch.zeros(
            (args.num_splits, self.envs_per_split * args.num_actors, len(self.summary_keys)),
            dtype=torch.float32).share_memory_().numpy()

    def _allocate(self, identity):
        slot_id = super()._allocate()

        if self._slot_indices[identity] != -1:
            # if this is not the very first allocation
            self._prev_slot_indices[identity] = self._slot_indices[identity]

        self._slot_indices[identity] = slot_id

    def _slot_closure(self, old_slots, new_slots):
        # when filling the first timestep of a new slot, copy data into the previous slot as bootstrap values
        # specifially, copied data includes all data needs to be bootstrapped except for rnn states (see storage specs)
        # and rewards, because rewards is 1-step behind other aligned data,
        # i.e., env.step() returns observations of the current step, but rewards of the previous step
        for storage_spec in self.storage_specs:
            name, _, _, bootstrap, _ = storage_spec
            if 'rnn_states' not in name and bootstrap:
                self.storage[name][old_slots, -1] = self.storage[name][new_slots, 0]

        self.rewards[old_slots, -1] = self.rewards[new_slots, 0]

        # NOTE: following 2 lines for debug only
        # with self._read_ready:
        #     assert np.all(self._is_busy[old_slots]) and np.all(self._is_busy[new_slots])
        super()._mark_as_readable(old_slots)


class LearnerBuffer(ReplayBuffer):
    def __init__(self, args, obs_space, share_obs_space, act_space):
        super().__init__(args, obs_space, share_obs_space, act_space)

        self.target_num_slots = self.num_consumers_to_notify = args.num_trainers
        self.num_slots = args.qsize
        # concatenate several slots from workers into a single batch,
        # which will be then sent to GPU for optimziation
        self.envs_per_seg = self.actors_per_group * self.envs_per_split
        self.envs_per_slot = args.slots_per_update * self.envs_per_seg

        self.slots_per_update = args.slots_per_update

        self.sample_reuse = args.sample_reuse

        super()._init_storage()        

        self._used_times = torch.zeros((self.num_slots, ), dtype=torch.int32).share_memory_().numpy()

        # summary block
        self.summary_block = torch.zeros((len(args.seg_addrs), len(self.summary_keys)),
                                         dtype=torch.float32).share_memory_().numpy()

        self._ptr_lock = mp.RLock()
        self._global_ptr = torch.zeros((2, ), dtype=torch.int32).share_memory_().numpy()

        self.gamma = np.float32(args.gamma)
        self.lmbda = np.float32(args.gae_lambda)

        # popart
        self._use_popart = args.use_popart
        if self._use_popart:
            self.value_normalizer = PopArt((1, ), self.args.num_trainers)
        else:
            self.value_normalizer = None

        self._use_advantage_normalization = args.use_advantage_normalization
        self._use_recurrent_policy = args.use_recurrent_policy

        if args.use_recurrent_policy:
            self.data_chunk_length = args.data_chunk_length
            assert self.episode_length % self.data_chunk_length == 0
            self.num_chunks = self.episode_length // self.data_chunk_length

            self.batch_size = self.envs_per_slot * self.num_chunks * self.num_agents

            log.info('Use recurrent policy. Batch size: {%d envs} * {%d chunks with length %d} * {%d agents} = %d',
                     self.envs_per_slot, self.num_chunks, self.data_chunk_length, self.num_agents, self.batch_size)

        else:
            self.batch_size = self.envs_per_slot * self.episode_length * self.num_agents
            log.info('Use feed forward policy. Batch size: {%d envs} * {%d timesteps} * {%d agents} = %d',
                     self.envs_per_slot, self.episode_length, self.num_agents, self.batch_size)

        self.num_mini_batch = args.num_mini_batch
        assert self.batch_size >= self.num_mini_batch and self.batch_size % self.num_mini_batch == 0
        self.mini_batch_size = self.batch_size // self.num_mini_batch

        if self.num_mini_batch > 1:
            log.info('Split a whole batch into %d minibatches. Each has size %d', self.num_mini_batch,
                     self.mini_batch_size)

    def _allocate(self):
        slot_id = super()._allocate()

        self._global_ptr[0] = slot_id
        self._global_ptr[1] = 0

    def put(self, seg_dict):
        # move pointer forward without waiting for the completion of copying
        with self._ptr_lock:
            slot_id = self._global_ptr[0].item()
            position_id = self._global_ptr[1].item()
            if position_id == self.slots_per_update - 1:
                self._allocate()
            else:
                self._global_ptr[1] += 1

        # copy data into main storage
        batch_slice = slice(position_id * self.envs_per_seg, (position_id + 1) * self.envs_per_seg)

        for k, v in seg_dict.items():
            self.storage[k][slot_id, :, batch_slice] = v

        self.total_timesteps += self.envs_per_seg * self.episode_length

        # mark the slot as readable if needed
        if position_id == self.slots_per_update - 1:
            with self._read_ready:
                self._used_times[slot_id] = 0
                super()._mark_as_readable(slot_id)

    def get(self, block=True, timeout=None):
        with self._read_ready:
            # randomly choose one slot from the oldest available slots
            slot_id = super().get(block, timeout, lambda x: np.random.choice(x[:self.target_num_slots]))
            # TODO: default reuse pattern is recycle, while it could be set to 'exhausting' or others
            # defer the timestamp such that the slot will be selected again only after
            # all readable slots are selected at least once
            self._time_stamp[slot_id] = np.max(self._time_stamp) + 1
        return slot_id, self.recurrent_generator(
            slot_id) if self._use_recurrent_policy else self.feed_forward_generator(slot_id)

    def close_out(self, slot_id):
        with self._read_ready:
            self._used_times[slot_id] += 1
            if self._used_times[slot_id] >= self.sample_reuse:
                self._used_times[slot_id] = 0
                super().close_out(slot_id)
            else:
                self._is_readable[slot_id] = 1
                self._is_busy[slot_id] = 0
                assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)

    def feed_forward_generator(self, slot):
        output_tensors = {}

        if self._use_popart:
            denormalized_values = self.value_normalizer.denormalize(self.values[slot])
        else:
            denormalized_values = self.values[slot]

        # compute value targets and advantages every time learner fetches data, because as the same slot
        # is reused, we need to recompute values of corresponding observations
        gae_return = compute_gae(self.rewards[slot], denormalized_values, self.masks[slot], self.fct_masks[slot],
                                 self.gamma, self.lmbda)
        v_target, advantage = gae_return.v_target, gae_return.advantage

        if self._use_advantage_normalization:
            advantage = masked_normalization(advantage, self.active_masks[slot, :-1])

        if self._use_popart:
            v_target = self.value_normalizer(v_target)

        for k in self.storage_keys:
            # we don't need rnn_states/fct_masks/rewards for MLP policy learning
            # fct_masks/rewards are only used in gae computation
            if 'rnn_states' not in k and k != 'fct_masks' and k != 'rewards':
                # flatten first 3 dims
                # [T, B, A, D] -> [T * B * A, D]
                output_tensors[k] = flatten(self.storage[k][slot, :self.episode_length], 3)

        output_tensors['v_target'] = flatten(v_target, 3)
        output_tensors['advantages'] = flatten(advantage, 3)

        # NOTE: following 2 lines are for debuggin only, which WILL SIGNIFICANTLY AFFECT SYSTEM PERFORMANCE
        # for k, v in output_tensors.items():
        #     assert np.all(1 - np.isnan(v)) and np.all(1 - np.isinf(v)), k

        if self.num_mini_batch == 1:
            yield output_tensors
        else:
            rand = torch.randperm(self.batch_size).numpy()
            for i in range(self.num_mini_batch):
                indice = rand[i * self.mini_batch_size:(i + 1) * self.mini_batch_size]
                yield {k: v[indice] for k, v in output_tensors.items()}

    def recurrent_generator(self, slot):
        output_tensors = {}

        if self._use_popart:
            denormalized_values = self.value_normalizer.denormalize(self.values[slot])
        else:
            denormalized_values = self.values[slot]

        # compute value targets and advantages every time learner fetches data, because as the same slot
        # is reused, we need to recompute values of corresponding observations
        gae_return = compute_gae(self.rewards[slot], denormalized_values, self.masks[slot], self.fct_masks[slot],
                                 self.gamma, self.lmbda)
        v_target, advantage = gae_return.v_target, gae_return.advantage

        if self._use_advantage_normalization:
            advantage = masked_normalization(advantage, self.active_masks[slot, :-1])

        if self._use_popart:
            v_target = self.value_normalizer(v_target)

        def _cast(x):
            # [T, B, A, D] -> [T, B * A, D] -> [L, B * A * (T/L), D]
            x = x.reshape(x.shape[0], -1, *x.shape[3:])
            return to_chunk(x, self.num_chunks)

        def _cast_h(h):
            # [T, B, A, rN, D] -> [T/L, B, A, rN, D] -> [T/L * B * A, rN, D] -> [rN, T/L * B * A, D]
            h = select(h, self.num_chunks)
            return h.reshape(-1, *h.shape[3:]).swapaxes(0, 1)

        for k in self.storage_keys:
            # we don't need fct_masks/rewards for policy learning, because they are used in v_target computation
            if k != 'fct_masks' and k != 'rewards':
                if 'rnn_states' in k:
                    output_tensors[k] = _cast_h(self.storage[k][slot, :self.episode_length])
                else:
                    output_tensors[k] = _cast(self.storage[k][slot, :self.episode_length])

        output_tensors['v_target'] = _cast(v_target)
        output_tensors['advantages'] = _cast(advantage)

        # NOTE: following 2 lines are for debuggin only, which WILL SIGNIFICANTLY AFFECT SYSTEM PERFORMANCE
        # for k, v in output_tensors.items():
        #     assert np.all(1 - np.isnan(v)) and np.all(1 - np.isinf(v)), k

        if self.num_mini_batch == 1:
            yield output_tensors
        else:
            rand = torch.randperm(self.batch_size).numpy()
            for i in range(self.num_mini_batch):
                indice = rand[i * self.mini_batch_size:(i + 1) * self.mini_batch_size]
                yield {k: v[indice] for k, v in output_tensors.items()}


class PolicyMixin:
    def insert(self, *args, **kwargs):
        ''' insert data returned by inference and env.step. '''
        raise NotImplementedError


class SharedPolicyMixin(PolicyMixin):
    def insert(self, timing, client_ids, obs, share_obs, rewards, dones, fct_masks=None, available_actions=None, **policy_outputs):
        slot_ids = np.zeros(len(client_ids), dtype=np.int32)

        # fill in the bootstrap step of a previous slot
        closure_slot_ids, old_closure_slot_ids = [], []
        # if a slot is full except for the bootstrap step, allocate a new slot for the corresponding client
        opening_slot_ids, opening_new_slot_ids = [], []

        with timing.add_time('inference/insert/acquire_lock'), timing.time_avg('inference/insert/acquire_lock_once'):
            self._insertion_idx_lock.acquire()

        with timing.add_time('inference/insert/get_indices'), timing.time_avg('inference/insert/get_indices_once'):
            for i, client_id in enumerate(client_ids):
                if self._slot_indices[client_id] == -1:
                    self._allocate(client_id)
                slot_ids[i] = self._slot_indices[client_id]

            ep_steps = self._ep_step[slot_ids]

        with timing.add_time('inference/insert/process_marginal'), timing.time_avg('inference/insert/process_marginal_once'):
            # advance 1 timestep
            self._ep_step[slot_ids] += 1

            for ep_step, slot_id, client_id in zip(ep_steps, slot_ids, client_ids):
                if ep_step == 0 and self._prev_slot_indices[client_id] != -1:
                    closure_slot_ids.append(slot_id)
                    old_closure_slot_ids.append(self._prev_slot_indices[client_id])

                if ep_step == self.episode_length - 1:
                    opening_slot_ids.append(slot_id)
                    self._allocate(client_id)
                    opening_new_slot_ids.append(self._slot_indices[client_id])

            self._ep_step[opening_slot_ids] = 0
            self._insertion_idx_lock.release()

        with timing.add_time('inference/insert/copy_data'), timing.time_avg('inference/insert/copy_data_once'):
            # env step returns
            self.share_obs[slot_ids, ep_steps] = share_obs
            self.obs[slot_ids, ep_steps] = obs
            self.rewards[slot_ids[ep_steps >= 1], ep_steps[ep_steps >= 1] - 1] = rewards[ep_steps >= 1]
            masks = 1 - np.all(dones, axis=2, keepdims=True)
            self.masks[slot_ids, ep_steps] = masks

            if hasattr(self, 'available_actions') and available_actions is not None:
                self.available_actions[slot_ids, ep_steps] = available_actions

            if hasattr(self, 'active_masks'):
                dones_cp = dones.copy()
                # deal with the auto-reset
                dones_cp[np.all(dones, axis=2).squeeze(-1)] = 0
                self.active_masks[slot_ids, ep_steps] = 1 - dones_cp

            if hasattr(self, 'fct_masks') and fct_masks is not None:
                self.fct_masks[slot_ids, ep_steps] = fct_masks

            # model inference returns
            rnn_mask = np.expand_dims(masks, -1)
            for k in policy_outputs.keys():
                if 'rnn_states' in k:
                    selected_idx = np.logical_and((ep_steps + 1) % self.data_chunk_length == 0, (ep_steps + 1) < self.episode_length)
                    if np.any(selected_idx):
                        self.storage[k][slot_ids[selected_idx], (ep_steps[selected_idx] + 1) // self.data_chunk_length] = (policy_outputs[k] * rnn_mask)[selected_idx]
                else:
                    self.storage[k][slot_ids, ep_steps] = policy_outputs[k]

        with timing.add_time('inference/insert/closure_and_opening'):
            if len(closure_slot_ids) > 0:
                self._slot_closure(old_closure_slot_ids, closure_slot_ids)

            if len(opening_slot_ids) > 0:
                for k in self.storage_keys:
                    if 'rnn_states' in k:
                        self.storage[k][opening_new_slot_ids, 0] = (policy_outputs[k] * rnn_mask)[ep_steps == self.episode_length - 1]

        self.total_timesteps += self.obs.shape[2]


class SharedWorkerBuffer(WorkerBuffer, SharedPolicyMixin):
    pass


# class SequentialPolicyMixin(PolicyMixin):
#     def get_actions(self, client):
#         slot_id = self._slot_hash[client]
#         ep_step = self._ep_step[slot_id]

#         pass

#     def get_policy_inputs(self, server_id, split_id):
#         slot_id = self._locate(server_id, split_id)
#         ep_step = self._ep_step[server_id, split_id]
#         agent_id = self._agent_ids[server_id, split_id]

#         return (self.share_obs[slot_id, ep_step, :, agent_id], self.obs[slot_id, ep_step, :, agent_id],
#                 self.rnn_states[slot_id, ep_step, :, agent_id], self.rnn_states_critic[slot_id, ep_step, :, agent_id],
#                 self.masks[slot_id, ep_step, :, agent_id], self.available_actions[slot_id, ep_step, :, agent_id])

#     def insert_before_inference(self,
#                                 server_id,
#                                 actor_id,
#                                 split_id,
#                                 share_obs,
#                                 obs,
#                                 rewards,
#                                 dones,
#                                 available_actions=None):
#         assert share_obs.shape == (self.envs_per_split, *self.share_obs.shape[4:]), (share_obs.shape,
#                                                                                     (self.envs_per_split,
#                                                                                      *self.share_obs.shape[4:]))
#         assert obs.shape == (self.envs_per_split, *self.obs.shape[4:]), (obs.shape, (self.envs_per_split,
#                                                                                     *self.obs.shape[4:]))

#         slot_id = self._locate(server_id, split_id)
#         ep_step = self._ep_step[server_id, split_id]
#         agent_id = self._agent_ids[server_id, split_id]
#         env_slice = slice(actor_id * self.envs_per_split, (actor_id + 1) * self.envs_per_split)

#         # env step returns
#         self.share_obs[slot_id, ep_step, env_slice, agent_id] = share_obs
#         self.obs[slot_id, ep_step, env_slice, agent_id] = obs

#         assert rewards.shape == (self.envs_per_split, self.num_agents, 1), (rewards.shape, (self.envs_per_split,
#                                                                                            self.num_agents, 1))
#         # accumulate reward first, and then record final accumulated reward when env terminates,
#         # because if current step is 'done', reported reward is from previous transition, which
#         # belongs to the previous episode (current step is the opening of the next episode)
#         self._reward_since_last_action[server_id, split_id, env_slice] += rewards
#         if ep_step >= 1:
#             is_done_before = self._env_done_trigger[server_id, split_id, env_slice] > 0
#             not_done_yet = (1 - is_done_before).astype(np.bool)

#             accumulated_reward = self._reward_since_last_action[server_id,
#                                           split_id, env_slice][not_done_yet, agent_id]
#             self.rewards[slot_id, ep_step - 1, env_slice][not_done_yet, agent_id] = accumulated_reward
#             self._reward_since_last_action[server_id, split_id, env_slice][not_done_yet, agent_id] = 0

#             saved_reward = self._reward_when_env_done[server_id, split_id, env_slice][is_done_before, agent_id]
#             self.rewards[slot_id, ep_step - 1, env_slice][is_done_before, agent_id] = saved_reward

#         # record final accumulated reward when env terminates
#         self._reward_when_env_done[server_id, split_id, env_slice][
#               dones.squeeze(-1)] = self._reward_since_last_action[server_id, split_id, env_slice][dones.squeeze(-1)]
#         self._reward_since_last_action[server_id, split_id, env_slice][dones.squeeze(-1)] = 0

#         if available_actions is not None:
#             assert available_actions.shape == (self.envs_per_split,
#                                                *self.available_actions.shape[4:]), (available_actions.shape,
#                                                                                     (self.envs_per_split,
#                                                                                      *self.available_actions.shape[4:]))
#             self.available_actions[slot_id, ep_step, env_slice, agent_id] = available_actions

#         assert dones.shape == (self.envs_per_split, 1), (dones.shape, (self.envs_per_split, 1))
#         assert dones.dtype == np.bool, dones.dtype
#         # once env is done, fill the next #agents timestep with 0 to mask bootstrap values
#         # env_done_trigger records remaining timesteps to be filled with 0
#         trigger = self._env_done_trigger[server_id, split_id, env_slice]
#         trigger[dones.squeeze(-1)] = self.num_agents
#         # NOTE: mask is initialized as all 1, hence we only care about filling 0
#         self.masks[slot_id, ep_step, env_slice, agent_id][trigger > 0] = 0
#         self._env_done_trigger[server_id, split_id, env_slice] = np.maximum(trigger - 1, 0)
#         assert np.all(self._env_done_trigger >= 0) and np.all(self._env_done_trigger <= self.num_agents)

#         # active_mask is always 1 because env automatically resets when any agent induces termination

#         if agent_id == self.num_agents - 1:
#             self.total_timesteps += self.envs_per_split

#     def insert_after_inference(self, server_id, split_id, value_preds, actions, action_log_probs, rnn_states,
#                                rnn_states_critic):
#         slot_id = self._locate(server_id, split_id)
#         ep_step = self._ep_step[server_id, split_id]
#         agent_id = self._agent_ids[server_id, split_id]

#         self.value_preds[slot_id, ep_step, :, agent_id] = value_preds

#         if ep_step == 0 and agent_id == self.num_agents - 1 and self._prev_q_idx[server_id, split_id] >= 0:
#             old_slot_id = self._locate_prev(server_id, split_id)
#             # fill bootstrap data in previous slot
#             self._slot_closure(old_slot_id, slot_id)

#         # model inference returns
#         self.actions[slot_id, ep_step, :, agent_id] = actions
#         self.action_log_probs[slot_id, ep_step, :, agent_id] = action_log_probs

#         rnn_mask = np.expand_dims(self.masks[slot_id, ep_step, :, agent_id], -1)
#         self.rnn_states[slot_id, ep_step + 1, :, agent_id] = rnn_states * rnn_mask
#         self.rnn_states_critic[slot_id, ep_step + 1, :, agent_id] = rnn_states_critic * rnn_mask

#         if agent_id == self.num_agents - 1:
#             self._agent_ids[server_id, split_id] = 0
#             self._ep_step[server_id, split_id] += 1
#             # section of this actor in current slot is full except for bootstrap step
#             if ep_step == self.episode_length - 1:
#                 self._ep_step[server_id, split_id] = 0

#                 new_slot_id = self._move_next(server_id, split_id)

#                 self._slot_opening(slot_id, new_slot_id)
#         else:
#             self._agent_ids[server_id, split_id] += 1

# class SequentialReplayBuffer(ReplayBuffer, SequentialPolicyMixin):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # following arrays may not be shared between processes because
#         # 1) only servers can access them (trainers should not access)
#         # 2) each server will access individual parts according to server_id, there's no communication among them
#         self._agent_ids = np.zeros((self.num_servers, self.num_splits), dtype=np.uint8)
#         self._reward_since_last_action = np.zeros(
#             (self.num_servers, self.num_splits, self.batch_size, self.num_agents, 1), dtype=np.float32)
#         self._reward_when_env_done = np.zeros_like(self._reward_since_last_action)
#         self._env_done_trigger = np.zeros((self.num_servers, self.num_splits, self.batch_size), dtype=np.int16)
