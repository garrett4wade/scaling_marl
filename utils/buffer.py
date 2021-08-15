import torch
import time
import numpy as np
import multiprocessing as mp
from queue import Empty
from algorithms.storage_registries import get_ppo_storage_specs, to_numpy_type, SUMMARY_KEYS
from algorithms.utils.modules import compute_gae, masked_normalization
from utils.utils import log
from algorithms.utils.transforms import flatten, to_chunk
from utils.popart import PopArt


class ReplayBuffer:
    def __init__(self, cfg, policy_id, num_agents, obs_space, share_obs_space, act_space):
        self.cfg = cfg
        self.policy_id = policy_id

        self.num_trainers = 0
        self.available_dsts = []
        for node_idx, local_config in self.cfg.learner_config.items():
            for i, (_, v) in enumerate(local_config.items()):
                if v == self.policy_id:
                    self.num_trainers += 1
                    self.available_dsts.append((int(node_idx), i))
        self.available_dsts = np.array(self.available_dsts, dtype=np.int32)

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        # system configuration
        self.num_actors = cfg.num_actors // cfg.num_tasks_per_node
        self.num_actor_groups = self.num_actors // self.cfg.actor_group_size
        self.num_splits = cfg.num_splits
        self.qsize = cfg.qsize

        self.envs_per_actor = cfg.envs_per_actor
        self.envs_per_split = cfg.envs_per_actor // cfg.num_splits
        self.data_chunk_length = cfg.data_chunk_length

        assert cfg.num_actors % cfg.num_tasks_per_node == 0
        assert cfg.envs_per_actor % self.num_splits == 0
        assert self.num_actors % self.cfg.actor_group_size == 0

        # storage shape configuration
        self.num_agents = num_agents
        self.episode_length = cfg.episode_length

        # TODO: support n-step bootstrap
        # self.bootstrap_step = bootstrap_step = cfg.bootstrap_step

    def _init_storage(self):
        # initialize storage
        # TODO: replace get_ppo_storage_specs with get_${algorithm}_storage_specs
        self.storage_specs, self.policy_input_keys, self.policy_output_keys = get_ppo_storage_specs(
            self.cfg, self.obs_space, self.share_obs_space, self.act_space)
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
        self.summary_keys = SUMMARY_KEYS[self.cfg.env_name]
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

    def _allocate_many(self, num_slots_to_allocate):
        slot_ids = []
        with self._read_ready:
            writable_slots = np.nonzero(self._is_writable)[0]
            if len(writable_slots) > 0:
                slot_ids.extend(writable_slots[:num_slots_to_allocate])
                # writable -> busy
                self._is_writable[slot_ids] = 0
            if len(slot_ids) < num_slots_to_allocate:
                res = num_slots_to_allocate - len(slot_ids)
                readable_slots = np.nonzero(self._is_readable)[0]
                assert len(readable_slots) > res, 'please increase qsize!'
                # replace the oldest readable slot, in a FIFO pattern
                res_slots = readable_slots[np.argsort(self._time_stamp[readable_slots])[:res]]
                # readable -> busy
                self._is_readable[res_slots] = 0
                self._time_stamp[res_slots] = 0
                slot_ids.extend(res_slots)
            self._is_busy[slot_ids] = 1
            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)
        return slot_ids

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
    def __init__(self, cfg, policy_id, num_agents, obs_space, share_obs_space, act_space):
        # NOTE: value target computation is deferred to centralized trainer in consistent with off-policy correction
        # e.g. V-trace and Retrace
        super().__init__(cfg, policy_id, num_agents, obs_space, share_obs_space, act_space)
        self.target_num_slots = 1
        self.num_consumers_to_notify = 1

        self.num_slots = self.qsize * self.num_actor_groups
        self.envs_per_seg = self.envs_per_slot = self.cfg.actor_group_size * self.envs_per_split

        super()._init_storage()

        # indices mapping client identity to slot id
        self._slot_indices = (self.num_slots * torch.ones(
            (self.num_splits * self.num_actor_groups, ), dtype=torch.int32)).share_memory_().numpy()
        self._prev_slot_indices = (self.num_slots * torch.ones(
            (self.num_splits * self.num_actor_groups, ), dtype=torch.int32)).share_memory_().numpy()

        # episode step record
        self._ep_step = torch.zeros((self.num_splits * self.num_actor_groups, ),
                                    dtype=torch.int32).share_memory_().numpy()
        self.num_chunks = self.episode_length // self.cfg.data_chunk_length
        self.data_chunk_length = self.cfg.data_chunk_length

        # Buffer insertion is invoked after copying actions to small shared memory for actors.
        # Since buffer insertion is slow, it is very possible that before the buffer insertion on policy worker #1
        # finishes, policy worker #2 starts buffer insertion of the same client (slot),
        # which causes confusion on ep_step and slot_id. One solution is adding a multiprocessing lock.
        self._insertion_idx_lock = mp.Lock()

        # destination buffer id of each slots
        self._destination = ((-1) * torch.ones((self.num_slots, 2), dtype=torch.int32)).share_memory_().numpy()
        self._cur_dst_idx = np.random.randint(len(self.available_dsts))

        # summary block
        # TODO: when task changes, we also need to reset summary block
        self.summary_block = torch.zeros(
            (self.num_splits, self.envs_per_split * self.num_actors, len(self.summary_keys)),
            dtype=torch.float32).share_memory_().numpy()

    def _allocate(self, identity):
        slot_id = super()._allocate()
        self._destination[slot_id] = self.available_dsts[self._cur_dst_idx]
        self._cur_dst_idx = (self._cur_dst_idx + 1) % len(self.available_dsts)

        self._prev_slot_indices[identity] = self._slot_indices[identity]
        self._slot_indices[identity] = slot_id

    def _allocate_many(self, identities):
        slot_ids = super()._allocate_many(len(identities))
        self._destination[slot_ids] = self.available_dsts[np.arange(
            self._cur_dst_idx, self._cur_dst_idx + len(identities)) % len(self.available_dsts)]
        self._cur_dst_idx = (self._cur_dst_idx + len(identities)) % len(self.available_dsts)

        self._prev_slot_indices[identities] = self._slot_indices[identities]
        self._slot_indices[identities] = slot_ids

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

    def get(self, block=True, timeout=None, reduce_fn=lambda x: x[0]):
        slot_id = super().get(block, timeout, reduce_fn)
        if slot_id is None:
            return None, (None, None)
        else:
            return slot_id, (self._destination[slot_id, 0].item(), self._destination[slot_id, 1].item())


class LearnerBuffer(ReplayBuffer):
    def __init__(self, cfg, policy_id, num_agents, obs_space, share_obs_space, act_space):
        super().__init__(cfg, policy_id, num_agents, obs_space, share_obs_space, act_space)

        # each trainer has its own buffer
        self.target_num_slots = self.num_consumers_to_notify = 1
        self.num_slots = cfg.qsize

        # concatenate several slots from workers into a single batch,
        # which will be then sent to GPU for optimziation
        self.envs_per_seg = self.cfg.actor_group_size * self.envs_per_split
        self.envs_per_slot = cfg.slots_per_update * self.envs_per_seg

        self.slots_per_update = cfg.slots_per_update

        self.sample_reuse = cfg.sample_reuse

        super()._init_storage()

        self._used_times = torch.zeros((self.num_slots, ), dtype=torch.int32).share_memory_().numpy()

        # summary block
        # TODO: move this summary block to task dispatcher
        self.summary_block = torch.zeros((len(cfg.seg_addrs[0]), len(self.summary_keys)),
                                         dtype=torch.float32).share_memory_().numpy()

        self._ptr_lock = mp.Lock()
        self._global_ptr = torch.zeros((2, ), dtype=torch.int32).share_memory_().numpy()
        with self._read_ready:
            self._is_writable[0] = 0
            self._is_busy[0] = 1
            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)

        self.gamma = np.float32(cfg.gamma)
        self.lmbda = np.float32(cfg.gae_lambda)

        # popart
        self._use_popart = cfg.use_popart
        if self._use_popart:
            self.value_normalizer = PopArt((1, ), self.num_trainers)
        else:
            self.value_normalizer = None

        self._use_advantage_normalization = cfg.use_advantage_normalization
        self._use_recurrent_policy = cfg.use_recurrent_policy

        if cfg.use_recurrent_policy:
            self.data_chunk_length = cfg.data_chunk_length
            assert self.episode_length % self.data_chunk_length == 0
            self.num_chunks = self.episode_length // self.data_chunk_length

            self.batch_size = self.envs_per_slot * self.num_chunks * self.num_agents

            log.info('Use recurrent policy. Batch size: {%d envs} * {%d chunks with length %d} * {%d agents} = %d',
                     self.envs_per_slot, self.num_chunks, self.data_chunk_length, self.num_agents, self.batch_size)

        else:
            self.batch_size = self.envs_per_slot * self.episode_length * self.num_agents
            log.info('Use feed forward policy. Batch size: {%d envs} * {%d timesteps} * {%d agents} = %d',
                     self.envs_per_slot, self.episode_length, self.num_agents, self.batch_size)

        self.num_mini_batch = cfg.num_mini_batch
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
            slot_id = super().get(block, timeout, lambda x: x[0])
            # TODO: default reuse pattern is recycle, while it could be set to 'exhausting' or others
            # defer the timestamp such that the slot will be selected again only after
            # all readable slots are selected at least once

            # self._time_stamp[slot_id] = np.max(self._time_stamp) + 1
        return slot_id

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
            # [T/L, B, A, rN, D] -> [T/L * B * A, rN, D] -> [rN, T/L * B * A, D]
            return h.reshape(-1, *h.shape[3:]).swapaxes(0, 1)

        for k in self.storage_keys:
            # we don't need fct_masks/rewards for policy learning, because they are used in v_target computation
            if k != 'fct_masks' and k != 'rewards':
                if 'rnn_states' in k:
                    output_tensors[k] = _cast_h(self.storage[k][slot])
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
    def insert(self, *cfg, **kwcfg):
        ''' insert data returned by inference and env.step. '''
        raise NotImplementedError


class SharedPolicyMixin(PolicyMixin):
    def prepare_rollout(self):
        self._allocate_many(np.arange(self.num_actor_groups * self.num_splits))

    def advance_indices(self, timing, client_ids, pause=False, **insert_data):
        client_ids = np.array(client_ids, dtype=np.int32)

        with timing.add_time('inference/insert/acquire_lock'), timing.time_avg('inference/insert/acquire_lock_once'):
            self._insertion_idx_lock.acquire()

        with timing.add_time('inference/insert/get_indices'), timing.time_avg('inference/insert/get_indices_once'):
            slot_ids = self._slot_indices[client_ids]
            ep_steps = self._ep_step[client_ids]
            prev_slot_ids = self._prev_slot_indices[client_ids]

        with timing.add_time('inference/insert/process_marginal'), timing.time_avg(
                'inference/insert/process_marginal_once'):
            # advance 1 timestep
            self._ep_step[client_ids] += 1

            # fill in the bootstrap step of a previous slot
            closure_choose = np.logical_and(ep_steps == 0, prev_slot_ids != self.num_slots)
            old_closure_slot_ids = prev_slot_ids[closure_choose]

            # if a slot is full except for the bootstrap step, allocate a new slot for the corresponding client
            opening_choose = ep_steps == self.episode_length - 1
            opening_clients = client_ids[opening_choose]

            if len(opening_clients) > 0:
                if not pause:
                    self._allocate_many(opening_clients)
                else:
                    # reset slot_indices and destinations if ready to pause
                    self._destination[slot_ids[opening_choose]] = -1
                    self._prev_slot_indices[opening_clients] = self._slot_indices[opening_clients]
                    self._slot_indices[opening_clients] = self.num_slots

                self._ep_step[opening_clients] = 0
            self._insertion_idx_lock.release()

        with timing.add_time('inference/insert/closure'):
            dones = insert_data['dones']
            masks = 1 - np.all(dones, axis=2, keepdims=True)

            if hasattr(self, 'active_masks'):
                dones_cp = dones.copy()
                # deal with the auto-reset
                dones_cp[np.all(dones, axis=2).squeeze(-1)] = 0
                active_masks = 1 - dones_cp
            else:
                active_masks = None

            if len(old_closure_slot_ids) > 0:
                # when filling the first timestep of a new slot, copy data into the previous slot as bootstrap values
                # specifially, copied data includes all data needs to be bootstrapped
                # and rewards, because rewards is 1-step behind other aligned data,
                # i.e., env.step() returns observations of the current step, but rewards of the previous step
                for storage_spec in self.storage_specs:
                    name, _, _, bootstrap, _ = storage_spec
                    if bootstrap:
                        if name in insert_data.keys():
                            self.storage[name][old_closure_slot_ids, -1] = insert_data[name][closure_choose]
                        else:
                            self.storage[name][old_closure_slot_ids, -1] = locals()[name][closure_choose]

                self.rewards[old_closure_slot_ids, -1] = insert_data['rewards'][closure_choose]

                # NOTE: following 2 lines for debug only
                # with self._read_ready:
                #     assert np.all(self._is_busy[old_slots]) and np.all(self._is_busy[new_slots])
                self._mark_as_readable(old_closure_slot_ids)

                if pause:
                    # reset prev slot indices if ready to pause
                    closure_clients = client_ids[closure_choose]
                    self._prev_slot_indices[closure_clients] = self.num_slots

            valid_choose = np.logical_not(closure_choose) if pause else np.ones(closure_choose.shape, dtype=np.bool)

        return slot_ids[valid_choose], ep_steps[valid_choose], masks, active_masks, valid_choose

    def insert(self,
               timing,
               slot_ids,
               ep_steps,
               valid_choose,
               obs,
               share_obs,
               rewards,
               masks,
               active_masks=None,
               fct_masks=None,
               available_actions=None,
               **policy_outputs_and_input_rnn_states):
        with timing.add_time('inference/insert/copy_data'), timing.time_avg('inference/insert/copy_data_once'):
            # env step returns
            self.share_obs[slot_ids, ep_steps] = share_obs[valid_choose]
            self.obs[slot_ids, ep_steps] = obs[valid_choose]
            self.rewards[slot_ids[ep_steps >= 1], ep_steps[ep_steps >= 1] - 1] = rewards[valid_choose][ep_steps >= 1]
            self.masks[slot_ids, ep_steps] = masks[valid_choose]

            if hasattr(self, 'available_actions') and available_actions is not None:
                self.available_actions[slot_ids, ep_steps] = available_actions[valid_choose]

            if hasattr(self, 'active_masks'):
                self.active_masks[slot_ids, ep_steps] = active_masks[valid_choose]

            if hasattr(self, 'fct_masks') and fct_masks is not None:
                self.fct_masks[slot_ids, ep_steps] = fct_masks[valid_choose]

            # model inference returns
            # we don't need to mask rnn states here because they will be masked when fed into wrapped RNN
            for k in policy_outputs_and_input_rnn_states.keys():
                if 'rnn_states' in k:
                    selected_idx = ep_steps % self.data_chunk_length == 0
                    if np.any(selected_idx):
                        self.storage[k][
                            slot_ids[selected_idx], ep_steps[selected_idx] //
                            self.data_chunk_length] = policy_outputs_and_input_rnn_states[k][valid_choose][selected_idx]
                else:
                    self.storage[k][slot_ids, ep_steps] = policy_outputs_and_input_rnn_states[k][valid_choose]

        self.total_timesteps += len(slot_ids)


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
#     def __init__(self, *cfg, **kwcfg):
#         super().__init__(*cfg, **kwcfg)
#         # following arrays may not be shared between processes because
#         # 1) only servers can access them (trainers should not access)
#         # 2) each server will access individual parts according to server_id, there's no communication among them
#         self._agent_ids = np.zeros((self.num_servers, self.num_splits), dtype=np.uint8)
#         self._reward_since_last_action = np.zeros(
#             (self.num_servers, self.num_splits, self.batch_size, self.num_agents, 1), dtype=np.float32)
#         self._reward_when_env_done = np.zeros_like(self._reward_since_last_action)
#         self._env_done_trigger = np.zeros((self.num_servers, self.num_splits, self.batch_size), dtype=np.int16)
