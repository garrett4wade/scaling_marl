import torch
import time
import numpy as np
import multiprocessing as mp
from queue import Empty
from algorithms.storage_registries import get_ppo_storage_specs, to_numpy_type


class ReplayBuffer:
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        # NOTE: value target computation is deferred to centralized trainer in consistent with off-policy correction
        # e.g. V-trace and Retrace

        # system configuration
        self.num_actors = num_actors = args.num_actors
        self.num_splits = num_splits = args.num_splits
        assert args.env_per_actor % args.num_splits == 0
        self.env_per_split = env_per_split = args.env_per_actor // args.num_splits
        self.qsize = qsize = args.qsize

        self.num_slots = num_slots = qsize * num_splits * num_actors

        # storage shape configuration
        self.num_agents = num_agents
        self.episode_length = ep_l = args.episode_length

        # TODO: support n-step bootstrap
        # self.bootstrap_step = bootstrap_step = args.bootstrap_step

        # initialize storage
        def shape_prefix(bootstrap):
            t = ep_l + 1 if bootstrap else ep_l
            return (num_slots, t, env_per_split, num_agents)

        # TODO: replace get_ppo_storage_specs with get_${algorithm}_storage_specs
        self.storage_specs, self.policy_input_keys, self.policy_output_keys = get_ppo_storage_specs(
            args, obs_space, share_obs_space, act_space)
        self.storage_keys = [storage_spec.name for storage_spec in self.storage_specs]
        self.storage_registries = {}

        for storage_spec in self.storage_specs:
            name, shape, dtype, bootstrap, init_value = storage_spec
            assert init_value == 0 or init_value == 1
            init_method = torch.zeros if init_value == 0 else torch.ones
            setattr(self, '_' + name, init_method((*shape_prefix(bootstrap), *shape), dtype=dtype).share_memory_())
            setattr(self, name, getattr(self, '_' + name).numpy())
            self.storage_registries[name] = ((*shape_prefix(bootstrap), *shape), to_numpy_type(dtype))

        # self._storage is torch.Tensor handle while storage is numpy.array handle
        # the 2 handles point to the same block of memory
        self._storage = {k: getattr(self, '_' + k) for k in self.storage_keys}
        self.storage = {k: getattr(self, k) for k in self.storage_keys}

        # hash table mapping client identity to slot id
        self._mp_mgr = mp.Manager()
        self._client_hash = self._mp_mgr.dict()
        self._prev_client_hash = self._mp_mgr.dict()

        # episode step record
        self._ep_step = torch.zeros((num_slots, ), dtype=torch.int32).share_memory_().numpy()

        # buffer indicators
        self._is_readable = torch.zeros((num_slots, ), dtype=torch.uint8).share_memory_().numpy()
        self._is_busy = torch.zeros((num_slots, ), dtype=torch.uint8).share_memory_().numpy()
        self._is_writable = torch.ones((num_slots, ), dtype=torch.uint8).share_memory_().numpy()
        assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)

        self._time_stamp = torch.zeros((num_slots, ), dtype=torch.float32).share_memory_().numpy()

        self._read_ready = mp.Condition()

        self.total_timesteps = torch.zeros((), dtype=torch.int64).share_memory_()

        # to read/write env-specific summary info, e.g. winning rate, scores
        self.summary_lock = mp.Lock()

    def get_utilization(self):
        with self._read_ready:
            available_slots = np.sum(self._is_readable)
        return available_slots / self.num_slots

    def _allocate(self, client):
        with self._read_ready:
            writable_slots = np.nonzero(self._is_writable)[0]
            if len(writable_slots) > 0:
                slot_id = writable_slots[0]
                # writable -> busy
                self._is_writable[slot_id] = 0
            else:
                readable_slots = np.nonzero(self._is_readable)[0]
                assert len(readable_slots) > 0, 'please increase qsize!'
                slot_id = readable_slots[0]
                # readable -> busy
                self._is_readable[slot_id] = 0
            self._is_busy[slot_id] = 1
            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)

        if client in self._client_hash.keys():
            self._prev_client_hash[client] = self._client_hash[client]

        self._client_hash[client] = slot_id

    def _opening(self, old_slots, new_slots):
        # when a slot (old_slot) is full, we should open up a new slot for data storage
        with self._read_ready:
            assert np.all(self._is_busy[old_slots]) and np.all(self._is_busy[new_slots])

        # copy rnn states of previous slots into new ones, which serve as the policy input at next inference time
        # note that other policy inputs (e.g. obs, share_obs) will be filled into new slots before the next inference
        # (in self.insert_before_inference)
        for k in self.storage_keys:
            if 'rnn_states' in k:
                self.storage[k][new_slots, 0] = self.storage[k][old_slots, -1]

    def _closure(self, old_slots, new_slots):
        # when filling the first timestep of a new slot, we should copy data into the previous slot as a bootstrap
        # specifially, copied data includes all data except for rnn states needs to be bootstrapped (see storage specs)
        # and rewards, because rewards is 1-step behind other aligned data,
        # i.e., env.step() returns observations of current step, and rewards of previous step
        for storage_spec in self.storage_specs:
            name, _, _, bootstrap, _ = storage_spec
            if 'rnn_states' not in name and bootstrap:
                self.storage[name][old_slots, -1] = self.storage[name][new_slots, 0]

        self.rewards[old_slots, -1] = self.rewards[new_slots, 0]

        with self._read_ready:
            assert np.all(self._is_busy[old_slots]) and np.all(self._is_busy[new_slots])
            # update indicator of current slot
            no_availble_before = np.sum(self._is_readable) < 1
            # readable -> busy (being written)
            self._is_readable[old_slots] = 1
            self._is_busy[old_slots] = 0
            self._time_stamp[old_slots] = time.time()
            # if reader is waiting for data, notify it
            if no_availble_before and np.sum(self._is_readable) >= 1:
                self._read_ready.notify(1)

    def get(self, block=True, timeout=None):
        with self._read_ready:
            if np.sum(self._is_readable) == 0 and not block:
                raise Empty
            self._read_ready.wait_for(lambda: np.sum(self._is_readable) >= 1, timeout=timeout)

            available_slots = np.nonzero(self._is_readable)[0]
            slot = available_slots[np.argsort(self._time_stamp[available_slots])[0]]

            # readable -> busy (being-read)
            self._is_readable[slot] = 0
            self._is_busy[slot] = 1

            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)
        return slot
    
    def get_many(self, timeout=None):
        with self._read_ready:
            self._read_ready.wait_for(lambda: np.sum(self._is_readable) >= 1, timeout=timeout)

            available_slots = np.nonzero(self._is_readable)[0]

            # readable -> busy (being-read)
            self._is_readable[available_slots] = 0
            self._is_busy[available_slots] = 1

            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)
        return available_slots

    def after_sending(self, slot_id):
        with self._read_ready:
            # reset indicator, busy (being-read) -> writable
            self._is_busy[slot_id] = 0
            self._is_writable[slot_id] = 1
            assert np.all(self._is_readable + self._is_busy + self._is_writable == 1)


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
    def get_actions(self, client):
        slot_id = self._client_hash[client]
        ep_step = self._ep_step[slot_id]

        if ep_step == 0:
            prev_slot_id = self._prev_client_hash[client]
            return self.actions[prev_slot_id, -1]

        return self.actions[slot_id, ep_step - 1]

    def get_policy_inputs(self, clients):
        slots = [self._client_hash[client] for client in clients]
        ep_steps = self._ep_step[slots]

        policy_inputs = {}
        for k in self.policy_input_keys:
            if hasattr(self, k):
                policy_inputs[k] = self.storage[k][slots, ep_steps]
        return policy_inputs

    def insert_before_inference(self, client, obs, share_obs, rewards, dones, infos=None, available_actions=None):
        if client not in self._client_hash.keys():
            self._allocate(client)
        slot_id = self._client_hash[client]
        ep_step = self._ep_step[slot_id]

        # env step returns
        self.share_obs[slot_id, ep_step] = share_obs
        self.obs[slot_id, ep_step] = obs
        if ep_step >= 1:
            self.rewards[slot_id, ep_step - 1] = rewards
        self.masks[slot_id, ep_step] = 1 - np.all(dones, axis=1, keepdims=True)

        if hasattr(self, 'available_actions') and available_actions is not None:
            self.available_actions[slot_id, ep_step] = available_actions

        if hasattr(self, 'active_masks'):
            dones_cp = dones.copy()
            dones_cp[np.all(dones, axis=1).squeeze(-1)] = 0
            self.active_masks[slot_id, ep_step] = 1 - dones_cp

        if hasattr(self, 'fct_masks') and infos is not None:
            force_terminations = np.array([[[agent_info['force_termination']] for agent_info in info]
                                           for info in infos])
            self.fct_masks[slot_id, ep_step] = 1 - force_terminations

        self.total_timesteps += self.env_per_split

    def insert_after_inference(self, clients, **policy_outputs):
        slots = [self._client_hash[client] for client in clients]
        ep_steps = self._ep_step[slots]

        # model inference returns
        rnn_mask = np.expand_dims(self.masks[slots, ep_steps], -1)
        for k in policy_outputs.keys():
            if 'rnn_states' in k:
                self.storage[k][slots, ep_steps + 1] = policy_outputs[k] * rnn_mask
            else:
                self.storage[k][slots, ep_steps] = policy_outputs[k]

        # closure on the previous slot
        closure_new_slots = []
        closure_old_slots = []
        for slot, ep_step, client in zip(slots, ep_steps, clients):
            if ep_step == 0 and client in self._prev_client_hash.keys():
                closure_old_slots.append(self._prev_client_hash[client])
                closure_new_slots.append(slot)
        if len(closure_new_slots) > 0:
            self._closure(closure_old_slots, closure_new_slots)

        # advance 1 timestep
        self._ep_step[slots] += 1

        # if a slot is full except for the bootstrap step, allocate a new slot for the corresponding client
        opening_clients = np.array(clients)[ep_steps == self.episode_length - 1]
        if len(opening_clients) > 0:
            opening_old_slots = [self._client_hash[client] for client in opening_clients]
            self._ep_step[opening_old_slots] = 0

            for client in opening_clients:
                self._allocate(client)

            opening_new_slots = [self._client_hash[client] for client in opening_clients]

            self._opening(opening_old_slots, opening_new_slots)


class SharedReplayBuffer(ReplayBuffer, SharedPolicyMixin):
    pass


# class SequentialPolicyMixin(PolicyMixin):
#     def get_actions(self, client):
#         slot_id = self._client_hash[client]
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
#         assert share_obs.shape == (self.env_per_split, *self.share_obs.shape[4:]), (share_obs.shape,
#                                                                                     (self.env_per_split,
#                                                                                      *self.share_obs.shape[4:]))
#         assert obs.shape == (self.env_per_split, *self.obs.shape[4:]), (obs.shape, (self.env_per_split,
#                                                                                     *self.obs.shape[4:]))

#         slot_id = self._locate(server_id, split_id)
#         ep_step = self._ep_step[server_id, split_id]
#         agent_id = self._agent_ids[server_id, split_id]
#         env_slice = slice(actor_id * self.env_per_split, (actor_id + 1) * self.env_per_split)

#         # env step returns
#         self.share_obs[slot_id, ep_step, env_slice, agent_id] = share_obs
#         self.obs[slot_id, ep_step, env_slice, agent_id] = obs

#         assert rewards.shape == (self.env_per_split, self.num_agents, 1), (rewards.shape, (self.env_per_split,
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
#             assert available_actions.shape == (self.env_per_split,
#                                                *self.available_actions.shape[4:]), (available_actions.shape,
#                                                                                     (self.env_per_split,
#                                                                                      *self.available_actions.shape[4:]))
#             self.available_actions[slot_id, ep_step, env_slice, agent_id] = available_actions

#         assert dones.shape == (self.env_per_split, 1), (dones.shape, (self.env_per_split, 1))
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
#             self.total_timesteps += self.env_per_split

#     def insert_after_inference(self, server_id, split_id, value_preds, actions, action_log_probs, rnn_states,
#                                rnn_states_critic):
#         slot_id = self._locate(server_id, split_id)
#         ep_step = self._ep_step[server_id, split_id]
#         agent_id = self._agent_ids[server_id, split_id]

#         self.value_preds[slot_id, ep_step, :, agent_id] = value_preds

#         if ep_step == 0 and agent_id == self.num_agents - 1 and self._prev_q_idx[server_id, split_id] >= 0:
#             old_slot_id = self._locate_prev(server_id, split_id)
#             # fill bootstrap data in previous slot
#             self._closure(old_slot_id, slot_id)

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

#                 self._opening(slot_id, new_slot_id)
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
