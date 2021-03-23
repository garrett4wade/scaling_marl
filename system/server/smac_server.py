import time
import numpy as np
import torch
from torch.distributed import rpc
from system.inference_server import InferenceServer


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACServer(InferenceServer):
    def __init__(self, rpc_rank, gpu_rank, weights_queue, buffer, config):
        super().__init__(rpc_rank, gpu_rank, weights_queue, buffer, config)

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
        self.load_weights(block=False)
        if init:
            # reset env
            obs, share_obs, available_actions = model_inputs
            rewards = np.zeros((self.env_per_split, self.num_agents, 1), dtype=np.float32)
            dones = np.zeros_like(rewards).astype(np.bool)
            infos = None
        else:
            obs, share_obs, rewards, dones, infos, available_actions = model_inputs
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.insert_before_inference(self.server_id, actor_id, split_id, share_obs, obs, rewards, dones,
                                            available_actions)
        if infos is not None:
            merged_info = {}
            for all_agent_info in infos:
                for k, v in all_agent_info[0].items():
                    if not isinstance(v, bool):
                        if k not in merged_info.keys():
                            merged_info[k] = v
                        else:
                            merged_info[k] += v
            with self.buffer.summary_lock:  # multiprocessing RLock
                self.buffer.battles_won[self.server_id, actor_id, split_id] = merged_info['battles_won']
                self.buffer.battles_game[self.server_id, actor_id, split_id] = merged_info['battles_game']

        def _unpack(action_batch_futures):
            action_batch = action_batch_futures.wait()
            batch_slice = slice(actor_id * self.env_per_split, (actor_id + 1) * self.env_per_split)
            return time.time(), action_batch[batch_slice]

        action_fut = self.future_outputs[split_id].then(_unpack)

        with self.locks[split_id]:
            self.queued_cnt[split_id] += 1
            if self.queued_cnt[split_id] >= self.num_actors:
                policy_inputs = self.buffer.get_policy_inputs(self.server_id, split_id)
                with torch.no_grad():
                    rollout_outputs = self.rollout_policy.get_actions(*map(
                        lambda x: x.reshape(self.rollout_batch_size * self.num_agents, *x.shape[2:]), policy_inputs))

                values, actions, action_log_probs, rnn_states, rnn_states_critic = map(
                    lambda x: _t2n(x).reshape(self.rollout_batch_size, self.num_agents, *x.shape[1:]), rollout_outputs)

                self.buffer.insert_after_inference(self.server_id, split_id, values, actions, action_log_probs,
                                                   rnn_states, rnn_states_critic)
                self.queued_cnt[split_id] = 0
                cur_future_outputs = self.future_outputs[split_id]
                self.future_outputs[split_id] = torch.futures.Future()
                cur_future_outputs.set_result(actions)

        return action_fut
