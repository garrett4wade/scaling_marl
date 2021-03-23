import time
import numpy as np
import torch
from torch.distributed import rpc
from system.inference_server import InferenceServer


def _t2n(x):
    return x.detach().cpu().numpy()


class HanabiServer(InferenceServer):
    def __init__(self, rpc_rank, gpu_rank, weights_queue, buffer, config):
        super().__init__(rpc_rank, gpu_rank, weights_queue, buffer, config)

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
        self.load_weights(block=False)
        if init:
            # reset env
            obs, share_obs, available_actions = model_inputs
            rewards = np.zeros((self.env_per_split, self.num_agents, 1), dtype=np.float32)
            dones = np.zeros((self.env_per_split, 1), dtype=np.bool)
            infos = [{} for _ in range(self.env_per_split)]
        else:
            obs, share_obs, rewards, dones, infos, available_actions = model_inputs
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.insert_before_inference(self.server_id, actor_id, split_id, share_obs, obs, rewards, dones,
                                            available_actions)

        elapsed_episode = total_scores = 0
        for done, info in zip(dones, infos):
            if done and 'score' in info.keys():
                elapsed_episode += 1
                total_scores += info['score']
        with self.buffer.summary_lock:  # multiprocessing RLock
            self.buffer.elapsed_episode[self.server_id, actor_id, split_id] += elapsed_episode
            self.buffer.total_scores[self.server_id, actor_id, split_id] += total_scores

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
                    rollout_outputs = self.rollout_policy.get_actions(*policy_inputs)

                values, actions, action_log_probs, rnn_states, rnn_states_critic = map(_t2n, rollout_outputs)

                self.buffer.insert_after_inference(self.server_id, split_id, values, actions, action_log_probs,
                                                   rnn_states, rnn_states_critic)
                self.queued_cnt[split_id] = 0
                cur_future_outputs = self.future_outputs[split_id]
                self.future_outputs[split_id] = torch.futures.Future()
                cur_future_outputs.set_result(actions)

        return action_fut
