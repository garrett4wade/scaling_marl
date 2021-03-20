import time
import numpy as np
import torch
from torch.distributed import rpc
from system.base_agent import Agent


def _t2n(x):
    return x.detach().cpu().numpy()


class HanabiAgent(Agent):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, rank, config, buffer):
        super().__init__(rank, config)
        self.buffer = buffer
        # TODO: consider how to deal with summary info
        self.scores = []

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
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
        self.buffer.insert_before_inference(actor_id, split_id, share_obs, obs, rewards, dones, available_actions)

        for done, info in zip(dones, infos):
            if done and 'score' in info.keys():
                self.scores.append(info['score'])

        def _unpack(action_batch_futures):
            action_batch = action_batch_futures.wait()
            batch_slice = slice(actor_id * self.env_per_split, (actor_id + 1) * self.env_per_split)
            return time.time(), action_batch[batch_slice]

        action_fut = self.future_outputs[split_id].then(_unpack)

        with self.locks[split_id]:
            self.queued_cnt[split_id] += 1
            if self.queued_cnt[split_id] >= self.num_actors:
                policy_inputs = self.buffer.get_policy_inputs(split_id)
                with torch.no_grad():
                    rollout_outputs = self.trainer.rollout_policy.get_actions(*policy_inputs)

                values, actions, action_log_probs, rnn_states, rnn_states_critic = map(_t2n, rollout_outputs)

                self.buffer.insert_after_inference(split_id, values, actions, action_log_probs, rnn_states,
                                                   rnn_states_critic)
                self.queued_cnt[split_id] = 0
                cur_future_outputs = self.future_outputs[split_id]
                self.future_outputs[split_id] = torch.futures.Future()
                cur_future_outputs.set_result(actions)

        return action_fut
