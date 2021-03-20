import time
import wandb
import numpy as np
import torch
from torch.distributed import rpc
from system.base_agent import Agent
from utils.buffer import SequentialReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class HanabiAgent(Agent):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super().__init__(config)
        self.buffer = SequentialReplayBuffer(self.all_args, self.num_agents, self.example_env.observation_space[0],
                                             self.share_observation_space, self.example_env.action_space[0],
                                             self.trainer.value_normalizer)
        self.scores = []

    def run(self):
        self.setup_actors()

        global_start = time.time()
        local_start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // (self.num_actors * self.env_per_split)

        last_total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            train_infos = self.train()

            # post process
            total_num_steps = self.buffer.total_timesteps.item()
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                recent_fps = int((total_num_steps - last_total_num_steps) / (end - local_start))
                global_avg_fps = int(total_num_steps / (end - global_start))
                print("\nGame Version {}, Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, "
                      "recent FPS {}, global average FPS {}.\n".format(self.all_args.hanabi_name, self.algorithm_name,
                                                                       self.experiment_name, episode, episodes,
                                                                       total_num_steps, self.num_env_steps, recent_fps,
                                                                       global_avg_fps))

                if self.env_name == "Hanabi":
                    average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
                    print("average score is {}.".format(average_score))

                    if self.use_wandb:
                        wandb.log(
                            {
                                'average_score': average_score,
                                'total_env_steps': total_num_steps,
                                'fps': recent_fps,
                            },
                            step=total_num_steps)
                    else:
                        self.writter.add_scalars('average_score', {'average_score': average_score}, total_num_steps)

                    last_total_num_steps = total_num_steps
                    local_start = time.time()

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

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

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_scores = []
        episode_cnt = 0

        obs, _, available_actions = self.eval_envs.reset()

        rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states.shape[-2:]),
                              dtype=np.float32)
        masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        env_done_trigger = np.zeros(self.n_eval_rollout_threads, dtype=np.int16)

        while episode_cnt < self.all_args.eval_episodes:
            for agent_id in range(self.num_agents):
                self.trainer.prep_rollout()
                policy_outputs = self.trainer.policy.act(obs,
                                                         rnn_states[:, agent_id],
                                                         masks[:, agent_id],
                                                         available_actions,
                                                         deterministic=True)
                action, new_rnn_state = map(_t2n, policy_outputs)

                # Obser reward and next obs
                obs, _, _, dones, infos, available_actions = self.eval_envs.step(action)

                env_done_trigger[dones.squeeze(-1)] = self.num_agents
                masks[:, agent_id][env_done_trigger > 0] = 0
                env_done_trigger = np.maximum(env_done_trigger - 1, 0)

                rnn_states[:, agent_id] = new_rnn_state * np.expand_dims(masks[:, agent_id], -1)

                for done, info in zip(dones, infos):
                    if done and 'score' in info.keys():
                        episode_cnt += 1
                        eval_scores.append(info['score'])

        eval_average_score = np.mean(eval_scores)
        print("eval average score is {}.".format(eval_average_score))
        if self.use_wandb:
            wandb.log({'eval_average_score': eval_average_score}, step=total_num_steps)
        else:
            self.writter.add_scalars('eval_average_score', {'eval_average_score': eval_average_score}, total_num_steps)
