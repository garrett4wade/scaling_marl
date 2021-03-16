import time
import wandb
import numpy as np
import torch
import itertools
from torch.distributed import rpc
from system.base_agent import Agent
from utils.buffer import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACAgent(Agent):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super().__init__(config)
        self.buffer = SharedReplayBuffer(self.all_args, self.num_agents, self.example_env.observation_space[0],
                                         self.share_observation_space, self.example_env.action_space[0],
                                         self.trainer.value_normalizer)
        self.all_agent0_infos = [[{} for _ in range(self.num_split)] for _ in range(self.num_actors)]

    def run(self):
        self.setup_actors()

        global_start = time.time()
        local_start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // (self.num_actors * self.env_per_split)

        last_battles_game = np.zeros(self.num_actors * self.num_split, dtype=np.float32)
        last_battles_won = np.zeros(self.num_actors * self.num_split, dtype=np.float32)
        last_total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            train_infos = self.train()

            # post process
            total_num_steps = self.buffer.total_timesteps
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                recent_fps = int((total_num_steps - last_total_num_steps) / (end - local_start))
                global_avg_fps = int(total_num_steps / (end - global_start))
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, "
                      "recent FPS {}, global average FPS {}.\n".format(self.all_args.map_name, self.algorithm_name,
                                                                       self.experiment_name, episode, episodes,
                                                                       total_num_steps, self.num_env_steps, recent_fps,
                                                                       global_avg_fps))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []

                    for i, info in enumerate(itertools.chain(*self.all_agent0_infos)):
                        if 'battles_won' in info.keys():
                            battles_won.append(info['battles_won'])
                            incre_battles_won.append(info['battles_won'] - last_battles_won[i])
                        if 'battles_game' in info.keys():
                            battles_game.append(info['battles_game'])
                            incre_battles_game.append(info['battles_game'] - last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(
                        incre_battles_game) > 0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log(
                            {
                                "incre_win_rate": incre_win_rate,
                                'total_env_steps': total_num_steps,
                                'fps': recent_fps,
                            },
                            step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                    last_battles_game = battles_game
                    last_battles_won = battles_won
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
            dones = np.zeros_like(rewards).astype(np.bool)
            infos = None
        else:
            obs, share_obs, rewards, dones, infos, available_actions = model_inputs
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.insert_before_inference(actor_id, split_id, share_obs, obs, rewards, dones, available_actions)
        if infos is not None:
            merged_info = {}
            for all_agent_info in infos:
                for k, v in all_agent_info[0].items():
                    if not isinstance(v, bool):
                        if k not in merged_info.keys():
                            merged_info[k] = v
                        else:
                            merged_info[k] += v
            self.all_agent0_infos[actor_id][split_id] = merged_info

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
                    rollout_outputs = self.trainer.rollout_policy.get_actions(*map(
                        lambda x: x.reshape(self.rollout_batch_size * self.num_agents, *x.shape[2:]), policy_inputs))

                values, actions, action_log_probs, rnn_states, rnn_states_critic = map(
                    lambda x: _t2n(x).reshape(self.rollout_batch_size, self.num_agents, *x.shape[1:]), rollout_outputs)

                self.buffer.insert_after_inference(split_id, values, actions, action_log_probs, rnn_states,
                                                   rnn_states_critic)
                self.queued_cnt[split_id] = 0
                cur_future_outputs = self.future_outputs[split_id]
                self.future_outputs[split_id] = torch.futures.Future()
                cur_future_outputs.set_result(actions)

        return action_fut

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = np.zeros((self.n_eval_rollout_threads, 1), dtype=np.float32)

        eval_obs, _, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            policy_inputs = (eval_obs, eval_rnn_states, eval_masks, eval_available_actions)

            policy_outputs = self.trainer.policy.act(*map(
                lambda x: x.reshape(self.n_eval_rollout_threads * self.num_agents, *x.shape[2:]), policy_inputs),
                                                     deterministic=True)
            eval_actions, eval_rnn_states = map(
                lambda x: _t2n(x).reshape(self.n_eval_rollout_threads, self.num_agents, *x.shape[1:]), policy_outputs)

            # Observe reward and next obs
            (eval_obs, _, eval_rewards, eval_dones, eval_infos,
             eval_available_actions) = self.eval_envs.step(eval_actions)

            # smac is shared-env, just record reward of agent 0
            one_episode_rewards += eval_rewards[:, 0]

            eval_dones_env = np.all(eval_dones, 1).squeeze(-1)
            eval_masks = np.broadcast_to(1 - np.all(eval_dones, axis=1, keepdims=True),
                                         (self.all_args.n_eval_rollout_threads, self.num_agents, 1))

            eval_rnn_states *= np.expand_dims(eval_masks, -1)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i].item())
                    one_episode_rewards[eval_i] = 0
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_env_infos = {'eval_average_episode_rewards': np.array(eval_episode_rewards)}
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
