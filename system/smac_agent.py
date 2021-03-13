import time
import wandb
import numpy as np
# from functools import reduce
import torch
import itertools
from torch.distributed import rpc
from system.base_agent import Agent


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACAgent(Agent):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super().__init__(config)
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
                                'avg_reward': train_infos["average_step_rewards"],
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

                # train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(
                #     lambda x, y: x * y, list(self.buffer.active_masks.shape))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @rpc.functions.async_execution
    def select_action(self, actor_id, split_id, model_inputs, init=False):
        if init:
            # reset env
            obs, share_obs, available_actions = model_inputs
            rewards = dones = infos = bad_masks = None
        else:
            obs, share_obs, rewards, dones, infos, available_actions, bad_masks = model_inputs
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.insert_before_inference(actor_id, split_id, share_obs, obs, rewards, dones, bad_masks,
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
                    self.trainer.prep_rollout()
                    rollout_outputs = self.trainer.rollout_policy.get_actions(*map(np.concatenate, policy_inputs))

                # [self.envs, agents, dim]
                def to_numpy(x):
                    return np.array(np.split(_t2n(x), self.env_per_split * self.num_actors))

                values, actions, action_log_probs, rnn_states, rnn_states_critic = tuple(map(to_numpy, rollout_outputs))
                self.buffer.insert_after_inference(split_id, values, actions, action_log_probs, rnn_states,
                                                   rnn_states_critic)
                self.queued_cnt[split_id] = 0
                cur_future_outputs = self.future_outputs[split_id]
                self.future_outputs[split_id] = torch.futures.Future()
                cur_future_outputs.set_result(actions)

        return action_fut

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        pass
        # eval_battles_won = 0
        # eval_episode = 0

        # eval_episode_rewards = []
        # one_episode_rewards = []

        # eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        # eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
        #                            dtype=np.float32)
        # eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # while True:
        #     self.trainer.prep_rollout()
        #     eval_actions, eval_rnn_states = \
        #         self.trainer.policy.act(np.concatenate(eval_obs),
        #                                 np.concatenate(eval_rnn_states),
        #                                 np.concatenate(eval_masks),
        #                                 np.concatenate(eval_available_actions),
        #                                 deterministic=True)
        #     eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
        #     eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

        #     # Obser reward and next obs
        #     (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos,
        #     eval_available_actions) = self.eval_envs.step(eval_actions)
        #     one_episode_rewards.append(eval_rewards)

        #     eval_dones_env = np.all(eval_dones, axis=1)

        #     eval_rnn_states[eval_dones_env] = 0

        #     eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #     eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
        #                                                   dtype=np.float32)

        #     for eval_i in range(self.n_eval_rollout_threads):
        #         if eval_dones_env[eval_i]:
        #             eval_episode += 1
        #             eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
        #             one_episode_rewards = []
        #             if eval_infos[eval_i][0]['won']:
        #                 eval_battles_won += 1

        #     if eval_episode >= self.all_args.eval_episodes:
        #         eval_episode_rewards = np.array(eval_episode_rewards)
        #         eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}
        #         self.log_env(eval_env_infos, total_num_steps)
        #         eval_win_rate = eval_battles_won / eval_episode
        #         print("eval win rate is {}.".format(eval_win_rate))
        #         if self.use_wandb:
        #             wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
        #         else:
        #             self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
        #         break
