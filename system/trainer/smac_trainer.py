import time
import wandb
import numpy as np
import torch
import torc.distributed as dist
import itertools
from system.base_trainer import Trainer


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACTrainer(Trainer):
    def __init__(self, rpc_rank, weights_queue, buffer, config):
        super().__init__(rpc_rank, weights_queue, buffer, config)

    def run(self):
        # synchronize weights of rollout policy before inference starts
        self.pack_off_weights()

        global_start = time.time()
        local_start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // (self.num_actors * self.env_per_split)

        last_battles_game = np.zeros(self.num_actors * self.num_split, dtype=np.float32)
        last_battles_won = np.zeros(self.num_actors * self.num_split, dtype=np.float32)
        last_total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.policy.lr_decay(episode, episodes)

            train_infos = self.training_step()
            self.pack_off_weights()

            # post process
            total_num_steps = self.buffer.total_timesteps.item()
            # save model
            if self.ddp_rank == 0 and (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if self.ddp_rank == 0 and episode % self.log_interval == 0:
                end = time.time()
                recent_fps = int((total_num_steps - last_total_num_steps) / (end - local_start) * self.num_trainers)
                global_avg_fps = int(total_num_steps / (end - global_start) * self.num_trainers)
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

                self.log_info(train_infos, total_num_steps)

            dist.barrier()
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        self.policy.eval_mode()
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = np.zeros((self.n_eval_rollout_threads, 1), dtype=np.float32)

        eval_obs, _, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            policy_inputs = (eval_obs, eval_rnn_states, eval_masks, eval_available_actions)

            policy_outputs = self.policy.act(*map(
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
                eval_env_infos = {
                    'eval_average_episode_rewards': np.mean(eval_episode_rewards),
                    "eval_win_rate": eval_battles_won / eval_episode
                }
                self.log_info(eval_env_infos, total_num_steps)
                print("eval win rate is {}.".format(eval_battles_won / eval_episode))
                break
