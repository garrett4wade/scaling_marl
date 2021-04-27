import time
import wandb
import numpy as np
import torch
import torch.distributed as dist
from system.base_trainer import Trainer


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACTrainer(Trainer):
    def __init__(self, rpc_rank, weights_queue, buffer, config):
        super().__init__(rpc_rank, weights_queue, buffer, config)

    def run(self):
        # synchronize weights of rollout policy before inference starts
        self.pack_off_weights()

        transition_per_batch = self.episode_length * self.num_actors * self.env_per_split * self.slots_per_update
        episodes = int(self.num_env_steps) // transition_per_batch // self.num_trainers

        global_tik = local_tik = time.time()
        last_battles_game = last_battles_won = last_total_num_steps = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.policy.lr_decay(episode, episodes)

            buffer_util = self.buffer.get_utilization()

            train_infos = self.training_step()
            self.pack_off_weights()

            total_num_steps = self.buffer.total_timesteps.item()
            consuemd_num_steps = (episode + 1) * transition_per_batch * self.num_trainers

            # save model
            if self.ddp_rank == 0 and (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if self.ddp_rank == 0 and episode % self.log_interval == 0:
                tok = time.time()
                recent_fps = int((total_num_steps - last_total_num_steps) / (tok - local_tik))
                global_avg_fps = int(total_num_steps / (tok - global_tik))
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, consumed num timesteps {}/{}, "
                      "recent FPS {}, global average FPS {}.\n".format(self.all_args.map_name, self.algorithm_name,
                                                                       self.experiment_name, episode, episodes,
                                                                       consuemd_num_steps, self.num_env_steps,
                                                                       recent_fps, global_avg_fps))

                assert self.env_name == "StarCraft2"
                with self.buffer.summary_lock:  # multiprocessing RLock
                    battles_won = sum(self.buffer.battles_won.values())
                    battles_game = sum(self.buffer.battles_game.values())
                recent_battles_won = battles_won - last_battles_won
                recent_battles_game = battles_game - last_battles_game

                recent_win_rate = recent_battles_won / recent_battles_game if recent_battles_game > 0 else 0.0
                print("recent winning rate is {}.".format(recent_win_rate))

                print('buffer utilization before training step: {}/{}'.format(
                    round(buffer_util * self.buffer.num_slots), self.buffer.num_slots))

                # as defined in https://cdn.openai.com/dota-2.pdf
                _t = self.log_interval if episode > 0 else 1
                recent_consumed_num_steps = _t * transition_per_batch * self.num_trainers
                recent_num_steps = total_num_steps - last_total_num_steps
                recent_sample_reuse = recent_consumed_num_steps / recent_num_steps

                global_sample_reuse = consuemd_num_steps / total_num_steps

                print('recent sample reuse: {:.2f}, global average sample reuse: {:.2f}'.format(
                    recent_sample_reuse, global_sample_reuse))

                if self.use_wandb:
                    wandb.log(
                        {
                            "recent_win_rate": recent_win_rate,
                            'total_env_steps': total_num_steps,
                            'fps': recent_fps,
                            'buffer_util': buffer_util,
                            'iteraion': episode + 1,
                            'sample_reuse': recent_sample_reuse,
                        },
                        step=consuemd_num_steps)
                else:
                    self.writter.add_scalars("recent_win_rate", {"recent_win_rate": recent_win_rate},
                                             consuemd_num_steps)

                last_battles_game = battles_game
                last_battles_won = battles_won
                last_total_num_steps = total_num_steps
                local_tik = time.time()

                self.log_info(train_infos, consuemd_num_steps)

            dist.barrier()
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(consuemd_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        self.policy.eval_mode()
        eval_battles_won = eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = np.zeros((self.n_eval_rollout_threads, 1), dtype=np.float32)

        eval_obs, _, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states.shape[-2:]),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while eval_episode < self.all_args.eval_episodes:
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

        summary_tensor = torch.Tensor([np.mean(eval_episode_rewards),
                                       eval_battles_won / eval_episode]).to(self.ddp_rank)
        dist.all_reduce(summary_tensor)

        if self.ddp_rank == 0:
            avg_reward = summary_tensor[0].item() / self.num_trainers
            avg_winning_rate = summary_tensor[1].item() / self.num_trainers
            eval_env_infos = {'eval_average_episode_rewards': avg_reward, "eval_win_rate": avg_winning_rate}
            self.log_info(eval_env_infos, total_num_steps)
            print("eval win rate is {}.".format(avg_winning_rate))
