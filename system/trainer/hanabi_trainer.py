import time
import wandb
import numpy as np
import torch
import torch.distributed as dist
from system.base_trainer import Trainer


def _t2n(x):
    return x.detach().cpu().numpy()


class HanabiTrainer(Trainer):
    def __init__(self, rpc_rank, weights_queue, buffer, config):
        super().__init__(rpc_rank, weights_queue, buffer, config)

    def run(self):
        # synchronize weights of rollout policy before inference starts
        self.pack_off_weights()

        transition_per_batch = self.episode_length * self.num_actors * self.env_per_split
        episodes = int(self.num_env_steps) // transition_per_batch // self.num_trainers

        global_tik = local_tik = time.time()
        last_total_num_steps = last_elapsed_episode = last_total_scores = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.policy.lr_decay(episode, episodes)

            buffer_util = self.buffer.get_utilization()

            train_infos = self.training_step()
            self.pack_off_weights()

            total_num_steps = self.buffer.total_timesteps.item()

            # save model
            if self.ddp_rank == 0 and (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if self.ddp_rank == 0 and episode % self.log_interval == 0:
                tok = time.time()
                recent_fps = int((total_num_steps - last_total_num_steps) / (tok - local_tik))
                global_avg_fps = int(total_num_steps / (tok - global_tik))
                print("\nGame Version {}, Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, "
                      "recent FPS {}, global average FPS {}.\n".format(self.all_args.hanabi_name, self.algorithm_name,
                                                                       self.experiment_name, episode, episodes,
                                                                       total_num_steps, self.num_env_steps, recent_fps,
                                                                       global_avg_fps))

                assert self.env_name == "Hanabi"
                with self.buffer.summary_lock:  # multiprocessing RLock
                    elapsed_episode = np.sum(self.buffer.elapsed_episode)
                    total_scores = np.sum(self.buffer.total_scores)
                recent_elapsed_episode = elapsed_episode - last_elapsed_episode
                recent_total_scores = total_scores - last_total_scores
                average_score = recent_total_scores / recent_elapsed_episode
                print("average score is {}.".format(average_score))

                print('buffer utilization before training step: {:.2f}'.format(buffer_util))

                if self.use_wandb:
                    wandb.log(
                        {
                            'average_score': average_score,
                            'total_env_steps': total_num_steps,
                            'fps': recent_fps,
                            'buffer_util': buffer_util,
                        },
                        step=total_num_steps)
                else:
                    self.writter.add_scalars('average_score', {'average_score': average_score}, total_num_steps)

                last_total_num_steps = total_num_steps
                last_elapsed_episode = elapsed_episode
                last_total_scores = total_scores
                local_tik = time.time()

                self.log_info(train_infos, total_num_steps)

            dist.barrier()
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        self.policy.eval_mode()
        eval_scores = []
        episode_cnt = 0

        obs, _, available_actions = self.eval_envs.reset()

        rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states.shape[-2:]),
                              dtype=np.float32)
        masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        env_done_trigger = np.zeros(self.n_eval_rollout_threads, dtype=np.int16)

        while episode_cnt < self.all_args.eval_episodes:
            for agent_id in range(self.num_agents):
                policy_outputs = self.policy.act(obs,
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

        avg_score = torch.Tensor([np.mean(eval_scores)]).to(self.ddp_rank)
        dist.all_reduce(avg_score)

        if self.ddp_rank == 0:
            avg_score = avg_score.item() / self.num_trainers
            self.log_info({'eval_average_score': np.mean(eval_scores)}, total_num_steps)
            print("eval average score is {}.".format(np.mean(eval_scores)))
