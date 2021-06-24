import wandb
import os
import torch
import zmq
import psutil
import signal
import numpy as np
from utils.utils import log
from utils.timing import Timing
from tensorboardX import SummaryWriter
import time
import torch.multiprocessing as mp

import torch.distributed as dist


class Trainer:
    """ Base class for training. """
    def __init__(self, rank, buffer, cfg, **kwargs):
        self.rank = rank
        self.cfg = cfg
        self.num_trainers = self.cfg.num_trainers

        # TODO: add eval
        # self.eval_envs = kwargs['eval_envs_fn'](self.rank, self.cfg)
        self.num_agents = self.cfg.num_agents

        # -------- parameters --------
        # names
        self.env_name = self.cfg.env_name
        self.algorithm_name = self.cfg.algorithm_name
        self.experiment_name = self.cfg.experiment_name
        # tricks
        self.use_linear_lr_decay = self.cfg.use_linear_lr_decay
        # system dataflow
        self.num_mini_batch = self.cfg.num_mini_batch
        self.num_actors = self.cfg.num_actors
        self.envs_per_actor = self.cfg.envs_per_actor
        self.num_splits = self.cfg.num_splits
        self.envs_per_split = self.envs_per_actor // self.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.episode_length = self.cfg.episode_length
        # interval
        self.use_eval = self.cfg.use_eval
        self.save_interval = self.cfg.save_interval
        self.eval_interval = self.cfg.eval_interval
        self.log_interval = self.cfg.log_interval
        # dir
        self.model_dir = self.cfg.model_dir
        # summay & render
        self.no_summary = self.cfg.no_summary
        self.use_wandb = self.cfg.use_wandb
        self.use_render = self.cfg.use_render

        self.train_for_env_steps = self.cfg.train_for_env_steps
        self.train_for_seconds = self.cfg.train_for_seconds
        self.transitions_per_batch = (self.episode_length * self.num_actors * self.envs_per_split *
                                      self.slots_per_update // self.cfg.num_policy_workers)
        self.train_for_episodes = self.train_for_env_steps // self.transitions_per_batch // self.num_trainers

        self.train_in_background = self.cfg.train_in_background
        # TODO: add training background thread
        assert not self.train_in_background

        self.training_tik = None
        self.logging_tik = None

        self.stop_experience_collection = False

        if self.rank == 0:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = kwargs["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        # policy network
        self.policy_fn = Policy
        self.policy = None
        self.policy_version = 0

        self.consumed_num_steps = 0
        self.received_num_steps = 0
        self.last_received_num_steps = None

        # algorithm
        self.algorithm_fn = TrainAlgo
        self.algorithm = None

        self.buffer = buffer

        self.initialized = False
        self.terminate = False

        self.model_weights_socket = None

        self.process = mp.Process(target=self._run)

    def _init(self):
        dist.init_process_group('nccl',
                                rank=self.rank,
                                world_size=self.cfg.num_trainers,
                                init_method=self.cfg.ddp_init_method)
        assert self.cfg.cuda and torch.cuda.is_available(), 'cpu training currently not supported'
        torch.cuda.set_device(self.rank)

        # policy network
        self.policy = self.policy_fn(self.rank,
                                     self.cfg,
                                     self.cfg.observation_space,
                                     self.cfg.share_observation_space,
                                     self.cfg.action_space,
                                     is_training=True)
        self.policy.train_mode()

        if self.model_dir is not None:
            self.restore()

        self.algorithm = self.algorithm_fn(self.cfg, self.policy)

        if self.rank == 0:
            self.model_weights_socket = zmq.Context().socket(zmq.PUB)
            model_port = self.cfg.model_weights_addr.split(':')[-1]
            self.model_weights_socket.bind('tcp://*:' + model_port)

        self.pack_off_weights()
        self.initialized = True
        log.debug('Sucessfully initializing Learner %d!', self.rank)

    def _terimiate(self):
        dist.destroy_process_group()

        self.terminate = True

    def _accumulated_too_much_experience(self):
        # TODO: add stop experience collection signal
        return False

    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        psutil.Process().nice(self.cfg.default_niceness)

        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(self.cfg.learner_num_threads)

        timing = Timing()

        if self.train_in_background:
            self.training_thread.start()
        else:
            self.init()
            log.error(
                'train_in_background set to False on learner %d! This is slow, use only for testing!',
                self.policy_id,
            )

        self.training_tik = self.logging_tik = time.time()

        while not self._should_end_training():
            try:
                if self._accumulated_too_much_experience():
                    # TODO: add stop experience collection signal
                    # if we accumulated too much experience, signal the policy workers to stop experience collection
                    # if not self.stop_experience_collection[self.policy_id]:
                    #     self.stop_experience_collection_num_msgs += 1
                    #     if self.stop_experience_collection_num_msgs >= 50:
                    #         log.info(
                    #             'Learner %d accumulated too much experience, stop experience collection! '
                    #             'Learner is likely a bottleneck in your experiment (%d times)',
                    #             self.policy_id,
                    #             self.stop_experience_collection_num_msgs,
                    #         )
                    #         self.stop_experience_collection_num_msgs = 0
                    # self.stop_experience_collection[self.policy_id] = True
                    pass

                elif self.stop_experience_collection:
                    pass
                    # TODO: add resume experience collection signal
                    # otherwise, resume the experience collection if it was stopped
                    # self.stop_experience_collection[self.policy_id] = False
                    # with self.resume_experience_collection_cv:
                    #     self.resume_experience_collection_num_msgs += 1
                    #     if self.resume_experience_collection_num_msgs >= 50:
                    #         log.debug('Learner %d is resuming experience collection!', self.policy_id)
                    #         self.resume_experience_collection_num_msgs = 0
                    #     self.resume_experience_collection_cv.notify_all()

                if not self.train_in_background:
                    train_infos = self.training_step(timing)
                    self.report(train_infos)
                    self.after_training_step()

                if self.policy_version % self.cfg.broadcast_interval == 0:
                    self.pack_off_weights()

                # self._experience_collection_rate_stats()

            except RuntimeError as exc:
                log.warning('Error in Learner: %d, exception: %s', self.rank, exc)
                log.warning('Terminate process...')
                self.terminate = True
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on Learner %d', self.rank)
                self.terminate = True
            except Exception:
                log.exception('Unknown exception in Learner %d', self.rank)
                self.terminate = True

        if self.train_in_background:
            self.experience_buffer_queue.put(None)
            self.training_thread.join()

        self._terminate()
        time.sleep(0.3)
        log.info('GPU learner timing: %s', timing)

    def run(self):
        raise NotImplementedError

    def eval(self):
        # TODO: conduct evaluation using inference server rather than trainer
        raise NotImplementedError

    def pack_off_weights(self):
        if self.rank == 0:
            # remove prefix 'module.' of DDP models
            numpy_state_dict = {k.replace('module.', ''): v.cpu().numpy() for k, v in self.policy.state_dict().items()}
            msg = []
            for k, v in numpy_state_dict.items():
                msg.extend([k.encode('ascii'), v])
            msg.append(str(self.policy_version).encode('ascii'))
            self.model_weights_socket.send_multipart(msg)
            log.debug('Broadcasting model weights...')

    def training_step(self, timing):
        log.info('buffer utilization before training step: {}/{}'.format(
            round(self.buffer.utilization * self.buffer.num_slots), self.buffer.num_slots))

        if self.use_linear_lr_decay:
            self.policy.lr_decay(self.policy_version, self.train_for_episodes)

        train_info = {}

        # TODO: use different summary keys for different algorithms
        summary_keys = ['value_loss', 'policy_loss', 'dist_entropy', 'grad_norm']
        for k in summary_keys:
            train_info[k] = 0

        with timing.add_time('training_step/synchronization'):
            dist.barrier()

        with timing.add_time('training_step/get_slot'):
            # only train popart parameter in the first epoch
            slot_id, data_generator = self.buffer.get()

        with timing.add_time('training_step/different_slots_assertion'):
            # ensure all process get different slot ids
            tensor_list = [torch.zeros(1).to(self.device) for _ in range(self.num_trainers)]
            dist.all_gather(tensor_list, torch.Tensor([slot_id]).to(self.device))
            slot_ids = torch.cat(tensor_list).tolist()
            assert len(np.unique(slot_ids)) == len(slot_ids)

        for sample in data_generator:
            with timing.add_time('training_step/algorithm_step'):
                infos = self.algorithm.step(sample)

            with timing.add_time('training_step/logging/loss_all_reduce'):
                for info in infos:
                    dist.all_reduce(info)

            with timing.add_time('training_step/logging/loss'):
                value_loss, policy_loss, dist_entropy, grad_norm = infos

                for k in summary_keys:
                    train_info[k] += locals()[k].item()

        with timing.add_time('training_step/logging/other_records'):
            train_info["average_step_rewards"] = np.mean(self.buffer.rewards[slot_id])
            train_info['dead_ratio'] = 1 - self.buffer.active_masks[slot_id].sum() / np.prod(
                self.buffer.active_masks[slot_id].shape)

            reduce_factor = self.num_mini_batch * self.num_trainers

            for k in summary_keys:
                train_info[k] /= reduce_factor

        with timing.add_time('training_step/close_out'):
            self.buffer.close_out(slot_id)
            self.policy_version += 1
            self.consumed_num_steps += self.transitions_per_batch

        return train_info

    def after_training_step(self):
        # save model
        if self.rank == 0 and (self.policy_version % self.save_interval == 0
                               or self.policy_version == self.train_for_episodes - 1):
            self.save()

        # log information
        if self.rank == 0 and self.policy_version % self.log_interval == 0:
            self.last_received_num_steps = self.received_num_steps
            self.received_num_steps = self.buffer.total_env_steps.item()

            recent_consumed_num_steps = self.log_interval * self.transitions_per_batch * self.num_trainers
            recent_received_num_steps = self.received_num_steps - self.last_received_num_steps

            recent_rollout_fps = int(recent_received_num_steps / (time.time() - self.logging_tik))
            global_avg_rollout_fps = int(self.received_num_steps / (time.time() - self.training_tik))

            recent_learning_fps = int(recent_consumed_num_steps / (time.time() - self.logging_tik))
            global_avg_learning_fps = int(self.consumed_num_steps / (time.time() - self.training_tik))

            log.info("Map {} Algo {} Exp {} updates {}/{} episodes, consumed num timesteps {}/{}, "
                     "recent rollout FPS {}, global average rollout FPS {}, "
                     "recent learning FPS {}, global average learning FPS {}.\n".format(
                         self.cfg.map_name, self.algorithm_name, self.experiment_name, self.policy_version,
                         self.train_for_episodes, self.consumed_num_steps, self.num_env_steps, recent_rollout_fps,
                         global_avg_rollout_fps, recent_learning_fps, global_avg_learning_fps))

            # assert self.env_name == "StarCraft2"
            # with self.buffer.summary_lock:  # multiprocessing RLock
            #     battles_won = np.sum(self.buffer.battles_won)
            #     battles_game = np.sum(self.buffer.battles_game)
            # recent_battles_won = battles_won - last_battles_won
            # recent_battles_game = battles_game - last_battles_game

            # recent_win_rate = recent_battles_won / recent_battles_game if recent_battles_game > 0 else 0.0
            # log.info("recent winning rate is {}.".format(recent_win_rate))

            # as defined in https://cdn.openai.com/dota-2.pdf

            recent_sample_reuse = recent_consumed_num_steps / recent_received_num_steps
            global_sample_reuse = self.consuemd_num_steps / self.received_num_steps

            log.info('recent sample reuse: {:.2f}, global average sample reuse: {:.2f}.'.format(
                recent_sample_reuse, global_sample_reuse))

            # if self.use_wandb:
            #     wandb.log(
            #         {
            #             "recent_win_rate": recent_win_rate,
            #             'total_env_steps': consumed_num_steps,
            #             'fps': recent_fps,
            #             'buffer_util': buffer_util,
            #             'iteraion': episode + 1,
            #             'sample_reuse': recent_sample_reuse,
            #         },
            #         step=consuemd_num_steps)
            # else:
            #     self.writter.add_scalars("recent_win_rate", {"recent_win_rate": recent_win_rate}, consuemd_num_steps)

            # last_battles_game = battles_game
            # last_battles_won = battles_won
            # self.last_consumed_num_steps = consumed_num_steps
            self.logging_tik = time.time()

        dist.barrier()
        # eval
        # if episode % self.eval_interval == 0 and self.use_eval:
        #     self.eval(consuemd_num_steps)

    def _should_end_training(self):
        end = self.consumed_num_steps > self.train_for_env_steps
        end |= self.policy_version > self.train_for_episodes
        end |= (time.time() - self.training_tik) > self.train_for_seconds

        # if self.cfg.benchmark:
        #     end |= self.total_env_steps_since_resume >= int(2e6)
        #     end |= sum(self.samples_collected) >= int(1e6)

        return end

    def save(self):
        """Save policy's actor and critic networks."""
        torch.save(self.policy.state_dict(), str(self.save_dir) + "/mdoel.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        self.policy.actor_critic.load_state_dict(torch.load(str(self.model_dir) + '/model.pt'))

    def report(self, infos, consumed_num_steps):
        if not self.no_summary:
            for k, v in infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=consumed_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, consumed_num_steps)
