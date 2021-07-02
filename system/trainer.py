import wandb
import os
import torch
import zmq
import psutil
import numpy as np
from utils.utils import log
from utils.timing import Timing
from tensorboardX import SummaryWriter
import time
import torch.multiprocessing as mp

import torch.distributed as dist
import yaml
import datetime


class Trainer:
    """ Base class for training. """
    def __init__(self, rank, gpu_rank, buffer, cfg, nodes_ready_events, **kwargs):
        self.node_idx = cfg.learner_node_idx
        self.rank = rank
        self.gpu_rank = gpu_rank
        self.device = torch.device(gpu_rank)
        self.cfg = cfg
        self.num_trainers = self.cfg.num_trainers
        # TODO: support CPU
        self.tpdv = dict(device=torch.device(gpu_rank), dtype=torch.float32)

        self.nodes_ready_events = nodes_ready_events

        # TODO: add eval
        # self.eval_envs = kwargs['eval_envs_fn'](self.rank, self.cfg)
        self.num_agents = self.cfg.num_agents

        self.buffer = buffer

        # -------- parameters --------
        # names
        self.env_name = self.cfg.env_name
        self.algorithm_name = self.cfg.algorithm_name
        self.experiment_name = self.cfg.experiment_name
        # summary
        self.summary_keys = self.buffer.summary_keys
        self.summary_idx_hash = {}
        for i, k in enumerate(self.summary_keys):
            self.summary_idx_hash[k] = i
            setattr(self, 'last_' + k, 0)
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
        self.slots_per_update = self.cfg.slots_per_update
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

        self.run_dir = kwargs["run_dir"]

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

        self.initialized = False
        self.terminate = False

        self.model_weights_socket = None

        self.process = mp.Process(target=self._run)

    def _init(self):
        if self.gpu_rank == 0:
            self.model_weights_socket = zmq.Context().socket(zmq.PUB)
            model_port = self.cfg.model_weights_addrs[self.node_idx].split(':')[-1]
            self.model_weights_socket.bind('tcp://*:' + model_port)

        if self.rank == 0:
            if not self.cfg.no_summary:
                self._init_summary()

            if not self.cfg.no_summary and self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

                config_file = open(os.path.join(self.log_dir, 'config.yaml'), 'w')
                yaml.dump(vars(self.cfg), config_file)
                config_file.close()

        # TODO: nccl does not work in multi-learner setting, need to figure out why
        dist.init_process_group('gloo',
                                rank=self.rank,
                                world_size=self.cfg.num_trainers,
                                init_method=self.cfg.ddp_init_method)
        log.debug('Learner {} ucessfully initialized process group!'.format(self.rank))
        assert self.cfg.cuda and torch.cuda.is_available(), 'cpu training currently not supported'
        torch.cuda.set_device(self.gpu_rank)

        # policy network
        self.policy = self.policy_fn(self.gpu_rank,
                                     self.cfg,
                                     self.cfg.observation_space,
                                     self.cfg.share_observation_space,
                                     self.cfg.action_space,
                                     is_training=True)
        self.policy.train_mode()

        if self.model_dir is not None:
            self.restore()

        self.algorithm = self.algorithm_fn(self.cfg, self.policy)

        for i, e in enumerate(self.nodes_ready_events):
            e.wait()
            if self.gpu_rank == 0:
                # the first learner in each node outputs debug info
                log.debug('Waiting for all nodes ready... {}/{} have already finished initialization...'.format(
                    i + 1, len(self.nodes_ready_events)))

        if self.gpu_rank == 0:
            self.pack_off_weights()
        self.initialized = True
        log.debug('Sucessfully initializing Learner %d!', self.rank)

    def _init_summary(self):
        algo = self.cfg.algorithm_name
        network_cls = 'rnn' if algo == 'rmappo' else 'mlp' if algo == 'mappo' else None
        postfix = 'r{}_'.format(str(self.cfg.sample_reuse)) + network_cls
        exp_name = str(self.cfg.experiment_name) + "_seed" + str(self.cfg.seed)
        if self.cfg.use_wandb:
            self.run = wandb.init(config=self.cfg,
                                  project=self.cfg.project_name,
                                  entity=self.cfg.user_name,
                                  name=exp_name,
                                  group=self.cfg.group_name,
                                  dir=str(self.run_dir),
                                  job_type="training",
                                  reinit=True)
        else:
            curr_run = exp_name + postfix + '_' + str(datetime.datetime.now()).replace(' ', '_')
            self.run_dir /= curr_run
            if not self.run_dir.exists():
                os.makedirs(str(self.run_dir))

    def _terminate(self):
        if self.rank == 0:
            self.model_weights_socket.close()
            if hasattr(self, 'run'):
                self.run.finish()

        dist.destroy_process_group()

    def _accumulated_too_much_experience(self):
        # TODO: add stop experience collection signal
        return False

    def _run(self):
        psutil.Process().nice(self.cfg.default_niceness)

        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(self.cfg.learner_num_threads)

        timing = Timing()

        if self.train_in_background:
            self.training_thread.start()
        else:
            self._init()

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
                    # TODO: multi-node training will get stuck when loading data to device
                    train_infos = self.training_step(timing)
                    self.report(train_infos)

                    self.maybe_save()

                    log_infos = self.maybe_log()
                    self.report(log_infos)

                    dist.barrier()

                # TODO: gpurank=0 may not indicate that this is the first learner on this node
                if self.policy_version % self.cfg.broadcast_interval == 0 and self.gpu_rank == 0:
                    # the first learner in each node broadcasts weights
                    self.pack_off_weights()

                # TODO: add eval
                # if episode % self.eval_interval == 0 and self.use_eval:
                #     self.eval(consuemd_num_steps)

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
        time.sleep(0.1)
        log.info('GPU learner timing: %s', timing)

    def eval(self):
        # TODO: conduct evaluation using inference server rather than trainer
        raise NotImplementedError

    def pack_off_weights(self):
        # remove prefix 'module.' of DDP models
        numpy_state_dict = {k.replace('module.', ''): v.cpu().numpy() for k, v in self.policy.state_dict().items()}
        msg = []
        for k, v in numpy_state_dict.items():
            msg.extend([k.encode('ascii'), v])
        msg.append(str(self.policy_version).encode('ascii'))
        self.model_weights_socket.send_multipart(msg)

        if self.policy_version % 10 == 0:
            log.debug('Broadcasting model weights...(ver. {})'.format(self.policy_version))

    def training_step(self, timing):
        buffer_util = self.buffer.utilization
        log.info('buffer utilization before training step: {}/{}'.format(round(buffer_util * self.buffer.num_slots),
                                                                         self.buffer.num_slots))

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
            tensor_list = [torch.zeros(2).to(self.device) for _ in range(self.num_trainers)]
            dist.all_gather(tensor_list, torch.Tensor([self.node_idx, slot_id]).to(self.device))
            slot_ids = torch.stack(tensor_list).cpu().numpy()
            assert len(np.unique(slot_ids, axis=0)) == len(slot_ids), (np.unique(slot_ids, axis=0), slot_ids)

        with timing.add_time('training_step/reanalyze'):
            if self.cfg.use_reanalyze:
                # re-compute values/rnn_states for learning (re-analysis in MuZero, burn-in in R2D2 etc.)
                with torch.no_grad():
                    # TODO: deal with MLP (no rnn_states/masks)
                    # TODO: deal with Hanabi (nonshared case)
                    share_obs = self.buffer.share_obs[slot_id]
                    rnn_states_critic = self.buffer.rnn_states_critic[slot_id][0]
                    masks = self.buffer.masks[slot_id]
                    reanalyze_inputs = {
                        'share_obs': share_obs.reshape(self.episode_length + 1, -1, *share_obs.shape[3:]),
                        'rnn_states_critic': rnn_states_critic.reshape(-1, *rnn_states_critic.shape[2:]).swapaxes(0, 1),
                        'masks': masks.reshape(self.episode_length + 1, -1, *masks.shape[3:]),
                    }
                    for k, v in reanalyze_inputs.items():
                        reanalyze_inputs[k] = torch.from_numpy(v).to(**self.tpdv)

                    values = self.policy.get_values(**reanalyze_inputs).cpu().numpy()
                    self.buffer.values[slot_id] = values.reshape(*self.buffer.values[slot_id].shape)

        for sample in data_generator:
            with timing.add_time('training_step/to_device'):
                # TODO: currently DDP training using gloo/nccl backend will be stuck here, need to resolve it
                for k, v in sample.items():
                    sample[k] = torch.from_numpy(v).to(**self.tpdv)

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

        return {**train_info, 'buffer_util': buffer_util}

    def maybe_save(self):
        if self.rank == 0 and (self.policy_version % self.save_interval == 0
                               or self.policy_version == self.train_for_episodes - 1):
            self.save()

    def maybe_log(self):
        log_infos = None
        # log information
        if self.rank == 0 and self.policy_version % self.log_interval == 0:
            self.last_received_num_steps = self.received_num_steps
            self.received_num_steps = self.buffer.total_timesteps.item()

            recent_consumed_num_steps = self.log_interval * self.transitions_per_batch * self.num_trainers
            recent_received_num_steps = self.received_num_steps - self.last_received_num_steps

            recent_rollout_fps = int(recent_received_num_steps / (time.time() - self.logging_tik))
            global_avg_rollout_fps = int(self.received_num_steps / (time.time() - self.training_tik))

            recent_learning_fps = int(recent_consumed_num_steps / (time.time() - self.logging_tik))
            global_avg_learning_fps = int(self.consumed_num_steps / (time.time() - self.training_tik))

            log.debug("Env {} Algo {} Exp {} updates {}/{} episodes, consumed num timesteps {}/{}, "
                      "recent rollout FPS {}, global average rollout FPS {}, "
                      "recent learning FPS {}, global average learning FPS {}.\n".format(
                          self.env_name, self.algorithm_name, self.experiment_name, self.policy_version,
                          self.train_for_episodes, self.consumed_num_steps, self.train_for_env_steps,
                          recent_rollout_fps, global_avg_rollout_fps, recent_learning_fps, global_avg_learning_fps))

            # as defined in https://cdn.openai.com/dota-2.pdf
            recent_sample_reuse = recent_consumed_num_steps / recent_received_num_steps
            global_sample_reuse = self.consumed_num_steps / self.received_num_steps

            log.debug('recent sample reuse: {:.2f}, global average sample reuse: {:.2f}.'.format(
                recent_sample_reuse, global_sample_reuse))

            log_infos = {
                'iteration': self.policy_version,
                'rollout_FPS': recent_rollout_fps,
                'learning_FPS': recent_learning_fps,
                'sample_reuse': recent_sample_reuse,
                'received_num_steps': self.received_num_steps
            }

            self.logging_tik = time.time()

            if self.env_name == 'StarCraft2':
                with self.buffer.summary_lock:
                    summary_info = self.buffer.summary_block.sum(0)
                elapsed_episodes = summary_info[self.summary_idx_hash['elapsed_episodes']]
                winning_episodes = summary_info[self.summary_idx_hash['winning_episodes']]
                episode_return = summary_info[self.summary_idx_hash['episode_return']]

                recent_elapsed_episodes = elapsed_episodes - self.last_elapsed_episodes
                recent_winning_episodes = winning_episodes - self.last_winning_episodes
                recent_episode_return = episode_return - self.last_episode_return

                if recent_elapsed_episodes > 0:
                    winning_rate = recent_winning_episodes / recent_elapsed_episodes
                    assert 0 <= winning_rate and winning_rate <= 1
                    avg_return = recent_episode_return / recent_elapsed_episodes
                    log.debug('Map: {}, Recent Winning Rate: {:.2%}, Avg. Return: {:.2f}.'.format(
                        self.cfg.map_name, winning_rate, avg_return))

                    self.last_elapsed_episodes = elapsed_episodes
                    self.last_winning_episodes = winning_episodes
                    self.last_episode_return = episode_return

                    log_infos = {**log_infos, 'train_winning_rate': winning_rate, 'train_episode_return': avg_return}
            else:
                raise NotImplementedError

        return log_infos

    def _should_end_training(self):
        end = self.terminate
        end |= self.consumed_num_steps > self.train_for_env_steps
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

    def report(self, infos):
        if infos is None or self.rank != 0:
            return

        if not self.no_summary:
            for k, v in infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=self.consumed_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, self.consumed_num_steps)
        else:
            log.info(infos)
