import os
import torch
import zmq
import psutil
import pathlib
import numpy as np
from queue import Queue, Empty
from utils.utils import log, TaskType, RWLock
from utils.buffer import LearnerBuffer
from utils.timing import Timing
import time
import torch.multiprocessing as mp

import torch.distributed as dist
from algorithms.registries import ALGORITHM_SUMMARY_KEYS


class Trainer:
    def __init__(self, cfg, gpu_rank, nodes_ready_events, trainer_ready_event, shm_state_dict):
        self.cfg = cfg
        self.gpu_rank = gpu_rank
        self.node_idx = self.cfg.learner_node_idx

        self.num_actors = self.cfg.num_actors // self.cfg.num_tasks_per_node
        self.envs_per_actor = cfg.envs_per_actor

        self.policy_id = self.cfg.learner_config[str(self.node_idx)][str(gpu_rank)]
        self.replicate_rank = self.trainer_idx = self.num_trainers = 0

        for node_idx, local_config in self.cfg.learner_config.items():
            if int(node_idx) < self.node_idx:
                self.trainer_idx += len(local_config)
                for gpu_idx, policy_id in local_config.items():
                    if policy_id == self.policy_id:
                        self.replicate_rank += 1
            elif int(node_idx) == self.node_idx:
                for gpu_idx, policy_id in local_config.items():
                    if int(gpu_idx) < self.gpu_rank:
                        self.trainer_idx += 1
                        if policy_id == self.policy_id:
                            self.replicate_rank += 1

        for _, local_config in self.cfg.learner_config.items():
            for _, policy_id in local_config.items():
                if policy_id == self.policy_id:
                    self.num_trainers += 1

        self.num_agents = len(self.cfg.policy2agents[str(self.policy_id)])
        example_agent = self.cfg.policy2agents[str(self.policy_id)][0]

        self.obs_space = self.cfg.observation_space[example_agent]
        self.act_space = self.cfg.action_space[example_agent]

        self.buffer = LearnerBuffer(self.cfg, self.policy_id, self.num_agents, self.obs_space, self.act_space)

        self.trainer_ready_event = trainer_ready_event

        # TODO: support CPU
        self.tpdv = dict(device=torch.device(gpu_rank), dtype=torch.float32)

        self.nodes_ready_events = nodes_ready_events

        # -------- parameters --------
        # names
        self.env_name = self.cfg.env_name
        self.algorithm_name = self.cfg.algorithm_name
        self.experiment_name = self.cfg.experiment_name
        # summary
        self.env_summary_keys = self.buffer.env_summary_keys
        self.env_summary_idx_hash = {}
        for i, k in enumerate(self.env_summary_keys):
            self.env_summary_idx_hash[k] = i
            setattr(self, 'last_' + k, 0)
        self.algorithm_summary_keys = ALGORITHM_SUMMARY_KEYS[self.cfg.algorithm_name]
        # tricks
        self.use_linear_lr_decay = self.cfg.use_linear_lr_decay
        # system dataflow
        self.num_mini_batch = self.cfg.num_mini_batch
        self.envs_per_actor = self.cfg.envs_per_actor
        self.num_splits = self.cfg.num_splits
        self.envs_per_split = self.envs_per_actor // self.num_splits
        assert self.envs_per_actor % self.num_splits == 0
        self.episode_length = self.cfg.episode_length
        self.slots_per_update = self.cfg.slots_per_update
        # interval
        self.save_interval = self.cfg.save_interval
        self.log_interval = self.cfg.log_interval
        # dir
        self.model_dir = self.cfg.model_dir
        self.save_dir = self.cfg.save_dir
        # summay & render
        # TODO: write a render script
        self.use_render = self.cfg.use_render

        self.train_for_env_steps = self.cfg.train_for_env_steps
        self.train_for_seconds = self.cfg.train_for_seconds
        self.transitions_per_batch = (self.episode_length * self.cfg.actor_group_size * self.envs_per_split *
                                      self.slots_per_update * self.num_agents * self.num_trainers)
        self.train_for_episodes = self.train_for_env_steps // self.transitions_per_batch

        self.training_tik = None
        self.logging_tik = None

        self.stop_experience_collection = False

        # initialize save dir
        if self.save_dir is None:
            self.save_dir = pathlib.Path('./models')
        else:
            self.save_dir = pathlib.Path(self.save_dir)
        self.save_dir /= self.cfg.env_name
        if self.cfg.env_name == 'StarCraft2':
            self.save_dir /= self.cfg.map_name
        self.save_dir = self.save_dir / cfg.algorithm_name / cfg.experiment_name / ('policy_' + str(self.policy_id))
        if not self.save_dir.exists():
            os.makedirs(str(self.save_dir))

        from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        # policy network
        self.policy_fn = Policy
        self.policy = None
        self.policy_version = (torch.ones(1, dtype=torch.int32) * (-1)).share_memory_()
        self.shm_state_dict = shm_state_dict

        self.consumed_num_steps = 0
        self.received_num_steps = 0
        self.last_received_num_steps = None

        # algorithm
        self.algorithm_fn = TrainAlgo
        self.algorithm = None

        self.initialized = False
        self.terminate = False

        self._context = None
        self.model_weights_socket = None
        self.reset_socket = None
        self.task_socket = None
        self.task_result_socket = None

        self.socket_identity = ('learner-' + str(self.policy_id)).encode('ascii')
        # TODO: modify this when processing tasks
        self.is_executing_task = False
        self.task_queue = Queue(8)

        self.batch_queue = mp.Queue(4 * 1000)
        self.value_tracer_queue = mp.Queue(4 * 1000)
        self.value_tracer_task_queues = [mp.JoinableQueue(1) for _ in range(self.cfg.num_value_tracers_per_trainer)]
        self.reanalyzer_task_queues = [mp.JoinableQueue(1) for _ in range(self.cfg.num_reanalyzers_per_trainer)]
        self.param_lock = RWLock()

        self.reanalyzers = []
        self.value_tracers = []

        self.process = mp.Process(target=self._run)

    def start_process(self):
        self.process.start()

    def _init(self):
        if self.replicate_rank == 0:
            self._context = zmq.Context()

            # the first trainer on each node will manage its slice of worker nodes
            self.model_weights_socket = self._context.socket(zmq.PUB)
            model_port = self.cfg.model_weights_addrs[self.policy_id].split(':')[-1]
            self.model_weights_socket.bind('tcp://*:' + model_port)

            self.reset_socket = self._context.socket(zmq.PUB)
            reset_port = self.cfg.reset_addrs[0].split(':')[-1]
            self.reset_socket.bind('tcp://*:' + reset_port)

            self.task_socket = self._context.socket(zmq.SUB)
            self.task_socket.connect(self.cfg.task_dispatcher_addr)
            self.task_socket.setsockopt(zmq.SUBSCRIBE, self.socket_identity)

            self.task_result_socket = self._context.socket(zmq.PUSH)
            self.task_result_socket.connect(self.cfg.task_result_addr)
            self.task_result_socket.send_multipart([self.socket_identity, str(self.policy_id).encode('ascii')])

        log.debug('Trainer {} initializing process group!'.format(self.trainer_idx))

        # os.environ['NCCL_DEBUG'] = 'info'
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
        os.environ['NCCL_IB_DISABLE'] = '1'
        # TODO: nccl with orthogonal initialization has a bug
        dist.init_process_group('nccl',
                                rank=self.replicate_rank,
                                world_size=self.num_trainers,
                                init_method=self.cfg.ddp_init_methods[self.policy_id])
        log.debug('Trainer {} sucessfully initialized process group!'.format(self.trainer_idx))
        assert self.cfg.cuda and torch.cuda.is_available(), 'cpu training currently not supported'
        torch.cuda.set_device(self.gpu_rank)

        # policy network
        self.policy = self.policy_fn(self.gpu_rank,
                                     self.cfg,
                                     self.obs_space,
                                     self.act_space,
                                     is_training=True)
        self.policy.train_mode()

        if self.model_dir is not None:
            self.restore()

        with self.param_lock.w_locked():
            self.policy_version += 1
            primal_state_dict = {k.replace('module.', ''): v.cpu() for k, v in self.policy.state_dict().items()}
            for k, v in self.shm_state_dict.items():
                v[:] = primal_state_dict[k]

        self.algorithm = self.algorithm_fn(self.cfg, self.policy)

        self.trainer_ready_event.set()

        log.debug('Waiting for all nodes ready...')
        for i, e in enumerate(self.nodes_ready_events):
            e.wait()
            if self.replicate_rank == 0:
                # the first trainer in each node outputs debug info
                log.debug('Waiting for all nodes ready... {}/{} have already finished initialization...'.format(
                    i + 1, len(self.nodes_ready_events)))

        if self.replicate_rank == 0:
            self.pack_off_weights()
        self.initialized = True
        log.debug('Sucessfully initializing Trainer %d!', self.trainer_idx)

    def process_task(self, task):
        # TODO: modify self.is_executing_tasks when processing tasks
        if task == TaskType.TRAIN:
            pass
        elif task == TaskType.TERMINATE:
            self.terminate = True
        else:
            raise NotImplementedError

    def _terminate(self):
        if self.replicate_rank == 0:
            self.model_weights_socket.close()
            self.reset_socket.close()
            self.task_socket.close()

        dist.destroy_process_group()

    def _run(self):
        psutil.Process().nice(self.cfg.default_niceness + 5)

        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(self.cfg.n_training_threads)

        timing = Timing()

        self._init()

        self.training_tik = self.logging_tik = time.time()

        try:
            while not self.terminate:
                # cl, tasks
                self.send_reset_task()

                if self.replicate_rank == 0:
                    try:
                        msg = self.task_socket.recv_multipart(flags=zmq.NOBLOCK)
                        self.task_queue.put(msg)
                    except zmq.ZMQError:
                        pass

                    if not self.is_executing_task:
                        try:
                            # TODO: here we don't process task except for TERMINATE
                            msg = self.task_queue.get_nowait()
                            task = int(msg[1].decode('ascii'))
                            self.process_task(task)
                            if self.terminate:
                                break
                        except Empty:
                            # log.warning('Trainer %d is not executing tasks and there are no tasks distributed to it!',
                            #             self.trainer_idx)
                            pass

                # TODO update CL archive

                train_infos = self.training_step(timing)

                self.maybe_save()

                log_infos = self.maybe_log(timing)

                self.report({**log_infos, **train_infos})

                dist.barrier()

                if self.policy_version % (self.cfg.sample_reuse *
                                            self.cfg.broadcast_interval) == 0 and self.replicate_rank == 0:
                    # the first trainer in each node broadcasts weights
                    self.pack_off_weights()

        except RuntimeError as exc:
            log.warning('Error in Trainer: %d, exception: %s', self.trainer_idx, exc)
            log.warning('Terminate process...')
            self.terminate = True
        except KeyboardInterrupt:
            log.warning('Keyboard interrupt detected on Trainer %d', self.trainer_idx)
            self.terminate = True
        except Exception:
            log.exception('Unknown exception in Trainer %d', self.trainer_idx)
            self.terminate = True

        self._terminate()
        time.sleep(0.1)
        log.info('GPU Trainer timing: %s', timing)

    def pack_off_weights(self):
        # remove prefix 'module.' of DDP models
        numpy_state_dict = {k.replace('module.', ''): v.cpu().numpy() for k, v in self.policy.state_dict().items()}
        msg = []
        for k, v in numpy_state_dict.items():
            msg.extend([k.encode('ascii'), v])
        msg.append(str(self.policy_version.item()).encode('ascii'))
        # print('********send_model', msg)
        self.model_weights_socket.send_multipart(msg)

        if self.policy_version.item() % 100 == 0:
            # print(numpy_state_dict['critic_rnn.rnn.weight_hh_l0'])
            log.debug('Broadcasting model weights...(ver. {})'.format(self.policy_version.item()))

    # cl, send tasks
    def send_reset_task(self):
        msg = []
        
        for _ in range(10000):
            numpy_msg = np.ones(5) * np.random.randint(0,10)
            msg.append(str(numpy_msg).encode('ascii'))
        self.reset_socket.send_multipart(msg)

    def training_step(self, timing):
        buffer_util = self.buffer.utilization
        if self.policy_version.item() % 40 == 0:
            log.info('buffer utilization before training step: {}/{}'.format(round(buffer_util * self.buffer.num_slots),
                                                                             self.buffer.num_slots))

        if self.use_linear_lr_decay:
            self.policy.lr_decay(self.policy_version.item(), self.train_for_episodes)

        train_info = {}

        self.algorithm_summary_keys = ['value_loss', 'policy_loss', 'dist_entropy', 'grad_norm']
        for k in self.algorithm_summary_keys:
            train_info[k] = 0

        with timing.add_time('training_step/synchronization'):
            self.policy.train_mode()
            dist.barrier()

        with timing.add_time('training_step/wait_for_batch'):
            # only train popart parameter in the first epoch
            slot_id = self.batch_queue.get()

        data_generator = self.buffer.recurrent_generator(
            slot_id) if self.cfg.use_recurrent_policy else self.buffer.feed_forward_generator(slot_id)

        with timing.time_avg('one_training_step'):
            for sample in data_generator:
                with timing.add_time('training_step/to_device'):
                    for k, v in sample.items():
                        sample[k] = torch.from_numpy(v).pin_memory().to(**self.tpdv, non_blocking=True)

                with timing.add_time('training_step/algorithm_step'):
                    # TODO use value_pred of samples, update cl archive
                    infos = self.algorithm.step(sample)
                    del sample

                with timing.add_time('training_step/logging/loss_all_reduce'):
                    for info in infos:
                        dist.all_reduce(info)

                with timing.add_time('training_step/logging/loss'):
                    value_loss, policy_loss, dist_entropy, grad_norm = infos

                    for k in self.algorithm_summary_keys:
                        train_info[k] += locals()[k].item()

        with timing.add_time('training_step/logging/other_records'):
            # train_info["average_step_rewards"] = np.mean(self.buffer.rewards[slot_id])
            # train_info['dead_ratio'] = 1 - self.buffer.active_masks[slot_id].sum() / np.prod(
            #     self.buffer.active_masks[slot_id].shape)

            reduce_factor = self.num_mini_batch * self.num_trainers

            for k in self.algorithm_summary_keys:
                train_info[k] /= reduce_factor

        with timing.add_time('training_step/close_out'):
            self.buffer.close_out(slot_id, self.policy_version.item())

            with self.param_lock.w_locked():
                self.policy_version += 1
                if self.policy_version.item() % (self.cfg.sample_reuse * self.cfg.broadcast_interval) == 0:
                    primal_state_dict = {k.replace('module.', ''): v.cpu() for k, v in self.policy.state_dict().items()}
                    for k, v in self.shm_state_dict.items():
                        v[:] = primal_state_dict[k]

            self.consumed_num_steps += self.transitions_per_batch

        return {**train_info, 'buffer_util': buffer_util, 'iteration': self.policy_version.item()}

    def maybe_save(self):
        if self.replicate_rank == 0 and (self.policy_version.item() % self.save_interval == 0
                                         or self.policy_version.item() == self.train_for_episodes - 1):
            self.save()

    def maybe_log(self, timing):
        log_infos = {}
        # log information
        if self.replicate_rank == 0 and self.policy_version.item() % self.log_interval == 1:
            self.last_received_num_steps = self.received_num_steps
            self.received_num_steps = self.buffer.total_timesteps.item()

            recent_consumed_num_steps = self.log_interval * self.transitions_per_batch
            recent_received_num_steps = self.received_num_steps - self.last_received_num_steps

            recent_rollout_fps = int(recent_received_num_steps / (time.time() - self.logging_tik))
            global_avg_rollout_fps = int(self.received_num_steps / (time.time() - self.training_tik))

            recent_learning_fps = int(recent_consumed_num_steps / (time.time() - self.logging_tik))
            global_avg_learning_fps = int(self.consumed_num_steps / (time.time() - self.training_tik))

            # as defined in https://cdn.openai.com/dota-2.pdf
            if recent_received_num_steps > 0:
                recent_sample_reuse = recent_consumed_num_steps / recent_received_num_steps
            else:
                recent_sample_reuse = np.nan
            global_sample_reuse = self.consumed_num_steps / self.received_num_steps / self.num_trainers

            log.debug("Env {} Algo {} Exp {} updates {}/{} episodes, consumed num timesteps {}/{}, "
                      "recent rollout FPS {}, global average rollout FPS {}, recent learning FPS {}, "
                      "global average learning FPS {}, recent sample reuse: {:.2f}, "
                      "global average sample reuse: {:.2f}, average training step time {}s.\n".format(
                          self.env_name, self.algorithm_name, self.experiment_name, self.policy_version.item(),
                          self.train_for_episodes, self.consumed_num_steps, self.train_for_env_steps,
                          recent_rollout_fps, global_avg_rollout_fps, recent_learning_fps, global_avg_learning_fps,
                          recent_sample_reuse, global_sample_reuse, timing.one_training_step))

            log_infos = {
                'rollout_FPS': recent_rollout_fps,
                'learning_FPS': recent_learning_fps,
                'sample_reuse': recent_sample_reuse,
                'received_num_steps': self.received_num_steps,
                'consumed_num_steps': self.consumed_num_steps,
            }

            self.logging_tik = time.time()

            if self.env_name == 'StarCraft2':
                with self.buffer.env_summary_lock:
                    summary_info = self.buffer.summary_block.sum(axis=(0, 1))
                elapsed_episodes = summary_info[self.env_summary_idx_hash['elapsed_episodes']]
                winning_episodes = summary_info[self.env_summary_idx_hash['winning_episodes']]
                episode_return = summary_info[self.env_summary_idx_hash['episode_return']]
                episode_length = summary_info[self.env_summary_idx_hash['episode_length']]

                recent_elapsed_episodes = elapsed_episodes - self.last_elapsed_episodes
                recent_winning_episodes = winning_episodes - self.last_winning_episodes
                recent_episode_return = episode_return - self.last_episode_return
                recent_episode_length = episode_length - self.last_episode_length

                if recent_elapsed_episodes > 0:
                    winning_rate = recent_winning_episodes / recent_elapsed_episodes
                    assert 0 <= winning_rate and winning_rate <= 1, winning_rate
                    avg_return = recent_episode_return / recent_elapsed_episodes
                    avg_ep_len = recent_episode_length / recent_elapsed_episodes
                    log.debug(
                        'Map: {}, Recent Winning Rate: {:.2%} ({}/{}), Avg. Episode Length {:.2f}, Avg. Return: {:.2f}.'
                        .format(self.cfg.map_name, winning_rate, int(recent_winning_episodes),
                                int(recent_elapsed_episodes), avg_ep_len, avg_return))

                    self.last_elapsed_episodes = elapsed_episodes
                    self.last_winning_episodes = winning_episodes
                    self.last_episode_return = episode_return
                    self.last_episode_length = episode_length

                    log_infos = {
                        **log_infos, 'train_winning_rate': winning_rate,
                        'train_episode_return': avg_return,
                        'train_episode_length': avg_ep_len
                    }
            elif self.cfg.env_name == 'HideAndSeek':
                with self.buffer.env_summary_lock:
                    summary_info = self.buffer.summary_block.sum(axis=(0, 1))
                env_summaries = {}
                elapsed_episodes = summary_info[self.env_summary_idx_hash['elapsed_episodes']]

                recent_elapsed_episodes = elapsed_episodes - self.last_elapsed_episodes

                if recent_elapsed_episodes > 0:
                    summary_str = ""
                    for k in self.env_summary_keys:
                        if k !=  'elapsed_episodes':
                            env_summaries[k] = (summary_info[self.env_summary_idx_hash[k]] - getattr(self, 'last_' + k)) / recent_elapsed_episodes
                            setattr(self, 'last_' + k, summary_info[self.env_summary_idx_hash[k]])
                            summary_str += ', {}: {:.2f}'.format(k.replace('_', ' ').title(), env_summaries[k])

                    log.debug('Map: %s%s. (%d)', self.cfg.map_name, summary_str, int(recent_elapsed_episodes))

                    self.last_elapsed_episodes = elapsed_episodes

                    log_infos = {**log_infos, **env_summaries}
            else:
                raise NotImplementedError

        return log_infos

    def save(self):
        """Save policy's actor and critic networks."""
        torch.save(self.policy.state_dict(), str(self.save_dir) + "/model_{}.pt".format(self.policy_version.item()))

    def restore(self):
        """Restore policy's networks from a saved model."""
        self.policy.actor_critic.load_state_dict(torch.load(str(self.model_dir) + '/model.pt'))

    def report(self, infos):
        if not infos or self.replicate_rank != 0:
            return

        data = np.zeros(len(infos), dtype=np.float32)
        data[:] = list(infos.values())
        msg = [self.socket_identity, str(self.policy_id).encode('ascii')] + [k.encode('ascii')
                                                                             for k in infos.keys()] + [data]

        self.task_result_socket.send_multipart(msg)
