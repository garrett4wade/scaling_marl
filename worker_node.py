import math
import multiprocessing
import os
import time
from collections import deque
from queue import Empty

import torch
from torch.multiprocessing import JoinableQueue as TorchJoinableQueue

from system.actor_worker import ActorWorker
from system.policy_worker import PolicyWorker
from system.transmitter import Transmitter
from system.actor_group_manager import ActorGroupManager
from utils.buffer import SharedWorkerBuffer
from utils.timing import Timing
from utils.utils import log, set_global_cuda_envvars, list_child_processes, kill_processes

if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue


class ExperimentStatus:
    SUCCESS, FAILURE, INTERRUPTED = range(3)


torch.multiprocessing.set_sharing_strategy('file_system')


class WorkerNode:
    def __init__(self, cfg, env_fn):

        # we should not use CUDA in the main thread, only on the workers
        set_global_cuda_envvars(cfg)
        self.cfg = cfg

        self.obs_space = cfg.observation_space
        self.share_obs_space = cfg.share_observation_space
        self.action_space = cfg.action_space
        self.num_agents = cfg.num_agents

        self.env_fn = env_fn

        # shared memory allocation
        self.buffer = SharedWorkerBuffer(self.cfg, self.obs_space, self.share_obs_space, self.action_space)

        self.actor_workers = []
        self.policy_workers = []
        self.transmitters = []

        # faster_fifo queue is initialized using BYTES!
        self.report_queue = MpQueue(40 * 1000 * 1000)
        # TODO: policy queue -> policy queues to support PBT
        self.policy_queue = MpQueue()

        self.num_actor_groups = self.cfg.num_actors // self.cfg.actor_group_size
        # TODO: here we only consider actions for policy-sharing
        act_shape = (self.cfg.envs_per_actor // self.cfg.num_splits, self.num_agents, 1)
        self.act_shms = [[
            torch.zeros(act_shape, dtype=torch.int32).share_memory_().numpy() for _ in range(self.cfg.num_splits)
        ] for _ in range(self.cfg.num_actors)]
        self.act_semaphores = [[multiprocessing.Semaphore(0) for _ in range(self.cfg.num_splits)]
                               for _ in range(self.cfg.num_actors)]

        # TODO: initialize env step outputs using config
        # following is just the case of StarCraft2 (policy-sharing environments)
        keys = ['obs', 'share_obs', 'rewards', 'available_actions', 'fct_masks', 'rnn_states', 'rnn_states_critic']
        self.envstep_output_shms = {}
        for k in keys:
            # TODO: some environments may not need available actions and fct_masks
            # if not hasattr(self.buffer, k):
            #     continue

            # buffer storage shape (num_slots, episode_length, num_envs, num_agents, *shape)
            shape = getattr(self.buffer, k).shape[2:]
            self.envstep_output_shms[k] = [[
                torch.zeros(shape, dtype=torch.float32).share_memory_().numpy() for _ in range(self.cfg.num_splits)
            ] for _ in range(self.num_actor_groups)]
        dones_shape = self.buffer.masks.shape[2:]
        self.envstep_output_shms['dones'] = [[
            torch.zeros(dones_shape, dtype=torch.float32).share_memory_().numpy() for _ in range(self.cfg.num_splits)
        ] for _ in range(self.num_actor_groups)]
        self.envstep_output_semaphores = [[multiprocessing.Semaphore(0) for _ in range(self.cfg.num_splits)]
                                          for _ in range(self.cfg.num_actors)]

        self.policy_worker_ready_events = [multiprocessing.Event() for _ in range(self.cfg.num_policy_workers)]

        self.samples_collected = 0

        # currently this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()

        self.report_interval = 5.0  # sec
        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes

        self.throughput_stats = [deque([], maxlen=stat_len) for stat_len in self.avg_stats_intervals]
        self.avg_stats = dict()
        self.stats = dict()  # regular (non-averaged) stats

    def create_actor_worker(self, idx, actor_queue):
        group_idx = idx // self.cfg.actor_group_size
        return ActorWorker(
            self.cfg,
            self.env_fn,
            self.num_agents,
            idx,
            self.buffer,
            task_queue=actor_queue,
            policy_queue=self.policy_queue,
            report_queue=self.report_queue,
            act_shm=self.act_shms[idx],
            act_semaphore=self.act_semaphores[idx],
            envstep_output_shm={k: v[group_idx]
                                for k, v in self.envstep_output_shms.items()},
            envstep_output_semaphore=self.envstep_output_semaphores[idx],
        )

    # noinspection PyProtectedMember
    def init_subset(self, indices, actor_queues):
        """
        Initialize a subset of actor workers (rollout workers) and wait until the first reset() is completed for all
        envs on these workers.

        This function will retry if the worker process crashes during the initial reset.

        :param indices: indices of actor workers to initialize
        :param actor_queues: task queues corresponding to these workers
        :return: initialized workers
        """

        reset_timelimit_seconds = self.cfg.reset_timeout_seconds

        workers = dict()
        last_env_initialized = dict()
        for i in indices:
            w = self.create_actor_worker(i, actor_queues[i])
            w.start_process()
            w.init()
            w.request_reset()
            workers[i] = w
            last_env_initialized[i] = time.time()

        total_num_envs = self.cfg.num_actors * self.cfg.envs_per_actor
        envs_initialized = [0] * self.cfg.num_actors
        workers_finished = set()

        while len(workers_finished) < len(workers):
            failed_worker = -1

            try:
                report = self.report_queue.get(timeout=1.0)

                if 'initialized_env' in report:
                    worker_idx, split_idx = report['initialized_env']
                    last_env_initialized[worker_idx] = time.time()
                    envs_initialized[worker_idx] += 1

                    log.debug(
                        'Progress for %d workers: %d/%d envs initialized...',
                        len(indices),
                        sum(envs_initialized),
                        total_num_envs,
                    )
                elif 'finished_reset' in report:
                    workers_finished.add(report['finished_reset'])
                elif 'critical_error' in report:
                    failed_worker = report['critical_error']
            except Empty:
                pass

            for worker_idx, w in workers.items():
                if worker_idx in workers_finished:
                    continue

                time_passed = time.time() - last_env_initialized[worker_idx]
                timeout = time_passed > reset_timelimit_seconds

                if timeout or failed_worker == worker_idx or not w.process.is_alive():
                    envs_initialized[worker_idx] = 0

                    log.error('Worker %d is stuck or failed (%.3f). Reset!', w.worker_idx, time_passed)
                    log.debug('Status: %r', w.process.is_alive())
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx])
                    new_worker.start_process()
                    new_worker.init()
                    new_worker.request_reset()

                    last_env_initialized[worker_idx] = time.time()
                    workers[worker_idx] = new_worker
                    del stuck_worker

        return workers.values()

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        """
        Initialize all types of workers and start their worker processes.
        """

        actor_queues = [MpQueue(2 * 1000 * 1000) for _ in range(self.cfg.num_actors)]

        if self.cfg.actor_group_size > 1:
            log.info('Initializing group managers...')
            self.group_managers = []
            for idx in range(self.num_actor_groups):
                s = slice(idx * self.cfg.actor_group_size, (idx + 1) * self.cfg.actor_group_size)
                gm = ActorGroupManager(self.cfg, idx, self.policy_queue, self.envstep_output_semaphores[s])
                gm.start()
                self.group_managers.append(gm)

        log.info('Initializing policy workers...')
        for i in range(self.cfg.num_policy_workers):
            policy_worker = PolicyWorker(
                i,
                self.cfg,
                self.obs_space,
                self.share_obs_space,
                self.action_space,
                self.buffer,
                self.policy_queue,
                self.report_queue,
                self.act_shms,
                self.act_semaphores,
                self.envstep_output_shms,
                self.policy_worker_ready_events[i],
            )
            self.policy_workers.append(policy_worker)
            policy_worker.start_process()

        log.info('Initializing actors...')

        # We support actor worker initialization in groups, which can be useful for some envs that
        # e.g. crash when too many environments are being initialized in parallel.
        # Currently the limit is not used since it is not required for any envs supported out of the box,
        # so we parallelize initialization as hard as we can.
        # If this is required for your environment, perhaps a better solution would be to use global locks,
        # like FileLock (see doom_gym.py)
        max_parallel_init = 32  # might be useful to limit this for some envs
        actor_indices = list(range(self.cfg.num_actors))
        for i in range(0, self.cfg.num_actors, max_parallel_init):
            workers = self.init_subset(actor_indices[i:i + max_parallel_init], actor_queues)
            self.actor_workers.extend(workers)

    def init_transmitters(self):
        transmitter_queues = []
        for i in range(self.cfg.num_transmitters):
            transmitter_queues.append(TorchJoinableQueue())

        for idx in range(self.cfg.num_transmitters):
            t = Transmitter(self.cfg, idx, transmitter_queues[i], self.buffer, self.policy_worker_ready_events)
            t.init()
            self.transmitters.append(t)

    def finish_initialization(self):
        """Wait until policy workers are fully initialized."""
        for w in self.policy_workers:
            log.debug('Waiting for policy worker %d to finish initialization...', w.worker_idx)
            w.initialized_event.wait()
            log.debug('Policy worker %d initialized!', w.worker_idx)

    def process_report(self, report):
        """Process stats from various types of workers."""

        if 'samples' in report:
            self.samples_collected += report['samples']

        if 'timing' in report:
            for k, v in report['timing'].items():
                if k not in self.avg_stats:
                    self.avg_stats[k] = deque([], maxlen=50)
                self.avg_stats[k].append(v)

        if 'stats' in report:
            self.stats.update(report['stats'])

    def report(self):
        """
        Called periodically (every X seconds, see report_interval).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """
        now = time.time()
        sample_throughputs = []
        for throughput_stat in self.throughput_stats:
            throughput_stat.append((now, self.samples_collected))
            if len(throughput_stat) > 1:
                past_moment, past_samples = throughput_stat[0]
                sample_throughput = (self.samples_collected - past_samples) / (now - past_moment)
            else:
                sample_throughput = math.nan
            sample_throughputs.append(sample_throughput)

        self.print_stats(sample_throughputs)

    def print_stats(self, sample_throughputs):
        log.debug('Throughput: {:.2f} (10 sec), {:.2f} (1 min), {:.2f} (5 mins). Samples: {}.'.format(
            *sample_throughputs, self.samples_collected))
        mem_stats = ''.join(['{}: {:.2f} MB, '.format(k, v) for k, v in self.stats.items()])[:-2]
        log.debug('Memory statistics: %s', mem_stats)
        timing_stats = ''.join(['{}: {:.4f} s, '.format(k, sum(v) / len(v)) for k, v in self.avg_stats.items()])[:-2]
        log.debug('Timing: %s', timing_stats)

    def _should_terminate(self):
        end = self.total_train_seconds > self.cfg.train_for_seconds

        if self.cfg.benchmark:
            end |= sum(self.samples_collected) >= int(1e6)

        return end

    def run(self):
        """
        This function contains the main loop of the algorithm, as well as initialization/cleanup code.

        :return: ExperimentStatus (SUCCESS, FAILURE, INTERRUPTED). Useful in testing.
        """

        status = ExperimentStatus.SUCCESS

        self.init_transmitters()
        self.init_workers()
        self.finish_initialization()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):
            try:
                while not self._should_terminate():
                    # TODO: termination should refer to task dispatcher through zmq socket
                    try:
                        reports = self.report_queue.get_many(timeout=0.1)
                        for report in reports:
                            self.process_report(report)
                    except Empty:
                        pass

                    if time.time() - self.last_report > self.report_interval:
                        self.report()

                        now = time.time()
                        self.total_train_seconds += now - self.last_report
                        self.last_report = now

            except Exception:
                log.exception('Exception in driver loop')
                status = ExperimentStatus.FAILURE
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected in driver loop, exiting...')
                status = ExperimentStatus.INTERRUPTED

        all_workers = self.actor_workers + self.policy_workers

        child_processes = list_child_processes()

        time.sleep(0.1)
        log.debug('Closing workers...')
        for i, w in enumerate(all_workers):
            w.close()
            time.sleep(0.01)
        for i, w in enumerate(all_workers):
            w.join()
        log.debug('Workers joined!')

        time.sleep(0.1)
        log.debug('Closing Transmitters...')
        for t in self.transmitters:
            t.close()
            time.sleep(0.01)
        for t in self.transmitters:
            t.join()
        log.debug('Transmitters joined!')

        # VizDoom processes often refuse to die for an unidentified reason, so we're force killing them with a hack
        kill_processes(child_processes)

        fps = self.samples_collected / timing.experience
        log.info('Collected %r, FPS: %.1f', self.samples_collected, fps)
        log.info('Timing: %s', timing)

        time.sleep(0.5)
        log.info('Done!')

        return status
